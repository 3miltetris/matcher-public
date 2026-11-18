"""
Contact search index — the store behind search_contacts
--------------------------------------------------------
data/all-contacts/ is ~3.7 GB, and ~79% of it is the embedding vectors
themselves (measured on an apollo parquet: 16.6 of 21 MB). No amount of
parallelism makes downloading that per query fit inside the claude.ai connector
timeout, so this module condenses each source folder into a compact index —
unit-normalised float16 vectors plus a small metadata table, deduped per company
within each blob — that search_contacts downloads once per build and then keeps
cached in the server process.

Layout (deliberately NOT under data/all-contacts/, which every contact reader
enumerates as a source):

    data/contact-index/{source}/manifest.json           -> the live build
    data/contact-index/{source}/{build_id}/vectors.npy  (n x dim float16)
    data/contact-index/{source}/{build_id}/meta.parquet (n rows, + `blob`)

The manifest records the {blob name: generation} map the build covered. The
query side uses the index only for blobs whose generation still matches and
scans anything else live, so a contact import that lands after the nightly build
is still searchable immediately. Builds are versioned and the manifest is written
LAST, so a query can never pair one build's vectors with another's metadata.

Build: python -m src.modules.tools.contact_index [--source apollo] [--force]
(the contact-index-job Cloud Run job runs it nightly; a source whose blobs are
unchanged since its last build is skipped).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.modules.tools import _common as cm, search as s

BUCKET = s.BUCKET
CONTACTS_PREFIX = s._CONTACTS_PREFIX
INDEX_PREFIX = 'data/contact-index/'
_META_COLS = ('company_key', 'company_name', 'website', 'industry', 'summary', 'blob')
_SCORE_CHUNK = 50_000          # rows per float16 -> float32 matmul slice
# Bump when the build output changes shape or meaning. A manifest of an older
# format is rebuilt even though its blobs are unchanged (2: dedup keyed on the
# website host, plus a per-source hosts.json for cross-folder `also_in`).
INDEX_FORMAT = 2

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
_LOAD_LOCKS: dict = {}         # one per cache key: a miss is downloaded once


def _manifest_path(source: str) -> str:
    return f'{INDEX_PREFIX}{source}/manifest.json'


# ── build ─────────────────────────────────────────────────────────────────────

def load_manifest(gcs_client, source: str, *, bucket: str = BUCKET) -> dict | None:
    blob = gcs_client.bucket(bucket).get_blob(_manifest_path(source))
    if blob is None:
        return None
    try:
        return json.loads(blob.download_as_bytes())
    except Exception:
        return None


def build_source(gcs_client, source: str, *, bucket: str = BUCKET,
                 force: bool = False) -> dict:
    """(Re)build one source folder's index. Skipped when the folder's blobs are
    exactly the ones the current manifest covers (unless `force`). Blobs that
    fail to read are left OUT of the manifest's blob map, so the query side keeps
    scanning them live instead of treating them as covered."""
    prefix = f'{CONTACTS_PREFIX}{source}/'
    blobs = [b for b in gcs_client.list_blobs(bucket, prefix=prefix)
             if b.name.endswith('.parquet')]
    current = {b.name: str(b.generation) for b in blobs}
    old = load_manifest(gcs_client, source, bucket=bucket)
    if (not force and old and old.get('blobs') == current
            and old.get('format') == INDEX_FORMAT):
        return {'source': source, 'skipped': 'up to date'}

    parts: list[tuple[np.ndarray, list[dict]]] = []
    covered: dict[str, str] = {}
    errors: list[str] = []
    for b in blobs:
        status, table = s._read_contact_blob(b)
        if status == 'error':
            errors.append(b.name)
            continue
        covered[b.name] = str(b.generation)
        if status != 'searched':
            continue
        mat, rowpos = s._embeddings_matrix(table)
        if mat.shape[0] == 0:
            continue
        record = s._contact_record(table, source)
        seen, keep_i, rows = set(), [], []
        for i, pos in enumerate(rowpos):
            rec = record(pos)
            if rec is None:
                continue
            ident = s._dedup_identity(rec['company_name'], rec['website'])
            if ident in seen:                 # one row per company per blob
                continue
            seen.add(ident)
            rec['blob'] = b.name
            keep_i.append(i)
            rows.append(rec)
        if keep_i:
            sub = mat[keep_i]
            norms = np.linalg.norm(sub, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            parts.append(((sub / norms).astype(np.float16), rows))
        del table, mat

    # Keep only the modal vector width, so one odd file cannot break the vstack.
    dims = [p[0].shape[1] for p in parts]
    dim = max(set(dims), key=dims.count) if dims else 0
    parts = [p for p in parts if p[0].shape[1] == dim]

    build_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    bucket_obj = gcs_client.bucket(bucket)
    n = 0
    hosts = sorted({s._site_host(r['website']) for p in parts for r in p[1]} - {''})
    if parts:
        vecs = np.vstack([p[0] for p in parts])
        rows = [r for p in parts for r in p[1]]
        n = int(vecs.shape[0])
        buf = io.BytesIO()
        np.save(buf, vecs, allow_pickle=False)
        bucket_obj.blob(f'{INDEX_PREFIX}{source}/{build_id}/vectors.npy') \
            .upload_from_string(buf.getvalue(), content_type='application/octet-stream')
        meta = pa.table({c: [r.get(c, '') for r in rows] for c in _META_COLS})
        mbuf = io.BytesIO()
        pq.write_table(meta, mbuf)
        bucket_obj.blob(f'{INDEX_PREFIX}{source}/{build_id}/meta.parquet') \
            .upload_from_string(mbuf.getvalue(), content_type='application/octet-stream')
        bucket_obj.blob(f'{INDEX_PREFIX}{source}/{build_id}/hosts.json')             .upload_from_string(json.dumps(hosts), content_type='application/json')

    manifest = {
        'format': INDEX_FORMAT,
        'source': source,
        'build_id': build_id if parts else None,
        'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'model': 'text-embedding-ada-002',
        'dim': dim,
        'has_embeddings': bool(parts),
        'rows': n,                     # internal only; never returned by a tool
        'blobs': covered,
    }
    # Written LAST: the switch to the new build is this one write.
    bucket_obj.blob(_manifest_path(source)).upload_from_string(
        json.dumps(manifest), content_type='application/json')

    # Drop builds older than the previous one (keep it for in-flight readers).
    keep_ids = {build_id, (old or {}).get('build_id')}
    for blob in gcs_client.list_blobs(bucket, prefix=f'{INDEX_PREFIX}{source}/'):
        seg = blob.name[len(f'{INDEX_PREFIX}{source}/'):].split('/', 1)
        if len(seg) == 2 and seg[0] not in keep_ids:
            try:
                blob.delete()
            except Exception:
                pass
    return {'source': source, 'build_id': manifest['build_id'], 'rows': n,
            'blobs': len(covered), 'errors': errors}


def build_all(gcs_client, *, bucket: str = BUCKET, force: bool = False,
              sources: list[str] | None = None) -> list[dict]:
    names = sources or cm.list_subfolders(gcs_client, CONTACTS_PREFIX, bucket=bucket)
    out = []
    for src in names:
        t0 = time.monotonic()
        try:
            res = build_source(gcs_client, src, bucket=bucket, force=force)
        except Exception as e:  # noqa: BLE001 — one source must not stop the rest
            res = {'source': src, 'error': f'{type(e).__name__}: {e}'}
        res['seconds'] = round(time.monotonic() - t0, 1)
        print(json.dumps(res), flush=True)
        out.append(res)
    return out


# ── query side ────────────────────────────────────────────────────────────────

def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
        _LOAD_LOCKS.clear()


def _npy_from_bytes(data: bytes) -> np.ndarray:
    """A read-only array VIEWING `data` (no copy). np.load(BytesIO(...)) copies,
    so it briefly holds the index twice — ~1.1 GB for apollo alone."""
    buf = io.BytesIO(data)
    version = np.lib.format.read_magic(buf)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(buf)
    else:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(buf)
    if fortran or dtype.hasobject:
        return np.load(io.BytesIO(data), allow_pickle=False)
    count = int(np.prod(shape)) if shape else 1
    return np.frombuffer(data, dtype=dtype, count=count, offset=buf.tell()).reshape(shape)


def _cached(key, gen: str, loader):
    """Single-flight cache: the first caller to miss on `key` runs `loader()`;
    concurrent callers for the same key wait for it instead of each downloading
    the same index (the boot prewarm and a first query would otherwise race)."""
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and hit['gen'] == gen:
            return hit
        lock = _LOAD_LOCKS.setdefault(key, threading.Lock())
    with lock:
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
        if hit and hit['gen'] == gen:
            return hit
        entry = loader()
        if entry is not None:
            entry['gen'] = gen
            with _CACHE_LOCK:
                _CACHE[key] = entry
        return entry


def _hosts_of_meta(meta) -> frozenset:
    return frozenset(s._site_host(w) for w in meta.column('website').to_pylist()) - {''}


def load_index(gcs_client, source: str, *, bucket: str = BUCKET) -> dict | None:
    """The cached index entry for a source, reloaded only when its manifest's
    generation changes (i.e. after a rebuild). None when there is no index or it
    is unreadable — the caller then scans that source live."""
    mblob = gcs_client.bucket(bucket).get_blob(_manifest_path(source))
    if mblob is None:
        return None

    def loader():
        try:
            manifest = json.loads(mblob.download_as_bytes())
            vecs = meta = codes = names = None
            hosts = frozenset()
            if manifest.get('has_embeddings') and manifest.get('build_id'):
                base = f"{INDEX_PREFIX}{source}/{manifest['build_id']}/"
                bkt = gcs_client.bucket(bucket)
                vecs = _npy_from_bytes(bkt.blob(base + 'vectors.npy').download_as_bytes())
                meta = pq.read_table(io.BytesIO(bkt.blob(base + 'meta.parquet').download_as_bytes()))
                if vecs.shape[0] != meta.num_rows:
                    return None
                names, codes = np.unique(np.asarray(meta.column('blob').to_pylist(), dtype=object),
                                         return_inverse=True)
                hosts = _hosts_of_meta(meta)
        except Exception:
            return None
        return {'manifest': manifest, 'vecs': vecs, 'meta': meta,
                'blob_names': names, 'blob_codes': codes, 'hosts': hosts}

    return _cached(('index', id(gcs_client), bucket, source), str(mblob.generation), loader)


def load_hosts(gcs_client, source: str, *, bucket: str = BUCKET) -> frozenset:
    """Just a source's set of website hosts — a few MB, for filling `also_in`
    across folders a search was not scoped to, without loading their vectors.
    Reuses a loaded full index when there is one. Empty when unavailable."""
    mblob = gcs_client.bucket(bucket).get_blob(_manifest_path(source))
    if mblob is None:
        return frozenset()
    gen = str(mblob.generation)
    with _CACHE_LOCK:
        full = _CACHE.get(('index', id(gcs_client), bucket, source))
    if full and full['gen'] == gen:
        return full['hosts']

    def loader():
        try:
            manifest = json.loads(mblob.download_as_bytes())
            if not (manifest.get('has_embeddings') and manifest.get('build_id')):
                return {'hosts': frozenset()}
            blob = gcs_client.bucket(bucket).get_blob(
                f"{INDEX_PREFIX}{source}/{manifest['build_id']}/hosts.json")
            if blob is None:
                return None              # pre-format-2 build: skip until rebuilt
            return {'hosts': frozenset(json.loads(blob.download_as_bytes()))}
        except Exception:
            return None

    entry = _cached(('hosts', id(gcs_client), bucket, source), gen, loader)
    return entry['hosts'] if entry else frozenset()


def score_index(entry: dict, source: str, qvec, valid_blobs: set, keep: int):
    """(candidates, any_rows): the top `keep` index rows of `valid_blobs` by
    cosine against `qvec`, in search_contacts' candidate shape."""
    vecs, meta = entry['vecs'], entry['meta']
    if vecs is None or not valid_blobs:
        return [], False
    valid_codes = np.array([i for i, nm in enumerate(entry['blob_names'])
                            if nm in valid_blobs], dtype=np.int64)
    mask = np.isin(entry['blob_codes'], valid_codes)
    if not mask.any():
        return [], False
    q = np.asarray(qvec, dtype=np.float32)
    if q.shape[0] != vecs.shape[1]:
        return [], True
    q = q / (np.linalg.norm(q) or 1.0)
    scores = np.empty(vecs.shape[0], dtype=np.float32)
    for start in range(0, vecs.shape[0], _SCORE_CHUNK):
        scores[start:start + _SCORE_CHUNK] = vecs[start:start + _SCORE_CHUNK].astype(np.float32) @ q
    scores[~mask] = -np.inf
    k = min(int(keep), int(mask.sum()))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    rows = meta.take(pa.array(top)).to_pylist()
    out = []
    for r, i in zip(rows, top):
        out.append({
            'company_key': r['company_key'], 'company_name': r['company_name'],
            'website': r['website'], 'source': source, 'industry': r['industry'],
            'summary': r['summary'], 'score': round(float(scores[i]), 3),
        })
    return out, True


# ── CLI (the contact-index-job entrypoint) ────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Build the search_contacts index.')
    ap.add_argument('--source', action='append', help='one source folder (repeatable)')
    ap.add_argument('--force', action='store_true', help='rebuild even if unchanged')
    ap.add_argument('--bucket', default=BUCKET)
    args = ap.parse_args(argv)
    from google.cloud import storage
    results = build_all(storage.Client(), bucket=args.bucket, force=args.force,
                        sources=args.source)
    failed = [r for r in results if r.get('error')]
    print(json.dumps({'sources': len(results), 'failed': [r['source'] for r in failed]}),
          flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
