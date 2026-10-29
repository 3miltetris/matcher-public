"""Topic loading, the in-memory index cache, and a real aligned match."""

import io

import pandas as pd

import src.modules.aspect_profile as ap
import src.modules.roadmap.matcher as mt


def _vec(i: int, dim: int = ap.EMBED_DIM) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


def _put_parquet(storage, path: str, df: pd.DataFrame) -> None:
    """Write a parquet blob straight through the fake (which has no
    upload_from_file, the method BucketManager.upload_file would use)."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    storage.bucket('test-bucket').blob(path).upload_from_string(
        buf.getvalue(), content_type='application/octet-stream')


def _topics_df():
    return pd.DataFrame([
        {'topic_number': 'T0', 'title': 'Autonomous sensors SBIR', 'agency': 'ARMY',
         'grant_summary': 'Compact sensors for robots.', 'due_date': '2099-06-01',
         'source': 'https://x/0', 'embeddings': _vec(0)},
        {'topic_number': 'T1', 'title': 'Quantum optics grant', 'agency': 'AFRL',
         'grant_summary': 'Photonics research.', 'due_date': '2099-06-01',
         'source': 'https://x/1', 'embeddings': _vec(1)},
    ])


def _profile_record():
    aspects = [{'label': 'MEMS sensing', 'kind': 'capability',
                'text': 'MEMS gas and acoustic sensors', 'keywords': 'sensors', 'markets': []}]
    return ap.build_profile_record(
        company_key=ap.company_key({'company_name': 'Acme', 'companyWebsite': 'acme.com'}),
        company_name='Acme', website='acme.com', profile_summary='Sensors co.',
        aspects=aspects, vectors=[_vec(0)], sources_used=['notes'],
        fingerprint='fp', model='test')


def test_load_topic_frame_stamps_agency(storage):
    _put_parquet(storage, 'data/all-topics/processed/DOD/t.parquet', _topics_df())
    df = mt.load_topic_frame(storage)
    assert len(df) == 2
    assert set(df['broad_agency']) == {'DOD'}
    assert 'grant_summary' in df.columns


def test_match_profile_finds_aligned_topic(storage):
    _put_parquet(storage, 'data/all-topics/processed/DOD/t.parquet', _topics_df())
    index = mt.build_topic_index(storage)
    assert index.n_topics == 2
    cands, skipped = mt.match_profile(_profile_record(), index, threshold=0.78)
    assert not cands.empty
    # The aligned topic (T0) is the top hit.
    top = cands.sort_values('aspect_score', ascending=False).iloc[0]
    assert top['topic_number'] == 'T0'
    assert top['aspect_score'] >= 0.78


def test_top_opportunities_limit_and_order(storage):
    _put_parquet(storage, 'data/all-topics/processed/DOD/t.parquet', _topics_df())
    index = mt.build_topic_index(storage)
    cands, _ = mt.match_profile(_profile_record(), index, threshold=0.0)
    opps = mt.top_opportunities(cands, limit=1)
    assert len(opps) == 1
    assert opps[0]['topic_number'] == 'T0'


def test_get_topic_index_caches(storage, monkeypatch):
    _put_parquet(storage, 'data/all-topics/processed/DOD/t.parquet', _topics_df())
    # Prime the cache, then make a second call that must NOT re-read GCS.
    mt.get_topic_index(storage, force=True)
    calls = {'n': 0}
    orig = mt.build_topic_index
    monkeypatch.setattr(mt, 'build_topic_index',
                        lambda *a, **k: (calls.__setitem__('n', calls['n'] + 1), orig(*a, **k))[1])
    mt.get_topic_index(storage, ttl_s=3600)   # within TTL → cached
    assert calls['n'] == 0


def test_empty_index_returns_no_match():
    empty = mt.TopicIndex.__new__(mt.TopicIndex)
    import numpy as np
    empty.matrix = np.zeros((0, ap.EMBED_DIM), np.float32)
    empty.meta = pd.DataFrame()
    cands, skipped = mt.match_profile(_profile_record(), empty)
    assert cands.empty and skipped
