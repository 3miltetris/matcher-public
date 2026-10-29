"""
roadmap.pipeline — generate one funding roadmap end to end.

  fetch  →  profile  →  match  →  [rerank]  →  deck  →  upload  →  writeback

Driven by a webhook that names only a contact; every field is re-read from
HubSpot with our token. Runs as a background task (the webhook returns at once),
so the ~2-minute target means *finishes* in ~2 min, not a blocking request.

Idempotent: a retried webhook for the same contact on the same day reuses the
PDF it already built and only re-writes the HubSpot property. A run record is
kept at roadmap/runs/{run_id}.json for observability.

Two modes:
  event     — no website fetch, no LLM re-rank (fastest; for the iPad kiosk)
  instantly — light homepage fetch + a 1-sample re-rank (richer; for campaigns)
"""

import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import src.modules.aspect_profile as ap
import src.modules.profile_build as pb
from src.modules.roadmap import deck, fetch
from src.modules.roadmap import hubspot_roadmap as hr
from src.modules.roadmap import matcher as mt

log = logging.getLogger('roadmap')

BUCKET         = 'cc-matcher-bucket-jeg-v1'
RUN_PREFIX     = 'roadmap/runs/'
PDF_PREFIX     = 'roadmap/pdfs/'
MODE_EVENT     = 'event'
MODE_INSTANTLY = 'instantly'

# A profile needs *something* to read; below this the description is too thin to
# profile and we return no_match rather than hallucinate.
_MIN_INPUT_CHARS = 20


@dataclass
class RoadmapDeps:
    storage_client: object
    bm: object
    embed: callable                      # str -> list[float]
    anth: object                         # anthropic.Anthropic
    hubspot: object                      # intake.hubspot_sync.HubSpot
    anthropic_api_key: str = ''          # for the re-rank call (am.run_rerank)
    bucket: str = BUCKET
    model: str = deck.DECK_MODEL
    assess_defense: bool = True
    target_aspects: int = 4
    max_markets: int = 4
    deck_limit: int = mt.DECK_LIMIT
    notify_fail: callable = None
    now: callable = lambda: datetime.now(timezone.utc)


def production_deps() -> RoadmapDeps:
    import anthropic
    from google.cloud import storage

    from src.modules.Embedding.text_embedder import TextProcessor
    from src.modules.GoogleBucketManager.bucket_manager import BucketManager
    from src.modules.intake import notify
    from src.modules.intake.hubspot_sync import HubSpot

    client   = storage.Client()
    tp       = TextProcessor(api_key=os.environ.get('OPENAI_API_KEY'))
    anth_key = os.environ.get('ANTHROPIC_API_KEY', '')
    return RoadmapDeps(
        storage_client=client,
        bm=BucketManager(BUCKET, client=client),
        embed=tp.get_embedding,
        anth=anthropic.Anthropic(api_key=anth_key),
        hubspot=HubSpot(),
        anthropic_api_key=anth_key,
        notify_fail=notify.step_failed,
    )


# ── Run record ───────────────────────────────────────────────────────────────

def _run_id(key: str, today: date) -> str:
    slug = re.sub(r'[^a-zA-Z0-9]+', '-', key).strip('-')[:60] or 'unknown'
    return f'roadmap_{today:%Y%m%d}_{slug}'


def _load_record(deps: RoadmapDeps, run_id: str) -> dict:
    try:
        obj, _ = deps.bm.download_json(f'{RUN_PREFIX}{run_id}.json')
        return obj or {}
    except Exception:
        return {}


def _save_record(deps: RoadmapDeps, record: dict) -> None:
    try:
        deps.bm.upload_json(f'{RUN_PREFIX}{record["run_id"]}.json', record)
    except Exception as e:
        log.warning('roadmap %s: record save failed: %s', record.get('run_id'), type(e).__name__)


def _pdf_filename(company: str, today: date) -> str:
    slug = re.sub(r'[^A-Za-z0-9]+', '-', company or 'roadmap').strip('-')[:40] or 'roadmap'
    return f'funding-roadmap-{slug}-{today.isoformat()}.pdf'


def _host_pdf(deps: RoadmapDeps, record: dict, pdf: bytes, company: str,
              today: date, public_base: str, warnings: list) -> str:
    """Store the PDF in GCS (durable copy) and publish it to HubSpot Files as the
    public link. Returns the link to write onto the contact — the HubSpot CDN URL
    when the upload succeeds, else our own serve route, so a roadmap is never lost
    to a Files-API hiccup."""
    token = record.get('serve_token') or secrets.token_urlsafe(24)
    blob  = f'{PDF_PREFIX}{token}.pdf'
    deps.storage_client.bucket(deps.bucket).blob(blob).upload_from_string(
        pdf, content_type='application/pdf')
    hub_url, file_id = '', ''
    try:
        hub_url, file_id = hr.upload_public_file(deps.hubspot, pdf,
                                                 _pdf_filename(company, today))
    except Exception as e:
        warnings.append(f'HubSpot Files upload failed ({type(e).__name__}); '
                        'using our serve-route link instead')
    url = hub_url or f'{public_base.rstrip("/")}/api/roadmap/{token}.pdf'
    record.update(serve_token=token, serve_blob=blob, pdf_url=url, hubspot_file_id=file_id)
    return url


# ── Pipeline ─────────────────────────────────────────────────────────────────

def generate_roadmap(deps: RoadmapDeps, *, contact_id: str | None = None,
                     email: str | None = None, mode: str = MODE_EVENT,
                     public_base: str = '') -> dict:
    """Build (or reuse) a roadmap for one contact and write the link back to
    HubSpot. Returns a result dict; never raises — failures are logged, written
    to the contact as roadmap_status='error', and notified."""
    today  = deps.now().date()
    key    = contact_id or email or 'unknown'
    run_id = _run_id(str(key), today)
    record = _load_record(deps, run_id)
    record.setdefault('run_id', run_id)
    record.update(mode=mode, updated_at=deps.now().isoformat(timespec='seconds'))
    warnings: list[str] = []

    try:
        # 1 · fetch form fields + deck (+ homepage on the instantly path)
        contact = hr.get_contact(deps.hubspot, contact_id=contact_id, email=email)
        inp     = hr.form_inputs(contact)
        cid     = inp['contact_id']
        record['company']     = inp['company']
        record['lead_source'] = inp.get('lead_source', '')

        # Reuse an already-built PDF (webhook retry) — only re-write the property.
        if record.get('pdf_url') and record.get('serve_blob'):
            hr.write_roadmap(deps.hubspot, cid, url=record['pdf_url'],
                             status=record.get('status', 'complete'),
                             generated_at=record.get('generated_at', today.isoformat()))
            log.info('roadmap %s: reused existing PDF', run_id)
            return _finish(deps, record, cid, warnings)

        # The deck is fetched with the HubSpot token (fetch guards it to HubSpot
        # hosts), never the anthropic key.
        deck_text, deck_reason = fetch.fetch_deck_text(
            inp['deck_url'], getattr(deps.hubspot, 'token', None))
        if deck_reason:
            warnings.append(f'deck: {deck_reason}')

        site_text = ''
        if mode == MODE_INSTANTLY:
            site_text, site_reason = fetch.fetch_website_text(inp['website'])
            if site_reason:
                warnings.append(f'website: {site_reason}')

        # 2 · build the capability profile
        material = ' '.join(x for x in (inp['description'], deck_text) if x).strip()
        if len(material) < _MIN_INPUT_CHARS and not site_text:
            return _no_input(deps, record, cid, today, public_base, warnings)

        texts = ap.notes_source_texts(material)
        if site_text:
            texts['website'] = ap._cap(site_text, 12000)

        prof_rec, prof_warn = pb.build_profile_from_texts(
            deps.anth, deps.embed,
            company_name=inp['company'] or 'Your company',
            website=inp['website'], state='', texts=texts,
            target_aspects=deps.target_aspects, max_markets=deps.max_markets,
            assess_defense=deps.assess_defense, assess_unexplored=False,
            max_unexplored=0, model=deps.model)
        warnings += prof_warn

        # 3 · match against the (cached) topic index
        index = mt.get_topic_index(deps.storage_client, bucket=deps.bucket)
        candidates, skipped = mt.match_profile(prof_rec, index)
        warnings += skipped

        # 4 · optional re-rank (instantly only)
        reranked = False
        if mode == MODE_INSTANTLY and not candidates.empty:
            candidates = mt.rerank(candidates, deps.anthropic_api_key,
                                   storage_client=deps.storage_client, samples=1)
            reranked = True

        opps = mt.top_opportunities(candidates, limit=deps.deck_limit, reranked=reranked)
        record['opportunities'] = len(opps)

        # 5 · deck narrative + PDF
        narrative, nar_warn = deck.generate_narrative(
            deps.anth, company_name=inp['company'] or 'Your company',
            profile_summary=prof_rec.get('profile_summary', ''),
            aspect_labels=prof_rec.get('aspect_labels', ''),
            opportunities=opps, model=deps.model)
        warnings += nar_warn
        ctx = deck.build_context(
            company_name=inp['company'] or 'Your company',
            profile_summary=prof_rec.get('profile_summary', ''),
            opportunities=opps, narrative=narrative, presentation_date=today)
        pdf = deck.render_pdf(ctx)

        # 6 · publish the PDF (HubSpot Files for the public link, GCS as backup)
        url = _host_pdf(deps, record, pdf, inp['company'] or 'Your company',
                        today, public_base, warnings)
        record.update(status='complete' if opps else 'no_match',
                      generated_at=today.isoformat(), warnings=warnings)
        _save_record(deps, record)

        # 7 · write the link back to HubSpot (the boss's workflow trigger)
        hr.write_roadmap(deps.hubspot, cid, url=url, status=record['status'],
                         generated_at=record['generated_at'])
        log.info('roadmap %s: complete (%d opportunities, status=%s)',
                 run_id, len(opps), record['status'])
        return _finish(deps, record, cid, warnings)

    except Exception as e:
        log.exception('roadmap %s: failed', run_id)
        record.update(status='error', error=f'{type(e).__name__}: {e}'[:500],
                      warnings=warnings)
        _save_record(deps, record)
        _mark_error(deps, contact_id, email, today)
        if deps.notify_fail:
            deps.notify_fail(run_id, record.get('company', '(unknown)'),
                             'roadmap', record['error'])
        return {'run_id': run_id, 'status': 'error', 'error': record['error']}


def _finish(deps: RoadmapDeps, record: dict, contact_id: str, warnings: list) -> dict:
    record['warnings'] = warnings
    _save_record(deps, record)
    return {'run_id': record['run_id'], 'status': record.get('status'),
            'pdf_url': record.get('pdf_url'), 'contact_id': contact_id,
            'opportunities': record.get('opportunities', 0), 'warnings': warnings}


def _no_input(deps: RoadmapDeps, record: dict, contact_id: str, today: date,
              public_base: str, warnings: list) -> dict:
    """No usable description, deck, or site — still produce the empty-state deck
    so the founder gets a courteous response and a follow-up can happen."""
    ctx = deck.build_context(company_name=record.get('company') or 'Your company',
                             profile_summary='', opportunities=[],
                             narrative={'why_now': '', 'items': {}},
                             presentation_date=today)
    pdf = deck.render_pdf(ctx)
    url = _host_pdf(deps, record, pdf, record.get('company') or 'Your company',
                    today, public_base, warnings)
    record.update(status='no_input', generated_at=today.isoformat(), opportunities=0)
    warnings.append('no usable description, deck, or website')
    hr.write_roadmap(deps.hubspot, contact_id, url=url, status='no_input',
                     generated_at=today.isoformat())
    return _finish(deps, record, contact_id, warnings)


def _mark_error(deps: RoadmapDeps, contact_id, email, today: date) -> None:
    """Best-effort error status on the contact, so a failed run is visible in
    HubSpot rather than silent."""
    try:
        cid = contact_id
        if not cid and email:
            cid = hr.get_contact(deps.hubspot, email=email).get('_id')
        if cid:
            hr.write_roadmap(deps.hubspot, cid, url='', status='error',
                             generated_at=today.isoformat())
    except Exception:
        pass


# ── Polling fallback ─────────────────────────────────────────────────────────
#
# When the HubSpot tier has no "Send a webhook" workflow action, a scheduled poll
# drives generation instead. roadmap_status is the queue: a contact with a
# description and no status is pending; claiming sets it to 'processing' so the
# next poll skips it; generate_roadmap sets a terminal status. A claim whose
# generation was killed is recovered once it goes stale.

POLL_STALE_MIN = 15


def _parse_dt(value) -> datetime | None:
    s = str(value or '').strip()
    if not s:
        return None
    for candidate in (s.replace('Z', '+00:00'), s[:10]):
        try:
            dt = datetime.fromisoformat(candidate)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _is_stale(generated_at, now: datetime, stale_min: int) -> bool:
    dt = _parse_dt(generated_at)
    return dt is None or (now - dt) >= timedelta(minutes=stale_min)


def claim_pending(deps: RoadmapDeps, *, limit: int = 10,
                  stale_min: int = POLL_STALE_MIN) -> list[str]:
    """Find up to `limit` contacts needing a roadmap (new submissions first, then
    stale 'processing' claims to recover killed runs), claim each, and return
    their ids. Synchronous and fast: claiming here means the caller can safely run
    generation in the background without two polls picking the same contact."""
    hr.ensure_output_properties(deps.hubspot)
    now = deps.now()
    picked = list(hr.pending_contacts(deps.hubspot, limit=limit))
    if len(picked) < limit:
        picked += [c for c in hr.processing_contacts(deps.hubspot, limit=limit)
                   if _is_stale(c.get(hr.PROP_GENERATED_AT), now, stale_min)]
    ids: list[str] = []
    for c in picked[:limit]:
        cid = c.get('_id')
        if not cid:
            continue
        try:
            hr.claim(deps.hubspot, cid, now.isoformat(timespec='seconds'))
            ids.append(cid)
        except Exception:
            log.exception('roadmap poll: claim failed for %s', cid)
    return ids


def generate_batch(deps: RoadmapDeps, ids: list[str], mode: str = MODE_EVENT,
                   public_base: str = '') -> list[dict]:
    """Generate roadmaps for a claimed batch, one at a time. Each is isolated:
    one failure never stops the rest."""
    results = []
    for cid in ids:
        try:
            results.append(generate_roadmap(deps, contact_id=cid, mode=mode,
                                            public_base=public_base))
        except Exception:
            log.exception('roadmap poll: generation failed for %s', cid)
    return results
