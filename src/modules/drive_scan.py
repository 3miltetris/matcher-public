"""
Drive scan + auto-assign (Phase 2.1)
------------------------------------
The shared-drive scan and fuzzy auto-assign used to live inside
views/drive_sync.py, so nothing unattended could discover new folders. Lifted
here verbatim (same `_INTERNAL` normalisation, same exact → containment → fuzzy
tiers via folder_match, same collision rule) so both the view and the scheduled
drive-sync-job call one implementation.

Streamlit-free and injected: the caller passes the Drive service (the view its
own, the job one built from ADC) and the lister is overridable for tests. The
rules the view relied on are preserved: existing assignments and skipped folders
are never overwritten, and a folder already assigned or skipped is left alone.
"""

from dataclasses import dataclass, field
from datetime import date

from src.modules import drive_client, folder_match


@dataclass
class ScanResult:
    assignments: dict                                    # the mutated assign_doc
    new_auto: dict = field(default_factory=dict)         # folder_id -> assignment entry
    new_unassigned: dict = field(default_factory=dict)   # folder_id -> review entry
    folders_seen: int = 0


def client_norms(name_by_key: dict[str, str]) -> dict[str, str]:
    """{client_key: company_name} -> {client_key: normalized name} for matching."""
    return {k: folder_match.normalize(v) for k, v in name_by_key.items()}


def scan_and_assign(
    svc, drive_id: str, sections: list[dict], assign_doc: dict,
    norms: dict[str, str], *, lister=drive_client.list_child_folders,
    today: str | None = None,
) -> ScanResult:
    """Scan `sections` ([{id, name}, ...]) of `drive_id`, auto-assign each new
    `{Client}_INTERNAL` folder to a client by name, and file the rest for review.
    Mutates `assign_doc` in place (keys `assignments`/`unassigned`/`skipped`) and
    returns the deltas. `norms` is `{client_key: normalized_name}`."""
    today = today or date.today().isoformat()
    assign_doc.setdefault('assignments', {})
    assign_doc.setdefault('unassigned', {})
    assign_doc.setdefault('skipped', {})

    newly_matched: dict[str, list[tuple[str, str, float, dict]]] = {}
    new_auto: dict[str, dict] = {}
    new_unassigned: dict[str, dict] = {}
    seen = 0

    for sec in sections:
        for folder in lister(svc, drive_id, sec['id']):
            fid = folder['id']
            seen += 1
            if fid in assign_doc['assignments'] or fid in assign_doc['skipped']:
                continue
            key, tier, score = folder_match.match_folder(
                folder_match.normalize(folder['name']), norms)
            meta = {'folder_name': folder['name'], 'section': sec['name']}
            if key:
                newly_matched.setdefault(key, []).append((fid, tier, score, meta))
            elif fid not in assign_doc['unassigned']:
                entry = {**meta, 'first_seen': today,
                         'note': f'no match >= {folder_match.FUZZY_THRESHOLD} (best {score:.2f})'}
                assign_doc['unassigned'][fid] = entry
                new_unassigned[fid] = entry

    # Collision rule (identical to the view): several new folders matching one
    # client are all kept only when every match is exact; otherwise the best
    # score wins and the rest go to review. A non-exact match to a client that
    # already has a folder assigned goes to review too.
    for key, matches in newly_matched.items():
        already_assigned = any(a.get('client_key') == key
                               for a in assign_doc['assignments'].values())
        all_exact = all(t == 'exact' for _, t, _, _ in matches)
        if all_exact:
            keep = matches
        elif already_assigned:
            keep = []
        else:
            keep = [max(matches, key=lambda m: m[2])]
        kept_ids = {m[0] for m in keep}
        for fid, tier, score, meta in matches:
            if fid in kept_ids:
                entry = {**meta, 'client_key': key, 'match_type': 'auto',
                         'assigned_at': today}
                assign_doc['assignments'][fid] = entry
                assign_doc['unassigned'].pop(fid, None)
                new_auto[fid] = entry
            else:
                entry = {**meta, 'first_seen': today,
                         'note': f'{tier} match ({score:.2f}) to already-matched client — review'}
                assign_doc['unassigned'][fid] = entry
                new_unassigned[fid] = entry

    return ScanResult(assign_doc, new_auto, new_unassigned, seen)
