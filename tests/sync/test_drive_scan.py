"""drive_scan.scan_and_assign — auto-assign, review routing, collision rule, idempotence."""

import src.modules.drive_scan as ds

NORMS = ds.client_norms({
    'Acme Robotics||acme.com': 'Acme Robotics',
    'Beta Systems||beta.com': 'Beta Systems',
})


def _lister(folders_by_section):
    def lister(svc, drive_id, section_id):
        return folders_by_section.get(section_id, [])
    return lister


def _run(folders_by_section, assign_doc=None, sections=None):
    assign_doc = {} if assign_doc is None else assign_doc
    sections = sections or [{'id': 's1', 'name': 'a-h'}]
    return ds.scan_and_assign(None, 'drv', sections, assign_doc, NORMS,
                              lister=_lister(folders_by_section), today='2026-10-20')


def test_auto_assigns_named_folder():
    r = _run({'s1': [{'id': 'f1', 'name': 'Acme Robotics_INTERNAL'}]})
    assert 'f1' in r.new_auto
    assert r.new_auto['f1']['client_key'] == 'Acme Robotics||acme.com'
    assert r.new_auto['f1']['match_type'] == 'auto'
    assert r.assignments['assignments']['f1']['client_key'] == 'Acme Robotics||acme.com'


def test_unmatched_goes_to_review():
    r = _run({'s1': [{'id': 'f9', 'name': 'Totally Unknown Co_INTERNAL'}]})
    assert 'f9' in r.new_unassigned
    assert 'f9' in r.assignments['unassigned']
    assert not r.new_auto


def test_existing_assignment_not_overwritten():
    doc = {'assignments': {'f1': {'client_key': 'Beta Systems||beta.com', 'match_type': 'manual'}}}
    r = _run({'s1': [{'id': 'f1', 'name': 'Acme Robotics_INTERNAL'}]}, assign_doc=doc)
    assert r.assignments['assignments']['f1']['client_key'] == 'Beta Systems||beta.com'  # kept
    assert 'f1' not in r.new_auto


def test_skipped_folder_left_alone():
    doc = {'skipped': {'f1': {'folder_name': 'Acme Robotics_INTERNAL'}}}
    r = _run({'s1': [{'id': 'f1', 'name': 'Acme Robotics_INTERNAL'}]}, assign_doc=doc)
    assert 'f1' not in r.new_auto
    assert 'f1' not in r.assignments['assignments']


def test_two_exact_matches_both_kept():
    r = _run({'s1': [
        {'id': 'f1', 'name': 'Acme Robotics_INTERNAL'},
        {'id': 'f2', 'name': 'Acme Robotics INTERNAL'},
    ]})
    assert set(r.new_auto) == {'f1', 'f2'}   # all exact -> keep both


def test_non_exact_collision_best_wins_rest_review():
    # Two fuzzy matches to Acme: the closer name wins, the other goes to review.
    r = _run({'s1': [
        {'id': 'f1', 'name': 'Acme Robotic_INTERNAL'},     # fuzzy
        {'id': 'f2', 'name': 'Acme Robotics Group_INTERNAL'},  # fuzzy, different score
    ]})
    assert len(r.new_auto) == 1
    assert len(r.new_unassigned) == 1


def test_rescan_is_idempotent():
    folders = {'s1': [{'id': 'f1', 'name': 'Acme Robotics_INTERNAL'}]}
    doc = {}
    _run(folders, assign_doc=doc)
    r2 = _run(folders, assign_doc=doc)       # same doc, second pass
    assert r2.new_auto == {}                  # already assigned, nothing new
    assert doc['assignments']['f1']['client_key'] == 'Acme Robotics||acme.com'
