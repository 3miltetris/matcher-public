from src.modules.intake import drive_folder as dfo


def test_folder_name_strips_bad_chars_and_keeps_spaces():
    assert dfo.folder_name('Acme / Robotics:  Inc.') == 'Acme Robotics Inc_INTERNAL'
    assert dfo.folder_name('Acme Robotics') == 'Acme Robotics_INTERNAL'


def _f(i, name):
    return {'id': i, 'name': name}


def test_exact_match_ignores_suffix_case_and_underscores_before_internal():
    folders = [_f('1', 'Acme Robotics_INTERNAL'), _f('2', 'Beta Labs_INTERNAL'),
               _f('3', 'Acme Robotics')]           # not an _INTERNAL folder
    found, tier = dfo.match_existing('ACME Robotics, LLC', folders)
    assert (found['id'], tier) == ('1', 'exact')


def test_ambiguous_exact_creates_new():
    folders = [_f('1', 'Acme_INTERNAL'), _f('2', 'Acme Inc_INTERNAL')]
    assert dfo.match_existing('Acme', folders) == (None, 'ambiguous')


def test_contains_reused_only_when_unique():
    folders = [_f('1', 'Acme Robotics Labs_INTERNAL'), _f('2', 'Beta_INTERNAL')]
    found, tier = dfo.match_existing('Acme Robotics', folders)
    assert (found['id'], tier) == ('1', 'contains')


def test_fuzzy_match_is_never_reused():
    folders = [_f('1', 'Acme Robotix_INTERNAL')]
    assert dfo.match_existing('Acme Robotics', folders) == (None, 'fuzzy')
