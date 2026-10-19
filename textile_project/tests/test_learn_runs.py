"""The tool's memory keeps runs apart by id: LoomLab's word on a run lands on that run, even beside another run
of the same name in the same second; rows without an id read as before."""
from textile import learn

R = {'name': 'design', 'size_px': [1000, 1000], 'areas': 300, 'missed': 0, 'tiny': 0, 'match': 99.9, 'line_mm': 0.17,
     'tried': [(0.17, 99.9, 300)], 'inks': [('F3E8CA', 'cream', 70.0), ('1A2F4C', 'navy', 30.0)], 'grain': 0.6,
     'woven': False, 'colours_limit': 8, 'detail': 'normal', 'snapped': {}}


def test_a_word_by_run_id_lands_on_that_run_only(tmp_path, monkeypatch):
    monkeypatch.setenv('TEXTILE_LEARN_LOG', str(tmp_path / 'log.jsonl'))
    assert learn.record(R, 'x.png', {}, [], run='a' * 32)
    assert learn.record(dict(R, areas=900), 'x.png', {}, [], run='b' * 32)      # same name, same second
    hit = learn.feedback('design', 'bad', 'lines too thick', run='a' * 32, extra={'by': 'LoomLab'})
    assert hit['run'] == 'a' * 32 and hit['features']['areas_per_mpx'] == 300.0     # the first run, not the latest
    rows = learn._read()
    assert rows[-1]['run'] == 'a' * 32 and rows[-1]['by'] == 'LoomLab'
    v = learn._verdicts(rows)
    runs = [x for x in rows if x['event'] == 'run']
    assert learn._verdict_of(v, runs[0])[0] == 'bad' and learn._verdict_of(v, runs[1]) is None
    assert learn.feedback('design', 'good', run='c' * 32) is None                # an unknown id: no word anywhere
    assert learn.feedback('design', 'good')['run'] == 'b' * 32                   # by name: the latest, as before
    assert '2 run, 2 par aapki raay' in learn.summary()


def test_a_half_line_from_a_crash_does_not_swallow_the_next_row(tmp_path, monkeypatch):
    monkeypatch.setenv('TEXTILE_LEARN_LOG', str(tmp_path / 'log.jsonl'))
    (tmp_path / 'log.jsonl').write_text('{"half a line', encoding='utf-8')
    assert learn.record(R, 'x.png', {}, [])
    assert [x['event'] for x in learn._read()] == ['run']
