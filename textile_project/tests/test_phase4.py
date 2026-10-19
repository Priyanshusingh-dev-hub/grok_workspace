"""Phase 4: `textile make` - original designs from a JSON config, flat
colours, seamless (all-over: both ways; panel: up and down)."""
import json

import numpy as np
import pytest
from PIL import Image

from conftest import ROOT
from textile import cli
from textile import make as mk
from textile.verify import seam_check

EXAMPLES = sorted((ROOT / 'examples').glob('*.json'))


def small(cfg, px=900):
    c = json.loads(json.dumps(cfg))
    w, h = c.get('size_px') or [round(v * c.get('dpi', 300)) for v in c['size_inch']]
    c['size_px'] = [px, round(px * h / w)]
    c['dpi'] = 300 * px // w or 1                       # the same design, drawn smaller
    return c


@pytest.mark.parametrize('path', EXAMPLES, ids=lambda p: p.stem)
def test_examples_are_flat_and_seamless(path):
    cfg = small(mk.load(path))
    index, pal, names = mk.make(cfg, log=lambda *a: None)
    rgb = pal[index]
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) == len(pal)   # no blend pixel, no empty colour
    seam = seam_check(rgb)
    if cfg.get('repeat') == 'panel':
        assert seam['top_bottom']['seamless']
    else:
        assert seam['seamless']


def test_make_writes_a_verified_package(tmp_path):
    assert cli.main(['make', str(ROOT / 'examples' / 'elephant_panel.json'), '--out', str(tmp_path)]) == 0
    r = json.loads((tmp_path / 'elephant_panel_report.json').read_text())
    assert r['size_px'] == [5400, 2700] and r['print_size_inch'] == [18.0, 9.0]
    assert r['verify']['passed'] and r['repeat'] == 'panel' and r['colors_in_final'] == 4
    assert (tmp_path / 'elephant_panel_repeat_preview.png').exists()


def test_the_same_config_draws_the_same_design():
    cfg = small(mk.load(ROOT / 'examples' / 'floral_allover.json'), 600)
    a, b = mk.make(cfg, log=lambda *a: None), mk.make(cfg, log=lambda *a: None)
    assert np.array_equal(a[0], b[0])
    cfg['seed'] = 99
    assert not np.array_equal(a[0], mk.make(cfg, log=lambda *a: None)[0])


def test_scatter_keeps_clear_of_the_main_motifs():
    cfg = {'size_px': [600, 600], 'dpi': 100, 'palette': {'g': '000000', 'a': 'FF0000', 'b': '00FF00'}, 'ground': 'g',
           'layers': [{'layout': 'straight', 'cols': 2, 'rows': 2, 'motif': 'dots', 'size': 30, 'colors': ['a']},
                      {'layout': 'scatter', 'per_sq_inch': 40, 'min_gap': 4, 'motif': 'dots', 'size': 2, 'colors': ['b']}]}
    index, pal, names = mk.make(cfg, log=lambda *a: None)
    green = names.index('b')
    y, x = np.nonzero(index == green)
    for cx in (150, 450):
        for cy in (150, 450):
            assert ((x - cx) ** 2 + (y - cy) ** 2).min() > 59 ** 2      # the 30 mm dots (r 59 px) stay clean


@pytest.mark.parametrize('bad,word', [
    ({'palette': {'g': 'zz0000', 'a': 'FF0000'}}, 'hex'),
    ({'ground': 'nahi'}, 'ground'),
    ({'layers': [{'layout': 'straight', 'motif': 'unicorn', 'colors': ['a']}]}, 'unicorn'),
    ({'layers': [{'layout': 'straight', 'motif': 'dots', 'colors': ['purple']}]}, 'purple'),
])
def test_a_bad_config_is_told_plainly(tmp_path, capsys, bad, word):
    cfg = {'size_px': [300, 300], 'palette': {'g': '000000', 'a': 'FF0000'}, 'ground': 'g',
           'layers': [{'layout': 'straight', 'motif': 'dots', 'colors': ['a']}]} | bad
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    assert cli.main(['make', str(tmp_path / 'c.json'), '--out', str(tmp_path / 'o')]) == 1
    out = capsys.readouterr().out
    assert out.startswith('STOP') and word in out
