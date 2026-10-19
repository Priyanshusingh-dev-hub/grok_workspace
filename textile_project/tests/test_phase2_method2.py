"""Phase 2, Method 2: `textile fill --method 2` against the prototype
(reference_code/method2_structure_fill_PROTOTYPE.py) on the tree panel."""
import json

import cv2
import numpy as np
from PIL import Image

from conftest import SAMPLES, TAG
from textile import cli


def _fill(out, *extra):
    return cli.main(['fill', '--method', '2', '--line', str(SAMPLES / 'tree_lineart.png'),
                     '--ref', str(SAMPLES / 'tree_ref.png'), '--out', str(out), '--name', 'tree', *extra])


def test_method2_is_the_prototype_with_the_colours_taken_from_the_reference(prototype2, tmp_path):
    assert _fill(tmp_path) == 0
    old = cv2.imread(str(prototype2 / 'depth.png'))[:, :, ::-1]
    new = np.asarray(Image.open(tmp_path / f'tree_final_{TAG}.png').convert('RGB'))
    assert np.array_equal(old, new)                       # every pixel, the colours included
    assert np.array_equal(np.load(prototype2 / 'cream.npy'), np.all(new == (232, 223, 210), -1))
    r = json.loads((tmp_path / 'tree_report.json').read_text())
    assert (r['ground_hex'], r['motif_hex']) == ('10100F', 'E8DFD2')
    assert r['regions'] == 1337 and r['ground_seeds'] == 75 and r['depth_hist'] == [105, 1187, 45]
    assert [(c['role'], c['coverage_percent']) for c in r['channels']] == [('ground', 68.78), ('motif', 31.22)]
    assert r['verify']['passed']


def test_method2_takes_given_colours(tmp_path):
    assert _fill(tmp_path, '--colors', '1B2A4A,F2E8CF', '--size', '1200') == 0
    r = json.loads((tmp_path / 'tree_report.json').read_text())
    assert (r['ground_hex'], r['motif_hex']) == ('1B2A4A', 'F2E8CF') and r['size_px'] == [1200, 1200]


def test_method2_wants_exactly_two_colours(tmp_path, capsys):
    assert _fill(tmp_path, '--colors', '000000') == 1
    assert 'do rang do' in capsys.readouterr().out
