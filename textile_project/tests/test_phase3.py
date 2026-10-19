"""Phase 3: `textile repeat` and `textile tile` against reference_code's
repeat_analyze.py and the deshear / tile-extract prototypes."""
import json
import re
import subprocess
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

from conftest import ROOT, SAMPLES
from textile import cli
from textile import repeat as rp
from textile import tile as tl
from textile.verify import seam_check

FLORAL = SAMPLES / 'ai_floral_allover.png'


@pytest.fixture(scope='module')
def analysis():
    return rp.analyze(FLORAL)


def test_repeat_is_the_scripts_analysis(analysis):
    out = subprocess.run([sys.executable, '-W', 'ignore', str(ROOT / 'reference_code' / 'repeat_analyze.py'), str(FLORAL)],
                         capture_output=True, text=True, check=True).stdout
    old = [tuple(float(v) for v in m) for m in
           re.findall(r'n=\s*(\d+)\s+dy=\s*([-\d.]+)\s+dx=\s*([-\d.]+)\s+score=([\d.]+)', out)]
    assert old == [(c['n'], c['dy'], c['dx'], c['score']) for c in analysis['clusters']]
    assert 'HALF-DROP' in out and analysis['type'] == 'half-drop'
    assert f"~ {analysis['W']:.0f} x {analysis['H']:.0f} px" in out              # 502 x 316
    assert f"drop {analysis['drop']:.0f} px" in out and f"(~{analysis['shear_deg']:.1f} deg)" in out
    assert (analysis['W'], analysis['H'], analysis['drop'], analysis['mirror']) == (501.7, 315.7, 157.9, False)
    assert analysis['avg_match'] == 0.70 and 'mirror nahi' in out


def test_a_panel_print_is_named_and_never_tiled(tmp_path):
    r = rp.analyze(SAMPLES / 'tree_ref.png')
    assert r['type'] == 'panel'
    assert cli.main(['tile', str(SAMPLES / 'tree_ref.png'), '--out', str(tmp_path), '--name', 'tree']) == 1
    assert not list(tmp_path.glob('*_final_*'))           # the analysis is kept, no tile cut
    assert json.loads((tmp_path / 'tree_repeat.json').read_text())['type'] == 'panel'


def test_tile_with_the_prototypes_settings_is_the_prototypes_tile(tmp_path):
    """The prototypes, unchanged but for their input path, on the sample;
    then `textile tile` with their fixed shear and size ranges."""
    d = tmp_path / 'proto'; d.mkdir()
    src = (ROOT / 'reference_code' / 'repeat_deshear_PROTOTYPE.py').read_text()
    (d / 'deshear.py').write_text(src.replace("'/mnt/user-data/uploads/23527.png'", repr(str(FLORAL))))
    for f in ('repeat_tile_extract_PROTOTYPE.py', 'seam_quilt_lib.py'):
        (d / f).write_text((ROOT / 'reference_code' / f).read_text())
    for f in ('deshear.py', 'repeat_tile_extract_PROTOTYPE.py'):
        subprocess.run([sys.executable, f], cwd=d, check=True, capture_output=True)
    out = tmp_path / 'new'
    assert cli.main(['tile', str(FLORAL), '--out', str(out), '--name', 'f', '--size', '1008',
                     '--shear-ratio', repr(29.0 / 315.8), '--w-range', '496:514', '--h-range', '308:326']) == 0
    old = cv2.imread(str(d / 'hd_tile.png'))
    new = cv2.imread(str(out / 'f_tile_native_504x320px.png'))
    assert np.array_equal(old, new)
    ds, valid = tl.deshear(cv2.imread(str(FLORAL)), 29.0 / 315.8)
    assert np.array_equal(ds, cv2.imread(str(d / 'deshear.png'))) and np.array_equal(valid, np.load(d / 'valid.npy'))


def test_tile_by_default_is_seamless_flat_and_verified(tmp_path):
    assert cli.main(['tile', str(FLORAL), '--out', str(tmp_path), '--name', 'f', '--size', '1500',
                     '--colors', '6', '--clean-ground', '30']) == 0
    r = json.loads((tmp_path / 'f_report.json').read_text())
    assert r['repeat']['type'] == 'half-drop' and r['deshear_ratio'] == round(28.4 / 315.7, 4)
    W, H = r['block_source_px']
    assert 494 <= W <= 510 and 308 <= H <= 324                 # the analysis' 502 x 316, +-8
    assert r['size_px'][0] == 1500 and r['verify']['passed'] and r['colors_in_final'] == 6
    assert r['seam']['seamless'] and r['native_seam']['seamless']
    assert (tmp_path / 'f_3x3_preview.png').exists()


def test_upscale_keeps_a_seamless_tile_seamless():
    y, x = np.mgrid[0:90, 0:120]
    t = (127 + 120 * np.stack([np.sin(2 * np.pi * x / 120), np.cos(2 * np.pi * y / 90),
                               np.sin(2 * np.pi * (x / 60 + y / 45))], -1)).astype(np.uint8)
    up = tl.upscale(t, 700)
    assert up.shape == (525, 700, 3) and seam_check(up)['seamless']


def test_deshear_straightens_either_way():
    im = np.zeros((100, 100, 3), np.uint8); im[:, 50] = 255           # a vertical line
    for s in (0.1, -0.1):
        sheared = tl.deshear(im, -s)[0]                                 # lean it
        back, valid = tl.deshear(sheared, s)
        assert valid.any() and back.shape[1] >= sheared.shape[1]
