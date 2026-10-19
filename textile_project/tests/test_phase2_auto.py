"""Phase 2: `textile fill --method 3` (a reference that has drifted) and
`--method auto` (Method 1, judged; Method 3 or 2 when it fails)."""
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from conftest import SAMPLES, TAG
from textile import cli
from textile import fill_method1 as m1
from textile import fill_method3 as m3
from textile import palette as pl
from textile.verify import coverage_diff


def bend(src, dst, most=80, seed=2):
    """The reference bent by a smooth random field, up to `most` px: the same
    drawing, drifted, as a re-generated reference often is."""
    img = cv2.imread(str(src)); H, W = img.shape[:2]
    rng = np.random.default_rng(seed)

    def field():
        f = cv2.resize(rng.normal(0, 1, (5, 5)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        return f / np.abs(f).max() * most
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    cv2.imwrite(str(dst), cv2.remap(img, gx + field(), gy + field(), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    return dst


def same(a, truth):
    """Share of pixels in the truth's colour (a's colours taken as the nearest
    truth colour: a bent reference is not flat, so k-means finds its inks a
    point or two off)."""
    tp = np.unique(truth.reshape(-1, 3), axis=0).astype(int)
    ap, inv = np.unique(a.reshape(-1, 3), axis=0, return_inverse=True)
    snap = tp[((ap[:, None].astype(int) - tp[None]) ** 2).sum(2).argmin(1)]
    return float((snap[inv.ravel()].reshape(truth.shape) == truth).all(-1).mean())


def rgb(f):
    index, pal, _ = pl.merge_stray(f.index, f.pal)
    return pal[index]


@pytest.fixture(scope='module')
def bent(tmp_path_factory):
    return bend(SAMPLES / 'floral_ref.png', tmp_path_factory.mktemp('bent') / 'floral_bent.png')


@pytest.fixture(scope='module')
def truth():
    """Method 1 on the floral's own reference, at 1500 px (quick)."""
    return rgb(m1.fill(str(SAMPLES / 'floral_lineart.png'), str(SAMPLES / 'floral_ref.png'), 1500, log=lambda *a: None))


def test_method3_lays_a_drifted_reference_back_onto_the_line_art(bent, truth):
    line = str(SAMPLES / 'floral_lineart.png')
    drifted = rgb(m1.fill(line, str(bent), 1500, force=True, log=lambda *a: None))
    f = m3.fill(line, str(bent), 1500, log=lambda *a: None)
    assert f.alignment_before < 0.8 and f.alignment_score > 0.95
    assert same(drifted, truth) < 0.92 < 0.98 < same(rgb(f), truth)


def test_method3_on_an_aligned_reference_is_method1(truth):
    f = m3.fill(str(SAMPLES / 'floral_lineart.png'), str(SAMPLES / 'floral_ref.png'), 1500, log=lambda *a: None)
    assert same(rgb(f), truth) > 0.999


def test_coverage_diff_is_half_the_summed_share_differences():
    pal = np.array([[0, 0, 0], [255, 255, 255]], np.uint8)
    ref = np.zeros((10, 10, 3), np.uint8); ref[:, :3] = 255           # 30% white
    index = np.zeros((10, 10), np.uint8); index[:, :5] = 1             # 50% white
    assert coverage_diff(ref, index, pal) == pytest.approx(20.0)


def _auto(out, name, *extra, ref=None):
    return cli.main(['fill', '--method', 'auto', '--line', str(SAMPLES / f'{name}_lineart.png'),
                     '--ref', str(ref or SAMPLES / f'{name}_ref.png'), '--out', str(out), '--name', name, *extra])


@pytest.mark.parametrize('name,extra,diff', [('floral', [], 1.7),
                                             ('star', ['--max-colors', '6', '--line-color', '120F06'], 3.2)])
def test_auto_keeps_method1_when_it_fits(tmp_path, name, extra, diff):
    assert _auto(tmp_path, name, *extra) == 0
    r = json.loads((tmp_path / f'{name}_report.json').read_text())
    assert r['method'] == 1 and r['auto']['chosen'] == 1 and r['auto']['coverage_diff'] == diff


def test_auto_sends_the_tree_panel_to_method2(prototype2, tmp_path):
    """Method 1 leaves the tree's motifs out (13.7+ points off), Method 3 finds
    nothing to shift to (a different drawing), so Method 2: the prototype's
    output, pixel for pixel, with the reason in the report."""
    assert _auto(tmp_path, 'tree') == 0
    r = json.loads((tmp_path / 'tree_report.json').read_text())
    assert r['method'] == 2 and r['auto']['chosen'] == 2 and r['auto']['only_two']
    assert r['auto']['method1']['coverage_diff'] > 6 and r['auto']['method3']['coverage_diff'] > 6
    old = cv2.imread(str(prototype2 / 'depth.png'))[:, :, ::-1]
    assert np.array_equal(old, np.asarray(Image.open(tmp_path / f'tree_final_{TAG}.png').convert('RGB')))


def test_auto_sends_a_drifted_reference_to_method3(tmp_path, bent):
    assert _auto(tmp_path, 'floral', '--size', '1500', ref=bent) == 0
    r = json.loads((tmp_path / 'floral_report.json').read_text())
    assert r['method'] == 3 and r['auto']['chosen'] == 3
    assert r['auto']['method1']['coverage_diff'] > 6 and r['alignment_score'] > 0.95
    assert r['auto']['coverage_diff'] < 6 and r['verify']['passed']
    # the sheet shows the method not chosen beside the one chosen
    assert Image.open(tmp_path / 'floral_compare.png').size[0] > 3 * 1200
