"""`textile edges`: every ink's outline redrawn smooth and hard-edged. Checked
on drawn shapes (known geometry) and through the CLI on files."""
import json

import cv2
import numpy as np
from PIL import Image

from textile import cli, edges


def shapes(noisy=False):
    idx = np.zeros((120, 160), np.uint8)
    idx[10:50, 10:70] = 1                                          # a clean rectangle
    cv2.fillPoly(idx, [np.array([[80, 20], [150, 40], [145, 80], [75, 60]], np.int32)], 2)   # slanted
    cv2.circle(idx, (40, 90), 15, 3, -1)
    idx[100, 70:150] = 1                                           # a 1 px line
    idx[110:114, 20:24] = 3                                        # a 4 px dot
    if noisy:
        rng = np.random.default_rng(2)
        edge = np.zeros(idx.shape, bool)
        edge[:, 1:] |= idx[:, 1:] != idx[:, :-1]
        edge[1:, :] |= idx[1:, :] != idx[:-1, :]
        edge[99:102] = False
        # every boundary pixel may take the ink of a neighbour two pixels away, a random way
        dy, dx = rng.integers(-2, 3, 2)
        other = np.roll(idx, (dy, dx), (0, 1))
        flip = edge & (rng.random(idx.shape) < 0.5)
        idx[flip] = other[flip]
    return idx


def test_a_clean_shape_is_unchanged_at_its_own_size():
    idx = np.zeros((80, 100), np.uint8)
    idx[10:40, 10:50] = 1
    idx[50:70, 20:90] = 2
    out, _ = edges.clean(idx, 1, 2)
    assert (out != idx).sum() <= 8                       # corners at most; edges stay exactly where they were


def test_only_the_same_inks_come_out_and_the_size_follows_the_scale():
    idx = shapes(noisy=True)
    for scale, shape in ((1, (120, 160)), (2, (240, 320)), (2.5, (300, 400)), (0.5, (60, 80))):
        out, rep = edges.clean(idx, scale, 2)
        assert out.shape == shape and rep['scale'] == scale
        assert set(np.unique(out)) <= set(np.unique(idx))      # no new ink; one ink per pixel by construction


def test_edges_settle_but_thin_lines_dots_and_corners_stay():
    idx = shapes(noisy=True)
    out, _ = edges.clean(idx, 1, 2)
    assert (out != idx).mean() < 0.04
    assert (out[100, 72:148] == 1).mean() > 0.9                # the 1 px line
    assert (out[108:116, 18:26] == idx[108:116, 18:26]).all()  # the small dot comes back exactly as it was



def test_corners_stay_sharp():
    idx = shapes(noisy=False)
    out, _ = edges.clean(idx, 4, 2)                            # the rectangle spans rows 40..199, cols 40..279 now
    # a corner rounded by a 3 source px radius (12 px here) would lose these pixels, 3 px in from each side
    assert out[43, 43] == 1 and out[43, 276] == 1 and out[196, 43] == 1 and out[196, 276] == 1


def test_round_shapes_do_not_melt():
    for r in (6, 10, 20):
        c = np.zeros((100, 100), np.uint8)
        cv2.circle(c, (50, 50), r, 1, -1)
        for strength in (1, 2, 3):
            out, _ = edges.clean(c, 1, strength)
            assert abs((out == 1).sum() / (c == 1).sum() - 1) < 0.08


def test_finer_grid_beats_plain_enlarging_on_a_drawn_truth():
    """A design drawn at 4x is the truth; its 1x sample is what a reduce gives.
    Redrawn clean it is closer to the truth than the pixels blown up. (At its
    own size the pixel grid is the limit: cleaning does not get closer there.)"""
    R = 4
    T = np.zeros((480, 480), np.uint8)
    cv2.fillPoly(T, [(np.array([[10, 40], [100, 20], [110, 70], [20, 90]]) * R).astype(np.int32)], 1)
    cv2.circle(T, (80 * R, 90 * R), 18 * R, 2, -1)
    cv2.line(T, (10 * R, 110 * R), (110 * R, 100 * R), 1, 2 * R, lineType=cv2.LINE_8)
    src = T[R // 2::R, R // 2::R]
    plain = (np.kron(src, np.ones((R, R), np.uint8)) != T).mean()
    clean, _ = edges.clean(src, R, 2)
    assert (clean != T).mean() < plain * 0.9


def test_specks_are_only_removed_when_asked():
    idx = np.zeros((60, 60), np.uint8)
    idx[20:40, 20:40] = 1
    idx[5, 5] = 2
    idx[50, 50:52] = 2
    assert (edges.clean(idx, 1, 2)[0] == 2).sum() == 3             # off by default: they may be real motifs
    out, rep = edges.clean(idx, 1, 2, speck=2)
    assert (out == 2).sum() == 0 and rep['specks_moved'] == 2


def test_a_repeat_is_cleaned_round_its_seam():
    idx = np.zeros((100, 100), np.uint8)
    cv2.circle(idx, (0, 50), 20, 1, -1)
    cv2.circle(idx, (99, 50), 20, 1, -1)
    out, _ = edges.clean(idx, 1, 2, wrap=(True, False))
    a, b = set(np.flatnonzero(out[:, 0] == 1)), set(np.flatnonzero(out[:, -1] == 1))
    assert len(a & b) / len(a | b) > 0.8


def test_cli_writes_the_package_at_the_mill_size(tmp_path):
    idx = shapes(noisy=True)
    pal = np.array([[240, 230, 210], [120, 30, 40], [40, 90, 60], [20, 20, 90]], np.uint8)
    src = tmp_path / 'noisy.png'
    Image.fromarray(pal[idx]).save(src)
    out = tmp_path / 'out'
    assert cli.main(['edges', str(src), '--out', str(out), '--size', '800', '--name', 'neat']) == 0
    rep = json.load(open(out / 'neat_report.json'))
    assert rep['size_px'] == [800, 600] and rep['dpi'] == 300 and rep['verify']['passed']
    assert rep['edges']['strength'] == 2 and rep['colors_in_final'] == 4
    final = np.asarray(Image.open(out / 'neat_final_800x600px_300dpi.png').convert('RGB'))
    assert {tuple(c) for c in final.reshape(-1, 3)} <= {tuple(c) for c in pal}


def test_cli_refuses_a_photo_and_warns_on_600_dpi(tmp_path, capsys):
    photo = tmp_path / 'photo.png'
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (50, 50, 3), np.uint8)).save(photo)
    assert cli.main(['edges', str(photo), '--out', str(tmp_path / 'o1')]) == 1
    assert 'flat design nahi' in capsys.readouterr().out
    flat = tmp_path / 'flat.png'
    Image.fromarray(np.where(shapes()[..., None] > 0, 200, 30).astype(np.uint8).repeat(3, 2)).save(flat)
    assert cli.main(['edges', str(flat), '--out', str(tmp_path / 'o2'), '--dpi', '600', '--size', '640']) == 0
    text = capsys.readouterr().out
    assert text.count('[CHETAVNI] 600 DPI') == 2                   # at the start and at the end
    rep = json.load(open(tmp_path / 'o2' / 'flat_clean_report.json'))
    assert rep['dpi'] == 600 and rep['size_px'][0] == 640
