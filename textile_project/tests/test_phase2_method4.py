"""Method 4: Method 1 with the line art's gaps sealed. Method 1 itself stays
byte-identical to reference_code (test_phase2_method1); here: seal 0 IS
Method 1, a gap no longer lets the ground flood a petal, and a shape the
sealing swallows keeps its colour."""
import cv2
import numpy as np
import pytest
from PIL import Image

from conftest import SAMPLES
from textile import fill_method1 as m1
from textile import fill_method4 as m4

GROUND, RED = (230, 210, 175), (200, 50, 40)


def pair(tmp_path, gap=2, size=200):
    """Beige ground, a red disc drawn in a black ring with a `gap` px break."""
    ref = np.zeros((size, size, 3), np.uint8); ref[:] = GROUND
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(xx - size / 2, yy - size / 2)
    ref[r < 50] = RED
    line = np.full((size, size), 255, np.uint8)
    ring = (r >= 50) & (r < 53)
    line[ring] = 0
    line[size // 2 - gap // 2:size // 2 + (gap + 1) // 2, size // 2 + 49:size // 2 + 55] = 255     # the break
    Image.fromarray(ref).save(tmp_path / 'ref.png'); Image.fromarray(line).save(tmp_path / 'line.png')
    return str(tmp_path / 'line.png'), str(tmp_path / 'ref.png')


def centre(f):
    return tuple(int(v) for v in f.pal[f.index[f.index.shape[0] // 2, f.index.shape[1] // 2]])


def test_a_gap_in_the_ring_lets_method1_flood_the_disc_and_method4_does_not(tmp_path):
    line, ref = pair(tmp_path)
    one = m1.fill(line, ref, 400, force=True, log=lambda *a: None)
    four = m4.fill(line, ref, 400, force=True, log=lambda *a: None)
    assert centre(one) == GROUND                      # the ground ran through the break and won the vote
    assert centre(four) == RED                        # sealed: the disc is its own area again
    # outside the ring and the ring's own pixels are unchanged
    assert (four.index == one.index).mean() > 0.80


def test_a_wide_break_is_not_sealed(tmp_path):
    line, ref = pair(tmp_path, gap=14)
    assert centre(m4.fill(line, ref, 400, force=True, log=lambda *a: None)) == GROUND


def test_seal_zero_is_method1(tmp_path):
    f1 = m1.fill(str(SAMPLES / 'floral_lineart.png'), str(SAMPLES / 'floral_ref.png'), 700, log=lambda *a: None)
    f4 = m4.fill(str(SAMPLES / 'floral_lineart.png'), str(SAMPLES / 'floral_ref.png'), 700, seal=0, log=lambda *a: None)
    assert np.array_equal(f1.index, f4.index) and np.array_equal(f1.pal, f4.pal) and f1.line_index == f4.line_index
    assert f1.regions == f4.regions and f1.alignment_score == f4.alignment_score


def test_sealing_does_not_lose_a_small_closed_shape(tmp_path):
    """A 5 px red sliver between two close lines: the closing swallows its
    area, its own vote puts it back."""
    size = 200
    ref = np.zeros((size, size, 3), np.uint8); ref[:] = GROUND
    ref[90:95, 40:160] = RED
    line = np.full((size, size), 255, np.uint8)
    line[88:90, 40:160] = 0; line[95:97, 40:160] = 0
    line[88:97, 40:42] = 0; line[88:97, 158:160] = 0           # closed at both ends: a real closed shape
    Image.fromarray(ref).save(tmp_path / 'r.png'); Image.fromarray(line).save(tmp_path / 'l.png')
    f = m4.fill(str(tmp_path / 'l.png'), str(tmp_path / 'r.png'), 200, seal=4, force=True, log=lambda *a: None)
    assert tuple(int(v) for v in f.pal[f.index[92, 100]]) == RED


def test_method4_refuses_what_method1_refuses(tmp_path):
    line, ref = pair(tmp_path)
    wide = np.zeros((100, 300, 3), np.uint8); Image.fromarray(wide).save(tmp_path / 'wide.png')
    with pytest.raises(m1.FillError, match='aspect'):
        m4.fill(line, str(tmp_path / 'wide.png'), 400, log=lambda *a: None)
