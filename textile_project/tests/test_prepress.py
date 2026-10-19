"""Optional prepress flags for `textile number` (all OFF by default).

Synthetic images only — no Hybrid Lab folder edits, no weakening of other suites.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from textile import prepress as pp
from textile.io_utils import save_png, to_image


def _flat_blocks(path: Path, size: int = 64):
    """Four solid colour blocks — clean plate boundaries for trap/underbase/film."""
    rgb = np.zeros((size, size, 3), np.uint8)
    half = size // 2
    rgb[:half, :half] = (200, 40, 40)       # red
    rgb[:half, half:] = (40, 180, 40)       # green
    rgb[half:, :half] = (40, 40, 200)       # blue
    rgb[half:, half:] = (245, 245, 245)     # near-white ground (skipped by selective underbase)
    save_png(to_image(rgb), str(path), 300)
    return rgb


def _hash_tree(root: Path, skip_names=()):
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name not in skip_names:
            rel = p.relative_to(root).as_posix().encode()
            h.update(rel)
            h.update(p.read_bytes())
    return h.hexdigest()


def test_flags_off_run_prepress_is_noop(tmp_path):
    flat = tmp_path / 'd_flat.png'
    _flat_blocks(flat)
    before = _hash_tree(tmp_path)
    r = pp.run_prepress(str(tmp_path), 'd')
    assert r['applied'] is False
    assert not (tmp_path / 'prepress').exists()
    assert _hash_tree(tmp_path) == before


def test_underbase_selective_skips_near_white(tmp_path):
    _flat_blocks(tmp_path / 'd_flat.png')
    r = pp.run_prepress(str(tmp_path), 'd', underbase=True, underbase_choke_px=1)
    assert r['underbase']['generated']
    assert r['underbase']['inks_skipped_near_white'] >= 1
    ub = np.asarray(Image.open(r['underbase']['path']).convert('L'))
    assert ub.shape == (64, 64)
    # near-white quadrant should be mostly empty after selective + choke
    assert int((ub[32:, 32:] > 0).sum()) < 64  # almost none in ground


def test_trap_adds_overlap_between_neighbours(tmp_path):
    _flat_blocks(tmp_path / 'd_flat.png')
    r0 = pp.run_prepress(str(tmp_path), 'd', trap_px=0)  # treated as off-ish via trap_px>0 check
    assert r0['applied'] is False or r0.get('trapping', {}).get('applied') is False
    r = pp.run_prepress(str(tmp_path), 'd', trap_px=2)
    assert r['trapping']['applied'] is True
    assert r['trapping']['overlap_pixels'] > 0
    trap_dir = tmp_path / 'prepress' / 'trapped_plates'
    assert trap_dir.is_dir() and list(trap_dir.glob('*.png'))


def test_film_black_on_white_with_registration(tmp_path):
    _flat_blocks(tmp_path / 'd_flat.png')
    r = pp.run_prepress(str(tmp_path), 'd', film=True)
    films = r['film_positives']
    assert len(films) >= 4
    im = np.asarray(Image.open(films[0]).convert('L'))
    # white canvas dominant; black ink + marks present
    assert im[0, 0] == 255
    assert (im == 0).any() and (im == 255).mean() > 0.5


def test_estimate_csv_and_txt(tmp_path):
    _flat_blocks(tmp_path / 'd_flat.png')
    r = pp.run_prepress(str(tmp_path), 'd', estimate=True)
    csv_p = Path(r['estimate'])
    assert csv_p.is_file()
    txt = csv_p.with_suffix('.txt')
    assert txt.is_file()
    text = csv_p.read_text(encoding='utf-8')
    assert 'mesh' in text and 'grams_per_100_prints' in text
    assert len(r['estimate_rows']) >= 4


def test_seam_check_tiled_preview_and_score(tmp_path):
    rgb = _flat_blocks(tmp_path / 'd_flat.png')
    # make a non-seamless tile: alter right edge
    bad = rgb.copy()
    bad[:, -1] = (0, 0, 0)
    save_png(to_image(bad), str(tmp_path / 'd_rangeen.png'), 300)
    r = pp.run_prepress(str(tmp_path), 'd', seam_check=True)
    assert r['seam_check']['applied']
    assert Path(r['seam_check']['tiled_preview']).is_file()
    data = json.loads(Path(r['seam_check']['path']).read_text(encoding='utf-8'))
    assert 'seam_mismatch_score' in data
    assert data['is_seamless_repeat'] is False


def test_cli_wires_prepress_flags():
    from textile import cli
    import inspect
    src = inspect.getsource(cli)
    for flag in ("'--underbase'", "'--trap'", "'--film'", "'--estimate'", "'--seam-check'"):
        assert flag in src
    assert 'run_prepress' in src


def test_mesh_and_ink_helpers():
    assert '110' in pp.recommend_mesh_count('underbase')
    assert '156' in pp.recommend_mesh_count('spot')
    g = pp.calculate_theoretical_ink_estimate(300, 300, 300, 0.5, 'spot')
    assert g is not None and g > 0


def test_choke_and_spread_roundtrip_shape():
    m = np.zeros((20, 20), np.uint8)
    m[5:15, 5:15] = 255
    c = pp.choke_mask(m, 2)
    s = pp.spread_mask(m, 2)
    assert c.shape == m.shape == s.shape
    assert int((c > 0).sum()) < int((m > 0).sum()) < int((s > 0).sum())
