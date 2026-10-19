"""Reconstruct: patch redraw + spray solver. Synthetic cases from the brief."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import Resource

from textile import reconstruct as rc
from textile.io_utils import save_png, to_image


def _save(rgb, path):
    save_png(to_image(rgb), str(path), 300)


def _disk(h, w, cy, cx, r, color, rgb):
    yy, xx = np.ogrid[:h, :w]
    rgb[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = color
    return rgb


def _radial_cv(mask):
    er = mask.copy()
    er[1:] &= mask[:-1]
    er[:-1] &= mask[1:]
    er[:, 1:] &= mask[:, :-1]
    er[:, :-1] &= mask[:, 1:]
    ys, xs = np.nonzero(mask & ~er)
    cy, cx = ys.mean(), xs.mean()
    rr = np.hypot(ys - cy, xs - cx)
    return float(rr.std() / (rr.mean() + 1e-6))


def test_cli_wires_reconstruct():
    import inspect
    from textile import cli
    src = inspect.getsource(cli)
    assert "reconstruct" in src and "cmd_reconstruct" in src and "run_reconstruct" in src


def test_jagged_circle_becomes_round_at_4x(tmp_path):
    h = w = 48
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (244, 236, 220)
    _disk(h, w, 24, 24, 14, (190, 40, 70), rgb)
    src = tmp_path / "circle.png"
    _save(rgb, src)
    r = rc.run_reconstruct(str(src), str(tmp_path / "out"), name="c", colors=4, scale=4, dpi=300, views=False)
    assert r["output_ge_source"]
    assert r["size_px"][0] >= w * 4 - 1
    comp = np.asarray(Image.open(r["preview"]).convert("RGB"))
    ink = comp[comp.shape[0] // 2, comp.shape[1] // 2]
    mask = np.linalg.norm(comp.astype(np.int16) - ink.astype(np.int16), axis=-1) < 30
    # The foreground disk, not the ground.
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    # largest interior component that isn't the border-touching ground
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    comp_id = int(sizes.argmax())
    cv = _radial_cv(lab == comp_id)
    assert cv < 0.08, cv
    assert r["edge_deviation_px"] < 2.5


def test_small_dots_kept_and_round(tmp_path):
    h, w = 80, 120
    rgb = np.full((h, w, 3), (250, 246, 236), np.uint8)
    centers = [(20, 18), (20, 48), (20, 78), (50, 30), (50, 70)]
    for cy, cx in centers:
        _disk(h, w, cy, cx, 3, (180, 30, 60), rgb)
    src = tmp_path / "dots.png"
    _save(rgb, src)
    r = rc.run_reconstruct(str(src), str(tmp_path / "out"), name="d", colors=4, scale=2, dpi=300, views=False)
    counts = r["counts"]
    assert counts["dots_source"] == len(centers)
    assert counts["dots_output"] == counts["dots_source"]
    assert r["flat_std"] == 0.0
    assert r["flat_stray_tone_percent"] == 0.0


def test_antialiased_edge_makes_no_third_ink(tmp_path):
    h, w = 40, 80
    black = np.array([16, 12, 14], np.uint8)
    cream = np.array([246, 236, 214], np.uint8)
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[:, :34] = black
    rgb[:, 38:] = cream
    for i, t in enumerate((0.25, 0.5, 0.75)):
        rgb[:, 34 + i] = np.clip(black.astype(np.float32) * (1 - t) + cream.astype(np.float32) * t, 0, 255).astype(np.uint8)
    src = tmp_path / "aa.png"
    _save(rgb, src)
    r = rc.run_reconstruct(str(src), str(tmp_path / "out"), name="aa", colors=6, scale=2, dpi=300, views=False)
    comp = np.asarray(Image.open(r["preview"]).convert("RGB"))
    uniq = np.unique(comp.reshape(-1, 3), axis=0)
    assert len(uniq) <= 2, uniq
    assert r["n_inks"] <= 2


def test_one_px_line_kept_and_thickened(tmp_path):
    h, w = 40, 90
    rgb = np.full((h, w, 3), (244, 236, 220), np.uint8)
    rgb[20, 8:82] = (20, 90, 40)
    src = tmp_path / "line.png"
    _save(rgb, src)
    # 1.2 mm at 300 DPI ≈ 14 px, wider than the 2px a 1px line becomes at 2x.
    r = rc.run_reconstruct(
        str(src), str(tmp_path / "out"), name="ln", colors=4, scale=2, dpi=300,
        min_feature_mm=1.2, views=False,
    )
    counts = r["counts"]
    assert counts["lines_source"] == 1
    assert counts["lines_output"] == 1
    comp = np.asarray(Image.open(r["preview"]).convert("RGB"))
    green = np.linalg.norm(comp.astype(np.int16) - np.array([20, 90, 40], np.int16), axis=-1) < 40
    # Thickness along the middle of the stroke.
    col = green[:, comp.shape[1] // 2]
    assert int(col.sum()) >= int(math.floor(r["min_feature_px"])) - 1
    # The stroke still spans the original run (not deleted, not a speck).
    xs = np.nonzero(green.any(0))[0]
    assert xs.max() - xs.min() > 70


def test_irregular_gradient_is_smooth_tone_not_bands(tmp_path):
    h, w = 48, 96
    rgb = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
        t = 0.15 + 0.75 * (0.5 + 0.5 * math.sin(x / w * math.pi * 2.2))
        t = float(np.clip(t, 0, 1))
        rgb[:, x] = (int(210 * t + 20), int(40 * t + 10), int(70 * t + 15))
    src = tmp_path / "grad.png"
    _save(rgb, src)
    r = rc.run_reconstruct(str(src), str(tmp_path / "out"), name="g", colors=4, scale=2, dpi=300, views=False)
    assert r["spray_percent"] > 20, r["spray_percent"]
    assert r["tone_pixels"] > 50
    # Continuous 8-bit, not a 3-step poster.
    tone_dir = tmp_path / "out" / "reconstruct" / "tone_channels"
    levels = 0
    for p in tone_dir.glob("*.png"):
        arr = np.asarray(Image.open(p).convert("L"))
        levels = max(levels, len(np.unique(arr)))
    assert levels > 16
    assert r["band_step"] < 15, r["band_step"]


def test_solid_ground_has_zero_std(tmp_path):
    h, w = 50, 70
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (18, 12, 22)
    rgb[10:40, 10:30] = (200, 40, 50)
    src = tmp_path / "solid.png"
    _save(rgb, src)
    r = rc.run_reconstruct(str(src), str(tmp_path / "out"), name="s", colors=4, scale=2, dpi=300, views=False)
    assert r["spray_percent"] < 2.0, r["spray_percent"]
    assert r["tone_pixels"] == 0
    assert r["flat_std"] == 0.0
    assert r["flat_stray_tone_percent"] == 0.0
    assert r["delta_e_cie76"]["flat_mean"] < 2.0


def test_psd_channels_only_darkest_first(tmp_path):
    h, w = 36, 48
    rgb = np.full((h, w, 3), (230, 220, 200), np.uint8)
    rgb[:18, :] = (12, 10, 14)
    rgb[8:28, 10:30] = (40, 120, 60)
    src = tmp_path / "psd.png"
    _save(rgb, src)
    r = rc.run_reconstruct(str(src), str(tmp_path / "out"), name="p", colors=4, scale=2, dpi=300, views=False)
    psd = PSDImage.open(r["package"]["psd"])
    assert list(psd) == []
    names = list(psd.image_resources.get(Resource.ALPHA_NAMES_UNICODE).data)
    assert len(names) == r["n_inks"]

    def lum(nm):
        hx = nm.strip().split()[-1]
        rr, gg, bb = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        return 0.2126 * rr + 0.7152 * gg + 0.0722 * bb

    lums = [lum(n) for n in names]
    assert lums == sorted(lums), names
    assert r["output_ge_source"]
    assert r["size_px"][0] >= w and r["size_px"][1] >= h


def test_ai_upscale_skips_cleanly_when_missing(tmp_path):
    rgb = np.full((16, 16, 3), (240, 230, 210), np.uint8)
    rgb[4:12, 4:12] = (20, 20, 20)
    src = tmp_path / "ai.png"
    _save(rgb, src)
    r = rc.run_reconstruct(
        str(src), str(tmp_path / "out"), name="ai", colors=3, scale=2, dpi=300,
        ai_upscale=True, views=False,
    )
    boost = r["quality_boost"]
    assert boost["ai"] is False
    assert "ai_skipped" in boost
    assert r["applied"] is True
