"""AUTO mode: FLAT / LINE / SPRAY region routing."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from textile import auto as am
from textile import cli
from textile.io_utils import save_png, to_image


def _auto_design(w=200, h=150):
    """Black ground, solid berry, thin green line, soft shaded petal, cream dots."""
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (8, 6, 10)
    rr, cc = np.ogrid[:h, :w]
    rgb[(rr - 35) ** 2 + (cc - 40) ** 2 <= 16 ** 2] = (190, 30, 40)
    # shaded petal
    for x in range(90, 170):
        t = (x - 90) / 80.0
        rgb[45:100, x] = (int(230 - 70 * t), int(150 - 90 * t), int(190 - 50 * t))
    # thin green outline
    rgb[50:95, 88:91] = (80, 170, 60)
    # cream dots
    for cy, cx in [(120, 40), (125, 55), (118, 70)]:
        rgb[(rr - cy) ** 2 + (cc - cx) ** 2 <= 4 ** 2] = (240, 225, 180)
    flat = rgb.copy()
    for x in range(90, 170):
        flat[45:100, x] = (200, 110, 150) if x < 130 else (160, 70, 110)
    return rgb, flat


def test_cli_wires_auto():
    import inspect
    src = inspect.getsource(cli)
    assert "'auto'" in src and "cmd_auto" in src and "run_auto" in src


def test_classify_routes_shade_and_keeps_ground_flat(tmp_path):
    rgb, flat = _auto_design()
    from textile import gradients as gr
    inks = gr.pick_inks(flat, 6, src_rgb=rgb)
    reg = am.classify_auto_regions(rgb, flat, inks)
    assert reg["fractions"]["flat"] > 10
    assert reg["fractions"]["spray"] > 1
    # black ground is flat, not spray
    ground = flat.max(axis=-1) < 20
    assert float(reg["shade_mask"][ground].mean()) < 0.05


def test_run_auto_flat_only_has_zero_tone_pixels(tmp_path):
    """Pure flat solids (no shading) → 0 spray/tone pixels."""
    w, h = 120, 90
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (20, 10, 30)
    rgb[20:70, 20:60] = (200, 40, 50)
    rgb[30:50, 80:110] = (240, 220, 180)
    src = tmp_path / "flat.png"
    save_png(to_image(rgb), str(src), 300)
    r = am.run_auto(str(src), str(tmp_path / "out"), name="f", colors=4, dpi=300, size=w)
    assert r["applied"] is True
    assert r.get("auto_mode") == "flat_line_solid_only"
    assert r.get("spray_pixels", 0) == 0
    assert r["tone_pixels"] == 0
    assert r["ground_std"] < 1.0


def test_run_auto_shaded_uses_spray_and_flat_ground(tmp_path):
    rgb, _flat = _auto_design()
    src = tmp_path / "s.png"
    save_png(to_image(rgb), str(src), 300)
    r = am.run_auto(str(src), str(tmp_path / "out"), name="s", colors=6, dpi=300, size=rgb.shape[1])
    assert r["applied"] is True
    assert r.get("auto_mode") == "flat_line_solid_plus_spray_shade"
    assert r.get("spray_pixels", 0) > 0
    assert r["ground_std"] < 1.0, r
    assert (r.get("delta_e_cie76") or {}).get("overall_mean") is not None


def test_edge_clean_keeps_dots_and_smooths_large(tmp_path):
    """Shared-SDF edges: large shape smoothed; tiny isolated dots kept; hi-res kept."""
    w, h = 160, 120
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (10, 8, 12)
    rgb[20:100, 30:110] = (200, 40, 50)  # large block (jagged ok)
    # stair-step the right edge
    for y in range(20, 100):
        rgb[y, 110 + (y % 3)] = (200, 40, 50)
    # tiny cream dots
    rgb[50:53, 20:23] = (240, 220, 180)
    rgb[80:83, 15:18] = (240, 220, 180)
    from textile import gradients as gr
    inks = gr.pick_inks(rgb, 4, src_rgb=rgb)
    before = rgb.copy()
    after, rep = am.smooth_flat_edges(rgb, inks, strength=2, scale=2.0, log=lambda m: None)
    assert rep.get("method") == "sdf_argmax"
    assert rep.get("kept_hires") is True
    assert after.shape[0] == rgb.shape[0] * 2 and after.shape[1] == rgb.shape[1] * 2
    # dots still present at 2x (approx mapped)
    assert (after[100:106, 40:46].max(axis=-1) > 200).any()
    # hi-res edge band differs from nearest-upsampled before (shared SDF moves boundary)
    before2 = np.asarray(Image.fromarray(before).resize(after.shape[1::-1], Image.Resampling.NEAREST))
    assert not np.array_equal(after[:, 216:232], before2[:, 216:232])
