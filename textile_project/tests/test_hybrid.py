"""Hybrid mode: FLAT solids + SHADED tonal overprint."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from textile import gradients as gr
from textile.io_utils import save_png, to_image


def _design_like(w=160, h=120):
    """Black ground + solid red berry + soft shaded petal."""
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (2, 3, 4)  # black ground
    # solid berry
    rr, cc = np.ogrid[:h, :w]
    rgb[(rr - 30) ** 2 + (cc - 30) ** 2 <= 12 ** 2] = (180, 20, 30)
    # soft shaded petal (real ramp)
    for x in range(70, 140):
        t = (x - 70) / 70.0
        rgb[40:90, x] = (int(240 - 80 * t), int(160 - 100 * t), int(180 - 40 * t))
    # cream scroll solid
    rgb[95:115, 20:60] = (240, 220, 190)
    flat = rgb.copy()
    # flat posterises the petal
    for x in range(70, 140):
        flat[40:90, x] = (200, 100, 140) if x < 105 else (160, 60, 100)
    return rgb, flat


def test_cli_wires_hybrid():
    from textile import cli
    import inspect
    src = inspect.getsource(cli)
    assert "'--hybrid'" in src
    assert "hybrid=" in src


def test_hybrid_flat_ground_no_mottle(tmp_path):
    rgb, flat = _design_like()
    h, w = rgb.shape[:2]
    save_png(to_image(rgb), str(tmp_path / "h.png"), 300)
    save_png(to_image(flat), str(tmp_path / "h_flat.png"), 300)
    r = gr.run_gradients(str(tmp_path), "h", str(tmp_path / "h.png"),
                         size_px=(w, h), dpi=300, colors_limit=6, hybrid=True)
    assert r["applied"] is True
    assert r.get("hybrid") is True
    assert r["flat_stray_tone_percent"] < 1.0
    assert r["flat_ground_std"] < 1.0, r
    # black ground in composite is spatially flat
    comp = np.asarray(Image.open(r["composite"]).convert("RGB"))
    black = (flat.max(axis=-1) < 20)
    core = np.asarray(Image.fromarray(black.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 0
    lum = 0.2126 * comp[..., 0] + 0.7152 * comp[..., 1] + 0.0722 * comp[..., 2]
    assert float(lum[core].astype(np.float64).std()) < 0.5
    # shaded petal still has tone variation (not a single flat colour)
    petal = np.zeros((h, w), bool)
    petal[40:90, 70:140] = True
    assert len(np.unique(comp[petal].reshape(-1, 3), axis=0)) > 5


def test_hybrid_default_on_gradients(tmp_path):
    """run_gradients hybrid=True by default."""
    import inspect
    sig = inspect.signature(gr.run_gradients)
    assert sig.parameters["hybrid"].default is True
