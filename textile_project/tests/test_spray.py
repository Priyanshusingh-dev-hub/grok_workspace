"""SPRAY mode: pure tonal pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from textile import gradients as gr
from textile import spray as sp
from textile.io_utils import save_png, to_image


def _airbrush(w=180, h=140, seed=3):
    """Soft overlapping airbrush blobs + fine speckle."""
    rng = np.random.default_rng(seed)
    paper = np.array([255.0, 255.0, 255.0])
    inks = np.array(
        [[220, 80, 120], [80, 160, 220], [40, 140, 70], [255, 200, 80], [180, 120, 200]],
        dtype=np.float64,
    )
    tones = []
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    centres = [(40, 50), (90, 120), (60, 90), (100, 40), (70, 70)]
    for i, (cy, cx) in enumerate(centres):
        dist = np.hypot(yy - cy, xx - cx)
        blob = np.exp(-(dist ** 2) / (2 * (28 + 8 * rng.random()) ** 2))
        blob = np.asarray(
            Image.fromarray((blob * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(4)),
            dtype=np.float64,
        ) / 255.0
        tones.append(blob * (0.35 + 0.5 * rng.random()))
    T = np.stack(tones, axis=-1)
    A = np.clip(1.0 - inks / paper, 0, 1).T
    abs_total = np.clip(T @ A.T, 0, 1)
    rgb = np.clip(paper * (1.0 - abs_total), 0, 255).astype(np.uint8)
    # fine colour speckle
    speck = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    m = rng.random((h, w)) < 0.04
    rgb = np.where(m[..., None], (0.7 * rgb + 0.3 * speck).astype(np.uint8), rgb)
    return rgb, inks.astype(np.uint8)


def test_fm_halftone_is_1bit_and_stable():
    cov = np.linspace(0, 1, 64 * 48, dtype=np.float64).reshape(48, 64)
    a = gr.fm_halftone(cov, seed=1)
    b = gr.fm_halftone(cov, seed=1)
    assert a.dtype == np.uint8 and set(np.unique(a)).issubset({0, 255})
    assert np.array_equal(a, b)
    # coverage ~ mean of film
    assert abs(a.mean() / 255.0 - 0.5) < 0.08


def test_spray_cli_wires():
    from textile import cli
    import inspect
    src = inspect.getsource(cli)
    assert "'spray'" in src and "cmd_spray" in src and "run_spray" in src


def test_spray_blotchy_de_under_3(tmp_path):
    from tests.test_gradients import _blotchy_noise
    rgb, _ = _blotchy_noise()
    h, w = rgb.shape[:2]
    src = tmp_path / "bl.png"
    save_png(to_image(rgb), str(src), 300)
    r = sp.run_spray(str(src), str(tmp_path / "out"), name="bl", colors=6, dpi=300, halftone="fm")
    assert r["applied"] is True
    assert r["halftone"] == "fm"
    assert r["delta_e_cie76"]["overall_mean"] < 3.0, r["delta_e_cie76"]
    assert Path(r["composite"]).is_file()
    assert r["package"]["merged"] is True
    assert Path(r["package"]["channels_zip"]).is_file()
    assert Path(r["package"]["spot_tif"]).is_file()
    assert Path(r["package"]["psd"]).is_file()
    # films are 1-bit FM
    film = np.asarray(Image.open(r["tone_plates"][0]["film_path"]).convert("L"))
    assert set(np.unique(film)).issubset({0, 255})
    # continuous tones
    tone = np.asarray(Image.open(r["tone_plates"][0]["tone_path"]).convert("L"))
    assert len(np.unique(tone)) > 30


def test_spray_airbrush_de_under_3(tmp_path):
    rgb, _ = _airbrush()
    h, w = rgb.shape[:2]
    src = tmp_path / "ab.png"
    save_png(to_image(rgb), str(src), 300)
    r = sp.run_spray(str(src), str(tmp_path / "out"), name="ab", colors=6, dpi=300, halftone="fm")
    assert r["applied"] is True
    assert r["delta_e_cie76"]["overall_mean"] < 3.0, r["delta_e_cie76"]
    # AM optional path also works
    r2 = sp.run_spray(str(src), str(tmp_path / "out_am"), name="abam", colors=5, dpi=300, halftone="am")
    assert r2["applied"] is True
    assert r2["tone_plates"][0]["halftone"] == "am"
