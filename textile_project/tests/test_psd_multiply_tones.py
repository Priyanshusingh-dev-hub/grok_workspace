"""Tone PSD: spot CHANNELS by default; optional --psd-layers keeps LINEAR_BURN stack."""
from __future__ import annotations

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import BlendMode, Resource

from textile import gradients as gr
from textile import spray as sp
from textile.io_utils import save_png, to_image


def _soft(w=100, h=80, seed=2):
    rng = np.random.default_rng(seed)
    inks = np.array([[200, 40, 80], [40, 120, 200], [40, 160, 60]], dtype=np.uint8)
    tones = []
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    for cy, cx in [(30, 30), (50, 70), (40, 50)]:
        dist = np.hypot(yy - cy, xx - cx)
        blob = np.exp(-(dist ** 2) / (2 * 22 ** 2))
        tones.append(blob * (0.5 + 0.3 * rng.random()))
    return gr.overprint_rgb(np.stack(tones, axis=-1), inks, (255, 255, 255))


def test_psd_channels_only_by_default(tmp_path):
    rgb = _soft()
    src = tmp_path / "s.png"
    save_png(to_image(rgb), str(src), 300)
    r = sp.run_spray(str(src), str(tmp_path / "out"), name="s", colors=4, dpi=300)
    psd = PSDImage.open(r["package"]["psd"])
    assert list(psd) == []  # no ink layers
    assert psd.channels >= 3 + 1
    names = psd.image_resources.get(Resource.ALPHA_NAMES_UNICODE)
    assert names is not None
    alpha_names = list(names.data)
    assert len(alpha_names) >= 1
    # darkest-first: parse trailing hex luminances
    def lum(nm):
        hx = nm.strip().split()[-1]
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    lums = [lum(n) for n in alpha_names]
    assert lums == sorted(lums), (alpha_names, lums)
    # planes readable
    planes = psd._record.image_data.get_data(psd._record.header)
    assert len(planes) == psd.channels


def test_psd_layers_opt_in_matches_preview(tmp_path):
    rgb = _soft()
    src = tmp_path / "s.png"
    save_png(to_image(rgb), str(src), 300)
    r = sp.run_spray(str(src), str(tmp_path / "out"), name="s", colors=4, dpi=300, use_layers=True)
    psd = PSDImage.open(r["package"]["psd"])
    layers = list(psd)
    assert layers[0].name.startswith("00 Paper")
    assert not any(l.name.startswith("COMPOSITE") for l in layers)
    tone_layers = [l for l in layers[1:] if "tone" in l.name.lower()]
    assert tone_layers and all(l.blend_mode == BlendMode.LINEAR_BURN for l in tone_layers)
    prev = np.asarray(Image.open(r["composite"]).convert("RGB"))
    comp = np.asarray(psd.composite(ignore_preview=True).convert("RGB"))
    assert gr.delta_e_cie76(prev, comp) < 1.0


def test_psd_composite_flag_adds_layer(tmp_path):
    rgb = _soft()
    src = tmp_path / "s.png"
    save_png(to_image(rgb), str(src), 300)
    r = sp.run_spray(str(src), str(tmp_path / "out"), name="s", colors=3, dpi=300,
                     include_composite=True, use_layers=True)
    assert any(l.name.startswith("COMPOSITE") for l in PSDImage.open(r["package"]["psd"]))
