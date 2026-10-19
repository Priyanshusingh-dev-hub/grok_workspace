"""--gradients multi-channel tonal overprint (OFF by default)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from textile import gradients as gr
from textile.io_utils import save_png, to_image


def _linear_gradient(path: Path, w=120, h=80):
    x = np.linspace(0, 1, w, dtype=np.float32)
    ramp = np.tile(x, (h, 1))
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = (30 + (220 * (1 - ramp))).astype(np.uint8)
    rgb[..., 1] = (40 + (200 * (1 - ramp))).astype(np.uint8)
    rgb[..., 2] = (80 + (150 * (1 - ramp))).astype(np.uint8)
    save_png(to_image(rgb), str(path), 300)
    return rgb


def _flat_blocks(path: Path, size=64):
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[: size // 2, : size // 2] = (200, 40, 40)
    rgb[: size // 2, size // 2 :] = (40, 180, 40)
    rgb[size // 2 :, : size // 2] = (40, 40, 200)
    rgb[size // 2 :, size // 2 :] = (240, 240, 240)
    save_png(to_image(rgb), str(path), 300)
    return rgb


def _hash_files(root: Path, ignore_dirs=("gradients",)):
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in ignore_dirs for part in p.parts):
            continue
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def test_tone_channels_are_greyscale_and_films_are_1bit(tmp_path):
    src = tmp_path / "g.png"
    rgb = _linear_gradient(src)
    # banded flat palette
    flat = np.zeros_like(rgb)
    edges = np.linspace(0, 120, 5, dtype=int)
    bands = [(220, 220, 240), (120, 140, 200), (60, 80, 160), (20, 30, 90)]
    for i, col in enumerate(bands):
        flat[:, edges[i]:edges[i + 1]] = col
    save_png(to_image(flat), str(tmp_path / "g_flat.png"), 300)
    r = gr.run_gradients(str(tmp_path), "g", str(src), size_px=(120, 80),
                         dpi=300, lpi=45, colors_limit=4)
    assert r["applied"] is True
    assert len(r["tone_channels"]) >= 1
    for e in r["tone_channels"]:
        tone = np.asarray(Image.open(e["tone_path"]).convert("L"))
        film = np.asarray(Image.open(e["film_path"]).convert("L"))
        assert tone.ndim == 2 and tone.dtype == np.uint8
        assert set(np.unique(film)).issubset({0, 255})
    # angles differ across inks
    angles = [e["angle_deg"] for e in r["tone_channels"]]
    if len(angles) > 1:
        assert len(set(angles)) == len(angles)


def test_flat_image_with_gradients_leaves_main_outputs_byte_identical(tmp_path):
    src = tmp_path / "f.png"
    _flat_blocks(src)
    save_png(to_image(np.asarray(Image.open(src))), str(tmp_path / "f_flat.png"), 300)
    (tmp_path / "f_colors.csv").write_text("Number,HEX,Colour,Share %,Group\n1,#C82828,red,25,\n", encoding="utf-8")
    save_png(Image.open(src), str(tmp_path / "f_rangeen.png"), 300)
    before = _hash_files(tmp_path)
    r = gr.run_gradients(str(tmp_path), "f", str(src), size_px=(64, 64),
                         dpi=300, lpi=45, colors_limit=8)
    after = _hash_files(tmp_path)
    assert before == after
    assert r["applied"] is False


def test_spot_colour_count_stays_within_colors_limit(tmp_path):
    src = tmp_path / "g.png"
    rgb = _linear_gradient(src, 100, 60)
    q = np.zeros_like(rgb)
    edges = np.linspace(0, 100, 6, dtype=int)
    palette = [(240, 240, 250), (180, 190, 220), (100, 120, 180), (50, 70, 140), (20, 30, 90)]
    for i, col in enumerate(palette):
        q[:, edges[i]:edges[i + 1]] = col
    save_png(to_image(q), str(tmp_path / "g_flat.png"), 300)
    r = gr.run_gradients(str(tmp_path), "g", str(src), size_px=(100, 60),
                         dpi=300, lpi=40, colors_limit=5)
    assert r["spot_colour_count"] <= 5
    assert r["applied"] is True


def test_cli_wires_gradient_flags():
    from textile import cli
    import inspect
    src = inspect.getsource(cli)
    assert "'--gradients'" in src and "'--halftone-lpi'" in src and "'--halftone-angle'" in src
    assert "run_gradients" in src


def test_overprint_composite_mean_delta_e_below_5(tmp_path):
    """Synthetic gradient + flowers: overprint composite matches source (dE < 5)."""
    w, h = 200, 140
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    t = 0.6 * y + 0.4 * x
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = (255 - 90 * t).astype(np.uint8)
    rgb[..., 1] = (245 - 160 * t).astype(np.uint8)
    rgb[..., 2] = (230 - 40 * t).astype(np.uint8)
    rgb[50:90, 70:120] = (30, 140, 60)
    rr, cc = np.ogrid[:h, :w]
    rgb[(rr - 40) ** 2 + (cc - 160) ** 2 <= 22 ** 2] = (220, 40, 90)
    src = tmp_path / "s.png"
    save_png(to_image(rgb), str(src), 300)
    # flat palette approximating the design
    flat = rgb.copy()
    # crude quantize to <= 8 colours for palette
    q = (flat.astype(np.int32) // 32) * 32
    q = np.clip(q, 0, 255).astype(np.uint8)
    save_png(to_image(q), str(tmp_path / "s_flat.png"), 300)
    r = gr.run_gradients(str(tmp_path), "s", str(src), size_px=(w, h),
                         dpi=300, lpi=45, colors_limit=8)
    assert r["applied"] is True
    de = r["delta_e_cie76"]
    assert de["overall_mean"] < 5.0, de
    assert de["gradient_mean"] < 5.0, de


def test_pixels_can_hold_multiple_overlapping_ink_tones(tmp_path):
    """A green leaf on a warm ground needs more than one ink's tone in places."""
    w, h = 80, 60
    rgb = np.zeros((h, w, 3), np.uint8)
    # soft warm ground gradient so the source is not "flat solids"
    for x in range(w):
        rgb[:, x] = (240 - x, 220 - x // 2, 200)
    rgb[20:45, 25:55] = (40, 160, 70)  # green
    # yellow-green soft blend into ground
    for i, x in enumerate(range(52, 62)):
        t = i / 10.0
        rgb[20:45, x] = (
            int(40 * (1 - t) + 240 * t),
            int(160 * (1 - t) + 200 * t),
            int(70 * (1 - t) + 180 * t),
        )
    src = tmp_path / "m.png"
    save_png(to_image(rgb), str(src), 300)
    flat = np.zeros_like(rgb)
    flat[:] = (240, 220, 200)
    flat[20:45, 25:55] = (40, 160, 70)
    flat[20:45, 55:60] = (200, 180, 40)
    flat[0:5, 0:5] = (40, 40, 200)  # extra ink available
    save_png(to_image(flat), str(tmp_path / "m_flat.png"), 300)
    r = gr.run_gradients(str(tmp_path), "m", str(src), size_px=(w, h),
                         dpi=300, lpi=45, colors_limit=6)
    assert r["applied"] is True
    # load tone channels; some pixels should have 2+ inks with tone > 5%
    tones = []
    for e in r["tone_channels"]:
        tones.append(np.asarray(Image.open(e["tone_path"]).convert("L")).astype(np.float32) / 255.0)
    stack = np.stack(tones, axis=-1)
    active = (stack > 0.05).sum(axis=-1)
    assert int((active >= 2).sum()) > 0


def _blotchy_noise(w=160, h=120, seed=7):
    """Irregular blotchy multi-hue field = real ink overprints (not a linear ramp).

    Returns (rgb, ink_palette_uint8).
    """
    rng = np.random.default_rng(seed)
    from PIL import Image as _I, ImageFilter as _F
    paper = np.array([255.0, 255.0, 255.0])
    inks = np.array([[220, 150, 180], [160, 190, 220], [40, 140, 60], [220, 50, 90],
                     [255, 220, 200]], dtype=np.float64)  # cream light ink
    tones = []
    for _ in range(len(inks)):
        n = rng.random((h // 6 + 2, w // 6 + 2))
        big = np.asarray(_I.fromarray((n * 255).astype(np.uint8), mode="L").resize((w, h), _I.Resampling.BICUBIC), dtype=np.float64) / 255.0
        big = np.asarray(_I.fromarray((big * 255).astype(np.uint8), mode="L").filter(_F.GaussianBlur(5)), dtype=np.float64) / 255.0
        tones.append(big * (0.25 + 0.6 * rng.random()))
    A = np.clip(1.0 - inks / paper, 0, 1).T
    T = np.stack(tones, axis=-1)
    abs_total = np.clip(T @ A.T, 0, 1)
    rgb = np.clip(paper * (1.0 - abs_total), 0, 255).astype(np.uint8)
    rgb[35:70, 40:75] = (40, 140, 60)
    rr, cc = np.ogrid[:h, :w]
    rgb[(rr - 85) ** 2 + (cc - 120) ** 2 <= 18 ** 2] = (220, 50, 90)
    return rgb, inks.astype(np.uint8)


def test_blotchy_irregular_gradient_follows_source_locally(tmp_path):
    """Blotchy/noisy tonal areas must stay continuous (no band steps) with dE < 3."""
    rgb, palette = _blotchy_noise()
    h, w = rgb.shape[:2]
    src = tmp_path / "b.png"
    save_png(to_image(rgb), str(src), 300)
    # flat built from the true ink palette (as numbering would provide)
    lab = rgb.astype(np.int32)
    # nearest palette colour per pixel
    d = ((lab[:, :, None, :] - palette[None, None, :, :].astype(np.int32)) ** 2).sum(-1)
    q = palette[d.argmin(-1)].copy()
    # guarantee every ink appears in the flat palette (numbering always lists them)
    for i, c in enumerate(palette):
        q.reshape(-1, 3)[i] = c
    save_png(to_image(q), str(tmp_path / "b_flat.png"), 300)
    r = gr.run_gradients(str(tmp_path), "b", str(src), size_px=(w, h),
                         dpi=300, lpi=45, colors_limit=8, max_inks_per_pixel=4)
    assert r["applied"] is True
    de = r["delta_e_cie76"]
    assert de["gradient_mean"] < 3.0, de
    assert de["overall_mean"] < 5.0, de
    bs = r["band_step"]
    # composite must not introduce much stronger steps than the blotchy source
    assert bs["ratio"] < 1.8, bs
    # tones are full 8-bit continuous (many unique levels, not ~32 steps)
    tone = np.asarray(Image.open(r["tone_channels"][0]["tone_path"]).convert("L"))
    assert len(np.unique(tone)) > 40


def test_motif_borders_never_paper(tmp_path):
    """2px motif borders must get motif ink/gradient tone — never bare paper when source has none."""
    w, h = 140, 100
    rgb = np.zeros((h, w, 3), np.uint8)
    # soft ground (not paper) so source is non-flat
    for x in range(w):
        rgb[:, x] = (245 - x // 3, 240 - x // 4, 230)
    # solid motifs with crisp edges (source border = motif colour, not paper)
    rgb[25:75, 15:55] = (40, 140, 60)
    rr, cc = np.ogrid[:h, :w]
    rgb[(rr - 50) ** 2 + (cc - 100) ** 2 <= 22 ** 2] = (220, 40, 90)
    src = tmp_path / "hb.png"
    save_png(to_image(rgb), str(src), 300)
    flat = rgb.copy()
    q = (flat.astype(np.int32) // 32) * 32
    q = np.clip(q, 0, 255).astype(np.uint8)
    for i, c in enumerate([(40, 140, 60), (220, 40, 90), (245, 240, 230), (200, 180, 160)]):
        q.reshape(-1, 3)[i] = c
    save_png(to_image(q), str(tmp_path / "hb_flat.png"), 300)
    r = gr.run_gradients(str(tmp_path), "hb", str(src), size_px=(w, h),
                         dpi=300, lpi=45, colors_limit=8)
    assert r["applied"] is True
    comp = np.asarray(Image.open(r["composite"]).convert("RGB"))
    motif = (
        (np.abs(rgb.astype(int) - (40, 140, 60)).sum(-1) < 8)
        | (np.abs(rgb.astype(int) - (220, 40, 90)).sum(-1) < 8)
    )
    dil = np.asarray(Image.fromarray(motif.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0
    ero = np.asarray(Image.fromarray(motif.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 0
    ring = dil & ~ero
    src_has_paper = np.max(np.abs(rgb.astype(int) - 255), axis=-1) <= 8
    ring_no_src_paper = ring & ~src_has_paper
    comp_paper = np.max(np.abs(comp.astype(int) - 255), axis=-1) <= 8
    halo = ring_no_src_paper & comp_paper
    assert int(halo.sum()) == 0, f"paper halo on motif border: {int(halo.sum())} px"
    assert r["delta_e_cie76"]["overall_mean"] < 5.0
