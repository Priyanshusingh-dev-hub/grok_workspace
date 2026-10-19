"""AUTO mode: per-region FLAT solid / LINE crisp / SPRAY tone.

CLI: python -m textile auto <img> --out <dir> --colors N

Reuses numbering (flat solids), gradients hybrid classify + tone solve, and
spray/package darkest-first plates. SPRAY tones are written only where the
source has real shading; a design with no shade gets 0 tone pixels (solid
plates only).
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
Image.MAX_IMAGE_PIXELS = None

from . import gradients as gr
from . import number as num
from . import package_channels as pkgch
from . import spray as sp
from . import edges as ed
from .io_utils import hex_of, save_png, to_image


def _local_range(lum_u8: np.ndarray, radius: int = 5) -> np.ndarray:
    im = Image.fromarray(lum_u8, mode="L")
    if radius >= 5:
        im = im.filter(ImageFilter.MedianFilter(3))
    mx = np.asarray(im.filter(ImageFilter.MaxFilter(radius)), dtype=np.float32)
    mn = np.asarray(im.filter(ImageFilter.MinFilter(radius)), dtype=np.float32)
    return mx - mn


def detect_line_mask(flat_rgb: np.ndarray, flat_ink_idx: np.ndarray, inks: np.ndarray) -> np.ndarray:
    """Thin strokes + small dots/stripes in the flat map → crisp solid (not soft spray)."""
    h, w = flat_ink_idx.shape
    ink_lums = np.array([gr._lum1(c) for c in inks], dtype=np.float64)
    thin = np.zeros((h, w), dtype=bool)
    for i in range(len(inks)):
        if ink_lums[i] < 40.0:
            continue  # dark ground edges are not "lines"
        m = flat_ink_idx == i
        if not m.any():
            continue
        core = gr._erode_bool(m, 1)
        thick = gr._dilate_bool(core, 1) if core.any() else np.zeros_like(m)
        thin |= m & ~thick
    try:
        from scipy import ndimage
        labeled, nlab = ndimage.label(flat_ink_idx >= 0)
        if nlab:
            areas = np.bincount(labeled.ravel())
            max_dot = max(40, (h * w) // 80000)
            small = np.zeros(nlab + 1, dtype=bool)
            for lab_i in range(1, nlab + 1):
                if areas[lab_i] <= max_dot:
                    ys, xs = np.where(labeled == lab_i)
                    ink_i = int(flat_ink_idx[ys[0], xs[0]])
                    if 0 <= ink_i < len(inks) and ink_lums[ink_i] >= 40.0:
                        small[lab_i] = True
            thin |= small[labeled]
    except Exception:
        pass
    return thin


def detect_shade_mask(
    src_rgb: np.ndarray,
    flat_rgb: np.ndarray,
    flat_mask: np.ndarray,
    line_mask: np.ndarray,
    inks: np.ndarray,
    range_min: float = 4.0,
) -> np.ndarray:
    """Soft tonal ramps + low-contrast translucent overlays (ghost flowers)."""
    del inks
    lum_u8 = np.clip(gr._luminance(src_rgb), 0, 255).astype(np.uint8)
    rng3 = _local_range(lum_u8, 3)
    rng5 = _local_range(lum_u8, 5)
    rng15 = _local_range(lum_u8, 15)
    paper = np.max(np.abs(src_rgb.astype(np.float64) - 255.0), axis=-1) <= 3.0
    soft_ramp = (rng5 >= range_min) & (rng15 >= rng5 + 2.0)
    hard_edge = (rng3 >= 8.0) & (rng15 <= rng3 + 6.0)
    # Ghost / translucent overlays: mild local change, smooth over wide window,
    # source differs from the flat posterisation (faded flowers on purple).
    src_lab = gr.rgb_to_lab(src_rgb).reshape(-1, 3)
    flat_lab = gr.rgb_to_lab(flat_rgb).reshape(-1, 3)
    de_flat = np.sqrt(((src_lab - flat_lab) ** 2).sum(-1)).reshape(src_rgb.shape[:2])
    ghost = (
        (rng5 >= 2.0) & (rng5 < 9.0) & (rng15 >= rng5 + 2.0)
        & (de_flat >= 5.0) & ~hard_edge & ~paper
    )
    shade = (soft_ramp | ghost) & ~flat_mask & ~line_mask & ~paper & ~hard_edge
    # Hard poster edges create a soft halo in range filters — keep that as solid, not spray
    near_hard = gr._dilate_bool(hard_edge, 2)
    shade = shade & ~near_hard
    return shade


def classify_auto_regions(
    src_rgb: np.ndarray,
    flat_rgb: np.ndarray,
    inks: np.ndarray,
) -> dict[str, Any]:
    """Return FLAT / LINE / SPRAY masks + flat ink index."""
    flat_mask, flat_ink = gr.classify_flat_shaded(src_rgb, flat_rgb, inks)
    ink_lums = np.array([gr._lum1(c) for c in inks], dtype=np.float64)
    dark = np.zeros(flat_mask.shape, dtype=bool)
    for i, lu in enumerate(ink_lums):
        if lu < 40.0:
            dark |= (flat_ink == i) & (gr._luminance(src_rgb) < 50)
    flat_mask = flat_mask | dark
    # Demote flat islands that sit on soft source ramps (cream petals), not hard edges
    lum_u8 = np.clip(gr._luminance(src_rgb), 0, 255).astype(np.uint8)
    rng3 = _local_range(lum_u8, 3)
    rng5 = _local_range(lum_u8, 5)
    rng15 = _local_range(lum_u8, 15)
    hard_edge = (rng3 >= 8.0) & (rng15 <= rng3 + 6.0)
    near_hard = gr._dilate_bool(hard_edge, 3)
    # Only when source actually differs from flat posterisation (true ramps/ghosts).
    # Hard-edge halos have high rng15 but de_flat≈0 when src≈flat.
    src_lab = gr.rgb_to_lab(src_rgb).reshape(-1, 3)
    flat_lab = gr.rgb_to_lab(flat_rgb).reshape(-1, 3)
    de_flat = np.sqrt(((src_lab - flat_lab) ** 2).sum(-1)).reshape(src_rgb.shape[:2])
    soft_cand = (
        flat_mask & ~dark & (rng15 >= 8.0) & (rng15 >= rng5 + 3.0)
        & (de_flat >= 6.0) & ~hard_edge & ~near_hard
    )
    # Only large soft components (petal bodies), not thin edge crumbs
    soft_in_flat = np.zeros_like(soft_cand)
    if soft_cand.any():
        from scipy import ndimage
        lab, n = ndimage.label(soft_cand)
        areas = np.bincount(lab.ravel())
        min_soft = max(200, (soft_cand.shape[0] * soft_cand.shape[1]) // 400)
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = areas[1:] >= min_soft
        soft_in_flat = keep[lab]
    flat_mask = flat_mask & ~soft_in_flat
    line_mask = detect_line_mask(flat_rgb, flat_ink, inks) & ~flat_mask
    shade_mask = detect_shade_mask(src_rgb, flat_rgb, flat_mask, line_mask, inks)
    shade_mask = shade_mask | soft_in_flat  # cream ramps → spray
    other = ~(flat_mask | line_mask | shade_mask)
    solid_mask = flat_mask | line_mask | other
    return {
        "flat_mask": flat_mask,
        "line_mask": line_mask,
        "shade_mask": shade_mask,
        "solid_mask": solid_mask,
        "flat_ink_idx": flat_ink,
        "fractions": {
            "flat": round(float(100.0 * flat_mask.mean()), 3),
            "line": round(float(100.0 * line_mask.mean()), 3),
            "spray": round(float(100.0 * shade_mask.mean()), 3),
            "solid": round(float(100.0 * solid_mask.mean()), 3),
        },
    }


def flat_to_index(flat_rgb: np.ndarray, inks: np.ndarray) -> np.ndarray:
    """Map each flat pixel to nearest ink index (uint8)."""
    h, w, _ = flat_rgb.shape
    ink_lab = gr.rgb_to_lab(inks.reshape(-1, 1, 3)).reshape(-1, 3)
    flat_lab = gr.rgb_to_lab(flat_rgb).reshape(-1, 3)
    d2 = ((flat_lab[:, None, :] - ink_lab[None, :, :]) ** 2).sum(-1)
    return np.argmin(d2, axis=1).astype(np.uint8).reshape(h, w)


def index_to_flat(index: np.ndarray, inks: np.ndarray) -> np.ndarray:
    out = np.zeros(index.shape + (3,), dtype=np.uint8)
    for i, ink in enumerate(inks):
        out[index == i] = ink
    return out


def remove_edge_specks(index: np.ndarray, max_px: int = 6) -> tuple[np.ndarray, int]:
    """Drop tiny crumb/sliver islands; keep compact real dots.

    An island ≤max_px is removed when it is a *sliver* (touches ≥2 other inks,
    or is very thin). Compact blobs that touch only one neighbour ink (a cream
    dot on a ground) are kept — those exist as dots in the source.
    """
    from scipy import ndimage
    out = index.copy()
    st = np.ones((3, 3), int)
    moved = 0
    K = int(index.max()) + 1
    for k in range(K):
        lab, n = ndimage.label(index == k, structure=st)
        if not n:
            continue
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        slices = ndimage.find_objects(lab)
        for c in range(1, n + 1):
            if sizes[c] > max_px or sizes[c] == 0:
                continue
            sl = slices[c - 1]
            y0, x0 = max(sl[0].start - 1, 0), max(sl[1].start - 1, 0)
            y1, x1 = min(sl[0].stop + 1, index.shape[0]), min(sl[1].stop + 1, index.shape[1])
            m = lab[y0:y1, x0:x1] == c
            ring = ndimage.binary_dilation(m, structure=st) & ~m
            nb = index[y0:y1, x0:x1][ring]
            nb = nb[nb != k]
            if len(nb) == 0:
                continue  # floating — keep
            n_inks = len(np.unique(nb))
            bh = sl[0].stop - sl[0].start
            bw = sl[1].stop - sl[1].start
            thin = min(bh, bw) <= 2 and max(bh, bw) >= 4
            # Compact single-neighbour island = real dot on a ground → keep
            if n_inks == 1 and not thin:
                continue
            out[y0:y1, x0:x1][m] = int(np.bincount(nb).argmax())
            moved += 1
    return out, moved


def reassign_line_inks_from_source(
    flat_ink_idx: np.ndarray,
    line_mask: np.ndarray,
    src_rgb: np.ndarray,
    inks: np.ndarray,
) -> np.ndarray:
    """Dark outline pixels → nearest true dark ink (source core median), never khaki AA mix.

    Light thin strokes (mustard stems, cream rims) keep their light inks. Only pixels
    whose *source* is dark are forced onto a dark ink — both LINE-tagged and any
    mid-tone flat ink that was an AA/khaki stand-in for a real dark outline.
    """
    out = flat_ink_idx.copy()
    ink_lums = np.array([gr._lum1(c) for c in inks], dtype=np.float64)
    dark_idx = [i for i, lu in enumerate(ink_lums) if lu < 70.0]
    if not dark_idx:
        return out
    ink_lab = gr.rgb_to_lab(inks.reshape(-1, 1, 3)).reshape(-1, 3)
    dark_lab = ink_lab[dark_idx]
    src_lum = gr._luminance(src_rgb)
    # Candidates: LINE tagged as dark in source, OR mid-ink pixels that are dark in source
    mid = np.zeros(flat_ink_idx.shape, dtype=bool)
    for i, lu in enumerate(ink_lums):
        if 50.0 <= lu <= 165.0:
            mid |= flat_ink_idx == i
    cand = (src_lum < 85.0) & (line_mask | mid)
    if not cand.any():
        return out
    core = gr._erode_bool(cand, 1)
    use = core if core.sum() >= 20 else cand
    med = np.median(src_rgb[use].astype(np.float64), axis=0)
    c_lab = gr.rgb_to_lab(np.clip(med, 0, 255).reshape(1, 1, 3).astype(np.uint8)).reshape(3)
    j = dark_idx[int(np.argmin(((dark_lab - c_lab) ** 2).sum(-1)))]
    out[cand] = j
    return out


def unify_dark_outline_rings(
    flat_ink_idx: np.ndarray,
    src_rgb: np.ndarray,
    inks: np.ndarray,
) -> np.ndarray:
    """Force thin dark rings around light fills onto ONE dark outline ink.

    Prevents purple/brown/black fragmentation of the same ring (shared SDF then
    keeps a single boundary). Large dark grounds are left untouched.
    """
    out = flat_ink_idx.copy()
    ink_lums = np.array([gr._lum1(c) for c in inks], dtype=np.float64)
    dark_idx = [i for i, lu in enumerate(ink_lums) if lu < 70.0]
    light_idx = [i for i, lu in enumerate(ink_lums) if lu > 140.0]
    if not dark_idx or not light_idx:
        return out
    dark = np.isin(flat_ink_idx, dark_idx)
    light = np.isin(flat_ink_idx, light_idx)
    if not dark.any() or not light.any():
        return out
    # Thin dark shell (ring/stroke), not the filled ground body
    core = gr._erode_bool(dark, 2)
    thin_dark = dark & ~core
    ring = thin_dark & gr._dilate_bool(light, 3)
    if not ring.any():
        return out
    # Grow one px into the dark side so the full ring width is covered
    ring = gr._dilate_bool(ring, 1) & dark
    use = gr._erode_bool(ring, 1)
    use = use if use.any() else ring
    med = np.median(src_rgb[use].astype(np.float64), axis=0)
    ink_lab = gr.rgb_to_lab(inks.reshape(-1, 1, 3)).reshape(-1, 3)
    dark_lab = ink_lab[dark_idx]
    c_lab = gr.rgb_to_lab(np.clip(med, 0, 255).reshape(1, 1, 3).astype(np.uint8)).reshape(3)
    j = dark_idx[int(np.argmin(((dark_lab - c_lab) ** 2).sum(-1)))]
    out[ring] = j
    return out


def smooth_index_sdf(
    index: np.ndarray,
    *,
    scale: float = 2.0,
    sigma_px: float = 1.25,
    log=print,
    tile_px: int = 768,
    overlap: int = 48,
) -> tuple[np.ndarray, dict]:
    """Supersample a label map with shared SDF boundaries (no gaps/slivers).

    For each tile: NEAREST-upsample labels → per-label signed distance → Gaussian
    smooth → argmax across labels at the hi-res grid. Neighbouring inks therefore
    share one boundary. Processes tiles so peak RAM stays modest.
    """
    from scipy import ndimage
    import gc

    scale = float(scale)
    H, W = index.shape
    oh, ow = int(round(H * scale)), int(round(W * scale))
    out_mp = oh * ow / 1e6
    # sigma in hi-res pixels
    sig = max(0.6, float(sigma_px) * scale)

    def _sdf_argmax_tile(tile: np.ndarray) -> np.ndarray:
        th, tw = tile.shape
        sh, sw = int(round(th * scale)), int(round(tw * scale))
        up = np.asarray(
            Image.fromarray(tile, mode="L").resize((sw, sh), Image.Resampling.NEAREST)
        )
        labels = np.unique(up)
        if len(labels) == 1:
            return up.astype(np.uint8)
        # scores: higher = more inside that label
        best_score = np.full((sh, sw), -1e9, dtype=np.float32)
        best_lab = np.zeros((sh, sw), dtype=np.uint8)
        for lab in labels:
            m = up == lab
            # EDT in pixels of the hi-res grid
            dist_in = ndimage.distance_transform_edt(m).astype(np.float32)
            dist_out = ndimage.distance_transform_edt(~m).astype(np.float32)
            sdf = dist_in - dist_out
            if sig > 0.5:
                sdf = ndimage.gaussian_filter(sdf, sigma=sig, mode="nearest")
            better = sdf > best_score
            best_score[better] = sdf[better]
            best_lab[better] = lab
        return best_lab

    if out_mp <= 12.0 or max(H, W) <= tile_px:
        cleaned = _sdf_argmax_tile(index)
        erep = {"method": "sdf_argmax", "tiled": False, "sigma_hi": sig}
    else:
        log(f"[auto] edges SDF: tiling {W}x{H} → {scale}× (tile={tile_px}, overlap={overlap}) …")
        cleaned = np.zeros((oh, ow), np.uint8)
        step = max(1, tile_px - overlap)
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                y1 = min(H, y0 + tile_px)
                x1 = min(W, x0 + tile_px)
                # pad read for EDT context
                py0 = max(0, y0 - overlap)
                px0 = max(0, x0 - overlap)
                py1 = min(H, y1 + overlap)
                px1 = min(W, x1 + overlap)
                pad = index[py0:py1, px0:px1]
                tclean = _sdf_argmax_tile(pad)
                # crop to the core tile in hi-res
                cy0 = int(round((y0 - py0) * scale))
                cx0 = int(round((x0 - px0) * scale))
                ch = int(round((y1 - y0) * scale))
                cw = int(round((x1 - x0) * scale))
                ty0 = int(round(y0 * scale))
                tx0 = int(round(x0 * scale))
                cleaned[ty0:ty0 + ch, tx0:tx0 + cw] = tclean[cy0:cy0 + ch, cx0:cx0 + cw]
                del tclean, pad
            if y0 % (step * 4) == 0:
                gc.collect()
        erep = {
            "method": "sdf_argmax", "tiled": True,
            "tile_px": tile_px, "overlap": overlap, "sigma_hi": sig,
        }
    erep["scale"] = float(scale)
    erep["outlines_smoothed"] = -1  # N/A for SDF path
    erep["kept_as_is"] = 0
    return cleaned.astype(np.uint8), erep


def drop_hires_islands(
    index: np.ndarray,
    *,
    max_px: int,
    keep_isolated: bool = True,
) -> tuple[np.ndarray, int]:
    """Drop islands smaller than max_px that touch another ink (crumbs/slivers).

    Isolated islands (no other-ink neighbour) are kept when keep_isolated — those
    are real dots. max_px should be ~(4*ss)**2 at the supersampled grid.
    """
    return remove_edge_specks(index, max_px=max_px)


def smooth_flat_edges(
    flat_rgb: np.ndarray,
    inks: np.ndarray,
    *,
    strength: int = 2,
    scale: float = 2.0,
    log=print,
    tile_px: int = 768,
    overlap: int = 48,
) -> tuple[np.ndarray, dict]:
    """Pen-tool edges via shared-label SDF at supersampled resolution.

    All inks compete in one SDF argmax so neighbours share a single boundary
    (no purple/cream slivers between ring and fill). Hi-res 1-bit plates are
    kept (never downsampled). Tiny edge crumbs (<(4·ss)² px) that touch another
    ink are removed; isolated dots are kept.
    """
    import gc
    del strength  # retained for API compat; SDF sigma is scale-linked
    index = flat_to_index(flat_rgb, inks)
    # Pre-clean crumbs at 1×
    index, n_speck_1x = remove_edge_specks(index, max_px=6)
    cleaned, erep = smooth_index_sdf(
        index, scale=scale, sigma_px=1.25, log=log, tile_px=tile_px, overlap=overlap,
    )
    # Hi-res island drop: ~(4*ss)^2
    ss = float(scale)
    max_hi = int(max(16, round((4.0 * ss) ** 2)))
    cleaned, n_speck_hi = drop_hires_islands(cleaned, max_px=max_hi, keep_isolated=True)
    cleaned = np.clip(cleaned, 0, len(inks) - 1).astype(np.uint8)
    # Build RGB flat; for huge canvases do it in row strips to limit peak RAM
    oh, ow = cleaned.shape
    if oh * ow > 40_000_000:
        flat2 = np.empty((oh, ow, 3), dtype=np.uint8)
        for y0 in range(0, oh, 512):
            y1 = min(oh, y0 + 512)
            flat2[y0:y1] = inks[cleaned[y0:y1]]
            if y0 % 2048 == 0:
                gc.collect()
    else:
        flat2 = index_to_flat(cleaned, inks)
    del cleaned, index
    gc.collect()
    erep = dict(erep)
    erep["edge_specks_removed"] = int(n_speck_1x + n_speck_hi)
    erep["edge_specks_1x"] = int(n_speck_1x)
    erep["edge_specks_hires"] = int(n_speck_hi)
    erep["island_max_px"] = int(max_hi)
    erep["kept_hires"] = True
    erep["hires_scale"] = float(scale)
    log(
        f"[auto] edges SDF: specks_1x={n_speck_1x} specks_hi={n_speck_hi} "
        f"(max_island={max_hi}) hires={flat2.shape[1]}x{flat2.shape[0]} "
        f"(scale={scale} tiled={erep.get('tiled')})"
    )
    return flat2, erep



def apply_auto_tones(
    tones: np.ndarray,
    regions: dict[str, Any],
    n_inks: int,
) -> np.ndarray:
    """Solid 100% ink on FLAT+LINE+other; keep continuous tones only on SPRAY."""
    flat_ink = regions["flat_ink_idx"]
    solid = regions["solid_mask"]
    out = gr.apply_hybrid_tones(tones, solid, flat_ink, n_inks, feather_px=0)
    return out


def _ground_std(composite: np.ndarray, flat_rgb: np.ndarray, solid_mask: np.ndarray) -> float:
    ground = (flat_rgb.max(axis=-1) < 40) & solid_mask
    if not ground.any():
        # purple/dark grounds: lowest-luminance solid band
        lum = gr._luminance(composite)
        dark = solid_mask & (gr._luminance(flat_rgb) < 45)
        ground = dark
    if not ground.any():
        return 0.0
    glum = gr._luminance(composite)[ground].astype(np.float64)
    return float(glum.std())


def _stream_hires_plates(
    *,
    after_path: str,
    tones_1x: np.ndarray,
    regions_1x: dict,
    inks: np.ndarray,
    w: int,
    h: int,
    w2: int,
    h2: int,
    out_dpi: int,
    adir: str,
    base: str,
    out_dir: str,
    halftone: str,
    write_psd: bool,
    include_composite: bool,
    use_layers: bool,
    paper: tuple,
    log,
) -> tuple[list, str, dict]:
    """Write 2× tone/film/composite from disk flat + 1× tones without holding full masks."""
    import gc
    from PIL import Image as _Image
    _Image.MAX_IMAGE_PIXELS = None
    TILE = 768
    tone_dir = os.path.join(adir, "tone_channels")
    film_dir = os.path.join(adir, "film")
    os.makedirs(tone_dir, exist_ok=True)
    os.makedirs(film_dir, exist_ok=True)
    solid_1x = (
        regions_1x["flat_mask"] | regions_1x["line_mask"]
        | ~(regions_1x["flat_mask"] | regions_1x["line_mask"] | regions_1x["shade_mask"])
    )
    fi1 = regions_1x["flat_ink_idx"]
    tone_plates: list[dict[str, Any]] = []
    after_im = _Image.open(after_path)
    for i, ink in enumerate(inks):
        hx = hex_of(ink)
        log(f"[auto] stream plate {i+1}/{len(inks)} #{hx}")
        tone_hi = _Image.new("L", (w2, h2))
        film_hi = _Image.new("L", (w2, h2))
        ch = tones_1x[..., i]
        mine = solid_1x & (fi1 == i)
        other = solid_1x & (fi1 != i)
        for y0 in range(0, h, TILE):
            for x0 in range(0, w, TILE):
                y1, x1 = min(h, y0 + TILE), min(w, x0 + TILE)
                Y0, X0, Y1, X1 = y0 * 2, x0 * 2, y1 * 2, x1 * 2
                tw, th = X1 - X0, Y1 - Y0
                u8 = np.clip(np.rint(ch[y0:y1, x0:x1] * 255), 0, 255).astype(np.uint8)
                cov = np.asarray(
                    _Image.fromarray(u8, "L").resize((tw, th), _Image.Resampling.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
                m_up = np.asarray(
                    _Image.fromarray(mine[y0:y1, x0:x1].astype(np.uint8) * 255, "L").resize(
                        (tw, th), _Image.Resampling.NEAREST
                    )
                ) > 0
                o_up = np.asarray(
                    _Image.fromarray(other[y0:y1, x0:x1].astype(np.uint8) * 255, "L").resize(
                        (tw, th), _Image.Resampling.NEAREST
                    )
                ) > 0
                cov[m_up] = 1.0
                cov[o_up] = 0.0
                tone_u8 = np.clip(np.rint(cov * 255), 0, 255).astype(np.uint8)
                tone_hi.paste(_Image.fromarray(tone_u8, "L"), (X0, Y0))
                if halftone == "am":
                    dots = gr.am_halftone(cov.astype(np.float64), lpi=45, dpi=out_dpi,
                                         angle_deg=gr._SCREEN_ANGLES[i % len(gr._SCREEN_ANGLES)])
                else:
                    dots = gr.fm_halftone(cov.astype(np.float64), seed=i + 1 + y0 + x0)
                film_hi.paste(_Image.fromarray(dots, "L"), (X0, Y0))
                del cov, tone_u8, dots, u8
        tone_path = os.path.join(tone_dir, f"TONE_{i+1:02d}_{hx}.png")
        film_path = os.path.join(film_dir, f"INK_{i+1:02d}_{hx}_film.png")
        save_png(tone_hi, tone_path, out_dpi)
        save_png(film_hi, film_path, out_dpi)
        tone_plates.append({
            "name": f"INK_{i+1:02d}", "hex": "#" + hx,
            "ink_rgb": [int(x) for x in ink],
            "tone_path": tone_path, "film_path": film_path,
            "mean_tone_percent": round(float(ch.mean() * 100), 3),
        })
        del tone_hi, film_hi
        gc.collect()

    # 1× composite for metrics; 2× preview snapped from flat file
    comp_1x = np.array(gr.overprint_rgb(tones_1x, inks, paper), copy=True)
    for i, ink in enumerate(inks):
        pix = solid_1x & (fi1 == i)
        if pix.any():
            comp_1x[pix] = ink
    comp = _Image.fromarray(comp_1x).resize((w2, h2), _Image.Resampling.BILINEAR)
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            y1, x1 = min(h, y0 + TILE), min(w, x0 + TILE)
            Y0, X0, Y1, X1 = y0 * 2, x0 * 2, y1 * 2, x1 * 2
            solid_t = solid_1x[y0:y1, x0:x1]
            if not solid_t.any():
                continue
            solid_up = np.asarray(
                _Image.fromarray(solid_t.astype(np.uint8) * 255, "L").resize(
                    (X1 - X0, Y1 - Y0), _Image.Resampling.NEAREST
                )
            ) > 0
            flat_t = np.asarray(after_im.crop((X0, Y0, X1, Y1)).convert("RGB"))
            c_t = np.array(comp.crop((X0, Y0, X1, Y1)), dtype=np.uint8, copy=True)
            c_t[solid_up] = flat_t[solid_up]
            comp.paste(_Image.fromarray(c_t), (X0, Y0))
    comp_path = os.path.join(adir, f"{base}_auto_composite.png")
    save_png(comp, comp_path, out_dpi)
    del comp, after_im
    gc.collect()
    pkg = sp.write_tone_package(
        out_dir, base, tone_plates, dpi=out_dpi, write_psd=write_psd,
        composite_path=comp_path, include_composite=include_composite,
        use_layers=use_layers, log=log,
    )
    # metrics at 1×
    return tone_plates, comp_path, {
        "comp_1x": comp_1x, "solid_1x": solid_1x, "package": pkg,
    }



def run_auto(
    image_path: str,
    out_dir: str,
    *,
    name: str | None = None,
    colors: int = 8,
    dpi: int = 300,
    size: int | None = None,
    allow_downscale: bool = False,
    min_shade_percent: float = 0.25,
    write_psd: bool = True,
    include_composite: bool = False,
    use_layers: bool = False,
    halftone: str = "fm",
    edge_scale: float = 2.0,
    log=print,
) -> dict[str, Any]:
    """Analyse design regions and apply FLAT / LINE / SPRAY only where present.

    NEVER downscales below the source pixel size unless ``allow_downscale=True``
    (explicit opt-in). Edge/film plates are kept at ``edge_scale``× (default 2×)
    supersampled 1-bit resolution; output DPI is ``dpi * edge_scale``.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = name or os.path.splitext(os.path.basename(image_path))[0]
    base = base.replace("#", "").replace(" ", "_") or "auto"
    src_im = Image.open(image_path).convert("RGB")
    nat_w, nat_h = src_im.size
    if size is None:
        work_w = nat_w
    else:
        work_w = int(size)
        if work_w < nat_w and not allow_downscale:
            log(f"[auto] REFUSING downscale {work_w} < native {nat_w}; "
                f"using full source ({nat_w}px). Pass allow_downscale=True to override.")
            work_w = nat_w
        elif work_w < nat_w:
            log(f"[auto] WARNING: allow_downscale — working at {work_w}px < native {nat_w}px")
    log(f"[auto] numbering flat solids ({work_w}px native={nat_w}x{nat_h}, colors={colors}) …")
    num_rep = num.number(
        image_path, out_dir, name=base, size=work_w, colours=colors,
        dpi=dpi, log=log, detail_rescue=True,
    )
    import gc
    gc.collect()
    w, h = int(num_rep["size_px"][0]), int(num_rep["size_px"][1])
    # Solid colour plates + PSD/TIF (darkest-first applied in package rewrite below)
    from .cli import number_package
    number_package(
        base, num_rep.get("sketch"), num_rep.get("csv"), num_rep.get("bold_numbers"),
        out_dir, (w, h), dpi, write_psd,
    )
    flat_path = os.path.join(out_dir, f"{base}_flat.png")
    src_1x = gr._load_rgb(image_path, (w, h))
    flat_1x = gr._load_rgb(flat_path)
    if flat_1x.shape[:2] != (h, w):
        flat_1x = np.asarray(Image.fromarray(flat_1x).resize((w, h), Image.Resampling.NEAREST), dtype=np.uint8)

    inks = gr.pick_inks(flat_1x, colors, src_rgb=src_1x)
    adir = os.path.join(out_dir, "auto")
    os.makedirs(adir, exist_ok=True)
    # Region analysis + tone solve on native 1x (hi-res bilinear would fake shade at edges)
    regions_1x = classify_auto_regions(src_1x, flat_1x, inks)
    fi_before = regions_1x["flat_ink_idx"]
    regions_1x["flat_ink_idx"] = reassign_line_inks_from_source(
        fi_before, regions_1x["line_mask"], src_1x, inks
    )
    regions_1x["flat_ink_idx"] = unify_dark_outline_rings(
        regions_1x["flat_ink_idx"], src_1x, inks
    )
    # Bake all dark-outline reassignments (LINE + mid-ink khaki stand-ins)
    flat_1x = np.array(flat_1x, copy=True)
    fi1 = regions_1x["flat_ink_idx"]
    changed = fi1 != fi_before
    if changed.any():
        flat_1x[changed] = inks[fi1[changed]]
    for i, ink in enumerate(inks):
        pix = regions_1x["line_mask"] & (fi1 == i)
        if pix.any():
            flat_1x[pix] = ink
    # Ring/fill contact band must stay SOLID (no spray AA cream/purple slivers)
    ink_lums = np.array([gr._lum1(c) for c in inks], dtype=np.float64)
    dark_m = np.isin(fi1, [i for i, lu in enumerate(ink_lums) if lu < 70.0])
    light_m = np.isin(fi1, [i for i, lu in enumerate(ink_lums) if lu > 140.0])
    contact = (
        (dark_m & gr._dilate_bool(light_m, 2)) | (light_m & gr._dilate_bool(dark_m, 2))
    )
    if contact.any():
        regions_1x["shade_mask"] = regions_1x["shade_mask"] & ~contact
        regions_1x["flat_mask"] = regions_1x["flat_mask"] | contact
        regions_1x["line_mask"] = regions_1x["line_mask"] & ~contact

    edge_scale = float(edge_scale)
    if not 1.0 <= edge_scale <= 4.0:
        raise ValueError("edge_scale must be 1..4")
    flat_before = flat_1x.copy()
    before_path = os.path.join(adir, f"{base}_flat_before_edges.png")
    save_png(to_image(flat_before), before_path, dpi)
    log(f"[auto] pen-tool edge clean KEEP {edge_scale:.0f}x hi-res 1-bit plates …")
    flat, edge_rep = smooth_flat_edges(flat_1x, inks, strength=3, scale=edge_scale, log=log)
    flat = np.array(flat, copy=True)
    h2, w2 = flat.shape[:2]
    out_dpi = int(round(dpi * edge_scale))
    # Hi-res source only for metrics / LINE core colour (not for shade classify)
    src = np.asarray(
        Image.fromarray(src_1x).resize((w2, h2), Image.Resampling.BILINEAR), dtype=np.uint8
    )
    after_path = os.path.join(adir, f"{base}_flat_after_edges.png")
    save_png(to_image(flat), after_path, out_dpi)
    save_png(to_image(flat), flat_path, out_dpi)
    stream = (w2 * h2) > 35_000_000
    if stream:
        log(f"[auto] large hi-res {w2}x{h2} — streaming plates (peak RAM cap)")
        import gc
        del flat
        gc.collect()

    frac = regions_1x["fractions"]
    log(f"[auto] regions: flat={frac['flat']}% line={frac['line']}% spray={frac['spray']}% "
        f"plates={w2}x{h2} @ {out_dpi} DPI ({edge_scale:.0f}x supersample)")
    report_size_note = {"hires_scale": edge_scale, "output_dpi": out_dpi, "base_dpi": int(round(out_dpi / edge_scale))}
    report: dict[str, Any] = {
        "name": base, "mode": "auto", "size_px": [w2, h2], "dpi": out_dpi,
        "colors": colors, "n_inks": int(len(inks)),
        "inks": ["#" + hex_of(c) for c in inks],
        "region_fractions": frac,
        "flat_before_edges": before_path, "flat_after_edges": after_path,
        "edges": edge_rep, "hires": report_size_note,
        "number": {k: num_rep.get(k) for k in ("name", "size_px", "colours", "missed") if k in num_rep},
        "applied": False,
    }
    use_spray = float(frac["spray"]) >= float(min_shade_percent)
    paper = (255, 255, 255)
    tone_plates: list[dict[str, Any]] = []
    comp_path = None
    w, h = w2, h2
    dpi = out_dpi

    if stream:
        # Classify already done at 1x; solve tones at 1x; stream 2x plates
        solid_1x = (
            regions_1x["flat_mask"] | regions_1x["line_mask"]
            | ~(regions_1x["flat_mask"] | regions_1x["line_mask"] | regions_1x["shade_mask"])
        )
        if not use_spray:
            log("[auto] no real shading → solid plates only (streamed hi-res)")
            tones_1x = np.zeros((regions_1x["flat_ink_idx"].shape[0], regions_1x["flat_ink_idx"].shape[1], len(inks)), np.float32)
            for i in range(len(inks)):
                tones_1x[..., i] = (regions_1x["flat_ink_idx"] == i).astype(np.float32)
            report["spray_pixels"] = 0
            report["tone_pixels"] = 0
            report["auto_mode"] = "flat_line_solid_only"
        else:
            log(f"[auto] solving spray tones at 1x on {frac['spray']}% …")
            tones_1x = gr.solve_tones(
                src_1x, inks, paper=paper, max_inks_per_pixel=max(5, len(inks)),
                iters=160, seal_motifs=False,
            )
            tones_1x = np.clip(tones_1x, 0.0, 1.0)
            tones_1x = apply_auto_tones(tones_1x, {
                "flat_ink_idx": regions_1x["flat_ink_idx"], "solid_mask": solid_1x,
            }, len(inks))
            report["spray_pixels"] = int(regions_1x["shade_mask"].sum())
            report["tone_pixels"] = report["spray_pixels"]
            report["auto_mode"] = "flat_line_solid_plus_spray_shade"
        tone_plates, comp_path, extra = _stream_hires_plates(
            after_path=after_path, tones_1x=tones_1x, regions_1x=regions_1x, inks=inks,
            w=src_1x.shape[1], h=src_1x.shape[0], w2=w2, h2=h2, out_dpi=out_dpi,
            adir=adir, base=base, out_dir=out_dir, halftone=halftone,
            write_psd=write_psd, include_composite=include_composite, use_layers=use_layers,
            paper=paper, log=log,
        )
        report["package"] = extra["package"]
        report["tone_plates"] = tone_plates
        # Stash 1x fields for metrics
        regions = {
            "flat_mask": regions_1x["flat_mask"],
            "line_mask": regions_1x["line_mask"],
            "shade_mask": regions_1x["shade_mask"],
            "solid_mask": solid_1x,
            "flat_ink_idx": regions_1x["flat_ink_idx"],
            "fractions": frac,
        }
        src = src_1x
        flat = gr._load_rgb(before_path)  # 1x for ground_std
        comp_path_metrics = None
        # Write a 1x composite path note; metrics use extra["comp_1x"] via saving
        save_png(to_image(extra["comp_1x"]), os.path.join(adir, f"{base}_auto_composite_1x.png"), int(round(out_dpi / edge_scale)))
        # Point metrics at 1x composite (same geometry as src)
        comp_path = os.path.join(adir, f"{base}_auto_composite_1x.png")
        # Also keep hi-res composite path in report
        report["composite_hires"] = os.path.join(adir, f"{base}_auto_composite.png")
    else:
        def _up_mask(m):
            return np.asarray(
                Image.fromarray(m.astype(np.uint8) * 255, mode="L").resize((w2, h2), Image.Resampling.NEAREST)
            ) > 0

        src = np.asarray(
            Image.fromarray(src_1x).resize((w2, h2), Image.Resampling.BILINEAR), dtype=np.uint8
        )
        line_hi = _up_mask(regions_1x["line_mask"])
        flat_ink = flat_to_index(flat, inks)
        fi_before_hi = flat_ink.copy()
        flat_ink = reassign_line_inks_from_source(flat_ink, line_hi, src, inks)
        flat_ink = unify_dark_outline_rings(flat_ink, src, inks)
        changed_hi = flat_ink != fi_before_hi
        if changed_hi.any():
            flat[changed_hi] = inks[flat_ink[changed_hi]]
        for i, ink in enumerate(inks):
            pix = line_hi & (flat_ink == i)
            if pix.any():
                flat[pix] = ink
        regions = {
            "flat_mask": _up_mask(regions_1x["flat_mask"]),
            "line_mask": line_hi,
            "shade_mask": _up_mask(regions_1x["shade_mask"]),
            "flat_ink_idx": flat_ink,
            "fractions": frac,
        }
        regions["solid_mask"] = regions["flat_mask"] | regions["line_mask"] | (
            ~(regions["flat_mask"] | regions["line_mask"] | regions["shade_mask"])
        )
        rmap = np.zeros((h2, w2, 3), dtype=np.uint8)
        rmap[regions["flat_mask"]] = (40, 180, 40)
        rmap[regions["line_mask"]] = (40, 100, 220)
        rmap[regions["shade_mask"]] = (220, 120, 40)
        rmap_path = os.path.join(adir, f"{base}_region_map.png")
        save_png(to_image(rmap), rmap_path, dpi)
        report["region_map"] = rmap_path

        if not use_spray:
            log("[auto] no real shading → solid plates only (0 tone pixels) at hi-res")
            comp = flat.copy()
            comp_path = os.path.join(adir, f"{base}_auto_composite.png")
            save_png(to_image(comp), comp_path, dpi)
            report["spray_pixels"] = 0
            report["tone_pixels"] = 0
            report["auto_mode"] = "flat_line_solid_only"
            tone_dir = os.path.join(adir, "tone_channels")
            os.makedirs(tone_dir, exist_ok=True)
            fi = regions["flat_ink_idx"]
            for i, ink in enumerate(inks):
                hx = hex_of(ink)
                cov = (fi == i).astype(np.float64)
                tone_u8 = (cov * 255.0).astype(np.uint8)
                tone_path = os.path.join(tone_dir, f"TONE_{i+1:02d}_{hx}.png")
                save_png(Image.fromarray(tone_u8, mode="L"), tone_path, dpi)
                tone_plates.append({
                    "name": f"INK_{i+1:02d}", "hex": "#" + hx,
                    "ink_rgb": [int(x) for x in ink],
                    "tone_path": tone_path, "film_path": tone_path,
                    "mean_tone_percent": round(float(cov.mean() * 100), 3),
                })
            pkg = sp.write_tone_package(
                out_dir, base, tone_plates, dpi=dpi, write_psd=write_psd,
                composite_path=comp_path, include_composite=include_composite,
                use_layers=use_layers, log=log,
            )
            report["package"] = pkg
            report["tone_plates"] = tone_plates
        else:
            log(f"[auto] solving spray tones at 1x on {frac['spray']}% shaded pixels …")
            tones_1x = gr.solve_tones(
                src_1x, inks, paper=paper, max_inks_per_pixel=max(5, len(inks)),
                iters=160, seal_motifs=False,
            )
            tones_1x = np.clip(tones_1x, 0.0, 1.0)
            tones_1x = apply_auto_tones(tones_1x, {
                "flat_ink_idx": regions_1x["flat_ink_idx"],
                "solid_mask": (
                    regions_1x["flat_mask"] | regions_1x["line_mask"]
                    | ~(regions_1x["flat_mask"] | regions_1x["line_mask"] | regions_1x["shade_mask"])
                ),
            }, len(inks))
            shade_1x = regions_1x["shade_mask"]
            t_shade = tones_1x[shade_1x]
            partial = ((t_shade > 0.02) & (t_shade < 0.98)).any(axis=-1) if shade_1x.any() else np.array([])
            report["spray_pixels"] = int(regions["shade_mask"].sum())
            report["tone_pixels"] = int(partial.sum()) if getattr(partial, "size", 0) else 0
            report["auto_mode"] = "flat_line_solid_plus_spray_shade"
            solid = regions["solid_mask"]
            fi = regions["flat_ink_idx"]
            tone_dir = os.path.join(adir, "tone_channels")
            film_dir = os.path.join(adir, "film")
            os.makedirs(tone_dir, exist_ok=True)
            os.makedirs(film_dir, exist_ok=True)

            def _up_cov(ch_1x: np.ndarray, ink_i: int) -> np.ndarray:
                u8 = np.clip(np.rint(ch_1x * 255.0), 0, 255).astype(np.uint8)
                cov = np.asarray(
                    Image.fromarray(u8, mode="L").resize((w2, h2), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
                mine = solid & (fi == ink_i)
                other = solid & (fi != ink_i)
                if mine.any():
                    cov[mine] = 1.0
                if other.any():
                    cov[other] = 0.0
                return cov

            for i, ink in enumerate(inks):
                hx = hex_of(ink)
                cov = _up_cov(tones_1x[..., i], i)
                tone_u8 = np.clip(np.rint(cov * 255.0), 0, 255).astype(np.uint8)
                tone_path = os.path.join(tone_dir, f"TONE_{i+1:02d}_{hx}.png")
                save_png(Image.fromarray(tone_u8, mode="L"), tone_path, dpi)
                if halftone == "am":
                    angle = gr._SCREEN_ANGLES[i % len(gr._SCREEN_ANGLES)]
                    dots = gr.am_halftone(cov.astype(np.float64), lpi=45, dpi=dpi, angle_deg=angle)
                else:
                    dots = gr.fm_halftone(cov.astype(np.float64), seed=i + 1)
                    angle = None
                film_path = os.path.join(film_dir, f"INK_{i+1:02d}_{hx}_film.png")
                save_png(Image.fromarray(dots, mode="L"), film_path, dpi)
                tone_plates.append({
                    "name": f"INK_{i+1:02d}", "hex": "#" + hx,
                    "ink_rgb": [int(x) for x in ink], "angle_deg": angle,
                    "tone_path": tone_path, "film_path": film_path,
                    "mean_tone_percent": round(float(cov.mean() * 100), 3),
                })
                del cov, tone_u8, dots
            comp_1x = np.array(gr.overprint_rgb(tones_1x, inks, paper), copy=True)
            solid_1x = (
                regions_1x["flat_mask"] | regions_1x["line_mask"]
                | ~(regions_1x["flat_mask"] | regions_1x["line_mask"] | regions_1x["shade_mask"])
            )
            fi1 = regions_1x["flat_ink_idx"]
            for i, ink in enumerate(inks):
                pix = solid_1x & (fi1 == i)
                if pix.any():
                    comp_1x[pix] = ink
            comp = np.array(
                Image.fromarray(comp_1x).resize((w2, h2), Image.Resampling.BILINEAR),
                dtype=np.uint8, copy=True,
            )
            for i, ink in enumerate(inks):
                pix = solid & (fi == i)
                if pix.any():
                    comp[pix] = ink
            comp_path = os.path.join(adir, f"{base}_auto_composite.png")
            save_png(to_image(comp), comp_path, dpi)
            del tones_1x, comp_1x
            pkg = sp.write_tone_package(
                out_dir, base, tone_plates, dpi=dpi, write_psd=write_psd,
                composite_path=comp_path, include_composite=include_composite,
                use_layers=use_layers, log=log,
            )
            report["package"] = pkg
            report["tone_plates"] = tone_plates

    # Metrics
    comp_rgb = np.asarray(Image.open(comp_path).convert("RGB"))
    de_all = gr.delta_e_cie76(src, comp_rgb)
    de_flat = gr.delta_e_cie76(src, comp_rgb, regions["flat_mask"]) if regions["flat_mask"].any() else 0.0
    de_line = gr.delta_e_cie76(src, comp_rgb, regions["line_mask"]) if regions["line_mask"].any() else 0.0
    de_spray = gr.delta_e_cie76(src, comp_rgb, regions["shade_mask"]) if regions["shade_mask"].any() else 0.0
    gstd = _ground_std(comp_rgb, flat, regions["solid_mask"])
    report["delta_e_cie76"] = {
        "overall_mean": round(de_all, 3),
        "flat_mean": round(float(de_flat), 3),
        "line_mean": round(float(de_line), 3),
        "spray_mean": round(float(de_spray), 3),
    }
    report["ground_std"] = round(gstd, 4)
    report["composite"] = comp_path
    report["applied"] = True
    rp = os.path.join(adir, f"{base}_auto_report.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    report["report"] = rp
    log(f"[auto] ΔE overall={de_all:.2f} flat={de_flat:.2f} line={de_line:.2f} spray={de_spray:.2f} "
        f"ground_std={gstd:.4f}")
    return report


def make_compare_jpg(
    source_path: str,
    auto_composite: str,
    out_path: str,
    zoom_path: str | None = None,
    panel_w: int = 600,
) -> tuple[str, str | None]:
    """source | auto side-by-side (+ optional zoom crop).

    NEVER downscales the auto composite (preserves supersampled edges). Source is
    upscaled to the auto plate size when they differ.
    """
    src = Image.open(source_path).convert("RGB")
    auto = Image.open(auto_composite).convert("RGB")
    if auto.size != src.size:
        # Upscale source to plate size — never crush hi-res plates back down
        src = src.resize(auto.size, Image.Resampling.BILINEAR)

    def fit(im):
        return im.resize((panel_w, int(round(im.height * panel_w / im.width))), Image.Resampling.LANCZOS)

    panels = [fit(src), fit(auto)]
    H = max(p.height for p in panels)
    gap = 8
    canvas = Image.new("RGB", (panel_w * 2 + gap, H + 28), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    for i, (im, lab) in enumerate(zip(panels, ["source", "auto"])):
        y = 24 + (H - im.height) // 2
        canvas.paste(im, (i * (panel_w + gap), y))
        dr.text((i * (panel_w + gap) + 8, 4), lab, fill=(0, 0, 0))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path, quality=90, optimize=True)

    zpath = zoom_path
    if zpath:
        arr = np.asarray(auto)
        # pick a high-contrast interior point
        lum = gr._luminance(arr)
        gy = np.abs(np.diff(lum.astype(np.float32), axis=0))
        score = np.pad(gy, ((0, 1), (0, 0)))
        # avoid borders
        m = np.zeros_like(score, dtype=bool)
        m[arr.shape[0] // 5: 4 * arr.shape[0] // 5, arr.shape[1] // 5: 4 * arr.shape[1] // 5] = True
        score2 = np.where(m, score, -1)
        j = int(np.argmax(score2))
        cy, cx = divmod(j, arr.shape[1])
        cw = min(220, arr.shape[0] // 4, arr.shape[1] // 4)
        y0, x0 = max(0, cy - cw // 2), max(0, cx - cw // 2)
        y1, x1 = min(arr.shape[0], y0 + cw), min(arr.shape[1], x0 + cw)

        def crop3(im):
            c = im.crop((x0, y0, x1, y1))
            return c.resize((c.width * 3, c.height * 3), Image.Resampling.NEAREST)

        sc, ac = crop3(src), crop3(auto)
        z = Image.new("RGB", (sc.width * 2 + 16, sc.height + 28), (255, 255, 255))
        zdr = ImageDraw.Draw(z)
        for i, (im, lab) in enumerate([(sc, "source 3x"), (ac, "auto 3x")]):
            z.paste(im, (i * (sc.width + 8), 24))
            zdr.text((i * (sc.width + 8) + 8, 4), lab, fill=(0, 0, 0))
        z.save(zpath, quality=90, optimize=True)
    return out_path, zpath


def make_edges_zoom(
    source_path: str,
    mid_path: str,
    after_path: str,
    out_path: str,
    *,
    zoom: int = 4,
    crop: int = 120,
    mid_label: str = "e392b54 4x",
    after_label: str = "new 4x",
) -> str:
    """4x zoom crop: source | mid (prior) | after (new pen-tool hi-res)."""
    src = Image.open(source_path).convert("RGB")
    mid = Image.open(mid_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")
    # Compare in after's pixel space (hi-res plates)
    if mid.size != after.size:
        mid = mid.resize(after.size, Image.Resampling.NEAREST)
    if src.size != after.size:
        src = src.resize(after.size, Image.Resampling.BILINEAR)
    arr = np.asarray(after)
    lum = gr._luminance(arr).astype(np.float32)
    from PIL import ImageFilter as _IF
    im = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), mode="L")
    rng = np.asarray(im.filter(_IF.MaxFilter(3)), dtype=np.float32) - np.asarray(
        im.filter(_IF.MinFilter(3)), dtype=np.float32
    )
    msk = np.zeros_like(rng, dtype=bool)
    msk[arr.shape[0] // 6: 5 * arr.shape[0] // 6, arr.shape[1] // 6: 5 * arr.shape[1] // 6] = True
    score = np.where(msk & (lum > 30) & (lum < 230), rng, -1.0)
    j = int(np.argmax(score))
    cy, cx = divmod(j, arr.shape[1])
    cw = min(crop, arr.shape[0] // 3, arr.shape[1] // 3)
    y0, x0 = max(0, cy - cw // 2), max(0, cx - cw // 2)
    y1, x1 = min(arr.shape[0], y0 + cw), min(arr.shape[1], x0 + cw)

    def cropz(im0):
        c = im0.crop((x0, y0, x1, y1))
        return c.resize((c.width * zoom, c.height * zoom), Image.Resampling.NEAREST)

    panels = [cropz(src), cropz(mid), cropz(after)]
    labels = ["source 4x", mid_label, after_label]
    gap = 8
    canvas = Image.new("RGB", (panels[0].width * 3 + gap * 2, panels[0].height + 28), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    for i, (im0, lab) in enumerate(zip(panels, labels)):
        canvas.paste(im0, (i * (panels[0].width + gap), 24))
        dr.text((i * (panels[0].width + gap) + 8, 4), lab, fill=(0, 0, 0))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path, quality=92, optimize=True)
    return out_path
