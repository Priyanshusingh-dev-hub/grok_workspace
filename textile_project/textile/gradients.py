"""Optional multi-channel tonal overprint for `textile number --gradients` (OFF by default).

When on: pick the ink set from the flat numbering palette (<= --colors), then for
every pixel solve non-negative tone amounts (0..1) of up to 3 inks whose
multiply/overprint best matches the SOURCE colour. One shape may hold several
overlapping ink tones. Outputs:
  - greyscale tone channel per ink (package / PSD / spot TIF)
  - 1-bit AM halftone film per ink at staggered screen angles
  - normal-view composite = continuous overprint on paper (smooth like source)

Does not change flat_index / separation / edge code. With --gradients off this
module is not called (byte-identical).
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .io_utils import hex_of, save_png, to_image

# Classic offset angles staggered to reduce moiré
_SCREEN_ANGLES = (15.0, 75.0, 0.0, 45.0, 22.5, 67.5, 7.5, 82.5)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float32)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    a = np.clip(rgb.astype(np.float64) / 255.0, 0.0, 1.0)
    mask = a <= 0.04045
    lin = np.empty_like(a)
    lin[mask] = a[mask] / 12.92
    lin[~mask] = ((a[~mask] + 0.055) / 1.055) ** 2.4
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) / 1.00000
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883
    eps, kappa = 216 / 24389, 24389 / 27

    def f(t):
        out = np.empty_like(t)
        m = t > eps
        out[m] = np.cbrt(t[m])
        out[~m] = (kappa * t[~m] + 16.0) / 116.0
        return out

    fx, fy, fz = f(x), f(y), f(z)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def delta_e_cie76(a_rgb: np.ndarray, b_rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    la, lb = rgb_to_lab(a_rgb), rgb_to_lab(b_rgb)
    d = np.sqrt(((la - lb) ** 2).sum(axis=-1))
    if mask is not None:
        if not np.any(mask):
            return 0.0
        return float(d[mask].mean())
    return float(d.mean())


def fm_halftone(
    coverage: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """FM / stochastic film (1-bit): blue-noise-ish via hashed threshold. 255 = ink."""
    if coverage.ndim != 2:
        raise ValueError("Expected 2D coverage")
    cov = np.clip(coverage.astype(np.float64), 0.0, 1.0)
    h, w = cov.shape
    # Deterministic spatial hash → stable film across runs (no RNG state)
    yy, xx = np.mgrid[0:h, 0:w]
    # Deterministic uint32 hash (wrap with dtype, no Python scalar overflow)
    n = (xx.astype(np.uint32) * np.uint32(374761393)
         + yy.astype(np.uint32) * np.uint32(668265263)
         + np.uint32(seed * 2654435761 & 0xFFFFFFFF))
    n ^= n >> np.uint32(13)
    n = n * np.uint32(1274126177)
    thr = (n & np.uint32(0xFFFF)).astype(np.float64) / 65535.0
    return np.where(cov >= thr, 255, 0).astype(np.uint8)


def am_halftone(
    coverage: np.ndarray,
    lpi: int = 45,
    dpi: int = 300,
    angle_deg: float = 45.0,
) -> np.ndarray:
    """AM round-dot halftone (1-bit): 255 = ink, 0 = no ink."""
    if coverage.ndim != 2:
        raise ValueError("Expected 2D coverage")
    cov = np.clip(coverage.astype(np.float64), 0.0, 1.0)
    h, w = cov.shape
    cell = max(2.0, float(dpi) / float(max(lpi, 1)))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    xr = xx * cos_a - yy * sin_a
    yr = xx * sin_a + yy * cos_a
    fx = np.abs(np.mod(xr, cell) - cell * 0.5)
    fy = np.abs(np.mod(yr, cell) - cell * 0.5)
    dist = np.hypot(fx, fy)
    radius = (cell * 0.5) * np.sqrt(cov)
    return np.where(dist <= radius, 255, 0).astype(np.uint8)


def _lum1(c) -> float:
    c = np.asarray(c, dtype=np.float64).reshape(3)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def pick_inks(
    flat_rgb: np.ndarray,
    colors_limit: int,
    paper_lum: float = 252.0,
    src_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Ink set from flat palette (preferred) + source; keep cream/light, skip pure paper."""
    flat_cols = np.unique(np.ascontiguousarray(flat_rgb, dtype=np.uint8).reshape(-1, 3), axis=0)
    flat_inks = [tuple(int(x) for x in c) for c in flat_cols if _lum1(c) < paper_lum]
    # If the flat already has a small ink set, use it — local tones then match the source overprint.
    if 1 <= len(flat_inks) <= colors_limit:
        out = list(flat_inks)
        # still add cream from source if missing
        if src_rgb is not None and len(out) < colors_limit:
            lum = _luminance(src_rgb)
            thr = float(np.percentile(lum, 90))
            light_px = src_rgb[lum >= thr]
            if len(light_px):
                cream = tuple(int(x) for x in np.median(light_px.astype(np.float64), axis=0).round())
                if 200 <= _lum1(cream) < paper_lum and cream not in out:
                    out.append(cream)
        return np.array(out[:colors_limit], dtype=np.uint8)
    chunks = [flat_cols]
    if src_rgb is not None:
        s = src_rgb.reshape(-1, 3)
        idx = np.linspace(0, len(s) - 1, min(8000, len(s)), dtype=int)
        chunks.append(s[idx])
    X = np.vstack(chunks).astype(np.float64)
    keep = np.array([_lum1(c) < paper_lum for c in X], dtype=bool)
    X = X[keep] if keep.any() else X
    if len(X) == 0:
        return np.array([[0, 0, 0]], dtype=np.uint8)
    n = min(colors_limit, len(X))
    if len(X) <= n:
        cols = np.unique(X.astype(np.uint8), axis=0)
        return cols[:colors_limit]
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n, n_init=4, random_state=0)
        km.fit(X)
        centers = np.clip(np.rint(km.cluster_centers_), 0, 255).astype(np.uint8)
    except Exception:
        # furthest-point sample fallback
        centers = [X[0]]
        for _ in range(n - 1):
            d = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
            centers.append(X[int(np.argmax(d))])
        centers = np.clip(np.rint(np.array(centers)), 0, 255).astype(np.uint8)
    # drop any cluster that landed on pure paper
    out = [tuple(int(x) for x in c) for c in centers if _lum1(c) < paper_lum]
    if not out:
        out = [tuple(int(x) for x in c) for c in centers]
    # inject a few vivid source colours so blotchy hues are not averaged away
    if src_rgb is not None and len(out) < colors_limit:
        lab = rgb_to_lab(src_rgb).reshape(-1, 3)
        chroma = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
        vivid_idx = np.argsort(-chroma)[:50]
        sflat = src_rgb.reshape(-1, 3)
        for j in vivid_idx:
            c = tuple(int(x) for x in sflat[j])
            if _lum1(c) >= paper_lum:
                continue
            if min(((np.array(c) - np.array(o)) ** 2).sum() for o in out) > 40 ** 2:
                out.append(c)
            if len(out) >= colors_limit:
                break
    return np.array(out[:colors_limit], dtype=np.uint8)


def _ink_absorptions(inks: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Per-ink RGB absorption columns (1 - ink/paper), shape (3, K). Linear local model."""
    p = np.clip(paper.astype(np.float64), 1.0, 255.0)
    i = np.clip(inks.astype(np.float64), 0.0, 255.0)
    return np.clip(1.0 - i / p, 0.0, 1.0).T  # (3, K)


def _dilate_bool(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    for _ in range(px):
        im = im.filter(ImageFilter.MaxFilter(3))
    return np.asarray(im) > 0


def solve_tones(
    src_rgb: np.ndarray,
    inks: np.ndarray,
    paper: tuple[int, int, int] = (255, 255, 255),
    max_inks_per_pixel: int = 5,
    iters: int = 100,
    motif_border_px: int = 2,
    seal_motifs: bool = True,
) -> np.ndarray:
    """Per-pixel continuous tones following the SOURCE locally (no step quantisation).

    Uses one ink set across the whole image (no per-pixel ink switching — that
    caused faint diagonal bands). Solid motifs and a 2px border are sealed with
    motif ink so borders never fall back to bare paper (white halos).
    """
    h, w, _ = src_rgb.shape
    k = len(inks)
    paper_a = np.array(paper, dtype=np.float64)
    A = _ink_absorptions(inks, paper_a)
    tgt = np.clip(src_rgb.astype(np.float64), 0.0, 255.0)
    p = np.clip(paper_a, 1.0, 255.0)
    b = np.clip(1.0 - tgt / p, 0.0, 1.0).reshape(-1, 3)
    n = b.shape[0]
    t = np.zeros((n, k), dtype=np.float64)
    At = A.T
    lr = 1.0 / (float((A * A).sum()) + 1e-6)
    for _ in range(iters):
        t = np.clip(t + lr * ((b - t @ At) @ A), 0.0, 1.0)
    for _ in range(55):
        err = b - np.clip(t @ At, 0.0, 1.0)
        t = np.clip(t + 0.85 * lr * (err @ A), 0.0, 1.0)
    # Soft sparsity only (no hard per-pixel top-N — that switched ink sets and banded)
    if k > max_inks_per_pixel:
        t = np.where(t < 0.02, 0.0, t)
    if not seal_motifs:
        paper_px = np.max(np.abs(tgt.reshape(-1, 3) - p), axis=1) <= 2.0
        t[paper_px] = 0.0
        return t.reshape(h, w, k)
    # Seal solid motifs + dilated border with motif ink (never bare paper on borders)
    ink_lab = rgb_to_lab(inks.reshape(-1, 1, 3)).reshape(-1, 3)
    src_lab = rgb_to_lab(src_rgb).reshape(-1, 3)
    d2 = ((src_lab[:, None, :] - ink_lab[None, :, :]) ** 2).sum(-1)
    nearest = np.argmin(d2, axis=1)
    mind = np.sqrt(d2[np.arange(n), nearest])
    lum = _luminance(src_rgb)
    im = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), mode="L")
    mx = np.asarray(im.filter(ImageFilter.MaxFilter(3)), dtype=np.float32)
    mn = np.asarray(im.filter(ImageFilter.MinFilter(3)), dtype=np.float32)
    solid_core = ((mx - mn) < 1.5) & (mind.reshape(h, w) < 5.0)
    # Propagate each core's majority ink outward by motif_border_px via iterative dilate
    nearest_img = nearest.reshape(h, w).astype(np.int32)
    fill_ink = np.full((h, w), -1, dtype=np.int32)
    fill_ink[solid_core] = nearest_img[solid_core]
    for _ in range(max(0, motif_border_px)):
        # dilate known ink labels: for each empty neighbour of a labelled pixel, take that ink
        known = fill_ink >= 0
        dil = _dilate_bool(known, 1)
        ring = dil & ~known
        if not ring.any():
            break
        # sample ink from a dilated known mask: max-filter on (ink+1) then -1
        lab_u8 = (fill_ink + 1).clip(0, 255).astype(np.uint8)
        # MaxFilter works for positive labels when labels are small ints
        prop = np.asarray(Image.fromarray(lab_u8, mode="L").filter(ImageFilter.MaxFilter(3)), dtype=np.int32) - 1
        fill_ink[ring] = prop[ring]
    solid_fill = fill_ink >= 0
    if solid_fill.any():
        idx = np.flatnonzero(solid_fill.ravel())
        t[idx] = 0.0
        t[idx, fill_ink.ravel()[idx]] = 1.0
    # true paper only (never clear sealed motif/border)
    paper_px = np.max(np.abs(tgt.reshape(-1, 3) - p), axis=1) <= 2.0
    paper_px &= ~solid_fill.ravel()
    t[paper_px] = 0.0
    return t.reshape(h, w, k)


def smooth_tones(tones: np.ndarray, radius: float = 0.4) -> np.ndarray:
    """At most a tiny ~1px denoise. Never heavy blur (that killed blotchy detail)."""
    if radius <= 0:
        return tones
    out = np.empty_like(tones)
    for k in range(tones.shape[2]):
        im = Image.fromarray(np.clip(tones[..., k] * 255.0, 0, 255).astype(np.uint8), mode="L")
        # Median 3x3 = 1px denoise, keeps irregular blotches
        im = im.filter(ImageFilter.MedianFilter(3))
        out[..., k] = np.asarray(im, dtype=np.float64) / 255.0
    return out


def overprint_rgb(tones: np.ndarray, inks: np.ndarray, paper: tuple[int, int, int]) -> np.ndarray:
    """Linear-absorption overprint (matches solve_tones).

    PSD tone plates use LINEAR_BURN of lerp(white, ink, t) layers, which
    reproduces this model. Photoshop MULTIPLY would be Π(1-t·a) (lighter).
    """
    paper_a = np.array(paper, dtype=np.float64)
    A = _ink_absorptions(inks, paper_a)
    abs_total = np.clip(tones @ A.T, 0.0, 1.0)
    out = paper_a * (1.0 - abs_total)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)



def _load_rgb(path: str, size_wh: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size_wh is not None:
        w, h = size_wh
        if im.size != (w, h):
            im = im.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)



def _erode_bool(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    for _ in range(px):
        im = im.filter(ImageFilter.MinFilter(3))
    return np.asarray(im) > 0


def classify_flat_shaded(
    src_rgb: np.ndarray,
    flat_rgb: np.ndarray,
    inks: np.ndarray,
    *,
    local_range_max: float = 1.0,
    near_flat_de: float = 2.0,
    min_flat_area: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Strict FLAT vs SHADED.

    FLAT only when local variance is ~0 AND source is within ~2 dE of one ink,
    over a reasonably large area (true solids: black ground, berries, cream).
    Subtle petal shading stays SHADED. Thin high-contrast outlines (e.g. green
    petal lines) are forced SHADED so they keep spray-quality tonal/ink strength.
    Returns (flat_mask HxW, flat_ink_idx HxW).
    """
    h, w, _ = src_rgb.shape
    if min_flat_area is None:
        min_flat_area = max(800, (h * w) // 500)
    ink_lab = rgb_to_lab(inks.reshape(-1, 1, 3)).reshape(-1, 3)
    flat_lab = rgb_to_lab(flat_rgb).reshape(-1, 3)
    src_lab = rgb_to_lab(src_rgb).reshape(-1, 3)
    d2f = ((flat_lab[:, None, :] - ink_lab[None, :, :]) ** 2).sum(-1)
    flat_ink = np.argmin(d2f, axis=1).astype(np.int32)
    paper_flat = np.max(np.abs(flat_rgb.reshape(-1, 3).astype(np.float64) - 255.0), axis=1) <= 4.0
    flat_ink[paper_flat] = -1
    src_to_ink = np.full(h * w, 999.0, dtype=np.float64)
    valid = flat_ink >= 0
    if valid.any():
        src_to_ink[valid] = np.sqrt(((src_lab[valid] - ink_lab[flat_ink[valid]]) ** 2).sum(-1))
    lum = _luminance(src_rgb)
    im = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), mode="L")
    im_den = im.filter(ImageFilter.MedianFilter(3))
    mx3 = np.asarray(im_den.filter(ImageFilter.MaxFilter(3)), dtype=np.float32)
    mn3 = np.asarray(im_den.filter(ImageFilter.MinFilter(3)), dtype=np.float32)
    mx5 = np.asarray(im_den.filter(ImageFilter.MaxFilter(5)), dtype=np.float32)
    mn5 = np.asarray(im_den.filter(ImageFilter.MinFilter(5)), dtype=np.float32)
    range3 = (mx3 - mn3).ravel()
    range5 = (mx5 - mn5).ravel()
    # Strict flat core: near-zero variance AND within ~2 dE of the flat ink
    flat_m = valid & (src_to_ink <= near_flat_de) & (range3 <= local_range_max) & (range5 <= local_range_max + 0.5)
    ink_lums = np.array([_lum1(c) for c in inks], dtype=np.float64)
    src_lum = lum.ravel()
    ink_img = flat_ink.reshape(h, w)

    # Drop small non-dark flat blobs (cream speckles / subtle petals mis-tagged)
    for i in range(len(inks)):
        region = (flat_m.reshape(h, w)) & (ink_img == i)
        area = int(region.sum())
        if area == 0:
            continue
        is_dark = ink_lums[i] < 40.0
        if (not is_dark) and area < min_flat_area:
            flat_m[region.ravel()] = False

    # Black ground: force solid despite JPEG noise (keep std = 0)
    for i in range(len(inks)):
        if ink_lums[i] >= 40.0:
            continue
        region = valid & (flat_ink == i) & (src_lum < 45)
        # large dark region in flat
        if int(region.sum()) >= min_flat_area:
            flat_m |= region & (src_to_ink < 25.0)

    flat_mask = flat_m.reshape(h, w)

    # Thin high-contrast outlines in the flat map → SHADED (keep spray-quality lines)
    # A "thin" ink run vanishes under 1px erode (no thick core nearby).
    thin_line = np.zeros((h, w), dtype=bool)
    for i in range(len(inks)):
        m = ink_img == i
        if not m.any() or ink_lums[i] < 40.0:
            continue  # don't treat black ground edge as a line
        core = _erode_bool(m, 1)
        if core.any():
            thick = _dilate_bool(core, 1)
            thin = m & ~thick
        else:
            thin = m  # entirely thin stroke
        thin_line |= thin
    # also: high local contrast in source stays shaded (outline edges)
    high_contrast = (range3 >= 4.0).reshape(h, w)
    flat_mask &= ~(thin_line | (high_contrast & thin_line))
    flat_mask &= ~thin_line

    # Fill only tiny JPEG holes inside DARK flat ground (1px), never cream/petal
    dark_flat = np.zeros((h, w), dtype=bool)
    for i in range(len(inks)):
        if ink_lums[i] < 40.0:
            dark_flat |= flat_mask & (ink_img == i)
    if dark_flat.any():
        dil = _dilate_bool(dark_flat, 1)
        hole = dil & ~flat_mask & (src_lum.reshape(h, w) < 45)
        if hole.any():
            # assign darkest ink
            dark_i = int(np.argmin(ink_lums))
            flat_mask[hole] = True
            ink_img[hole] = dark_i
            flat_ink = ink_img.ravel()

    return flat_mask, ink_img


def apply_hybrid_tones(
    tones: np.ndarray,
    flat_mask: np.ndarray,
    flat_ink_idx: np.ndarray,
    n_inks: int,
    feather_px: int = 2,
) -> np.ndarray:
    """FLAT core → 100% single ink; 1–2px boundary keeps tonal solve (feather into shade)."""
    out = tones.copy()
    h, w, k = out.shape
    assert k == n_inks
    # Feather: only the eroded core is forced solid; ring stays shaded tones
    core = _erode_bool(flat_mask, max(0, feather_px)) if feather_px > 0 else flat_mask
    # Thin-line rescue: if a shaded pixel's flat ink is a vivid outline colour and
    # tone is weak, snap that single ink to full (keeps green outlines crisp)
    fm = core.ravel()
    fi = flat_ink_idx.ravel()
    idx = np.flatnonzero(fm & (fi >= 0) & (fi < k))
    if idx.size:
        out.reshape(-1, k)[idx] = 0.0
        out.reshape(-1, k)[idx, fi[idx]] = 1.0
    return out


def boost_thin_outline_tones(
    tones: np.ndarray,
    flat_rgb: np.ndarray,
    flat_ink_idx: np.ndarray,
    inks: np.ndarray,
    flat_mask: np.ndarray,
) -> np.ndarray:
    """Keep thin flat-map strokes (green outlines) at near-full ink when shaded."""
    h, w, k = tones.shape
    ink_img = flat_ink_idx
    ink_lums = np.array([_lum1(c) for c in inks], dtype=np.float64)
    out = tones
    thin = np.zeros((h, w), dtype=bool)
    for i in range(k):
        if ink_lums[i] < 40.0:
            continue
        m = ink_img == i
        if not m.any():
            continue
        core = _erode_bool(m, 1)
        thick = _dilate_bool(core, 1) if core.any() else np.zeros_like(m)
        thin |= m & ~thick
    # only where we did NOT force flat solid
    ring = thin & ~flat_mask
    if not ring.any():
        return out
    out = out.copy()
    for i in range(k):
        pix = ring & (ink_img == i)
        if not pix.any():
            continue
        # raise this ink toward full; leave others (overprint ok) but ensure presence
        cur = out[..., i]
        cur = np.where(pix, np.maximum(cur, 0.92), cur)
        out[..., i] = cur
    return out


def flat_stray_tone_percent(tones: np.ndarray, flat_mask: np.ndarray, flat_ink_idx: np.ndarray) -> float:
    """% of flat pixels that have >1% tone on a non-assigned ink."""
    h, w, k = tones.shape
    fm = flat_mask.ravel()
    fi = flat_ink_idx.ravel()
    idx = np.flatnonzero(fm & (fi >= 0))
    if idx.size == 0:
        return 0.0
    t = tones.reshape(-1, k)[idx]
    # zero the assigned ink, see residual
    rows = np.arange(len(idx))
    residual = t.copy()
    residual[rows, fi[idx]] = 0.0
    stray = (residual.max(axis=1) > 0.01)
    return float(100.0 * stray.mean())



def run_gradients(
    out_dir: str,
    name: str,
    design_path: str,
    *,
    size_px: tuple[int, int],
    dpi: int = 300,
    lpi: int = 45,
    angle_deg: float = 15.0,
    colors_limit: int = 8,
    max_inks_per_pixel: int = 5,
    hybrid: bool = True,
    log=print,
) -> dict[str, Any]:
    """Solve multi-channel tones, write films + composite + tone channel list for package merge.

    hybrid=True (default): FLAT regions (low variance, near palette ink) get 100% solid
    single ink like the numbering flat; SHADED regions keep tonal overprint. Kills
    mottling on solid grounds while preserving petal gradients.
    """
    w, h = size_px
    flat_path = os.path.join(out_dir, f"{name}_flat.png")
    if not os.path.isfile(flat_path):
        raise FileNotFoundError(f"gradients needs {flat_path}")
    flat = _load_rgb(flat_path)
    src = _load_rgb(design_path, (w, h))
    if flat.shape[:2] != (h, w):
        flat = np.asarray(Image.fromarray(flat).resize((w, h), Image.Resampling.NEAREST), dtype=np.uint8)

    # Pure paper base for density math; cream/light colours stay as printable inks
    # so low tone = paper showing (cream top-left must not turn pink).
    paper = (255, 255, 255)
    inks = pick_inks(flat, colors_limit, src_rgb=src)
    report: dict[str, Any] = {
        "name": name,
        "size_px": [w, h],
        "dpi": dpi,
        "lpi": lpi,
        "colors_limit": colors_limit,
        "n_inks": int(len(inks)),
        "spot_colour_count": int(len(inks)),
        "applied": False,
        "files": [],
        "tone_plates": [],
        "tone_channels": [],
        "mode": "hybrid_flat_solid_plus_tonal_shade" if hybrid else "multi_channel_tonal_overprint",
    }
    if len(inks) == 0:
        log("[gradients] no inks found; numbering outputs unchanged")
        return report

    # Skip if source is already nearly flat solid blocks (byte-identical path)
    if _is_essentially_flat(src):
        log("[gradients] source is flat solids; numbering outputs unchanged")
        gdir = os.path.join(out_dir, "gradients")
        os.makedirs(gdir, exist_ok=True)
        rp = os.path.join(gdir, f"{name}_gradients_report.json")
        with open(rp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        report["files"].append(rp)
        report["report"] = rp
        return report

    log(f"[gradients] solving tones for {len(inks)} inks, max {max_inks_per_pixel}/pixel "
        f"(hybrid={'on' if hybrid else 'off'}) …")
    # Continuous tonal solve without motif-seal (hybrid applies solid flats cleanly)
    tones = solve_tones(
        src, inks, paper=paper, max_inks_per_pixel=max_inks_per_pixel,
        seal_motifs=not hybrid,  # hybrid replaces seal with flat/shaded classify
    )
    tones = np.clip(tones, 0.0, 1.0)
    flat_mask = None
    flat_ink_idx = None
    if hybrid:
        flat_mask, flat_ink_idx = classify_flat_shaded(src, flat, inks)
        # Feather 2px into SHADE: solid snap only on eroded core; dark ground unfeathered
        ink_lums = np.array([_lum1(c) for c in inks], dtype=np.float64)
        dark_flat = np.zeros(flat_mask.shape, dtype=bool)
        for i, lu in enumerate(ink_lums):
            if lu < 40.0:
                dark_flat |= flat_mask & (flat_ink_idx == i)
        flat_core = _erode_bool(flat_mask, 2) | dark_flat
        tones = apply_hybrid_tones(tones, flat_core, flat_ink_idx, len(inks), feather_px=0)
        tones = boost_thin_outline_tones(tones, flat, flat_ink_idx, inks, flat_core)
        flat_mask = flat_core  # metrics / stray / ground use solid core only
        # thin shaded structures (stems) next to flats: keep tonal solve — already in tones
        # re-seal only the 1px motif border of FLAT against paper holes
        log(f"[gradients] hybrid: flat={100.0*flat_mask.mean():.1f}% shaded={100.0*(~flat_mask).mean():.1f}%")

    gdir = os.path.join(out_dir, "gradients")
    os.makedirs(gdir, exist_ok=True)
    film_dir = os.path.join(gdir, "film")
    tone_dir = os.path.join(gdir, "tone_channels")
    os.makedirs(film_dir, exist_ok=True)
    os.makedirs(tone_dir, exist_ok=True)

    for i, ink in enumerate(inks):
        hx = hex_of(ink)
        angle = _SCREEN_ANGLES[i % len(_SCREEN_ANGLES)] if angle_deg == 15.0 else (angle_deg + i * 30.0) % 90.0
        cov = tones[..., i]
        tone_u8 = np.clip(np.rint(cov * 255.0), 0, 255).astype(np.uint8)  # full 8-bit continuous
        tone_path = os.path.join(tone_dir, f"TONE_{i+1:02d}_{hx}.png")
        save_png(Image.fromarray(tone_u8, mode="L"), tone_path, dpi)
        dots = am_halftone(cov, lpi=lpi, dpi=dpi, angle_deg=angle)
        film_path = os.path.join(film_dir, f"INK_{i+1:02d}_{hx}_film.png")
        save_png(Image.fromarray(dots, mode="L"), film_path, dpi)
        entry = {
            "name": f"INK_{i+1:02d}",
            "hex": "#" + hx,
            "ink_rgb": [int(x) for x in ink],
            "angle_deg": angle,
            "lpi": lpi,
            "tone_path": tone_path,
            "film_path": film_path,
            "mean_tone_percent": round(float(cov.mean() * 100), 3),
        }
        report["tone_plates"].append(entry)
        report["tone_channels"].append(entry)
        report["files"].extend([tone_path, film_path])

    composite = overprint_rgb(tones, inks, paper)
    comp_path = os.path.join(gdir, f"{name}_gradient_composite.png")
    save_png(to_image(composite), comp_path, dpi)
    report["files"].append(comp_path)
    report["composite"] = comp_path

    dots_stack = np.zeros((h, w, len(inks)), dtype=np.float64)
    for i, e in enumerate(report["tone_plates"]):
        d = np.asarray(Image.open(e["film_path"]).convert("L"), dtype=np.float64) / 255.0
        dots_stack[..., i] = d
    dots_prev = overprint_rgb(dots_stack, inks, paper)
    dots_path = os.path.join(gdir, f"{name}_gradient_dots_preview.png")
    save_png(to_image(dots_prev), dots_path, dpi)
    report["files"].append(dots_path)
    report["dots_preview"] = dots_path

    de_all = delta_e_cie76(src, composite)
    # gradient-ish pixels: local tone range in source
    lum = _luminance(src)
    im = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), mode="L")
    mx = np.asarray(im.filter(ImageFilter.MaxFilter(5)), dtype=np.float32)
    mn = np.asarray(im.filter(ImageFilter.MinFilter(5)), dtype=np.float32)
    grad_m = (mx - mn) >= 2.0
    solid_m = (mx - mn) < 2.0
    de_grad = delta_e_cie76(src, composite, grad_m) if grad_m.any() else de_all
    de_solid = delta_e_cie76(src, composite, solid_m) if solid_m.any() else de_all
    bs_src = band_step_metric(src)
    bs_comp = band_step_metric(composite)
    report["delta_e_cie76"] = {
        "gradient_mean": round(de_grad, 3),
        "motif_mean": round(de_solid, 3),
        "overall_mean": round(de_all, 3),
    }
    report["band_step"] = {
        "source": round(bs_src, 3),
        "composite": round(bs_comp, 3),
        "ratio": round(bs_comp / max(bs_src, 1e-6), 3),
    }
    log(f"[gradients] ΔE76 mean: gradient={de_grad:.2f} solid={de_solid:.2f} overall={de_all:.2f}")
    log(f"[gradients] band-step: src={bs_src:.2f} comp={bs_comp:.2f} ratio={bs_comp/max(bs_src,1e-6):.2f}")
    if hybrid and flat_mask is not None:
        # black/near-black ground: spatial luminance std (near 0 = no mottling)
        ink_lum = np.array([_lum1(c) for c in inks])
        dark_i = int(np.argmin(ink_lum))
        # prefer flat-image dark pixels (true ground) that were sealed flat
        flat_dark = (flat.max(axis=-1) < 40) & flat_mask
        ground = flat_dark if flat_dark.any() else (flat_mask & (flat_ink_idx == dark_i))
        if ground.any():
            glum = _luminance(composite)[ground].astype(np.float64)
            gstd = float(glum.std())
            report["flat_ground_std"] = round(gstd, 4)
            report["flat_ground_pixels"] = int(ground.sum())
            report["flat_ground_nunique"] = int(len(np.unique(composite[ground].reshape(-1, 3), axis=0)))
            log(f"[gradients] hybrid flat-ground lum-std={gstd:.4f} nunique={report['flat_ground_nunique']}")
        stray = flat_stray_tone_percent(tones, flat_mask, flat_ink_idx)
        report["flat_stray_tone_percent"] = round(stray, 4)
        report["flat_region_percent"] = round(float(100.0 * flat_mask.mean()), 3)
        report["hybrid"] = True
        # save flat/shade debug mask
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mask_rgb[flat_mask] = (40, 180, 40)
        mask_rgb[~flat_mask] = (220, 80, 40)
        mp = os.path.join(gdir, f"{name}_hybrid_flat_shade_mask.png")
        save_png(to_image(mask_rgb), mp, dpi)
        report["files"].append(mp)
        report["hybrid_mask"] = mp
        log(f"[gradients] hybrid flat stray-tone={stray:.3f}%")

    # CSV listing
    csv_path = os.path.join(gdir, f"{name}_tone_colors.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["Plate", "HEX", "Angle", "MeanTone%", "ToneFile", "FilmFile"])
        for e in report["tone_plates"]:
            wr.writerow([e["name"], e["hex"], e["angle_deg"], e["mean_tone_percent"],
                         os.path.basename(e["tone_path"]), os.path.basename(e["film_path"])])
    report["files"].append(csv_path)

    report["applied"] = True
    report["inks"] = ["#" + hex_of(c) for c in inks]
    report["paper_rgb"] = list(paper)
    rp = os.path.join(gdir, f"{name}_gradients_report.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    report["report"] = rp
    report["files"].append(rp)
    log(f"[gradients] {len(inks)} tone channels + films -> {gdir}")
    return report


def _is_essentially_flat(rgb: np.ndarray, max_unique: int = 16) -> bool:
    """True for solid-block images (flags-on should leave main outs untouched)."""
    cols = np.unique(rgb.reshape(-1, 3), axis=0)
    if len(cols) > max_unique:
        return False
    lum = _luminance(rgb)
    im = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), mode="L")
    mx = np.asarray(im.filter(ImageFilter.MaxFilter(3)), dtype=np.float32)
    mn = np.asarray(im.filter(ImageFilter.MinFilter(3)), dtype=np.float32)
    # solids: most pixels have zero local range (only shared edges are high)
    return float(np.median(mx - mn)) < 0.5


def band_step_metric(rgb: np.ndarray, window: int = 8) -> float:
    """Max step in box-filtered luminance (visible banding → large value)."""
    lum = _luminance(rgb).astype(np.float64)
    im = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), mode="L")
    blur = np.asarray(im.filter(ImageFilter.BoxBlur(max(1, window // 2))), dtype=np.float64)
    # first differences along rows, cols, and both diagonals
    peaks = []
    for arr in (blur, blur.T,
                np.diagonal(blur, offset=0),  # short; also use rolled diags via diffs on shifts
                ):
        a = np.asarray(arr, dtype=np.float64).ravel() if arr.ndim > 1 else np.asarray(arr, dtype=np.float64)
        if a.size < 3:
            continue
        d = np.abs(np.diff(a))
        if d.size:
            peaks.append(float(np.percentile(d, 99)))
    # full diagonal-ish: row-wise and col-wise already cover band artefacts
    d_row = np.abs(np.diff(blur, axis=1))
    d_col = np.abs(np.diff(blur, axis=0))
    peaks.append(float(np.percentile(d_row, 99)))
    peaks.append(float(np.percentile(d_col, 99)))
    return float(max(peaks)) if peaks else 0.0


def strip_dot_coverage(dots: np.ndarray, n_bins: int = 10) -> list[float]:

    h, w = dots.shape
    edges = np.linspace(0, w, n_bins + 1, dtype=int)
    return [float((dots[:, edges[i]:edges[i + 1]] > 0).mean()) for i in range(n_bins)]
