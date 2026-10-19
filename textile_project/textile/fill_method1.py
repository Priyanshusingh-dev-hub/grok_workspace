"""Colour fill, Method 1 (aligned line art + reference): reference_code/
method1_colorfill.py as a module. The logic is the script's, step for step:

  1. the reference's flat palette (k-means if it is not flat)
  2. the reference as palette indices, resized NEAREST to the output size
  3. the line art: Lanczos up, a light 3x3 blur, threshold -> line mask
  4. alignment: how many reference colour edges fall on the lines
  5. closed areas (4-connected) take the reference's majority colour there
  6. line pixels take the colour most common under the lines (or --line-color)

Only two things differ from the script, both approved by the user: images
are decoded with cv2.imdecode from bytes (the same pixels as cv2.imread, but
any Windows path opens), and a design that is not square keeps its
proportions (width = size) instead of being stretched square.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from . import palette as pl
from .io_utils import read_cv2, rgb_of


class FillError(Exception):
    """A fill that must stop, with the reason in the user's words."""


@dataclass
class Fill:
    index: np.ndarray            # H x W palette indices
    pal: np.ndarray              # K x 3 uint8
    line_index: int
    reference_was_flat: bool
    alignment_score: float
    regions: int
    doubtful: list = field(default_factory=list)      # (region, size, purity)
    debug: np.ndarray | None = None                   # the doubtful regions in red, 1200 px
    reference: np.ndarray | None = None               # the reference, 1200 px wide (compare sheet)


def alignment_score(lines, ref_idx):
    """Reference ke color-boundaries kitne % line art ki lines ke paas padte hain.
    Aligned images me ye high hota hai; shift / alag design par low."""
    b = np.zeros_like(lines)
    b[:, 1:] |= ref_idx[:, 1:] != ref_idx[:, :-1]
    b[1:, :] |= ref_idx[1:, :] != ref_idx[:-1, :]
    near = cv2.dilate(lines.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return float((b & near).sum() / max(b.sum(), 1))


def output_size(w, h, size):
    """(width, height): `size` wide, the design's proportions kept."""
    return size, max(1, round(size * h / w))


def fill(line_path, ref_path, size=3535, max_colors=16, min_share=0.0005, line_threshold=150,
         line_color='auto', min_align=0.55, force=False, log=print) -> Fill:
    line = read_cv2(line_path, cv2.IMREAD_GRAYSCALE)
    ref_bgr = read_cv2(ref_path, cv2.IMREAD_COLOR)
    if line is None or ref_bgr is None:
        raise FillError('image read nahi hui - path check karo.')
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    log(f"[input] line art {line.shape[1]}x{line.shape[0]}, reference {ref.shape[1]}x{ref.shape[0]}")

    ar_l = line.shape[1] / line.shape[0]
    ar_r = ref.shape[1] / ref.shape[0]
    if abs(ar_l - ar_r) > 0.01:
        raise FillError(f"aspect ratio alag hai (line {ar_l:.3f} vs ref {ar_r:.3f}). "
                        "Dono same crop ke hone chahiye.")
    W, H = output_size(line.shape[1], line.shape[0], size)

    # 1. palette
    pal, is_flat = pl.extract_palette(ref, max_colors, min_share, log=log)
    K = len(pal)

    # 2. reference -> palette index at target size (NEAREST: koi naya color na bane)
    ref_idx = pl.map_to_palette(ref, pal)
    ref_idx = cv2.resize(ref_idx, (W, H), interpolation=cv2.INTER_NEAREST)

    # 3. line art upscale + threshold (sirf 3x3 halka blur jaggies ke liye; NO boundary smoothing)
    g = cv2.resize(line, (W, H), interpolation=cv2.INTER_LANCZOS4)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    lines = g < line_threshold
    del g

    # 4. alignment check
    score = alignment_score(lines, ref_idx)
    log(f"[align] score = {score:.2f} (>= {min_align} chahiye)")
    if score < min_align and not force:
        raise FillError("line art aur reference aligned nahi lag rahe. Images check karo "
                        "(crop/shift/rotation/alag design). Jaan-boojh kar chalana ho to --force do.")

    # 5. regions + majority vote
    lab, n = ndimage.label(~lines)  # 4-connectivity: diagonal se color leak nahi hota
    votes = np.bincount(lab.ravel().astype(np.int64) * K + ref_idx.ravel(),
                        minlength=(n + 1) * K).reshape(n + 1, K)
    region_color = votes.argmax(1)
    out = region_color[lab].astype(np.uint8)

    # 6. line color
    if line_color == "auto":
        lc = int(np.bincount(ref_idx[lines], minlength=K).argmax())
    else:
        tgt = rgb_of(line_color).astype(int)
        lc = int(((pal.astype(int) - tgt) ** 2).sum(1).argmin())
    out[lines] = lc

    # 7. leak / doubtful regions report
    sizes = votes.sum(1)
    purity = votes.max(1) / np.maximum(sizes, 1)
    bad = [(int(i), int(sizes[i]), float(purity[i])) for i in range(1, n + 1)
           if sizes[i] > W * H * 0.002 and purity[i] < 0.6]
    debug = None
    if bad:
        log(f"[warn] {len(bad)} bade regions me reference ke multiple colors mile (purity < 60%). "
            "Ho sakta hai line tooti ho aur color leak hua ho -> debug image dekho.")
        dbg = pal[out].copy()
        mask = np.isin(lab, [b[0] for b in bad])
        dbg[mask] = (dbg[mask] * 0.3 + np.array([255, 0, 0]) * 0.7).astype(np.uint8)
        dw, dh = (1200, 1200) if W == H else (1200, max(1, round(1200 * H / W)))
        debug = np.asarray(Image.fromarray(dbg).resize((dw, dh), Image.NEAREST))
    del lab
    rw, rh = (1200, 1200) if W == H else (1200, max(1, round(1200 * H / W)))
    ref_view = np.asarray(Image.fromarray(ref).resize((rw, rh), Image.NEAREST))
    return Fill(out, pal, lc, bool(is_flat), score, int(n), bad, debug, ref_view)
