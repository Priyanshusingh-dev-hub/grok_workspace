"""Colour fill, Method 3 (line art + a reference that has drifted, any number
of colours): ROADMAP Phase 2's "Method 2 for 3+ colours, so a little
misalignment is tolerated".

The reference is first laid onto the line art, a little at a time, then
filled exactly as Method 1 fills (each closed area takes the majority colour
there, lines the colour most common under them):

  1. palette and line mask as Method 1 (k-means if the reference is not
     flat; Lanczos up, 3x3 blur, < threshold)
  2. register: for every tile of the design, the shift (within `reach` px)
     that lays the most reference colour edges on the line art's lines;
     tiles with too few edges take their nearest neighbour's shift, the
     shifts are median-filtered, and the reference index map is moved by the
     smooth field, NEAREST only (no new, mixed colours). Three passes:
     coarse (256 px tiles, up to `reach`), then 128 px tiles (16 px), then
     64 px tiles (6 px), each on the result of the one before.
  3. Method 1's vote on the moved reference; doubtful areas (big, < 60% one
     colour) in red, as Method 1 marks them.

Measured on the floral with its reference bent by a smooth random field
(up to 80 px at 3535): Method 1 gets 87.4% of the pixels right (alignment
0.53), this 99.6%; on the unbent reference both give the same design. It
does NOT fix a reference that is a different drawing (another AI generation,
motifs elsewhere, like the tree panel): there the shapes do not match
anywhere to shift to, and Method 2's structure fill (2 colours) is the way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from . import palette as pl
from .fill_method1 import FillError, alignment_score, output_size
from .fill_method2 import _shift
from .io_utils import read_cv2, rgb_of

BASE = 3535          # the pass sizes below are for a 3535 px design; others scale


@dataclass
class Fill3:
    index: np.ndarray
    pal: np.ndarray
    line_index: int
    reference_was_flat: bool
    alignment_before: float      # Method 1's score on the reference as given
    alignment_score: float       # the same score after registering
    max_shift_px: int
    regions: int
    doubtful: list = field(default_factory=list)
    debug: np.ndarray | None = None
    reference: np.ndarray | None = None


def _edges(idx):
    b = np.zeros(idx.shape, bool)
    b[:, 1:] |= idx[:, 1:] != idx[:, :-1]
    b[1:, :] |= idx[1:, :] != idx[:-1, :]
    return b


def _pool(mask, w, h):
    """True where any pixel of the cell is True."""
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0


def shift_field(lines, ref_idx, scale, reach, tile, win, min_edge=0.01):
    """Per-pixel (fx, fy): how far the reference must move so its colour
    edges fall on the lines, from one best shift per tile."""
    H, W = lines.shape
    h, w = -(-H // scale), -(-W // scale)
    near = cv2.dilate(_pool(lines, w, h).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    e = _pool(_edges(ref_idx), w, h)
    t = max(4, tile // scale)
    R = max(1, -(-reach // scale))
    ty, tx = -(-h // t), -(-w // t)

    def per_tile(a):
        a = np.pad(a, ((0, ty * t - h), (0, tx * t - w))).reshape(ty, t, tx, t).sum((1, 3)).astype(np.float64)
        return ndimage.uniform_filter(a, 2 * win + 1, mode='constant') if win else a

    best = np.full((ty, tx), -9.0)
    bdx, bdy = np.zeros((ty, tx)), np.zeros((ty, tx))
    total = per_tile(e)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            s = _shift(e, -dy, -dx, h, w)            # s[y, x] = e[y - dy, x - dx]
            score = per_tile(s & near) / np.maximum(per_tile(s), 1) - 0.001 * np.hypot(dx, dy)
            m = score > best
            best[m], bdx[m], bdy[m] = score[m], dx, dy
    weak = total < min_edge * (t * (2 * win + 1)) ** 2
    for a in (bdx, bdy):
        if weak.all():
            a[:] = 0
            continue
        if weak.any():                                # nearest tile that could tell
            _, (iy, ix) = ndimage.distance_transform_edt(weak, return_indices=True)
            a[:] = a[iy, ix]
        a[:] = ndimage.median_filter(a, 3, mode='nearest')
    big = (tx * t * scale, ty * t * scale)
    fx = cv2.resize((bdx * scale).astype(np.float32), big, interpolation=cv2.INTER_LINEAR)[:H, :W]
    fy = cv2.resize((bdy * scale).astype(np.float32), big, interpolation=cv2.INTER_LINEAR)[:H, :W]
    return fx, fy, float(weak.mean())


def register(lines, ref_idx, reach=96, log=print):
    """The reference index map laid onto the line art (three passes)."""
    H, W = lines.shape
    f = W / BASE
    passes = ((4, reach, 256, 0), (2, 16, 128, 1), (1, 6, 64, 1))
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    cur, most = ref_idx, 0
    for scale, r, tile, win in passes:
        fx, fy, weak = shift_field(lines, cur, scale, max(1, round(r * f)), max(16, round(tile * f)), win)
        cur = cv2.remap(cur, gx - fx, gy - fy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        step = int(round(max(np.abs(fx).max(), np.abs(fy).max())))
        most += step
        log(f'[register] {round(tile * f)} px tiles: shift up to {step} px, '
            f'{weak * 100:.0f}% tiles me kam edges (padosi ka shift liya)')
    return cur, most


def fill(line_path, ref_path, size=3535, max_colors=16, min_share=0.0005, line_threshold=150,
         line_color='auto', reach=96, pal=None, log=print) -> Fill3:
    line = read_cv2(line_path, cv2.IMREAD_GRAYSCALE)
    ref_bgr = read_cv2(ref_path, cv2.IMREAD_COLOR)
    if line is None or ref_bgr is None:
        raise FillError('image read nahi hui - path check karo.')
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    ar_l, ar_r = line.shape[1] / line.shape[0], ref.shape[1] / ref.shape[0]
    if abs(ar_l - ar_r) > 0.01:
        raise FillError(f"aspect ratio alag hai (line {ar_l:.3f} vs ref {ar_r:.3f}). "
                        "Dono same crop ke hone chahiye.")
    W, H = output_size(line.shape[1], line.shape[0], size)

    # 1. palette (or Method 1's, when auto mode already has it) + line mask
    if pal is None:
        pal, is_flat = pl.extract_palette(ref, max_colors, min_share, log=log)
    else:
        is_flat = False
    K = len(pal)
    ref_idx = cv2.resize(pl.map_to_palette(ref, pal), (W, H), interpolation=cv2.INTER_NEAREST)
    g = cv2.resize(line, (W, H), interpolation=cv2.INTER_LANCZOS4)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    lines = g < line_threshold
    del g

    # 2. lay the reference onto the line art
    before = alignment_score(lines, ref_idx)
    ref_idx, most = register(lines, ref_idx, reach, log=log)
    score = alignment_score(lines, ref_idx)
    log(f'[align] score {before:.2f} -> {score:.2f} (reference khiska kar)')

    # 3. Method 1's fill on the moved reference
    lab, n = ndimage.label(~lines)
    votes = np.bincount(lab.ravel().astype(np.int64) * K + ref_idx.ravel(),
                        minlength=(n + 1) * K).reshape(n + 1, K)
    out = votes.argmax(1)[lab].astype(np.uint8)
    if line_color == 'auto':
        lc = int(np.bincount(ref_idx[lines], minlength=K).argmax())
    else:
        lc = int(((pal.astype(int) - rgb_of(line_color).astype(int)) ** 2).sum(1).argmin())
    out[lines] = lc

    sizes = votes.sum(1)
    purity = votes.max(1) / np.maximum(sizes, 1)
    bad = [(int(i), int(sizes[i]), float(purity[i])) for i in range(1, n + 1)
           if sizes[i] > W * H * 0.002 and purity[i] < 0.6]
    rw, rh = (1200, 1200) if W == H else (1200, max(1, round(1200 * H / W)))
    debug = None
    if bad:
        log(f'[warn] {len(bad)} bade regions me reference ke multiple colors mile (purity < 60%) '
            '-> debug image dekho.')
        dbg = pal[out].copy()
        mask = np.isin(lab, [b[0] for b in bad])
        dbg[mask] = (dbg[mask] * 0.3 + np.array([255, 0, 0]) * 0.7).astype(np.uint8)
        debug = np.asarray(Image.fromarray(dbg).resize((rw, rh), Image.NEAREST))
    del lab
    view = np.asarray(Image.fromarray(ref).resize((rw, rh), Image.NEAREST))
    return Fill3(out, np.asarray(pal, np.uint8), lc, bool(is_flat), before, score, most, int(n), bad, debug, view)
