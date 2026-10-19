"""Colour fill, Method 2 (misaligned line art + reference): reference_code/
method2_structure_fill_PROTOTYPE.py as a module. Two colours, ground and
motif; the reference gives the colour scheme, the line art gives every shape.

The logic is the prototype's, step for step (its numbers are the defaults):
  1. line art -> line mask (Lanczos up, 3x3 blur, < threshold), regions 4-connected
  2. which regions face each other across a line: look up to `reach` px along
     8 directions from every region pixel that touches a line
  3. regions touching the image edge are the ground (depth 0); so are big
     closed regions the reference shows dark: area > seed_area and dark >
     seed_dark, or in the side bands area > side_area and dark > side_dark
     (never small ones: petals would come out hollow)
  4. depth by BFS over neighbours: odd = motif, even = ground
  5. a line pixel is ground only if every region it faces is motif (the
     divider between petals), else motif (a lone stem still shows)

Changed from the prototype, as ROADMAP Phase 2 asks: paths and numbers are
arguments, the two colours come from the reference (the most common colour is
the ground) unless given, and a design that is not square keeps its
proportions. With --colors 10100F,E8DFD2 it is the prototype's output.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from . import palette as pl
from .fill_method1 import FillError, output_size
from .io_utils import read_cv2, rgb_of

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
PROTOTYPE_COLOURS = ('10100F', 'E8DFD2')     # INK (16,16,15) and cream (232,223,210)


@dataclass
class Fill2:
    index: np.ndarray        # 0 = ground, 1 = motif
    pal: np.ndarray          # [ground, motif]
    regions: int
    seeds: int               # big dark regions made ground from the reference
    unreached: int
    depth_hist: list
    reference: np.ndarray    # 1200 px view for the compare sheet


def _shift(a, dy, dx, H, W):
    out = np.zeros_like(a)
    ys = slice(max(dy, 0), H + min(dy, 0)); yd = slice(max(-dy, 0), H + min(-dy, 0))
    xs = slice(max(dx, 0), W + min(dx, 0)); xd = slice(max(-dx, 0), W + min(-dx, 0))
    out[yd, xd] = a[ys, xs]
    return out


def reference_colours(ref_rgb, min_share=0.0005, log=print):
    """[ground, motif] from the reference: its two colours (k-means 2 unless it
    is already flat), the more common one first."""
    pal, _ = pl.extract_palette(ref_rgb, 2, min_share, log=log)
    if len(pal) == 1:
        raise FillError('reference me sirf ek rang hai; Method 2 ko ground aur motif do rang chahiye.')
    cnt = pl.coverage(pl.map_to_palette(ref_rgb, pal), len(pal))
    order = np.argsort(-cnt)[:2]
    return pal[order]


def fill(line_path, ref_path, size=3535, line_threshold=150, reach=14, dark_level=110,
         seed_area=6000, seed_dark=0.8, side_band=0.27, side_area=9000, side_dark=0.55,
         colors=None, log=print) -> Fill2:
    line = read_cv2(line_path, cv2.IMREAD_GRAYSCALE)
    ref_grey = read_cv2(ref_path, cv2.IMREAD_GRAYSCALE)
    ref_bgr = read_cv2(ref_path, cv2.IMREAD_COLOR)
    if line is None or ref_grey is None:
        raise FillError('image read nahi hui - path check karo.')
    W, H = output_size(line.shape[1], line.shape[0], size)

    g = cv2.resize(line, (W, H), interpolation=cv2.INTER_LANCZOS4); g = cv2.GaussianBlur(g, (3, 3), 0)
    lines = g < line_threshold
    L, n = ndimage.label(~lines); L = L.astype(np.int32)
    log(f'[method2] regions {n}')

    Fs = []; edges = set()
    for dy, dx in DIRS:
        F = np.zeros_like(L)
        for k in range(reach, 0, -1):
            sk = _shift(L, dy * k, dx * k, H, W); F = np.where(sk != 0, sk, F)
        Fs.append(F)
        nxt = _shift(L, dy, dx, H, W)
        m = (L != 0) & (nxt == 0) & (F != 0) & (F != L)
        a = L[m]; b = F[m]
        pr = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], 1), axis=0)
        edges.update(map(tuple, pr))
    adj = [[] for _ in range(n + 1)]
    for a, b in edges: adj[a].append(b); adj[b].append(a)
    border = set(np.unique(np.concatenate([L[0], L[-1], L[:, 0], L[:, -1]]))) - {0}

    refb = cv2.resize(ref_grey, (W, H), interpolation=cv2.INTER_AREA) < dark_level
    area = np.bincount(L.ravel(), minlength=n + 1)
    blk = np.bincount(L.ravel(), weights=refb.ravel(), minlength=n + 1) / np.maximum(area, 1)
    cx = ndimage.center_of_mass(np.ones_like(L), L, range(1, n + 1)); cx = np.array([0] + [c[1] for c in cx])
    side = (cx < side_band * W) | (cx > (1 - side_band) * W)
    big = [r for r in range(1, n + 1) if (area[r] > seed_area and blk[r] > seed_dark)
           or (side[r] and area[r] > side_area and blk[r] > side_dark)]
    log(f'[method2] extra ground seeds {len(big)}')
    border |= set(big)
    depth = np.full(n + 1, -1); q = deque()
    for r in border: depth[r] = 0; q.append(r)
    while q:
        r = q.popleft()
        for t in adj[r]:
            if depth[t] < 0: depth[t] = depth[r] + 1; q.append(t)
    unreached = int((depth[1:] < 0).sum())
    hist = np.bincount(depth[1:][depth[1:] >= 0]).tolist()
    log(f'[method2] unreached {unreached}, depth hist {hist}')
    depth[depth < 0] = 1
    cream_r = (depth % 2 == 1); cream_r[0] = False
    creamR = cream_r[L]
    allc = np.ones((H, W), bool); anyf = np.zeros((H, W), bool)
    for F in Fs:
        f = F != 0; anyf |= f
        allc &= np.where(f, cream_r[F], True)
    lineCream = lines & ~(allc & anyf)
    motif = np.where(lines, lineCream, creamR)

    if colors:
        pal = np.array([rgb_of(c) for c in colors], np.uint8)
    else:
        pal = reference_colours(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB), log=log)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    rw, rh = (1200, 1200) if W == H else (1200, max(1, round(1200 * H / W)))
    view = np.asarray(Image.fromarray(ref_rgb).resize((rw, rh), Image.NEAREST))
    return Fill2(motif.astype(np.uint8), pal, int(n), len(big), unreached, hist, view)
