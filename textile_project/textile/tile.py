"""Repeat unit (Kaam C): reference_code/repeat_deshear_PROTOTYPE.py,
repeat_tile_extract_PROTOTYPE.py and seam_quilt_lib.py as a module.

The logic is the prototypes', step for step:
  1. deshear: x' = x - s*y (+ s*H0 so nothing leaves the left edge), Lanczos,
     with a mask of the pixels that came from the image (`valid`)
  2. crop search: every block size W x H in the given ranges (step 4) and
     every position (step 12) whose block + overlap is all valid; cost = MSE
     between the top strip and the strip one block below, plus the left
     strip and the strip one block right (overlap o = 56)
  3. the 25 cheapest crops are seam-cut (a min-cost path through the overlap,
     left-right then top-bottom) and the cheapest seam wins

Changed from the prototypes: paths, the shear and the size ranges are
arguments. By default s = shear / H from `textile repeat`, and the sizes are
the analysis' W and H +-8 px (ROADMAP: "W, H +-8"); the prototype searched
fixed ranges (496-512 x 308-324) and a fixed s = 29/315.8, which the tests
pass to show the result is the prototype's own. A negative shear is
desheared the other way (the prototype only had a positive one).

New, from CLAUDE.md Kaam C steps 5-6 (no prototype): the 3 x 3 preview, the
upscale (wrap-padded Lanczos and a light unsharp, so the edges still meet)
and the optional clean ground.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter


def seam_path(cost):
    R, C = cost.shape; E = cost.copy(); back = np.zeros((R, C), int)
    for r in range(1, R):
        prev = E[r - 1]
        l = np.r_[np.inf, prev[:-1]]; m = prev; rr = np.r_[prev[1:], np.inf]
        st = np.vstack([l, m, rr]); k = st.argmin(0)
        back[r] = np.arange(C) + k - 1; E[r] += st.min(0)
    p = np.zeros(R, int); p[-1] = E[-1].argmin()
    for r in range(R - 1, 0, -1): p[r - 1] = back[r, p[r]]
    return p, E[-1].min() / R


def make(img, x0, y0, W, H, o):
    C = img[y0:y0 + H + o, x0:x0 + W + o].copy()
    A, B = C[:, :o], C[:, W:W + o]; p, ch = seam_path(((A - B) ** 2).sum(-1))
    T = C[:, :W].copy()
    for r in range(H + o): T[r, :p[r]] = B[r, :p[r]]
    A, B = T[:o], T[H:H + o]; q, cv = seam_path(((A - B) ** 2).sum(-1).T)
    out = T[:H].copy()
    for c in range(W): out[:q[c], c] = B[:q[c], c]
    return out, ch + cv


def deshear(im, s):
    """The image straightened (x' = x - s*y) and the mask of real pixels."""
    H0, W0 = im.shape[:2]
    off = s * H0 if s > 0 else 0.0
    M = np.float32([[1, -s, off], [0, 1, 0]])
    Wd = int(W0 + abs(s) * H0) + 1
    ds = cv2.warpAffine(im, M, (Wd, H0), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0))
    valid = cv2.warpAffine(np.ones((H0, W0), np.uint8), M, (Wd, H0), flags=cv2.INTER_NEAREST) > 0
    return ds, valid


def extract(ds, valid, w_range, h_range, o=56, pos_step=12, top=25, log=print):
    """The seamless block: (tile float32, seam cost, (mse, W, H, x0, y0))."""
    ds = ds.astype(np.float32)
    cands = []
    for W in w_range:
        for H in h_range:
            for y0 in range(0, ds.shape[0] - H - o, pos_step):
                for x0 in range(0, ds.shape[1] - W - o, pos_step):
                    if not valid[y0:y0 + H + o, x0:x0 + W + o].all(): continue
                    a = np.mean((ds[y0:y0 + o, x0:x0 + W] - ds[y0 + H:y0 + H + o, x0:x0 + W]) ** 2)
                    b = np.mean((ds[y0:y0 + H, x0:x0 + o] - ds[y0:y0 + H, x0 + W:x0 + W + o]) ** 2)
                    cands.append((a + b, W, H, x0, y0))
    if not cands:
        raise ValueError('is size ka block image me poora nahi aata (image chhoti hai ya jhukav bahut).')
    cands.sort()
    log(f'[tile] {len(cands)} crops jaanche, sabse achhe 25 par seam-cut')
    best = None
    for c in cands[:top]:
        _, W, H, x0, y0 = c; t, cost = make(ds, x0, y0, W, H, o)
        if best is None or cost < best[1]: best = (t, cost, c)
    return best


def tiled(rgb, n=3):
    """The block laid n x n, straight (a half-drop is inside the block)."""
    return np.tile(rgb, (n, n, 1))


def upscale(rgb, width, pad=16):
    """Lanczos to `width` (height in proportion) on a wrap-padded copy, so the
    left edge is resampled knowing the right one; then a light unsharp."""
    h, w = rgb.shape[:2]
    height = max(1, round(width * h / w))
    if (width, height) == (w, h):
        return rgb
    p = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode='wrap')
    fx, fy = width / w, height / h
    big = Image.fromarray(p).resize((round(p.shape[1] * fx), round(p.shape[0] * fy)), Image.LANCZOS)
    x0, y0 = round(pad * fx), round(pad * fy)
    big = big.crop((x0, y0, x0 + width, y0 + height))
    return np.asarray(big.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2)))


def clean_ground(rgb, tol):
    """Pixels within `tol` (RGB distance) of the ground colour become exactly
    that colour: AI grain on a plain ground, gone; nothing else moves. The
    ground: the median of the most common colour bin (16 levels a channel),
    as a noisy ground has no one exact most common colour."""
    q = (rgb // 16).reshape(-1, 3).astype(np.int32)
    key = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    top = np.bincount(key).argmax()
    ground = np.median(rgb.reshape(-1, 3)[key == top], axis=0).round().astype(int)
    near = np.sqrt(((rgb.astype(int) - ground) ** 2).sum(-1)) <= tol
    out = rgb.copy()
    out[near] = ground
    return out, ground, float(near.mean())
