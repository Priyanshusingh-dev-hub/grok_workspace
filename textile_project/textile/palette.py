"""Flat palettes: the reference's own colours, the nearest one for every pixel,
and stray colours folded into the big ones.

`extract_palette` and `map_to_palette` are method1_colorfill.py's, unchanged
(the user's tested logic). `merge_stray` is new (ROADMAP Phase 1).
"""
from __future__ import annotations

import numpy as np

STRAY_SHARE = 0.0005      # under 0.05% of the pixels a colour is a stray, not a channel


def extract_palette(ref_rgb, max_colors, min_share, log=print):
    """Reference ke flat colors nikaalo. Agar reference already flat hai to exact colors,
    warna k-means se max_colors tak."""
    flat = ref_rgb.reshape(-1, 3)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    share = counts / counts.sum()
    keep = share >= min_share
    if keep.sum() <= max_colors and share[keep].sum() > 0.97:
        log(f"[palette] Reference flat hai: {keep.sum()} exact colors mile.")
        return cols[keep].astype(np.uint8), True
    log(f"[palette] Reference flat nahi hai ({len(cols)} unique colors). "
        f"k-means se {max_colors} colors bana raha hoon - result zaroor check karna.")
    from sklearn.cluster import KMeans
    sample = flat[np.random.default_rng(0).choice(len(flat), min(len(flat), 400000), replace=False)]
    km = KMeans(max_colors, n_init=4, random_state=0).fit(sample.astype(np.float32))
    pal = np.unique(np.clip(np.rint(km.cluster_centers_), 0, 255).astype(np.uint8), axis=0)
    return pal, False


def map_to_palette(img_rgb, pal):
    """Har pixel ko nearest palette index par map karo (chunked, memory safe)."""
    h, w, _ = img_rgb.shape
    out = np.empty((h, w), np.uint8)
    p = pal.astype(np.int32)
    step = max(1, 4_000_000 // (w * len(pal)))
    for y in range(0, h, step):
        blk = img_rgb[y:y + step].astype(np.int32)
        out[y:y + step] = ((blk[:, :, None, :] - p[None, None]) ** 2).sum(-1).argmin(-1)
    return out


def coverage(index, n):
    """Pixels per palette entry."""
    return np.bincount(index.ravel(), minlength=n)


def merge_stray(index, pal, min_share=STRAY_SHARE):
    """Fold every colour under `min_share` of the pixels into the nearest big
    colour (RGB distance, as map_to_palette measures), so no channel is made
    for a few hundred stray pixels. Unused colours leave the palette; the big
    ones keep their order.

    Returns (index, palette, merged) where merged lists
    {'hex', 'pixels', 'into'} for every colour folded in."""
    pal = np.asarray(pal, np.uint8)
    cnt = coverage(index, len(pal))
    total = max(int(cnt.sum()), 1)
    big = np.nonzero(cnt / total >= min_share)[0]
    if len(big) == 0:                       # nothing is big: keep the most used one
        big = np.array([int(cnt.argmax())])
    target = np.arange(len(pal))
    merged = []
    p = pal.astype(np.int32)
    for k in range(len(pal)):
        if k in big:
            continue
        near = big[((p[big] - p[k]) ** 2).sum(1).argmin()]
        target[k] = near
        if cnt[k]:
            merged.append({'hex': _hex(pal[k]), 'pixels': int(cnt[k]), 'into': _hex(pal[near])})
    new_pos = {int(k): i for i, k in enumerate(big)}
    remap = np.array([new_pos[int(target[k])] for k in range(len(pal))], np.uint8)
    return remap[index], pal[big].copy(), merged


def dropped_colours(rgb, pal):
    """The exact colours of a flat image that are not in `pal`, each with its
    pixel count and the palette colour map_to_palette sends it to."""
    cols, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    have = {tuple(int(v) for v in c) for c in pal}
    out = []
    p = np.asarray(pal, np.int32)
    for c, n in zip(cols, counts):
        if tuple(int(v) for v in c) in have:
            continue
        near = pal[((p - c.astype(np.int32)) ** 2).sum(1).argmin()]
        out.append({'hex': _hex(c), 'pixels': int(n), 'into': _hex(near)})
    return sorted(out, key=lambda m: -m['pixels'])


def _hex(rgb):
    return '{:02X}{:02X}{:02X}'.format(*(int(v) for v in rgb))
