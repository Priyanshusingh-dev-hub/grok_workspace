"""Clean edges ("kinare saaf"): every ink's outline redrawn as a smooth,
hard-edged shape, optionally on a finer pixel grid.

The prints this is for come from a Reduce or a fill of an AI image: the edge
of every shape is a noisy staircase, a pixel or two off the line the drawing
meant. Gaussian smoothing is not the cure (the mill's rule: it rounds corners
and melts dots). This keeps what makes a drawing look drawn:

  - the outline of each ink is traced (OpenCV contours, holes included) and
    smoothed ALONG itself, not across it, so a stripe keeps its width
  - corners are found first (two straight arms meeting at an angle: noise has
    no straight arms) and held fixed, so a petal tip stays sharp
  - thin parts (under 3 px) may move only a little, and shapes too small to
    have an outline stay exactly as they were, so a dot is not smoothed away
  - the shapes are drawn back WITHOUT anti-aliasing (fillPoly, LINE_8), on a
    grid `scale` times finer, so the pixel staircase of the source is gone

One ink per pixel holds by construction: shapes are painted biggest first,
each over the ones before, and the few pixels no shape covers take their
neighbour's ink. No new colour is ever made (the output is an index map of
the same inks).

`strength` 1 (gentle) / 2 / 3 (strong) sets how far an edge may move and how
long a stretch is averaged. Optionally (`speck`, off by default) the
8-connected islands of a few pixels first go to the ink around them: they are
often real small motifs, so that is the operator's call (LoomLab's tiny-dot
check is the tool for printability).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

# sigma: px of outline averaged; move: the most an edge may shift (px); thin: the most a part
# under 3 px wide may shift
STRENGTH = {1: dict(sigma=1.5, move=1.0, thin=0.4),
            2: dict(sigma=2.5, move=2.0, thin=0.5),
            3: dict(sigma=3.5, move=3.0, thin=0.6)}
WRAP = 32                      # a seamless repeat is wrapped this many px before tracing
MIN_POINTS = 12                # a shape with a shorter outline is kept as it is
MIN_AREA = 24
_SHIFT = 4                     # fillPoly sub-pixel bits
_OUT = 0.1                     # outward shift (px), well under 0.5: fillPoly also paints the pixels its border touches (swept on a drawn-at-4x truth)


def _settings(strength):
    s = int(strength)
    if s not in STRENGTH:
        raise ValueError(f'strength 1, 2 ya 3 hona chahiye (mila {strength}).')
    return STRENGTH[s]


def drop_specks(index, max_px):
    """8-connected islands of `max_px` px or fewer go to the ink most of their
    border touches. Returns (new index, number of islands moved)."""
    out = index.copy()
    moved = 0
    st = np.ones((3, 3), int)
    for k in range(int(index.max()) + 1):
        lab, n = ndimage.label(index == k, structure=st)
        if not n:
            continue
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        small = np.flatnonzero(sizes[1:] <= max_px) + 1
        if not len(small):
            continue
        slices = ndimage.find_objects(lab)
        for c in small:
            sl = slices[c - 1]
            y0, x0 = max(sl[0].start - 1, 0), max(sl[1].start - 1, 0)
            y1, x1 = sl[0].stop + 1, sl[1].stop + 1
            m = lab[y0:y1, x0:x1] == c
            ring = ndimage.binary_dilation(m, structure=st) & ~m
            nb = index[y0:y1, x0:x1][ring]
            nb = nb[nb != k]
            if len(nb):
                out[y0:y1, x0:x1][m] = np.bincount(nb).argmax()
                moved += 1
    return out, moved


def find_corners(p, arm=6, deg=50, dev_max=0.9):
    """Indices of the real corners of a closed outline `p` (n, 2). A corner has
    two STRAIGHT arms (each within `dev_max` px of its chord over `arm` points)
    that meet at more than `deg` degrees; where several qualify the sharpest wins."""
    n = len(p)
    if n < 3 * arm:
        return []
    i = np.arange(n)
    j = np.arange(arm + 1)
    back = p[(i[:, None] - arm + j[None, :]) % n]
    fwd = p[(i[:, None] + j[None, :]) % n]

    def straight(a):
        c = a[:, -1] - a[:, 0]
        r = a - a[:, :1]
        cross = np.abs(r[:, :, 0] * c[:, None, 1] - r[:, :, 1] * c[:, None, 0])
        return cross.max(1) / (np.linalg.norm(c, axis=1) + 1e-9), c

    db, cb = straight(back)
    df, cf = straight(fwd)
    cos = (cb * cf).sum(1) / ((np.linalg.norm(cb, axis=1) + 1e-9) * (np.linalg.norm(cf, axis=1) + 1e-9))
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ok = (db <= dev_max) & (df <= dev_max) & (ang > deg)
    ang = np.where(ok, ang, 0)
    out = []
    half = arm // 2
    for k in np.flatnonzero(ok):
        if ang[k] >= ang[np.arange(k - half, k + half + 1) % n].max() and not (out and k - out[-1] <= half):
            out.append(int(k))
    if len(out) > 1 and out[0] + n - out[-1] <= half:
        out.pop()
    return out


def _kernel(sigma):
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum(), r


def _blur_closed(p, sigma):
    k, r = _kernel(sigma)
    ext = np.vstack([p[-r:], p, p[:r]])
    return np.stack([np.convolve(ext[:, d], k, mode='valid') for d in (0, 1)], 1)


def _blur_open(seg, sigma):
    """An outline stretch whose two ends are corners: the ends stay where they
    are (odd reflection about each end)."""
    k, r = _kernel(sigma)
    m = len(seg)
    pad = min(r, m - 1)
    kk = k[r - pad:r + pad + 1]
    kk = kk / kk.sum()
    ext = np.vstack([2 * seg[0] - seg[pad:0:-1], seg, 2 * seg[-1] - seg[-2:-pad - 2:-1]])
    return np.stack([np.convolve(ext[:, d], kk, mode='valid') for d in (0, 1)], 1)


def smooth_outline(p, sigma, move, corners=None):
    """Closed outline `p` (n, 2) averaged along itself; corner points stay put,
    and no point moves more than `move` px (an array: per point).

    One Gaussian pass shrinks a round shape (a dot would melt), so the
    averaging is done as 2*G(p) - G(G(p)): the same smoothing of the wobble,
    with the shrinking taken back out (Taubin's lambda|mu with mu = -1)."""
    n = len(p)
    if corners is None:
        corners = find_corners(p)
    sigma = min(sigma, n / 10.0)               # never longer than the outline itself
    if sigma < 0.8:
        return p
    out = p.copy()
    if not corners:
        g = _blur_closed(p, sigma)
        out = 2 * g - _blur_closed(g, sigma)
    else:
        for a in range(len(corners)):
            s, e = corners[a], corners[(a + 1) % len(corners)]
            idx = np.arange(s, e + 1 if e > s else e + n + 1) % n
            seg = p[idx]
            if len(seg) < 4:
                continue
            g = _blur_open(seg, sigma)
            out[idx] = 2 * g - _blur_open(g, sigma)
    d = out - p
    dn = np.linalg.norm(d, axis=1, keepdims=True)
    cap = np.broadcast_to(np.asarray(move, float), (n,))[:, None]
    return p + d * np.minimum(1.0, cap / np.maximum(dn, 1e-9))


def _outward(q):
    """`q` moved half a pixel out of its shape: contour points are the centres of
    the boundary pixels, half a pixel inside the true edge. OpenCV runs every
    outline (an outer one and a hole alike) with its shape on the same side,
    so one fixed normal is outward for all of them."""
    t = np.roll(q, -1, 0) - np.roll(q, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return q - _OUT * np.stack([t[:, 1], -t[:, 0]], 1)


def _fill_gaps(out, covered, nearest_of):
    """Pixels no shape covers take the ink of a covered neighbour, ring by ring;
    anything still bare (not reachable) takes the plain upscaled index."""
    ys, xs = np.nonzero(~covered)
    h, w = out.shape
    for _ in range(64):
        if not len(ys):
            return
        got = np.zeros(len(ys), bool)
        val = np.zeros(len(ys), out.dtype)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            yy, xx = np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)
            ok = covered[yy, xx] & ~got
            val[ok] = out[yy[ok], xx[ok]]
            got |= ok
        out[ys[got], xs[got]] = val[got]
        covered[ys[got], xs[got]] = True
        ys, xs = ys[~got], xs[~got]
    if len(ys):
        out[ys, xs] = nearest_of(ys, xs)


def clean(index, scale=1, strength=2, wrap=(False, False), speck=None):
    """The index map `index` (H, W; values < 255) with clean edges, drawn on a
    grid `scale` times finer (any factor from 0.25 to 8; 1 keeps the size):
    (out index, report dict). `wrap` = (x, y): the design repeats along that
    axis, so its edges are cleaned round the seam. `speck` = remove islands of up
    to this many px first (0, the default: none; measured on a real design, removing
    2 px islands cost 2.5 match points because they are real small motifs)."""
    cfg = _settings(strength)
    scale = float(scale)
    if not 0.25 <= scale <= 8:
        raise ValueError('scale 0.25 se 8 ke beech hona chahiye.')
    index = np.ascontiguousarray(index, np.uint8)
    H, W = index.shape
    px, py = (min(WRAP, W) if wrap[0] else 0), (min(WRAP, H) if wrap[1] else 0)
    work = np.pad(index, ((py, py), (px, px)), mode='wrap') if (px or py) else index
    speck = int(speck or 0)
    moved = 0
    if speck:
        work, moved = drop_specks(work, speck)
    h, w = work.shape
    ow, oh = max(1, round(w * scale)), max(1, round(h * scale))
    sx, sy = ow / w, oh / h                       # exactly the grid the output has
    K = int(work.max()) + 1
    out = np.zeros((oh, ow), np.uint8)
    covered = np.zeros((oh, ow), bool)
    k3 = np.ones((3, 3), np.uint8)
    smoothed = kept = 0
    for k in np.argsort(-np.bincount(work.ravel(), minlength=K), kind='stable'):
        mask = (work == k).astype(np.uint8)
        if not mask.any():
            continue
        thin = cv2.dilate(((mask > 0) & ~(cv2.dilate(cv2.erode(mask, k3), k3) > 0)).astype(np.uint8), k3) > 0
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        shapes, raw = [], []
        for c in contours:
            c = c[:, 0, :].astype(np.float64)
            if len(c) < MIN_POINTS or cv2.contourArea(c.astype(np.float32)) < MIN_AREA:
                raw.append(c.astype(np.int32))
                continue
            yy = np.clip(c[:, 1].astype(int), 0, h - 1)
            xx = np.clip(c[:, 0].astype(int), 0, w - 1)
            cap = np.where(thin[yy, xx], cfg['thin'], cfg['move'])
            shapes.append(_outward(smooth_outline(c, cfg['sigma'], cap)))
            smoothed += 1
        drawn = np.zeros((oh, ow), np.uint8)
        if shapes:
            pts = [np.round(np.stack([(q[:, 0] + 0.5) * sx - 0.5, (q[:, 1] + 0.5) * sy - 0.5], 1) * (1 << _SHIFT)
                            ).astype(np.int32) for q in shapes]
            cv2.fillPoly(drawn, pts, 1, lineType=cv2.LINE_8, shift=_SHIFT)
        if raw:
            small = np.zeros((h, w), np.uint8)
            cv2.fillPoly(small, raw, 1, lineType=cv2.LINE_8)
            small &= mask                       # a dot comes back exactly, a 1 px line too
            kept += len(raw)
            drawn |= cv2.resize(small, (ow, oh), interpolation=cv2.INTER_NEAREST)
        on = drawn > 0
        out[on] = k
        covered |= on

    def nearest(ys, xs):
        return work[np.minimum((ys / sy).astype(int), h - 1), np.minimum((xs / sx).astype(int), w - 1)]

    _fill_gaps(out, covered, nearest)
    if px or py:
        x0, y0 = round(px * sx), round(py * sy)
        out = out[y0:y0 + round(H * scale), x0:x0 + round(W * scale)]
    out = np.ascontiguousarray(out)
    return out, {'scale': round(scale, 3), 'strength': int(strength), 'specks_moved': int(moved),
                 'outlines_smoothed': smoothed, 'kept_as_is': kept}
