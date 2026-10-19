"""The sketch drawn as smooth curves: every part's outline traced, smoothed along
itself (a round edge stays round, a corner with two straight arms stays a sharp
corner: textile edges' own `find_corners` / `smooth_outline`), and drawn in
crisp, dark, anti-aliased strokes, plus the same lines as an SVG that stays sharp
at any zoom. For looking at, sharing and drawing over: the sketch `textile
paint` reads stays the pixel one (`NAME_sketch_seal0.png`), whose areas are the
numbers.

  bold(lab, line_patch, ...)  -> (image H x W uint8, svg text)

`lab` is the design's parts (H x W, 0 unused), `line_patch[i]` says part i is
itself a drawn line (a thin dark outline), filled solid instead of outlined.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import edges as ed
from .io_utils import unique_rgb

EDGE_SHIFT = 0.5        # contour points are boundary-pixel centres: half a pixel out is the true edge


def _outlines(mask):
    """The closed outlines (outer and holes) of one part's mask, as float (n, 2) x,y on the true pixel edge."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        p = c[:, 0, :].astype(np.float64)
        if len(p) >= 3:
            t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
            t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
            p = p - EDGE_SHIFT * np.stack([t[:, 1], -t[:, 0]], 1)
        out.append(p)
    return out


STRAIGHT_DEV = 0.9      # a stretch within this many source px of its chord is a straight line, drawn straight
CIRCLE_IOU = 0.90       # a closed outline that fills >= this share of its best-fit circle (and vice versa) is drawn as that circle
CIRCLE_DEV = 0.12       # ...and no point of it strays more than this share of the radius (+1.5 px): a toothed or scalloped ring keeps its teeth
POLY_DEV = 0.03        # a polygon: no outline point farther than this share of sqrt(area) (+1.5 px) from its straight sides
POLY_MIN_SIDE = 6      # a shape smaller than this (sqrt of its area, px) is a dot, never turned into a polygon
FAIR_MAX_RADIUS = 0.3   # fairing sigma on a closed outline is at most this share of its radius (perimeter / 2 pi)
SMALL_ROUND_PER = 50.0  # an outline shorter than this (source px round) with at most two corners, none sharper than 110 degrees: one round curve
LEAF_MIN_ASPECT = 1.6   # a leaf / petal is at least this much longer (tip to tip) than it is wide; a near-round dot with a bump is not
CORNER_DEG = 60.0       # a bend sharper than this (turning angle) is a design corner and stays sharp; a gentler one is eased into the curve
FAIR_SIGMA = 2.5        # fairing of every smooth stretch, in source px along the curve: slope changes gradually, small jogs and wobbles ease out
PRUNE_SRC = 4.0         # a stretch between two corners shorter than this (source px) is a notch or an ear, not a design corner
BLOB_SMOOTH = 1.0       # extra smoothing for a small closed blob; 2.5 flattened real dots, so it is off (as the version the user liked)
SPLINE_SMOOTH = 1.6     # spline smoothing, in source px of allowed wobble per point


def _spline(seg, scale, closed):
    """One stretch (n, 2) fitted by a smoothing cubic spline: a sine-like wave, a long arc, a spiral all come out
    as one flowing curve, the source's pixel stairs (and the stairs' wobble) averaged away. Ends are held when
    the stretch is open (they are corners)."""
    from scipy.interpolate import splprep, splev
    n = len(seg)
    pts = seg
    if closed:
        pts = np.vstack([seg, seg[:1]])
    # drop repeated points (splprep needs distinct ones)
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6]
    pts = pts[keep]
    if len(pts) < 5:
        return seg
    w = np.ones(len(pts))
    blob = BLOB_SMOOTH if closed and n < 60 * scale else 1.0   # a small closed blob (a border dot): 1.0 = as every curve
    if not closed:
        w[0] = w[-1] = 50.0                              # ends stay where they are
    try:
        tck, u = splprep([pts[:, 0], pts[:, 1]], w=w, s=len(pts) * (SPLINE_SMOOTH * blob * scale) ** 2 * 0.25,
                         per=1 if closed else 0, k=3)
    except Exception:
        return seg
    m = max(int(n), 8)
    t = np.linspace(0, 1, m, endpoint=not closed)
    x, y = splev(t, tck)
    return np.stack([x, y], 1)


def circle_of(q, size, iou=CIRCLE_IOU):
    """The perfect circle (polygon, float x,y) for a closed outline `q` that is about a circle, else None.
    Least-squares circle through the outline; 'about' = the shape and the circle overlap by at least `iou`
    (intersection over union, measured on pixels) and no point strays far from it (CIRCLE_DEV). An outline cut by the sheet's edge is never a circle."""
    if iou <= 0 or len(q) < 12:
        return None
    H, W = size
    if q[:, 0].min() < 1.5 or q[:, 1].min() < 1.5 or q[:, 0].max() > W - 1.5 or q[:, 1].max() > H - 1.5:
        return None
    x, y = q[:, 0], q[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones(len(q))], 1)
    try:
        (a, b, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    r2 = c + a * a + b * b
    if not np.isfinite(r2) or r2 <= 1:
        return None
    r = float(np.sqrt(r2))
    if np.abs(np.hypot(x - a, y - b) - r).max() > CIRCLE_DEV * r + 1.5:
        return None
    x0, y0 = int(np.floor(min(x.min(), a - r))) - 2, int(np.floor(min(y.min(), b - r))) - 2
    x1, y1 = int(np.ceil(max(x.max(), a + r))) + 2, int(np.ceil(max(y.max(), b + r))) + 2
    shape = (y1 - y0, x1 - x0)
    mine = np.zeros(shape, np.uint8)
    ring = np.zeros(shape, np.uint8)
    cv2.fillPoly(mine, [np.round((q - (x0, y0)) * 16).astype(np.int32)], 1, shift=4)
    cv2.circle(ring, (int(round((a - x0) * 16)), int(round((b - y0) * 16))), int(round(r * 16)), 1, -1, shift=4)
    union = np.count_nonzero(mine | ring)
    if union == 0 or np.count_nonzero(mine & ring) / union < iou:
        return None
    n = max(64, int(2 * np.pi * r * 4))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([a + r * np.cos(t), b + r * np.sin(t)], 1)


def _seg_dist(pts, poly):
    """Distance of every point (n, 2) to the closed polygon's outline (vertices (m, 2)): the nearest side."""
    best = np.full(len(pts), np.inf)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        d = b - a
        L2 = float(d @ d)
        t = np.clip(((pts - a) @ d) / L2, 0, 1) if L2 > 0 else np.zeros(len(pts))
        best = np.minimum(best, np.linalg.norm(pts - (a + t[:, None] * d), axis=1))
    return best


def _strays(pts, poly, tol, chunk=64):
    """`_seg_dist(pts, poly).max() > tol` without measuring every point against every side (an oval's outline has
    thousands: that was 88 of a run's 206 s; now ~1 s). A point within `tol` of a corner is within `tol` of the
    outline. The others, farthest from any corner first, are measured only against the sides that could come
    within `tol` (an end within tol + the longest side), by `_seg_dist`'s own sums. The same answer but when the
    farthest point sits within a rounding error (1e-15) of `tol` (random test: 3 of 8000 at tol = max +- 1 ulp)."""
    if len(pts) == 0:
        return False
    tree = cKDTree(poly)
    d, _ = tree.query(pts)
    far = np.flatnonzero(d > tol * (1 - 1e-9))         # the tree's sums may round a hair below numpy's: margin
    if not len(far):
        return False
    far = far[np.argsort(-d[far], kind='stable')]
    m = len(poly)
    reach = (tol + float(np.linalg.norm(np.roll(poly, -1, 0) - poly, axis=1).max())) * (1 + 1e-9) + 1e-9
    for s in range(0, len(far), chunk):
        P = pts[far[s:s + chunk]]
        near = set()
        for v in tree.query_ball_point(P, reach):
            near.update(v)
        sides = sorted({i for v in near for i in (v, (v - 1) % m)})
        best = np.full(len(P), np.inf)
        for i in sides:
            a, b = poly[i], poly[(i + 1) % m]
            dd = b - a
            L2 = float(dd @ dd)
            t = np.clip(((P - a) @ dd) / L2, 0, 1) if L2 > 0 else np.zeros(len(P))
            best = np.minimum(best, np.linalg.norm(P - (a + t[:, None] * dd), axis=1))
        if (best > tol).any():
            return True
    return False


def _fit_sides(q, idx, verts):
    """Refine a polygon: each side is the least-squares line through its own outline points (the ends, near the
    corners, left out), neighbours' lines meet in the new corners. Falls back to the plain vertex when two sides
    are almost parallel or the meeting point strays."""
    m = len(verts)
    lines = []
    n = len(q)
    for i in range(m):
        i0, i1 = idx[i], idx[(i + 1) % m]
        span = (i1 - i0) % n or n
        cut = int(span * 0.15)
        pts = q[[(i0 + j) % n for j in range(cut, span - cut + 1)]]
        if len(pts) < 3:
            return verts
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], float), np.array([vx, vy], float)))
    out = []
    for i in range(m):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        v = verts[i]
        if abs(den) > 0.2:
            t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
            c = p1 + t * d1
            side = max(np.linalg.norm(verts[i] - verts[i - 1]), np.linalg.norm(verts[(i + 1) % m] - verts[i]))
            if np.linalg.norm(c - v) < 0.2 * side:
                v = c
        out.append(v)
    return np.array(out)


def _soft_corner(poly, interior_max=150.0):
    """True when some corner of the polygon is open wider than `interior_max` degrees (the turn there is
    under 30 degrees): a sign the 'sides' are really one curve approximated in pieces."""
    m = len(poly)
    for i in range(m):
        u, v = poly[i - 1] - poly[i], poly[(i + 1) % m] - poly[i]
        c = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
        if np.degrees(np.arccos(np.clip(c, -1, 1))) > interior_max:
            return True
    return False


def polygon_of(q, size, iou=CIRCLE_IOU):
    """The clean polygon (vertices (m, 2), 3 to 6 straight sides) for a closed outline `q` that already is one
    (a triangle, a rectangle, a diamond, a hexagon...), else None. Judged on the outline itself: the polygon
    must overlap the shape by `iou` AND every outline point must lie close to its sides (POLY_DEV), so a leaf,
    a petal or any shape with a curved side is never straightened, and a polygon with a little wobble is."""
    if iou <= 0 or len(q) < 12:
        return None
    H, W = size
    if q[:, 0].min() < 1.5 or q[:, 1].min() < 1.5 or q[:, 0].max() > W - 1.5 or q[:, 1].max() > H - 1.5:
        return None
    A = cv2.contourArea(q.astype(np.float32))
    if A < POLY_MIN_SIDE ** 2:
        return None
    per = cv2.arcLength(q.astype(np.float32), True)
    tol = POLY_DEV * np.sqrt(A) + 1.5
    for frac in (0.06, 0.045, 0.03, 0.02):                  # fewest corners first
        ap = cv2.approxPolyDP(q.astype(np.float32).reshape(-1, 1, 2), frac * per, True)[:, 0, :].astype(float)
        if not 3 <= len(ap) <= 5:
            continue
        idx = [int(np.argmin(np.linalg.norm(q - v, axis=1))) for v in ap]
        if sorted(idx) != idx and len(set(idx)) == len(idx):
            order = np.argsort(idx)
            ap, idx = ap[order], [idx[j] for j in order]
        if len(set(idx)) != len(idx):
            continue
        poly = _fit_sides(q, idx, ap)
        if _soft_corner(poly):
            continue                                        # a corner that is nearly straight is a curve cut in pieces
        if _strays(q, poly, tol):
            continue
        x0, y0 = int(np.floor(min(q[:, 0].min(), poly[:, 0].min()))) - 2, int(np.floor(min(q[:, 1].min(), poly[:, 1].min()))) - 2
        x1, y1 = int(np.ceil(max(q[:, 0].max(), poly[:, 0].max()))) + 2, int(np.ceil(max(q[:, 1].max(), poly[:, 1].max()))) + 2
        mine = np.zeros((y1 - y0, x1 - x0), np.uint8)
        new = np.zeros_like(mine)
        cv2.fillPoly(mine, [np.round((q - (x0, y0)) * 16).astype(np.int32)], 1, shift=4)
        cv2.fillPoly(new, [np.round((poly - (x0, y0)) * 16).astype(np.int32)], 1, shift=4)
        union = np.count_nonzero(mine | new)
        if union and np.count_nonzero(mine & new) / union >= iou:
            return poly
    return None


def _raster_iou(q, poly, iou):
    """True when the shape `q` and `poly` (both closed outlines, float x,y) overlap by at least `iou`."""
    x0 = int(np.floor(min(q[:, 0].min(), poly[:, 0].min()))) - 2
    y0 = int(np.floor(min(q[:, 1].min(), poly[:, 1].min()))) - 2
    x1 = int(np.ceil(max(q[:, 0].max(), poly[:, 0].max()))) + 2
    y1 = int(np.ceil(max(q[:, 1].max(), poly[:, 1].max()))) + 2
    mine = np.zeros((y1 - y0, x1 - x0), np.uint8)
    new = np.zeros_like(mine)
    cv2.fillPoly(mine, [np.round((q - (x0, y0)) * 16).astype(np.int32)], 1, shift=4)
    cv2.fillPoly(new, [np.round((poly - (x0, y0)) * 16).astype(np.int32)], 1, shift=4)
    union = np.count_nonzero(mine | new)
    return bool(union) and np.count_nonzero(mine & new) / union >= iou


def _inside_sheet(q, size):
    H, W = size
    return not (q[:, 0].min() < 1.5 or q[:, 1].min() < 1.5 or q[:, 0].max() > W - 1.5 or q[:, 1].max() > H - 1.5)


def oval_of(q, size, iou=CIRCLE_IOU):
    """The clean ellipse (polygon points) for a closed outline that is about an oval (a long dot, a plain oval
    petal with rounded ends), else None: same two tests as the others (overlap >= iou, no point strays)."""
    if iou <= 0 or len(q) < 12 or not _inside_sheet(q, size):
        return None
    A = cv2.contourArea(q.astype(np.float32))
    if A < POLY_MIN_SIDE ** 2:
        return None
    (cx, cy), (w, h), ang = cv2.fitEllipse(q.astype(np.float32))
    a, b = w / 2, h / 2
    if min(a, b) < 2 or max(a, b) / min(a, b) > 6:
        return None
    t = np.linspace(0, 2 * np.pi, max(64, int(2 * np.pi * max(a, b) * 4)), endpoint=False)
    th = np.radians(ang)
    poly = np.stack([cx + a * np.cos(t) * np.cos(th) - b * np.sin(t) * np.sin(th),
                     cy + a * np.cos(t) * np.sin(th) + b * np.sin(t) * np.cos(th)], 1)
    if _strays(q, poly, POLY_DEV * np.sqrt(A) + 1.5) or not _raster_iou(q, poly, iou):
        return None
    return poly


def _bezier_side(side):
    """One side of a leaf (points from tip to tip) as a quadratic curve through both tips, its single control
    point found by least squares (arc length as the parameter). Returns (control point, chord midpoint)."""
    P0, P2 = side[0], side[-1]
    d = np.linalg.norm(np.diff(side, axis=0), axis=1)
    t = np.r_[0, np.cumsum(d)] / max(d.sum(), 1e-9)
    w = 2 * t * (1 - t)
    base = ((1 - t) ** 2)[:, None] * P0 + (t ** 2)[:, None] * P2
    C = ((side - base) * w[:, None]).sum(0) / max((w * w).sum(), 1e-9)
    return C, (P0 + P2) / 2


def leaf_of(q, size, scale, iou=CIRCLE_IOU):
    """The clean leaf / petal (polygon points) for a closed outline with two pointed tips and a curved side
    between them, else None. Each side is fitted by a smooth quadratic arc through the tips; when both sides
    bow out about the same (within 25%) they are made equal, so a petal comes out symmetric. Only when the
    result still overlaps the shape by `iou` and no outline point strays (POLY_DEV)."""
    if iou <= 0 or len(q) < 24 or not _inside_sheet(q, size):
        return None
    A = cv2.contourArea(q.astype(np.float32))
    if A < (2 * POLY_MIN_SIDE) ** 2:
        return None
    corners = ed.find_corners(q, arm=max(5, int(round(2 * scale))), deg=45, dev_max=max(0.9, 0.35 * scale))
    if len(corners) != 2:
        return None
    i0, i1 = sorted(corners)
    n = len(q)
    side1, side2 = q[i0:i1 + 1], np.vstack([q[i1:], q[:i0 + 1]])
    if len(side1) < 8 or len(side2) < 8:
        return None
    P0, P2 = q[i0], q[i1]
    chord = P2 - P0
    L = np.linalg.norm(chord)
    if L < 8:
        return None
    nrm = np.array([-chord[1], chord[0]]) / L
    if L / max(np.ptp(q @ nrm), 1e-6) < LEAF_MIN_ASPECT:
        return None                                       # tip to tip is not clearly longer than wide: a round thing, not a leaf
    C1, M = _bezier_side(side1)
    C2, _ = _bezier_side(side2[::-1])                     # side 2 walked tip0 -> tip1 too
    d1, d2 = (C1 - M) @ nrm, (C2 - M) @ nrm
    tol = POLY_DEV * np.sqrt(A) + 1.5

    def curve(C):
        u = np.linspace(0, 1, max(24, int(L)))[:, None]
        return (1 - u) ** 2 * P0 + 2 * u * (1 - u) * C + u ** 2 * P2

    tries = []
    if d1 * d2 < 0 and 0.75 <= min(abs(d1), abs(d2)) / max(abs(d1), abs(d2)):
        m = (abs(d1) + abs(d2)) / 2
        tries.append((M + np.sign(d1) * m * nrm, M + np.sign(d2) * m * nrm))     # a symmetric petal first
    tries.append((C1, C2))
    for c1, c2 in tries:
        poly = np.vstack([curve(c1), curve(c2)[::-1]])
        if not _strays(q, poly, tol) and _raster_iou(q, poly, iou):
            return poly
    return None


def _resample(pts, step=1.0, closed=False):
    """The polyline resampled at equal arc-length steps (closed: the closing segment included)."""
    q = np.vstack([pts, pts[:1]]) if closed else pts
    d = np.linalg.norm(np.diff(q, axis=0), axis=1)
    q = q[np.r_[True, d > 1e-9]]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    t = np.arange(0, s[-1], step) if closed else np.r_[np.arange(0, s[-1], step), s[-1]]
    return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1)


def fair(curve, sigma, closed):
    """The curve eased by a Gaussian along its own length (sigma px): the slope then changes gradually, a
    small jog or wobble becomes a gentle S, and the curve is not shortened noticeably (sigma is a few px, a
    curve's radius tens). An open stretch keeps its end points and their slope (odd reflection)."""
    c = _resample(curve, 1.0, closed)
    if sigma <= 0 or len(c) < 8:
        return c
    if closed:
        sigma = min(sigma, FAIR_MAX_RADIUS * len(c) / (2 * np.pi))     # a small closed shape is not eaten: shrink ~ sigma^2 / 2R
        return np.stack([ndimage.gaussian_filter1d(c[:, k], sigma, mode='wrap') for k in (0, 1)], 1)
    pad = int(min(len(c) - 2, 4 * sigma + 2))
    e = np.pad(c, ((pad, pad), (0, 0)), mode='reflect', reflect_type='odd')
    o = np.stack([ndimage.gaussian_filter1d(e[:, k], sigma) for k in (0, 1)], 1)[pad:-pad]
    o[0], o[-1] = c[0], c[-1]
    return o


def _turn(p, i, arm):
    """Turning angle (degrees, 0 = straight on) of outline `p` at index i."""
    n = len(p)
    a, b = p[(i - arm) % n] - p[i], p[(i + arm) % n] - p[i]
    c = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return 180 - np.degrees(np.arccos(np.clip(c, -1, 1)))


def _prune_corners(p, corners, arm, min_len):
    """Corners that only make a tiny stretch (a notch, a spur, an ear of the traced edge) are noise: of every
    stretch shorter than min_len the weaker end corner goes, so the curve runs through it."""
    n = len(p)
    cs = sorted(corners)
    while len(cs) >= 2:
        L = []
        for k in range(len(cs)):
            i, j = cs[k], cs[(k + 1) % len(cs)]
            idx = np.arange(i, j + 1 if j > i else j + n + 1) % n
            L.append(np.linalg.norm(np.diff(p[idx], axis=0), axis=1).sum())
        k = int(np.argmin(L))
        if L[k] >= min_len:
            break
        a, b = cs[k], cs[(k + 1) % len(cs)]
        cs.remove(a if _turn(p, a, arm) <= _turn(p, b, arm) else b)
    return cs


def smooth(p, scale, fair_sigma=FAIR_SIGMA):
    """An outline smoothed for a sketch. Real corners (two straight arms) split it into stretches; each stretch
    is a straight line when it never leaves its chord, else a smoothing spline. A round shape with no corner
    is one closed spline. A speck-sized outline (a 2-3 px part) that the spline shrinks to nothing keeps its own
    points: drawn filled, an empty one crashed OpenCV (a slit's tiny piece, kept since lines are rescued)."""
    q = _smooth(p, scale, fair_sigma)
    return q if len(q) >= 3 else p


def _smooth(p, scale, fair_sigma):
    if len(p) < 8:
        return p
    arm = max(5, int(round(2 * scale)))
    corners = ed.find_corners(p, arm=arm, deg=CORNER_DEG, dev_max=max(0.9, 0.35 * scale))
    if PRUNE_SRC > 0 and len(corners) >= 2:
        corners = _prune_corners(p, corners, arm, PRUNE_SRC * scale)
    n = len(p)
    sg = fair_sigma * scale
    if 1 <= len(corners) <= 2 and corners and SMALL_ROUND_PER > 0:
        per = np.linalg.norm(np.roll(p, -1, 0) - p, axis=1).sum()
        if per < SMALL_ROUND_PER * scale and all(_turn(p, i, arm) < 110 for i in corners):
            corners = []                                   # a dot-sized outline with a soft bump: a dot, not a teardrop
    if not corners:
        return fair(_spline(p, scale, True), sg, True)
    out = []
    for a in range(len(corners)):
        s, e = corners[a], corners[(a + 1) % len(corners)]
        idx = np.arange(s, e + 1 if e > s else e + n + 1) % n
        seg = p[idx]
        if len(seg) < 4:
            out.append(seg)
            continue
        c = seg[-1] - seg[0]
        L = np.linalg.norm(c)
        dev = (np.abs((seg[:, 0] - seg[0, 0]) * c[1] - (seg[:, 1] - seg[0, 1]) * c[0]).max() / L) if L > 1e-6 else 9
        if dev <= STRAIGHT_DEV * scale:
            t = np.linspace(0, 1, max(2, int(L)))[:, None]
            out.append(seg[0] + t * c)                  # a straight edge stays dead straight
        else:
            out.append(fair(_spline(seg, scale, False), sg, False))
    return np.vstack(out)


def _owned_runs(owned):
    """Index runs [(array of indices)] of the True stretches of a closed outline's `owned` flags, or None when
    every point is owned (the outline is drawn whole, closed)."""
    n = len(owned)
    if owned.all():
        return None
    if not owned.any():
        return []
    starts = np.flatnonzero(owned & ~np.roll(owned, 1))
    runs = []
    for st in starts:
        j = st
        run = []
        while owned[j % n] and len(run) < n:
            run.append(j % n)
            j += 1
        if len(run) >= 2:
            runs.append(np.array(run))
    return runs


def bold(lab, line_patch, scale, width_px, keep_core=None, k=1, circle=CIRCLE_IOU, polygons=CIRCLE_IOU, motifs=CIRCLE_IOU, stats=None,
         fair_sigma=FAIR_SIGMA, dedup=True):
    """The smooth, bold sketch: every part's smoothed outline stroked `width_px` wide in black on white,
    anti-aliased; a part that is a drawn line is filled black. `keep_core` (H x W bool): pixels painted white
    again afterwards, so a small part (a dot, a thin petal) the bold stroke would fill keeps its white middle.
    `polygons`: the same for a triangle / rectangle / diamond / hexagon (3-5 straight sides), 0 = off.
    `motifs`: the same for an oval (ellipse) and a leaf / petal (two pointed tips, curved sides), 0 = off.
    `dedup`: a boundary shared by two parts is stroked once (by the bigger part), not once per part: two
    separately smoothed copies of one edge drew a doubled line with slivers between them.
    `fair_sigma`: how strongly each smooth stretch is faired, in source px (0 = off).
    `stats`: a dict that gets how many outlines became circles / polygons / stayed smoothed curves.
    `circle`: an outline that is that share (0.9 = 90%) a circle is drawn as a perfect one (0 = off).
    `k`: the canvas is drawn k times bigger, straight from the curves (a blank slate, not an enlarged image):
    every coordinate and the stroke are multiplied by k, so the lines stay as crisp as the first px.
    Returns (image, svg text; the SVG stays in the design's own units)."""
    H, W = lab.shape
    img = np.full((H * k, W * k), 255, np.uint8)
    S = 16                                              # cv2 shift 4: 1/16 px sub-pixel positions
    boxes = ndimage.find_objects(lab)
    area = np.bincount(lab.ravel())
    fills, strokes, svg = [], [], []
    for i, sl in enumerate(boxes, 1):
        if sl is None:
            continue
        y0, x0 = max(sl[0].start - 1, 0), max(sl[1].start - 1, 0)
        y1, x1 = min(sl[0].stop + 1, H), min(sl[1].stop + 1, W)
        for p in _outlines(lab[y0:y1, x0:x1] == i):
            g = p + (x0, y0)
            if dedup and not line_patch[i] and len(g) >= 8:
                t = np.roll(g, -1, 0) - np.roll(g, 1, 0)
                t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
                out = g + 0.6 * np.stack([-t[:, 1], t[:, 0]], 1)               # a step outwards: the part on the other side
                nb = lab[np.clip(np.round(out[:, 1]).astype(int), 0, H - 1), np.clip(np.round(out[:, 0]).astype(int), 0, W - 1)]
            else:
                nb = None
            q = circle_of(g, (H, W), circle)                  # about a circle: a true circle
            kind = 'circles'
            if q is None:
                q, kind = polygon_of(g, (H, W), polygons), 'polygons'      # about a triangle / rectangle...: a clean one
            if q is None:
                q, kind = oval_of(g, (H, W), motifs), 'ovals'               # a plain oval: a true ellipse
            if q is None:
                q, kind = leaf_of(g, (H, W), scale, motifs), 'leaves'       # a leaf / petal: two clean arcs, tip to tip
            if q is None:
                q, kind = smooth(p, scale, fair_sigma) + (x0, y0), 'smooth'
            if stats is not None:
                stats[kind] = stats.get(kind, 0) + 1
            if line_patch[i]:
                fills.append(q)
            elif nb is None:
                strokes.append((q, None))
            else:
                other = nb[cKDTree(g).query(q)[1]]
                owned = ((other == 0) | (other == i) | line_patch[other] | (area[i] > area[other])
                         | ((area[i] == area[other]) & (i < other)))
                strokes.append((q, owned))
    if dedup:
        # A point a bigger part does not draw is dropped ONLY when another loop's drawn line runs right beside
        # it (within half a stroke): that line covers it. Anywhere else (a thin part the other side's lookup
        # hopped over, a snapped shape that moved) it stays, so a boundary is never left undrawn.
        mine = [(q, o) for q, o in strokes if o is not None]
        if mine and any(o.any() for _, o in mine):
            pts = np.vstack([q[o] for q, o in mine if o.any()])
            who = np.concatenate([np.full(int(o.sum()), j) for j, (q, o) in enumerate(mine) if o.any()])
            tree = cKDTree(pts)
            kept = []
            j = 0
            for q, o in strokes:
                if o is None:
                    kept.append((q, None))
                    continue
                if not o.all():
                    d, idx = tree.query(q, distance_upper_bound=0.5 * width_px)
                    near = np.isfinite(d) & (who[np.minimum(idx, len(who) - 1)] != j)
                    o = o | ~near
                kept.append((q, o))
                j += 1
            strokes = kept
    pieces = []                                         # (points, closed): what is stroked, in the sheet's own units
    for q, owned in strokes:
        runs = None if owned is None else _owned_runs(owned)
        if runs is None:
            pieces.append((q, True))
        else:
            pieces += [(q[r], False) for r in runs]
    for q, closed in pieces:
        cv2.polylines(img, [np.round(q * k * S).astype(np.int32)], closed, 0, thickness=int(width_px * k),
                      lineType=cv2.LINE_AA, shift=4)
    if keep_core is not None:
        kc = keep_core if k == 1 else cv2.resize(keep_core.astype(np.uint8), (W * k, H * k),
                                                 interpolation=cv2.INTER_NEAREST) > 0
        img[kc] = 255
    for q in fills:
        cv2.fillPoly(img, [np.round(q * k * S).astype(np.int32)], 0, lineType=cv2.LINE_AA, shift=4)
        cv2.polylines(img, [np.round(q * k * S).astype(np.int32)], True, 0, thickness=2 * k,
                      lineType=cv2.LINE_AA, shift=4)

    def path(q, closed=True):
        a = cv2.approxPolyDP(q.astype(np.float32).reshape(-1, 1, 2), 0.3, closed)
        a = q if a is None or len(a) == 0 else a[:, 0, :]               # a degenerate piece (OpenCV gives None): keep its points
        return 'M' + ' L'.join(f'{x:.1f} {y:.1f}' for x, y in a) + (' Z' if closed else '')

    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    svg.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    svg.append(f'<g fill="none" stroke="black" stroke-width="{width_px}" stroke-linejoin="round" stroke-linecap="round">')
    svg += [f'<path d="{path(q, closed)}"/>' for q, closed in pieces]
    svg.append('</g><g fill="black" stroke="none">')
    svg += [f'<path d="{path(q)}"/>' for q in fills]
    svg.append('</g></svg>')
    return img, '\n'.join(svg)


def colour_fill(rgb, k, sigma=1.2):
    """A flat-colour image (H x W x 3, few colours) drawn k times bigger with smooth edges: each colour's mask is
    blurred a little, scaled up (cubic) and the strongest colour wins each pixel, so edges follow the same kind of
    curve as the outlines instead of the source's stairs. One colour per pixel, no mixed colours."""
    H, W = rgb.shape[:2]
    cols, inv = unique_rgb(rgb, return_inverse=True)
    inv = inv.reshape(H, W)
    best = np.full((H * k, W * k), -1.0, np.float32)
    idx = np.zeros((H * k, W * k), np.uint8)
    for c in range(len(cols)):
        m = cv2.GaussianBlur((inv == c).astype(np.float32), (0, 0), sigma)
        m = cv2.resize(m, (W * k, H * k), interpolation=cv2.INTER_CUBIC) if k > 1 else m
        up = m > best
        best[up] = m[up]
        idx[up] = c
    return cols.astype(np.uint8)[idx]


def colour_sketch(colour, bold_img):
    """The bold lines drawn over the colour fill: where the line is dark the colour goes dark (multiply), so the
    anti-aliased edges blend into the colour instead of leaving a white halo."""
    return (colour.astype(np.uint16) * bold_img[..., None] // 255).astype(np.uint8)
