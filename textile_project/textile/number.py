"""Number a COLOURED design: every patch of one colour gets a number.

`textile number design.png --out o/`: the user's way to plan a sketch from a
finished colour design. The design is read at --size (3535 px wide by
default), cleaned of weave/print grain (a 5 px median at its own scale), cut
to its distinct inks (paint._ref_palette: k-means 16 in CIELAB, shades closer
than dE 30 are one ink, at most --colors), and every connected patch of one
ink (4-connected, as the sketch areas are) is an area. Grain is melted (see
`patches`: specks under --min-area px, and thin or ragged flecks lying inside
one other patch; small round dots are kept). Then:

  NAME_numbers.png   the coloured design, each patch outlined, its number on it
                     (paint's numbers sheet: inside when it fits, else a dot and
                     a blue number beside it; it grows until every number has a place)
  NAME_flat.png      the design as those flat inks (what was numbered)
  NAME_sketch_seal0.png        the colours taken away: a 2 px black line where two patches meet, on
                     white. Its areas, as `textile paint` finds them (seal 0: the name says so), ARE the numbers
  NAME_sketch_numbers.png      that sketch with the numbers
  NAME_colors.csv    Number, HEX, Colour, share %: the same CSV `textile paint`
                     reads, so a sketch drawn to these patches can be painted with it

Nothing is printed from here: no package, no mill file. It is a map to work from.
"""
from __future__ import annotations

import csv
import os

import cv2
import numpy as np
from scipy import ndimage

from . import names as nm
from . import paint as pt
from . import palette as pl
from .fill_method1 import FillError, output_size
from .io_utils import hex_of, read_cv2, safe_name, save_png, to_image


def _label_max(values, lab, n):
    """Per label 1..n, the largest of `values` (>= 0) on it: ndimage.maximum's answer, without its sort of every
    pixel (2 s a call at 3535 px; this 0.04 s)."""
    out = np.zeros(n + 1)
    np.maximum.at(out, lab.ravel(), values.ravel())
    return out[1:]


def _label(index):
    """4-connected patches of one ink, numbered from 1 across all inks."""
    lab = np.zeros(index.shape, np.int32)
    n = 0
    for k in range(int(index.max()) + 1):
        l, m = ndimage.label(index == k)
        lab[l > 0] = l[l > 0] + n
        n += m
    return lab, n


LINE_RESCUE = 10             # a clean design's slant line, 8-connected, at least this many px at its own size...
LINE_RESCUE_GRAIN = 2.0      # ...only in a crisp picture: in a cloth photo (the peacock mockup, grain 3.6) the weave's
                             # threads are such lines too, and kept they came back as grey speckle over the ground
LINE_RESCUE_ONE_INK = 0.75   # a thin line's outside is this much ONE other ink (a slit in a leaf), not two (a fringe)


def _rescued_lines(index, lab, n, cand, min_len, max_half=None):
    """Per patch (n + 1 bools): a melting candidate that is a piece of a LINE, so it stays. A 1-2 px line drawn
    slant is a chain of 4-connected bits that each look like a speck; joined 8-connected, the chain is one line
    of `min_len`+ px. It is kept only when what lies around it is mostly one other ink (LINE_RESCUE_ONE_INK):
    a dark slit in a cream leaf is cream on both sides, while an edge's anti-alias fringe runs between two
    different inks (cream one side, olive the other) and still melts. On the user's rust/purple design the
    leaves' dark slits came out broken into dashes (match 81.1 -> 84.5 with this), with no halo on the bench."""
    keep = np.zeros(n + 1, bool)
    if not cand.any():
        return keep
    H, W = index.shape
    K = int(index.max()) + 1
    eight = np.ones((3, 3), bool)
    for k in np.unique(index[cand[lab]]):
        mk = index == k
        l8, m8 = ndimage.label(mk, structure=eight)
        size = np.bincount(l8.ravel(), minlength=m8 + 1)
        comps, others = [], []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            a = l8[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
            b = index[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
            m = (a > 0) & (b != k)
            comps.append(a[m])
            others.append(b[m].astype(np.int64))
        c, o = np.concatenate(comps), np.concatenate(others)
        if not len(c):
            continue
        touch = np.bincount(c * K + o, minlength=(m8 + 1) * K).reshape(m8 + 1, K)
        line = (size >= min_len) & (touch.max(1) >= LINE_RESCUE_ONE_INK * np.maximum(touch.sum(1), 1))
        if max_half is not None:                   # the operator set the smallest part: only a really thin piece is a line
            half = ndimage.maximum(ndimage.distance_transform_edt(mk), l8, np.arange(m8 + 1))
            line &= np.asarray(half) <= max_half
        line[0] = False
        comp_of = np.zeros(n + 1, np.int64)
        comp_of[lab[mk]] = l8[mk]
        sel = np.flatnonzero(cand & (comp_of > 0))
        keep[sel] |= line[comp_of[sel]]
    return keep


def _to_nearest_colour(index, move, src_lab, pal_lab, rounds=50):
    """`index` with the `move` pixels given away: each takes, among the inks of the kept pixels touching it
    (3 x 3), the one closest to the pixel's OWN colour in the picture (`src_lab`), working inwards a ring at a
    time. A cream / purple anti-alias pixel that read as rust becomes cream or purple by what it looks like,
    not by which side happens to be nearer (a slit's dark bits went to the cream round it that way)."""
    index = index.copy()
    todo = move.copy()
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(rounds):
        if not todo.any():
            break
        best = np.full(index.shape, np.inf, np.float32)
        choice = np.full(index.shape, -1, np.int32)
        for k in range(len(pal_lab)):
            have = ((index == k) & ~todo).astype(np.uint8)
            if not have.any():
                continue
            near = cv2.dilate(have, k3).view(bool) & todo
            if not near.any():
                continue
            d = np.full(index.shape, np.inf, np.float32)
            d[near] = np.linalg.norm(src_lab[near] - pal_lab[k], axis=1)
            better = d < best
            best[better] = d[better]
            choice[better] = k
        got = choice >= 0
        if not got.any():
            break
        index[got] = choice[got]
        todo &= ~got
    if todo.any():                                   # nothing kept anywhere near: the nearest kept pixel's ink
        _, (iy, ix) = ndimage.distance_transform_edt(todo, return_indices=True)
        index[todo] = index[iy[todo], ix[todo]]
    return index


def patches(index, min_area=250, thin=3.5, enclosed_max=3000, rim=0.0, rim_area=None, src_lab=None, pal=None,
            lines=0, strict_lines=False, keep_dots=0.0):
    """(labels H x W int32 from 1, n, the cleaned ink index): the design's areas, its grain melted.

    A patch goes (its pixels to the nearest other patch) when it is under
    `min_area` px, or when it lies wholly inside ONE other patch (a fleck in a
    leaf) and is thin (no point more than `thin` px from its edge: a weave line)
    or small and ragged (under `enclosed_max` px and over 2.5x the area of the
    widest circle in it). A small ROUND patch inside another stays: a flower's
    centre, a dot. Patches that touch two or more others (a leaf, the ground
    and the dark outlines that join it) are never melted, so two leaves parted
    by a dark line stay two areas. On the user's woven jaal photo: 1066 areas,
    most of them weave, -> 426, the yellow flower's dark centre and the pink
    flower's dots and ring kept.

    A clean design passes its picture (`src_lab`, `pal`) and `lines`: then a piece of a slant 1-2 px line is
    not melted (`_rescued_lines`, `lines` px long 8-connected), and a melted pixel takes the touching ink that
    looks most like it (`_to_nearest_colour`) instead of the nearest patch's."""
    pal_lab = nm._lab(pal.astype(np.float64)) if src_lab is not None else None
    for _ in range(5):
        lab, n = _label(index)
        area = np.bincount(lab.ravel(), minlength=n + 1)
        bd = np.zeros(lab.shape, bool)
        dx, dy = lab[:, 1:] != lab[:, :-1], lab[1:, :] != lab[:-1, :]
        bd[:, 1:] |= dx
        bd[:, :-1] |= dx
        bd[1:, :] |= dy
        bd[:-1, :] |= dy
        thick = np.zeros(n + 1)
        thick[1:] = _label_max(ndimage.distance_transform_edt(~bd), lab, n)
        a = np.concatenate([lab[:, 1:][dx], lab[1:, :][dy]])
        b = np.concatenate([lab[:, :-1][dx], lab[:-1, :][dy]])
        key = np.unique(np.r_[a, b].astype(np.int64) * (n + 1) + np.r_[b, a])   # (patch, neighbour) pairs, once
        pairs = np.stack([key // (n + 1), key % (n + 1)], 1)
        neigh = np.bincount(pairs[:, 0], minlength=n + 1)
        one = np.zeros(n + 1, np.int64)
        one[pairs[:, 0]] = pairs[:, 1]                    # for a patch with one neighbour: that neighbour
        edge = np.zeros(n + 1, bool)
        edge[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = True
        inside = (neigh == 1) & ~edge
        ragged = area / (np.pi * np.maximum(thick, 0.5) ** 2) > 2.5
        melt = (area < min_area) | (inside & ((thick < thin) | ((area < enclosed_max) & ragged))) | ((thick < rim) & (area < (np.inf if rim_area is None else rim_area)))
        melt[0] = False
        if keep_dots and pal is not None:         # small-detail rescue: a distinct round dot / flower centre / bud
            dpx = dot_islands(index, pal, keep_dots, lab, n)
            melt &= ~np.bincount(lab[dpx], minlength=n + 1).astype(bool)
        if lines:
            melt &= ~_rescued_lines(index, lab, n, melt, lines, 1.5 if strict_lines else None)
        if not melt.any():
            return lab, n, index
        if src_lab is not None:
            index = _to_nearest_colour(index, melt[lab], src_lab, pal_lab).astype(index.dtype)
            continue
        ink = np.zeros(n + 1, np.int64)
        ink[lab.ravel()] = index.ravel()
        e = melt & inside
        ink[e] = ink[one[e]]                              # a fleck takes the ink of the patch it lies in
        index = ink[lab].astype(index.dtype)
        rest = (melt & ~inside)[lab]
        if rest.any():
            _, (iy, ix) = ndimage.distance_transform_edt(rest, return_indices=True)
            index[rest] = index[iy[rest], ix[rest]]
    lab, n = _label(index)
    return lab, n, index


LINE_WIDTH = 1.25      # a patch at most this many source px from middle to edge (and long) is a drawn line
BLEND_MAX_SHARE = 0.03 # an edge's blend is a sliver of the design (paisley 1.4-2.6%); a busy design's real green,
                       # gold, teal are thin everywhere too but cover 5-14% (the user's peacock lost them all)
BLEND_THIN = 0.6       # an ink with 60%+ of its pixels in edge-thin patches...
BLEND_DE_RIM = 20      # ...or within 20 when 95%+ of it is rim (a black outline is ~60% rim: stays)
BLEND_DE = 10          # ...and within dE 10 of the line between two other inks is their blend, not an ink
# An ink of ANY share (under 10%) is a blend too when it is all rim with no core and an exact mix: 80%+ of it
# in edge-thin patches, at most 3% of it 3+ px inside itself, within dE 2.5 of the line between two inks. Soft
# AI edges make wide blends: the rust/purple design's maroon 3.6%, the ginkgo's silver 5.8%, the black sprigs'
# grey 3.3% passed the 3% cap and came out as screens of their own. The busy peacock jaal's real thin teal / green /
# gold are all-rim too but no exact mix (dE 4.6, 10.6, 13); its cream-grey 14.5% (dE 3.1, 78% rim) is the closest.
BLEND_RIM_ONLY = 0.8
BLEND_CORE = 0.03
BLEND_EXACT_DE = 2.5
BLEND_EXACT_MAX_SHARE = 0.10
WOVEN_GRAIN = 6.0      # median dE a 3x3 median makes: the woven photo 8.2; clean designs 0-0.7, a busy fine one 4.4
CLEAN_SAME_DE = 12     # a clean design's inks that close are one ink (its black outline and navy fill: kept apart)


LINE_AUTO = (0.17, 0.35, 0.5)   # --line-mm auto: tried in turn, the closest paint-back kept
LINE_CORE = 1.5        # a part keeps at least this many px from its middle to the line (a 3 px core)


def _wide_line(lab, bd, half, dt=None):
    """The line where parts meet, `half` px into each side, but never so wide that a
    part loses its core or is cut in two at a narrow neck: a part where the full width
    would do that keeps the plain 1+1 px line (bd) on its side. Moti line big parts
    me, aur bareek daane / patli patti salamat. `dt`: the distance from `bd`, when the
    caller has it already (--line-mm auto draws three widths on one map)."""
    if half <= 1:
        return bd
    if dt is None:
        dt = ndimage.distance_transform_edt(~bd)
    wide = bd | (dt < half)
    n = int(lab.max())
    core = (lab > 0) & ~wide
    # each part must keep exactly one piece of core, at least LINE_CORE deep
    cl, cn = ndimage.label(core)
    pieces = np.unique(lab[core].astype(np.int64) * (cn + 1) + cl[core])     # (part, core piece) pairs, once each
    has = np.bincount(pieces // (cn + 1), minlength=n + 1)
    deep = np.zeros(n + 1)
    deep[1:] = _label_max(dt, lab, n)
    ok = (has == 1) & (deep >= half + LINE_CORE)
    ok[0] = True
    return np.where(ok[lab], wide, bd)


SHADE_THIN = 3.0        # a shaded edge up to 3 source px from middle to edge (the leaves' dark dashes were 2-3)
RIM_SHARE = 0.25       # ...sharing at least a quarter of its border with that part
SHADE_DARK_L = 25.0    # ...and near black
RIM_NEAR_DE = 20       # a thin rim this close in colour to the part it edges is that part's darker edge


def shade_rims(index, pal, thin):
    """A thin patch (no point more than `thin` px from its edge) whose ink is within
    RIM_NEAR_DE of a patch it shares RIM_SHARE+ of its border with is that patch's shaded
    edge, not a part: it takes that patch's ink. Only a DARKER rim, and under dE 20: a lighter
    vein, or a truth floral's olive vein in a green leaf (24.4 apart), is a real part. The paisley's navy leaves had a
    near-black edge (black-navy 18.7) that came out as short thick black dashes all
    over the sketch; its real black outlines round the maroon flowers (black-maroon
    48.7) stay."""
    L = nm._lab(pal.astype(np.float64))
    for _ in range(2):
        lab, n = _label(index)
        bd = np.zeros(lab.shape, bool)
        dx, dy = lab[:, 1:] != lab[:, :-1], lab[1:, :] != lab[:-1, :]
        bd[:, 1:] |= dx
        bd[:, :-1] |= dx
        bd[1:, :] |= dy
        bd[:-1, :] |= dy
        thick = np.zeros(n + 1)
        thick[1:] = _label_max(ndimage.distance_transform_edt(~bd), lab, n)
        a = np.concatenate([lab[:, 1:][dx], lab[1:, :][dy], lab[:, :-1][dx], lab[:-1, :][dy]]).astype(np.int64)
        b = np.concatenate([lab[:, :-1][dx], lab[:-1, :][dy], lab[:, 1:][dx], lab[1:, :][dy]]).astype(np.int64)
        key, cnt = np.unique(a * (n + 1) + b, return_counts=True)
        pa, pb = key // (n + 1), key % (n + 1)
        ink = np.zeros(n + 1, np.int64)
        ink[lab.ravel()] = index.ravel()
        tot = np.bincount(pa, weights=cnt, minlength=n + 1)
        # among the patches sharing at least RIM_SHARE of its border, the one closest in colour
        dE = np.linalg.norm(L[ink[pa]] - L[ink[pb]], axis=1)
        # a shaded edge is darker than what it edges, and near black (L < SHADE_DARK_L): on the degraded truth
        # floral a dark green vein in an olive leaf is as close (18.4) as the paisley's black to its navy (18.7)
        darker = (L[ink[pa], 0] < L[ink[pb], 0]) & (L[ink[pa], 0] < SHADE_DARK_L)
        ok = (cnt >= RIM_SHARE * tot[pa]) & (dE < RIM_NEAR_DE) & darker
        main = np.zeros(n + 1, np.int64)
        best = np.full(n + 1, np.inf)
        for x, y, e in zip(pa[ok], pb[ok], dE[ok]):
            if e < best[x]:
                best[x], main[x] = e, y
        near = main > 0
        move = (thick <= thin) & near & (main > 0)
        move[0] = False
        if not move.any():
            break
        ink[move] = ink[main[move]]
        index = ink[lab].astype(index.dtype)
    return index


RIDGE_MIN = 20.0       # a line pixel differs from its two sides' mean by this much (RGB distance)...
RIDGE_SIDES = 0.5      # ...while its two sides are alike (apart by at most half that, or RIDGE_SIDE_ABS)
RIDGE_SIDE_ABS = 15.0
RIDGE_ANGLE = 0.15     # the line's ink lies on the ray from the sides through the pixel (off it by <= 15%)
RIDGE_ALPHA = 0.2      # and the pixel is at least 20% of the way to it (a fainter one is shading or grain)
RIDGE_OFFSETS = (1, 2) # sides 1 px away (a 1 px line) or 2 px (a line over two pixels)


def line_inks(rgb, pal, index, skip=None):
    """(H, W) int32: for a pixel of a thin line that its nearest ink loses, the ink the line is drawn in; -1
    elsewhere. A 1-2 px line in an AI picture is blurred into the ground: its pixels are a MIX of the line's ink
    and the ground's (in RGB, where the picture was blended), and the nearest ink is often the ground (a cream
    stamen on a black flower vanished) or an in-between ink (a black hairline on cream became sage, then melted).
    A pixel is a line's when, across some direction (4 of them, sides 1 or 2 px off), its two sides are alike and
    it differs from them: then the line's ink is the first ink beyond it on the ray from the sides through it.
    An edge is no line (its two sides differ), nor shading (too faint, RIDGE_ALPHA). A pixel whose own ink is
    already such an ink is left alone: of three near-maroons the first on the ray stole the ginkgo's maroon lines.
    Truth bench (7 designs incl. hairlines): agreement 99.18 -> 99.30 (soft: 98.41 -> 98.76), thin parts kept
    23 -> 36% (16 -> 32%), dark hairlines on cream 67 -> 91% (5 -> 89%). Real pictures: the sprigs' stamens, the
    peacock frame's leaf veins and the ginkgo's maroon lines come back; line art comes out ~8% bolder (jaal)."""
    H, W, _ = rgb.shape
    img = rgb.astype(np.float32)
    inks = pal.astype(np.float32)
    pad = max(RIDGE_OFFSETS)
    P = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    best_t = np.full((H, W), np.inf, np.float32)
    best_k = np.full((H, W), -1, np.int32)
    own_ok = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for s in RIDGE_OFFSETS:
            A = P[pad - s * dy:pad - s * dy + H, pad - s * dx:pad - s * dx + W]
            B = P[pad + s * dy:pad + s * dy + H, pad + s * dx:pad + s * dx + W]
            side = (A + B) / 2
            v = img - side
            nv = np.linalg.norm(v, axis=2)
            cand = (nv >= RIDGE_MIN) & (np.linalg.norm(A - B, axis=2) <= np.maximum(RIDGE_SIDE_ABS, RIDGE_SIDES * nv))
            if not cand.any():
                continue
            ys, xs = np.nonzero(cand)
            sv, vv = side[ys, xs], v[ys, xs]
            vv2 = np.maximum((vv * vv).sum(1), 1e-9)
            for k in range(len(inks)):
                if skip is not None and skip[k]:      # an ink kept only for its dots (a blend grey) draws no line
                    continue
                w = inks[k] - sv
                t = (w * vv).sum(1) / vv2
                ok = (t >= 0.9) & (t <= 1 / RIDGE_ALPHA) & \
                     (np.linalg.norm(w - t[:, None] * vv, axis=1) <= RIDGE_ANGLE * np.linalg.norm(w, axis=1))
                better = ok & (t < best_t[ys, xs])
                best_t[ys[better], xs[better]] = t[better]
                best_k[ys[better], xs[better]] = k
                mine = ok & (index[ys, xs] == k)
                own_ok[ys[mine], xs[mine]] = True
    best_k[own_ok] = -1
    return best_k


def dot_islands(index, pal, min_area, lab=None, n=None, de_min=None):
    """H x W bool, the small-detail rescue's DOTS: parts lying inside ONE other part (DOT_HOST of the px 1-3 px
    round them are that part's; a 1 px anti-alias sliver on the rim does not count), not on the picture's edge, at
    least `min_area` px, round (area <= DOT_RAGGED x their widest circle's) and DOT_DE (dE2000) or more from the
    part round them: a grey dot in a navy block, a light dot on a black flower centre, a teal bud."""
    if lab is None:
        lab, n = _label(index)
    K = len(pal)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    small = area < max(min_area, 1) * 400               # (a big part is no dot; skip the work)
    small[0] = False
    if not small[lab].any():
        return np.zeros(lab.shape, bool)
    bd = np.zeros(lab.shape, bool)
    dx, dy = lab[:, 1:] != lab[:, :-1], lab[1:, :] != lab[:-1, :]
    bd[:, 1:] |= dx
    bd[:, :-1] |= dx
    bd[1:, :] |= dy
    bd[:-1, :] |= dy
    thick = np.zeros(n + 1)
    thick[1:] = _label_max(ndimage.distance_transform_edt(~bd), lab, n)
    # an anti-alias sliver round a dot (1 px wide, small: the melt's `rim` pass takes it anyway) is no neighbour
    sliver = (thick <= RIM) & (area < 30)
    sliver[0] = False
    cnt = np.zeros((n + 1) * K, np.int64)
    for d in (1, 2, 3):                                 # which inks lie 1-3 px round each part
        for a, b, ia, ib in ((lab[:, d:], lab[:, :-d], index[:, d:], index[:, :-d]),
                             (lab[d:, :], lab[:-d, :], index[d:, :], index[:-d, :])):
            m = (a != b) & (small[a] | small[b])
            ma, mb = m & ~sliver[b], m & ~sliver[a]
            cnt += np.bincount(a[ma].astype(np.int64) * K + ib[ma], minlength=(n + 1) * K)
            cnt += np.bincount(b[mb].astype(np.int64) * K + ia[mb], minlength=(n + 1) * K)
    cnt = cnt.reshape(n + 1, K)
    host = cnt.argmax(1)
    share = cnt.max(1) / np.maximum(cnt.sum(1), 1)
    edge = np.zeros(n + 1, bool)
    edge[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = True
    ink = np.zeros(n + 1, np.int64)
    ink[lab.ravel()] = index.ravel()
    PL = nm._lab(pal.astype(np.float64))
    de = nm.delta_e2000(PL[ink], PL[host])
    dot = (small & (share >= DOT_HOST) & ~edge & (area >= min_area) & (de >= (DOT_DE if de_min is None else de_min))
           & (area / (np.pi * (thick + 1) ** 2) <= DOT_RAGGED))   # (thick: px from the middle to the rim px)
    return dot[lab]


RARE_DE = 20.0         # small-detail rescue: picture px this far (dE2000) from the ink they were given...
RARE_MIN_PX = 3        # ...at least this many touching (8-connected) are a colour the palette missed (a few teal
RARE_GROUP_DE = 10.0   # dots never reach k-means); groups this close are one colour...
RARE_BLEND_DE = 6.0    # ...and one within this (CIELAB) of the line between two inks round it is their edge blend
RARE_BLEND_RGB = 14.0  # (or within this in RGB: a red star's soft rim on cream is a mix in RGB, curved in CIELAB)


def _between(c, PL, c_rgb=None, P_rgb=None):
    """Is colour `c` (CIELAB) a mix of two of the inks `PL`: within RARE_BLEND_DE of the line joining them in
    CIELAB, or (given RGB) within RARE_BLEND_RGB of it in RGB, where pictures blend their edges."""
    if c_rgb is not None:
        for i in range(len(P_rgb)):
            for j in range(i + 1, len(P_rgb)):
                d = P_rgb[j] - P_rgb[i]
                t = float(np.dot(c_rgb - P_rgb[i], d) / max(np.dot(d, d), 1e-9))
                if 0.0 < t < 1.0 and np.linalg.norm(P_rgb[i] + t * d - c_rgb) < RARE_BLEND_RGB:
                    return True
    for i in range(len(PL)):
        for j in range(i + 1, len(PL)):
            d = PL[j] - PL[i]
            t = float(np.dot(c - PL[i], d) / max(np.dot(d, d), 1e-9))
            if 0.0 < t < 1.0 and np.linalg.norm(PL[i] + t * d - c) < RARE_BLEND_DE:
                return True
    return False


def rare_rescue(rgb, pal, index, colours, src_lab=None):
    """(pal, index, flagged): colours the palette missed get an ink of their own while --colors allows (a yellow
    flower centre of 8 px, a rare teal dot); the rest come back in `flagged` [(rgb, px)] to be reported, never
    silently lost. A group whose colour lies between two inks round it is an edge blend and is skipped."""
    from . import palette as pl
    L = src_lab if src_lab is not None else nm._lab(rgb.reshape(-1, 3).astype(np.float64)).reshape(rgb.shape).astype(np.float32)
    PL = nm._lab(pal.astype(np.float64))
    far = nm.delta_e2000(L.astype(np.float64), PL[index]) >= RARE_DE
    if far.sum() < RARE_MIN_PX:
        return pal, index, []
    k3 = np.ones((3, 3), np.uint8)
    rng = sum(cv2.dilate(L[..., c], k3) - cv2.erode(L[..., c], k3) for c in range(3))
    lab, n = ndimage.label(far, structure=np.ones((3, 3)))
    area = np.bincount(lab.ravel(), minlength=n + 1)
    groups = []                                       # [lab colour, px, labels]
    for i in np.flatnonzero(area >= RARE_MIN_PX):
        if i == 0:
            continue
        sel = lab == i
        r = rng[sel]
        c = L[sel][r <= np.percentile(r, 30)].mean(0).astype(np.float64)   # its most solid px speak for it
        for gr in groups:
            if float(nm.delta_e2000(gr[0], c)) < RARE_GROUP_DE:
                gr[1] += int(area[i]); gr[2].append(i)
                break
        else:
            groups.append([c, int(area[i]), [i]])
    groups.sort(key=lambda gr: -gr[1])
    added, flagged = [], []
    k5 = np.ones((5, 5), np.uint8)
    for c, npx, ids in groups:
        m = np.isin(lab, ids)
        ring = (cv2.dilate(m.astype(np.uint8), k5) > 0) & ~m
        rc = np.bincount(index[ring], minlength=len(pal)).astype(float)
        around = np.flatnonzero(rc >= 0.1 * max(rc.sum(), 1))
        r = rng[m]
        col = np.clip(np.rint(rgb[m][r <= np.percentile(r, 30)].mean(0)), 0, 255).astype(np.uint8)
        P_rgb = np.vstack([pal, np.array(added, np.uint8).reshape(-1, 3)]).astype(np.float64)
        if (_between(c, PL[around], col.astype(np.float64), P_rgb[around])
                or min(float(nm.delta_e2000(c, q)) for q in PL) < RARE_DE):
            continue
        if len(pal) + len(added) < colours:
            added.append(col)
            PL = np.vstack([PL, nm._lab(col.astype(np.float64))[None]])
        else:
            flagged.append((tuple(int(v) for v in col), npx))
    if added:
        pal = np.vstack([pal, np.array(added, np.uint8)])
        index = pl.map_to_palette(rgb, pal)
    return pal, index, flagged


DOT_PURE_DE = 10.0     # a kept dot of a blend ink looks like that ink (its px average this close, dE2000): a grey
                       # dot is that grey; a scrap of a soft black rim on cream is any grey from black to cream


def _true_dots(mask, index, src_lab, pal_lab):
    """`mask` limited to its 8-connected parts whose picture colour (mean) is within DOT_PURE_DE of their ink."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    cnt = np.maximum(np.bincount(lab.ravel(), minlength=n + 1), 1)
    mean = np.stack([np.bincount(lab.ravel(), weights=src_lab[..., c].ravel(), minlength=n + 1) / cnt
                     for c in range(3)], 1)
    ink = np.zeros(n + 1, np.int64)
    ink[lab.ravel()] = index.ravel()
    ok = nm.delta_e2000(mean, pal_lab[ink]) < DOT_PURE_DE
    ok[0] = False
    return ok[lab]


def blend_inks(index, pal, rim, islands=False):
    """Inks that are only the blend along edges: most of their pixels (BLEND_THIN) lie
    in patches no thicker than `rim` px, AND their colour lies between two other
    inks (within BLEND_DE of the line joining them, not at its ends). A thin black
    outline is thin but no mix of two others, so it stays; the paisley's silver
    (black over cream) and coffee (black over maroon) go."""
    K = len(pal)
    lab, n = _label(index)
    bd = np.zeros(lab.shape, bool)
    dx, dy = lab[:, 1:] != lab[:, :-1], lab[1:, :] != lab[:-1, :]
    bd[:, 1:] |= dx
    bd[:, :-1] |= dx
    bd[1:, :] |= dy
    bd[:-1, :] |= dy
    thick = np.zeros(n + 1)
    thick[1:] = _label_max(ndimage.distance_transform_edt(~bd), lab, n)
    thin_px = np.bincount(index.ravel(), weights=(thick[lab] <= rim).ravel(), minlength=K)
    all_px = np.maximum(np.bincount(index.ravel(), minlength=K), 1)
    L = nm._lab(pal.astype(np.float64))
    out = np.zeros(K, bool)
    share = all_px / max(index.size, 1)
    # a small ink kept for its solid dots (under MIN_INK_SHARE) mixes with nothing: the truth floral's maroon
    # centres made its thin green stems look 'between maroon and olive' and they went. (Asking that a blend touch
    # both its inks failed the other way: real blends often touch one ink and another blend, as in the paisley.)
    parent = share >= MIN_INK_SHARE
    isl = np.zeros(K)
    if islands:                                    # small-detail rescue: px in parts that touch ONE other part only
        a = np.concatenate([lab[:, 1:][dx], lab[1:, :][dy]])
        b = np.concatenate([lab[:, :-1][dx], lab[:-1, :][dy]])
        key = np.unique(np.r_[a, b].astype(np.int64) * (n + 1) + np.r_[b, a])
        neigh = np.bincount(key // (n + 1), minlength=n + 1)
        edge = np.zeros(n + 1, bool)
        edge[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = True
        one = (neigh == 1) & ~edge
        isl = np.bincount(index.ravel(), weights=one[lab].ravel(), minlength=K) / all_px
    for k in range(K):
        frac = thin_px[k] / all_px[k]
        if frac < BLEND_THIN or isl[k] >= ISLAND_SHARE:   # a grey dot in a navy block is a dot, not navy+white's blend
            continue
        if share[k] >= BLEND_MAX_SHARE:
            mk = index == k
            if (frac < BLEND_RIM_ONLY or share[k] >= BLEND_EXACT_MAX_SHARE
                    or (ndimage.distance_transform_edt(mk)[mk] >= 3).mean() > BLEND_CORE):
                continue
            de = BLEND_EXACT_DE                              # a big all-rim ink: only an exact mix is a blend
        else:
            de = BLEND_DE_RIM if frac >= 0.95 else BLEND_DE  # all rim: a looser mix still counts (paisley coffee 14)
        for i in range(K):
            for j in range(i + 1, K):
                if k in (i, j) or not (parent[i] and parent[j]):
                    continue
                d = L[j] - L[i]
                t = float(np.dot(L[k] - L[i], d) / max(np.dot(d, d), 1e-9))
                gap = np.linalg.norm(L[i] + t * d - L[k])
                # the loose all-rim match (dE 10-20) only well inside the pair: near an end it is a shade of that
                # end (the truth floral's thin dark-green stems, t 0.90 from rose to olive at dE 18.6, went)
                if 0.1 < t < 0.9 and gap < de and (gap < BLEND_DE or 0.2 < t < 0.8):
                    out[k] = True
    return out


def merge_similar_inks(index, pal):
    """An ink that is small (under learn.SIMILAR_SHARE % of the design) and within learn.SIMILAR_DE of a bigger
    one is the same colour on cloth: its pixels go to that bigger ink (the closest such). Returns (index, how many
    inks were merged, [(from hex, to hex)]). Same rule `learn.advise` names as `similar_inks`."""
    from . import learn
    share = np.bincount(index.ravel(), minlength=len(pal)) / index.size * 100
    lab = nm._lab(pal.astype(np.float64))
    to = np.arange(len(pal))
    moved = []
    for a in np.argsort(share):
        if share[a] <= 0 or share[a] >= learn.SIMILAR_SHARE:
            continue
        best, bd = None, learn.SIMILAR_DE
        for b in range(len(pal)):
            if b != a and share[b] > share[a] and to[b] == b:
                d = float(np.linalg.norm(lab[a] - lab[b]))
                if d < bd:
                    best, bd = b, d
        if best is not None:
            to[a] = best
            share[best] += share[a]
            share[a] = 0
            moved.append((hex_of(pal[a]), hex_of(pal[best])))
    return to[index].astype(index.dtype), len(moved), moved


SHADE_DE = 10.0        # merge-shades (default on): an ink this close (dE2000) to a bigger one...
SHADE_TOUCH = 0.30     # ...whose border lies this much along it is that ink's shading (the AI's darker red at a
                       # petal's base, a gold leaf's orange edge): one screen. Two motifs that merely sit near each
                       # other touch along little of their border (06's olive leaves and maroon dots: 1-29%)
SHADE_LINE_MM = 0.35   # an ink with under SHADE_CORE of its pixels this far inside it is a drawn line (a cream
SHADE_CORE = 0.2       # lattice on a beige ground, dE 8.8, all border on the ground), never shading: kept


def merge_shade_inks(index, pal, px_mm):
    """Merge near-identical shades (closer than SHADE_DE, SHADE_TOUCH of its border
    along it, not line-shaped) into the bigger ink: one screen per colour, as a printer separates a watercolour
    design, instead of hard-edged blotches where the picture's shading crossed an ink boundary. Smallest first,
    so a shade of a shade ends in the main ink; the main ink keeps its colour. On by default for `textile number`
    (opt out with --no-merge-shades): a real two-tone motif may need the opt-out. Returns (index, [(from hex, to hex)])."""
    K = len(pal)
    share = np.bincount(index.ravel(), minlength=K).astype(np.int64)
    T = np.zeros((K, K), np.int64)                              # 4-neighbour contacts between inks
    for a, b in ((index[:, 1:], index[:, :-1]), (index[1:, :], index[:-1, :])):
        m = a != b
        key = a[m].astype(np.int64) * K + b[m]
        c = np.bincount(key, minlength=K * K).reshape(K, K)
        T += c + c.T
    lab = nm._lab(pal.astype(np.float64))
    to = np.arange(K)
    moved = []
    for s in np.argsort(share, kind='stable'):
        if share[s] == 0 or to[s] != s:
            continue
        border = T[s].sum()
        best, bd = None, SHADE_DE
        for b in range(K):
            if b == s or to[b] != b or share[b] <= share[s] or border == 0 or T[s, b] < SHADE_TOUCH * border:
                continue
            d = float(nm.delta_e2000(lab[s:s + 1], lab[b:b + 1])[0])
            if d < bd:
                best, bd = b, d
        if best is None:
            continue
        mask = (to[index] == s).astype(np.uint8)
        inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)[mask > 0] >= SHADE_LINE_MM * px_mm
        if inside.mean() < SHADE_CORE:
            continue
        to[to == s] = best
        share[best] += share[s]
        share[s] = 0
        T[best] += T[s]
        T[:, best] += T[:, s]
        T[best, best] = 0
        T[s] = 0
        T[:, s] = 0
        moved.append((hex_of(pal[s]), hex_of(pal[best])))
    return to[index].astype(index.dtype), moved


SAME_INK_DE = 3.0      # two inks closer than this (dE2000, a just-noticeable difference) are one ink: k-means's
                       # complete-linkage groups left twin navies (1.8 apart) and a tile's twin black-blues (3.2)
MIN_INK_SHARE = 0.003  # an ink under 0.3% of the design is folded into the nearest...
SMALL_INK_CORE = 0.5   # ...unless half its pixels lie 2+ px inside it (solid dots / shapes, not an edge's rim)
SMALL_INK_DE = 20      # and it is far (dE2000) from every other ink: the truth floral's maroon centres, 0.28%
SMALL_KEEP_DE = 8      # small-detail rescue: a small ink this far from every other ink (not a twin / shade)...
SMALL_KEEP_SOLID = 0.25  # ...with this share of its pixels PURE (within SMALL_PURE_DE of the ink: a petal's, a dot's
SMALL_PURE_DE = 5.0    # own flat colour; an edge's anti-alias blend runs through every mix, few px on any one) keeps
                       # its own screen
DOT_MIN_MM2 = 0.03     # small-detail rescue: a round island of its own ink inside ONE other part is kept when at least
DOT_MIN_SRC_PX = 3     # this big printed (0.03 mm2 ~ a 0.2 mm dot at --detail normal; scaled with the --detail /
                       # --min-area chosen) and this many picture px, though under the min-area
DOT_DE = 12            # ...and this far (dE2000) from the part around it (a darker fleck of the same red still melts)
DOT_HOST = 0.6         # ...lying inside it: this share of the px 1-3 px round it are that one part's
DOT_RAGGED = 2.5       # ...and round: area at most this x its widest circle's (a JPEG speck is ragged)
ISLAND_SHARE = 0.5     # a 'blend' ink with half its pixels in islands inside ONE part is dots, not an edge's blend
SOLID_RANGE = 30       # a pixel whose 3x3 neighbourhood spans under this (summed CIELAB L+a+b range) is solid


def _pure_share(rgb, mask, ink_lab):
    """Share of the `mask` pixels whose own colour is within SMALL_PURE_DE (dE2000) of the ink (`ink_lab`)."""
    px = rgb[mask]
    if not len(px):
        return 0.0
    return float((nm.delta_e2000(nm._lab(px.astype(np.float64)), ink_lab) < SMALL_PURE_DE).mean())


def solid_pixels(rgb, cap=400000):
    """The design's solid pixels (inside a part, not on an edge), as an N x 1 x 3 image for the palette: a
    busy design's pixels are mostly edge mixes (the user's peacock jaal, motifs 5-15 px with dark outlines) and
    k-means on all of them found muddy greys instead of its green, rose and gold. All pixels when too few
    (under 3%) are solid."""
    L = nm._lab(rgb.reshape(-1, 3).astype(np.float64)).reshape(rgb.shape).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    rng = sum(cv2.dilate(L[..., c], k) - cv2.erode(L[..., c], k) for c in range(3))
    m = rng < SOLID_RANGE
    sel = rgb[m] if m.mean() >= 0.03 else rgb.reshape(-1, 3)
    if len(sel) > cap:
        sel = sel[np.random.default_rng(0).choice(len(sel), cap, replace=False)]
    return sel.reshape(-1, 1, 3)


def grain(rgb):
    """How grainy a picture is: the median CIELAB change a 3x3 median makes, at <= 1254 px."""
    if rgb.shape[1] > 1600:
        rgb = cv2.resize(rgb, (1254, max(1, round(1254 * rgb.shape[0] / rgb.shape[1]))), interpolation=cv2.INTER_AREA)
    a = nm._lab(rgb.reshape(-1, 3).astype(np.float64))
    b = nm._lab(cv2.medianBlur(rgb, 3).reshape(-1, 3).astype(np.float64))
    return float(np.median(np.linalg.norm(a - b, axis=1)))


DETAIL_MM2 = {'kam': 3.0, 'normal': 0.4, 'zyada': 0.1}   # a part smaller than this (mm2 at 300 DPI) is melted
WOVEN_DETAIL = 4.0     # a woven photo's grain is coarser: its smallest part is 4x that (normal: 1.6 mm2 ~ 220 px)
RIM = 1.01             # at the design's own size an edge's blend is 1 px wide (its middle 1 px from the edge)
SEP_GREY = 60          # a separator (two colours meeting, no outline) is drawn this grey; an outline 0 (black)


def _merge_twins(rgb, pal, index):
    """Inks closer than SAME_INK_DE are merged (closest pair first) into their pixels' mean; the design is mapped
    again. Two screens of one colour help nobody."""
    merged = False
    while len(pal) > 1:
        L = nm._lab(pal.astype(np.float64))
        d = nm.delta_e2000(L[:, None], L[None])
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] >= SAME_INK_DE:
            break
        cnt = np.bincount(index.ravel(), minlength=len(pal)).astype(float)
        w = cnt[[i, j]] / max(cnt[[i, j]].sum(), 1)
        pal = pal.copy()
        pal[i] = np.clip(np.rint(w[0] * pal[i] + w[1] * pal[j]), 0, 255)
        pal = np.delete(pal, j, 0)
        index = np.where(index == j, i, index)
        index = np.where(index > j, index - 1, index).astype(index.dtype)
        merged = True
    if merged:
        index = pl.map_to_palette(rgb, pal)
    return pal, index


SMALL_PX = 20          # an object under this many picture px across (the square root of its area)...
SMALL_THIN = 2.5       # ...or this thin (px from its middle to its edge: a stem, a fine line) is a SMALL object:
SMALL_PAD = 2          # it and this many px around it are drawn from the picture's own anti-aliasing (`draw_small`)
SMALL_UNMIX_RMAX = 30.0    # a px that no 'ground + its own ink' mix explains within this (RGB) keeps the plain shares
SMALL_SHADE_DE = 15.0      # a small part this close (dE2000) to the ink around it is that ink's shading (a darker red
                           # patch in a red petal, 7.5; in an orange-red one, 12.1), not an object: edges draw it. A real
                           # small object stands out more (a black dot on red, gold on maroon: 30+); one that does
                           # not is only drawn as before
SMALL_SPECK = 2.0          # a bit drawn smaller than this many picture px is no detail the picture can carry (AI
                           # texture: dark streaks in a sage band): there the edges' own drawing stands


def small_zone(index, pal, max_px=SMALL_PX, thin=SMALL_THIN, pad=SMALL_PAD):
    """The picture's px of small objects (one-ink parts, 8-connected, under `max_px` across or at most `thin` from
    middle to edge), grown by `pad` px so their outline is drawn as one curve with what is around it. A part in a
    shade of the ink most around it (dE2000 < SMALL_SHADE_DE: the darker red patch in a red petal) is the picture's
    shading, not an object: the edges' smooth outline stays (drawn from the anti-aliasing, the shading's own grain
    made its edge ragged)."""
    K = len(pal)
    L = nm._lab(pal.astype(np.float64))
    near = nm.delta_e2000(L[:, None], L[None]) < SMALL_SHADE_DE
    z = np.zeros(index.shape, bool)
    for k in np.unique(index):
        lab, n = ndimage.label(index == k, structure=np.ones((3, 3)))
        if n == 0:
            continue
        area = np.bincount(lab.ravel(), minlength=n + 1)
        dt = cv2.distanceTransform(np.pad((index == k).astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        deep = np.r_[0.0, _label_max(dt, lab, n)]
        small = (np.sqrt(area) < max_px) | (deep <= thin)
        small[0] = False
        if near[k].any() and small.any():      # what is around each part: its 1 px ring's commonest ink
            ring = ndimage.grey_dilation(lab, size=(3, 3))
            m = (ring > 0) & (index != k)
            cnt = np.bincount(ring[m].astype(np.int64) * K + index[m], minlength=(n + 1) * K).reshape(n + 1, K)
            around = cnt.argmax(1)
            small &= ~((cnt.max(1) > 0) & near[k][around])
        z |= small[lab]
    if pad:
        z = cv2.dilate(z.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0
    return z


def ink_coverage(rgb, index, pal):
    """(K, h, w): how much of each picture px each ink covers. The picture was anti-aliased in RGB, so a px on an
    edge is (1 - t) a + t b of the two inks it lies between, t its colour's projection from a to b: where the edge
    runs INSIDE the px, which a hard label per px throws away. a = its own ink, b = the ink most beside it (3 x 3);
    and a small object's px is read against its ground (the ink most around it, 9 x 9): a blurred dot's middle is
    0.8 gold, not 1. Each px keeps the ink its label gave it (the palette's own, tested choice): only HOW MUCH of
    the px it covers is read. Re-reading a px as another ink's mix (a tan blur on beige as gold) found more tiny dots
    on synthetic designs, but tore a real sage band (between navy and cream in colour) on a real one and dropped a
    thin sage-only ink altogether: colour alone cannot tell a mix from a real in-between ink."""
    h, w = index.shape
    K = len(pal)
    P = pal.astype(np.float32)
    X = rgb.astype(np.float32)
    pad = np.pad(index, 1, mode='edge')
    cnt = np.zeros((K, h, w), np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                nbr = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                for k in range(K):
                    cnt[k] += (nbr == k) & (index != k)
    other = cnt.argmax(0)
    has = cnt.max(0) > 0
    a = P[index]
    d = P[other] - a
    t = np.clip(np.where(has, ((X - a) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-6), 0), 0, 1).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    cov = np.zeros((K, h, w), np.float32)
    cov[index, yy, xx] = 1 - t
    cov[other, yy, xx] += np.where(has, t, 0)
    # against the ground: the ink most around (9 x 9) -> the px's own ink
    box = np.stack([cv2.boxFilter((index == k).astype(np.float32), -1, (9, 9), normalize=False) for k in range(K)])
    bg = box.argmax(0)
    g = P[bg]
    dk = a - g
    tg = np.clip(((X - g) * dk).sum(-1) / np.maximum((dk * dk).sum(-1), 1e-6), 0, 1).astype(np.float32)
    rg = np.linalg.norm(X - (g + tg[..., None] * dk), axis=-1)
    use = (index != bg) & (rg <= SMALL_UNMIX_RMAX)
    gc = np.zeros_like(cov)
    gc[bg, yy, xx] = 1 - tg
    gc[index, yy, xx] += tg
    return cov, gc, use


def _sample(field, ys, xs, s):
    """Cubic samples of a picture-size field at drawing px (ys, xs) (cv2.resize's own centre mapping), in pieces
    (cv2.remap takes at most 32767 px a row)."""
    n = len(xs)
    cols = 8192
    rows = max(1, -(-n // cols))
    fill = rows * cols - n
    mx = np.concatenate([((xs + 0.5) / s - 0.5).astype(np.float32), np.zeros(fill, np.float32)]).reshape(rows, cols)
    my = np.concatenate([((ys + 0.5) / s - 0.5).astype(np.float32), np.zeros(fill, np.float32)]).reshape(rows, cols)
    out = np.empty((rows, cols), np.float32)
    for r0 in range(0, rows, 2048):
        out[r0:r0 + 2048] = cv2.remap(field, mx[r0:r0 + 2048], my[r0:r0 + 2048], cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
    return out.ravel()[:n]


def draw_small(rgb, index, pal, big, max_px=SMALL_PX):
    """Small objects redrawn, big ones left exactly as `big` (the picture's labels `index` drawn at big's size by
    `edges`): in the small zone (`small_zone`) each ink's share of every picture px (`ink_coverage`) is
    interpolated (cubic) to the drawing's grid and the ink with the most wins, so the edge falls where the
    picture's anti-aliasing puts it. A 6 px dot keeps its size and roundness, a star its points, a stem stays one
    line (the edges' outline smoothing rounded them into blobs and dashes). Bits under SMALL_SPECK picture px, and
    every edge between two shades of one colour (SMALL_SHADE_DE), keep the edges' drawing: AI texture and shading,
    whose grain made them ragged here. Same inks as `big`: only shapes change. Measured on dense truth designs
    (~7600 objects of every size each, degraded like an AI picture), object IoU by size: drawn 2.4x, 8-12 px across
    0.78 -> 0.88, 12-20 px 0.88 -> 0.94; drawn 4.9x (a mill sheet from a ChatGPT picture) 0.81 -> 0.91, 0.88 -> 0.96.
    Truth bench v2 a little better in every setting; the user's 14 designs all closer to their picture (+0.1 to
    +1.5), no ink lost."""
    H, W = big.shape
    h, w = index.shape
    s = W / w
    zone = small_zone(index, pal, max_px)
    if not zone.any():
        return big
    cov, gc, use = ink_coverage(rgb, index, pal)
    sel = zone & use
    cov[:, sel] = gc[:, sel]
    Z = cv2.resize(zone.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.nonzero(Z)
    best = np.full(len(ys), -1.0, np.float32)
    arg = np.zeros(len(ys), np.uint8)
    for k in range(len(pal)):
        v = _sample(cov[k], ys, xs, s)
        m = v > best
        best[m] = v[m]
        arg[m] = k
    out = big.copy()
    # between two shades of one colour (a red petal and its darker red) the edges' smooth line stands: that edge
    # is the picture's shading, and its grain made it ragged; the new line is for edges between real colours
    Lb = nm._lab(pal.astype(np.float64))
    shade = nm.delta_e2000(Lb[:, None], Lb[None]) < SMALL_SHADE_DE
    old = big[ys, xs]
    out[ys, xs] = np.where(shade[arg, old], old, arg)
    if SMALL_SPECK:                            # a bit under SMALL_SPECK picture px: the edges' drawing there
        tiny = SMALL_SPECK * s * s
        for k in np.unique(arg):
            lab, n = ndimage.label(out == k, structure=np.ones((3, 3)))
            area = np.bincount(lab.ravel(), minlength=n + 1)
            bad = area < tiny
            bad[0] = False
            if bad.any():
                m = bad[lab] & Z
                out[m] = big[m]
    return out


RESCUE_REPORT = {}     # the last flat_index's small-detail notes: 'rare_flagged' [(rgb, px)] colours with no ink left


def flat_index(rgb, W, H, colours, keep_px, woven, smooth=True, grain_level=0.0, small_px=SMALL_PX, strict_lines=False,
               rescue=True):
    """(palette K x 3 uint8, index H x W uint8): the design as flat inks at W x H, its grain / edge blends
    melted. `keep_px` = the smallest part, in px at W x H; `grain_level` = `grain(rgb)`. This is all of `number`'s
    colour work (the sketch, numbers and plates are drawn from it), split out so it can be measured on its own.
    `rescue` (default on; --no-detail-rescue) keeps a clean design's small details: a small distinct SOLID ink
    (petals, a rare teal dot) its own screen, a grey dot inside a dark part (no edge blend), and a round distinct
    dot / centre under the --detail min-area (down to DOT_MIN_MM2 printed). Off = the earlier behaviour."""
    from . import edges as ed
    from . import palette as pl
    scale = W / rgb.shape[1]
    if woven:
        # a woven photo (as tuned on the user's jaal): its grain cleaned at its own scale, then everything at
        # --size, flecks melted there (on the photo's own grid they came back as 2042 parts, not 425)
        big = cv2.resize(cv2.medianBlur(rgb, 5), (W, H), interpolation=cv2.INTER_AREA if rgb.shape[1] > W
                         else cv2.INTER_LANCZOS4)
        small = cv2.resize(big, (min(W, 1000), max(1, round(min(W, 1000) * H / W))), interpolation=cv2.INTER_AREA)
        pal = pt._ref_palette(small, colours)
        index = pl.map_to_palette(big, pal)
        _, _, index = patches(index, max(20.0, keep_px), rim=0.8 * scale)
        if smooth:                                 # its ragged woven edges smoothed along themselves
            index, _ = ed.clean(index.astype(np.uint8), 1, 3)
    else:
        # a clean design: inks and parts read at its own size, as it is (a median ate its dots; an area average
        # invented blend shades), then drawn at --size with every outline smoothed (textile edges)
        pal = pt._ref_palette(solid_pixels(rgb), colours, CLEAN_SAME_DE)
        index = pl.map_to_palette(rgb, pal)
        pal, index = _merge_twins(rgb, pal, index)
        # an ink barely used (under MIN_INK_SHARE) is no screen of its own: its pixels go to the nearest ink
        share = np.bincount(index.ravel(), minlength=len(pal)) / index.size
        keep = share >= MIN_INK_SHARE
        for k in np.flatnonzero(~keep & (share > 0)):           # a small but real ink of its own
            mk = index == k
            L = nm._lab(pal.astype(np.float64))
            far = min(float(nm.delta_e2000(L[k], L[j])) for j in range(len(pal)) if j != k) >= SMALL_INK_DE
            if far and (ndimage.distance_transform_edt(mk)[mk] >= 2).mean() >= SMALL_INK_CORE:
                keep[k] = True
            elif rescue and not keep[k]:
                near = min(float(nm.delta_e2000(L[k], L[j])) for j in range(len(pal)) if j != k)
                if near >= SMALL_KEEP_DE and _pure_share(rgb, mk, L[k]) >= SMALL_KEEP_SOLID:
                    keep[k] = True
        if not keep.all() and keep.any():
            pal = pal[keep]
            index = pl.map_to_palette(rgb, pal)
        src_lab = nm._lab(rgb.reshape(-1, 3).astype(np.float64)).reshape(rgb.shape).astype(np.float32)
        RESCUE_REPORT.clear()
        if rescue:                                 # a colour k-means never saw (a few px of yellow centres)
            pal, index, RESCUE_REPORT['rare_flagged'] = rare_rescue(rgb, pal, index, colours, src_lab)
        # an edge's anti-alias blend is no ink: its pixels go to the touching ink they look most like
        pal_lab = nm._lab(pal.astype(np.float64))
        dot_only = np.zeros(len(pal), bool)
        for _ in range(3):                         # again: a chain of blends (cream, 2 shades, purple) goes link by link
            blend = blend_inks(index, pal, RIM, islands=rescue)
            if not blend.any() or blend.all():
                break
            move = blend[index]
            used = ~blend                          # and it leaves the palette (no empty ink, no empty screen)
            if rescue:                             # ...but its dots inside one other part stay (a grey ink that is
                stay = move & dot_islands(index, pal, DOT_MIN_SRC_PX, de_min=RARE_DE)  # both a blend and grey dots)
                stay &= _true_dots(stay, index, src_lab, pal_lab)
                move &= ~stay
                kept = np.bincount(index[stay], minlength=len(pal)).astype(bool)
                dot_only |= kept & blend
                used |= kept
            index = _to_nearest_colour(index, move, src_lab, pal_lab).astype(index.dtype)
            pal = pal[used]
            pal_lab = pal_lab[used]
            dot_only = dot_only[used]
            index = (np.cumsum(used) - 1)[index].astype(index.dtype)
        if grain_level < LINE_RESCUE_GRAIN:       # a thin line blurred into the ground gets its own ink back
            lk = line_inks(rgb, pal, index, dot_only if rescue else None)
            index = np.where(lk >= 0, lk, index).astype(index.dtype)
        dots = max(DOT_MIN_SRC_PX, DOT_MIN_MM2 * keep_px / max(DETAIL_MM2['normal'], 1e-9) / scale ** 2) if rescue else 0.0
        keep_px = max(2.0, keep_px / scale ** 2)
        index = shade_rims(index, pal, SHADE_THIN)
        _, _, index = patches(index, keep_px, thin=0, enclosed_max=0, rim=RIM, rim_area=30, src_lab=src_lab, pal=pal,
                              lines=LINE_RESCUE if grain_level < LINE_RESCUE_GRAIN else 0, strict_lines=strict_lines,
                              keep_dots=min(dots, keep_px))
        src_index = index
        if smooth and scale != 1:
            index, _ = ed.clean(index.astype(np.uint8), scale, 2)
        if index.shape != (H, W):
            index = cv2.resize(index.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        if smooth and small_px and scale > 1:      # small objects from the picture's own anti-aliasing
            index = draw_small(rgb, src_index.astype(np.uint8), pal, index.astype(np.uint8), small_px)
        if rescue:                                 # an ink kept for dots that all melted after: no empty screen
            used = np.bincount(index.ravel(), minlength=len(pal)) > 0
            if not used.all():
                pal = pal[used]
                index = (np.cumsum(used) - 1)[index].astype(index.dtype)
    return pal, index


def design_match(rgb, flat, sample=200_000):
    """0-100: how close the flat design (any size) is to the picture it came from, on the picture's own grid
    (the flat box-shrunk to it), by mean CIEDE2000 (100 = the same, a mean of 25 = 0): LoomLab's Reduce match.
    The sketch's own `match` (99.9%) only says the sketch + CSV give back the FLAT design; this says how much of
    the picture the flat inks keep. An AI picture's shading and grain are not flat ink, so clean designs come out
    ~85-93 here (the user's 26 designs: 79-94), and 95+ is not reachable from such a picture."""
    h, w = rgb.shape[:2]
    small = flat if flat.shape[:2] == (h, w) else cv2.resize(flat, (w, h), interpolation=cv2.INTER_AREA)
    a, b = rgb.reshape(-1, 3), small.reshape(-1, 3)
    step = max(1, len(a) // sample)
    de = nm.delta_e2000(nm._lab(a[::step].astype(np.float64)), nm._lab(b[::step].astype(np.float64)))
    return round(float(max(0.0, 100 * (1 - de.mean() / 25))), 1)


def number(design_path, out_dir, name=None, size=3535, colours=8, detail='normal', smooth=True, line_mm='auto',
           separators=True, min_area=None, dpi=300, bold_mm=0.5, bold_scale=1, circle=0.9, polygons=0.9, motifs=0.9, merge_similar=False, fair=2.5,
           merge_shades=True, small_px=SMALL_PX, log=print, detail_rescue=True):
    """Number a coloured design and draw its sketch. See the module doc; returns a dict of what was made."""
    rgb = read_cv2(design_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FillError('design read nahi hua - path check karo.')
    if detail not in DETAIL_MM2:
        raise FillError(f"detail '{detail}' nahi: kam, normal ya zyada")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    name = safe_name(name or os.path.splitext(os.path.basename(design_path))[0])
    os.makedirs(out_dir, exist_ok=True)
    W, H = output_size(rgb.shape[1], rgb.shape[0], size)
    scale = W / rgb.shape[1]
    g = grain(rgb)
    woven = g >= WOVEN_GRAIN
    log(f"[number] grain {g:.1f}: " + ('photo / buna kapda - daane saaf kiye' if woven else 'saaf design - bareek detail rakhi'))

    px_mm = dpi / 25.4
    keep_mm2 = (min_area / px_mm ** 2) if min_area else DETAIL_MM2[detail] * (WOVEN_DETAIL if woven else 1)
    pal, index = flat_index(rgb, W, H, colours, keep_mm2 * px_mm ** 2, woven, smooth, g, small_px, bool(min_area),
                            rescue=detail_rescue)
    for col, px in RESCUE_REPORT.get('rare_flagged', []):
        log(f'[number] rare colour #%02X%02X%02X ({px} px) ko apni ink nahi mili: --colors badhao' % col)
    shaded = []
    if merge_shades:                               # an ink's shading (darker red at a petal's base): one screen
        index, shaded = merge_shade_inks(index, pal, px_mm)
        for src, dst in shaded:
            log(f'[number] shade: {src} ko {dst} me mila diya (usi rang ka gehra/halka shade, ek screen)')
    if merge_similar:                              # a small ink next to a bigger one of the same look: one screen, not two
        index, n_merged, moved = merge_similar_inks(index, pal)
        for src, dst in moved:
            log(f'[number] chhota rang {src} ko {dst} me mila diya (ek hi rang jaisa dikhta hai)')
    lab, _ = _label(index)

    # 4. the sketch. A part that is itself a line (a drawn outline: at most ~2.5 source px wide and long, not a
    # dot) is drawn solid black, as a sketch draws it (outlined on both sides it left a 1-2 px strip that broke
    # into hundreds of bits); where two other parts meet, a line --line-mm wide, grey when --separators
    # (it prints as the colour beside it) and black otherwise
    bd = np.zeros(lab.shape, bool)
    dx, dy = lab[:, 1:] != lab[:, :-1], lab[1:, :] != lab[:-1, :]
    bd[:, 1:] |= dx
    bd[:, :-1] |= dx
    bd[1:, :] |= dy
    bd[:-1, :] |= dy
    p_n = int(lab.max())
    dist_bd = ndimage.distance_transform_edt(~bd)          # used for the parts' widths, every line width, the bold cores
    thick = np.zeros(p_n + 1)
    thick[1:] = _label_max(dist_bd, lab, p_n)
    p_area = np.bincount(lab.ravel(), minlength=p_n + 1)
    is_line = (thick <= LINE_WIDTH * scale) & (p_area >= 4 * np.pi * np.maximum(thick, 1) ** 2)
    is_line[0] = False
    # only the darkest ink's thin patches are drawn lines (outlines are dark); a thin LIGHT part (a cream vein in
    # a navy leaf) stays a part with its own number, or the CSV's 'lines' ink would print it dark
    p_ink = np.zeros(p_n + 1, np.int64)
    p_ink[lab.ravel()] = index.ravel()
    darkest = int(np.argmin(nm._lab(pal.astype(np.float64))[:, 0]))
    is_line &= p_ink == darkest
    line_px = is_line[lab]
    line_ink = (hex_of(pal[int(np.bincount(index[line_px], minlength=len(pal)).argmax())])
                if line_px.any() else None)
    flat = pal[index]
    sketch_path = os.path.join(out_dir, f'{name}_sketch_seal0.png')
    K = len(pal)

    def draw(mm):
        """The sketch with lines `mm` wide, its areas, their inks and the CSV text, and how close the sketch
        painted with that CSV (as `textile paint` will) comes back to the flat design."""
        # the line sits ON the boundary, one px each side (bd), widened evenly: drawn on one side only it moved
        # every edge by a px and the bench's errors sat on the lines (42-65% of them)
        lw = max(2, round(mm * px_mm))                         # under 2 px a grey line breaks where it runs slant
        edge = _wide_line(lab, bd, lw / 2, dist_bd) & ~line_px
        sk = np.full(lab.shape, 255, np.uint8)
        sk[edge] = SEP_GREY if separators else 0
        sk[line_px] = 0
        save_png(to_image(sk), sketch_path, dpi)
        # the numbers are the areas `textile paint` itself finds in that sketch (seal 0, from the file name)
        rg = pt.find_regions(sketch_path, size=W, seal=0, log=lambda m: None)
        # a sliver the lines cut off (smaller than --detail's smallest part) is no part of the design: it is
        # inked into the line beside it (a separator when --separators, so it prints as its neighbour)
        sliver = rg.area < max(20.0, keep_mm2 * px_mm ** 2)
        sliver[0] = False
        if sliver.any():
            sk[sliver[rg.lab]] = SEP_GREY if separators else 0
            save_png(to_image(sk), sketch_path, dpi)
            rg = pt.find_regions(sketch_path, size=W, seal=0, log=lambda m: None)
        inside = rg.lab > 0
        votes = np.bincount(rg.lab[inside].astype(np.int64) * K + index[inside],
                            minlength=(rg.n + 1) * K).reshape(-1, K)
        ik = votes.argmax(1)                                   # each area takes the ink most of it had
        text = '\n'.join(f'{i}={hex_of(pal[ik[i]])}' for i in range(1, rg.n + 1))
        text += f'\nlines={line_ink or "fill"}' + ('\nseparators=fill' if separators else '')
        back, bpal, _ = pt.paint(rg, *pt.plan(rg, pt.parse_colors(text))[:3])
        back_rgb = bpal[back]
        m = float((np.abs(back_rgb.astype(int) - flat.astype(int)).sum(-1) < 30).mean() * 100)
        return sk, rg, ik, m, back_rgb

    # --line-mm auto: the sketch is drawn at each width and the one that paints back closest to the design is
    # kept (a tie, under 0.05 points, goes to the thicker, clearer line)
    widths = LINE_AUTO if line_mm in (None, 'auto') else (float(line_mm),)
    tried = []
    best = None
    for mm in widths:
        sk, rg, ik, m, back_rgb = draw(mm)
        tried.append((mm, round(m, 2), int(rg.n)))
        if len(widths) > 1:
            log(f'[number] line {mm} mm: {rg.n} hisse, sketch se wapas design {m:.2f}%')
        if best is None or m > best[3] + 0.05 or (abs(m - best[3]) <= 0.05 and mm > best[0]):
            best = (mm, sk, rg, m, ik, back_rgb)
    mm, sketch, reg, match, ink, back_rgb = best
    if len(widths) > 1:
        log(f'[number] chuna: {mm} mm (sabse zyada match)')
    save_png(to_image(sketch), sketch_path, dpi)
    black = np.where(sketch < 255, 0, 255).astype(np.uint8)  # the same sketch all in black, to show or share
    save_png(to_image(black), os.path.join(out_dir, f'{name}_sketch_black.png'), dpi)
    save_png(to_image(back_rgb), os.path.join(out_dir, f'{name}_rangeen.png'), dpi)   # what paint will make
    # the bold sketch: the same parts' outlines as smooth curves (round stays round, corners stay sharp), drawn
    # dark and thick, anti-aliased, plus an SVG. Thin parts (under 1.2 strokes wide) keep a white core, so a bold
    # stroke never fills a small petal or dot.
    from . import curves as cv
    wpx = max(2, round(bold_mm * px_mm))
    dtb = dist_bd
    thin = (thick < 1.2 * wpx)[lab] & (lab > 0)
    core = thin & (dtb > 1.5)
    snapped = {}
    bold_img, bold_svg = cv.bold(lab, is_line, scale, wpx, core, k=bold_scale, circle=circle, polygons=polygons,
                                 motifs=motifs, stats=snapped, fair_sigma=fair)
    bold_path = os.path.join(out_dir, f'{name}_sketch_bold.png')
    bold_dpi = dpi * bold_scale                           # same inches, k times the pixels
    save_png(to_image(bold_img), bold_path, bold_dpi)
    with open(os.path.join(out_dir, f'{name}_sketch_bold.svg'), 'w', encoding='utf-8') as fh:
        fh.write(bold_svg)
    colour_big = cv.colour_fill(flat, bold_scale)
    bold_col_path = os.path.join(out_dir, f'{name}_sketch_bold_rangeen.png')
    save_png(to_image(cv.colour_sketch(colour_big, bold_img)), bold_col_path, bold_dpi)
    bold_rgb = np.stack([bold_img] * 3, -1)
    if bold_scale > 1:                                    # numbers on the big sheet: the parts scaled up with it
        import dataclasses
        k = bold_scale
        reg_k = dataclasses.replace(
            reg, lab=cv2.resize(reg.lab, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST),
            lines=cv2.resize(reg.lines.astype(np.uint8), None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST) > 0,
            area=reg.area * k * k, raw=None)
    else:
        reg_k = reg
    bn_path, _ = pt._numbers(reg_k, out_dir, name, bold_rgb, kind='sketch_bold_numbers', full=True, ink=bold_img)
    n = reg.n
    area = reg.area
    big = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA if rgb.shape[1] > W else cv2.INTER_LANCZOS4)
    path, missed = pt._numbers(reg, out_dir, name, big, kind='numbers', outline=True, full=True)
    sk_path, _ = pt._numbers(reg, out_dir, name, None, kind='sketch_numbers', full=True)
    letters = pt.maps(reg, out_dir, name, numbers=False, template=False)
    save_png(to_image(flat), os.path.join(out_dir, f'{name}_flat.png'), dpi)
    csv_path = os.path.join(out_dir, f'{name}_colors.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['Number', 'HEX', 'Colour', 'Share %', 'Group'])
        for i in range(1, n + 1):
            c = pal[ink[i]]
            gi = reg.group[i]
            w.writerow([i, '#' + hex_of(c), nm.colour_name(c), round(area[i] / (W * H) * 100, 3),
                        reg.letters[gi] if gi >= 0 else ''])
        if line_ink:                                      # the drawn outlines print in their own ink
            w.writerow(['lines', '#' + line_ink, 'sketch ki kaali line (design ki outline ka rang)', '', ''])
        else:
            w.writerow(['lines', 'fill', 'sketch ki line: paas ke hisse ka rang (koi outline nahi)', '', ''])
        if separators:
            w.writerow(['separators', 'fill', 'grey line: do rangon ki seema, paas ka rang', '', ''])

    tiny = int((area[1:] < DETAIL_MM2['zyada'] * px_mm ** 2).sum())
    real = design_match(rgb, flat)
    log(f'[number] {W}x{H} px, {n} hisse ({tiny} bahut chhote), {len(reg.letters)} group'
        + (f', outline {line_ink} = kaali line' if line_ink else '') + f'; sketch se wapas design: {match:.1f}% match; asli design se mel {real}%')
    shares = np.bincount(index.ravel(), minlength=len(pal)) / index.size * 100
    return {'name': name, 'size_px': [W, H], 'areas': int(n), 'missed': int(missed), 'numbers': path,
            'sketch': sketch_path, 'sketch_numbers': sk_path, 'letters': letters['map'], 'match': round(match, 1),
            'tiny': tiny, 'groups': len(reg.letters), 'line_mm': mm, 'bold': bold_path, 'bold_numbers': bn_path, 'bold_colour': bold_col_path, 'snapped': snapped, 'tried': tried,
            'grain': round(float(g), 2), 'woven': bool(woven), 'colours_limit': colours, 'detail': detail,
            'csv': csv_path, 'design_match': real, 'shades_merged': shaded, 'inks': [(hex_of(c), nm.colour_name(c), round(float(s), 2)) for c, s in zip(pal, shares) if s > 0.05]}
