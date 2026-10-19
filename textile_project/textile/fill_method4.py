"""Colour fill, Method 4: Method 1 with the gaps in the line art sealed.

AI line art is rarely a closed drawing: a stroke breaks for a pixel or two,
and the reference's ground colour runs through the break into a petal, which
then votes (and fills) as ground. Measured on an AI floral pair whose line
art lines up with its reference at 0.96: with the gaps open, whole petals
and leaves came out the ground colour.

The vote is Method 1's, step for step (reference_code/method1 stays as it
is, and `fill_method1` is still byte-identical to it). The one change is
which areas vote: the line mask is CLOSED (dilate then erode, an ellipse of
`seal` px) before the areas are labelled, so a break narrower than ~2 x seal
no longer joins two areas. Then:

  - the real lines are untouched: they are drawn from the original mask, in
    the colour most common under them, as Method 1 does
  - a pixel the closing added (the seal itself) takes the colour of the
    nearest area
  - an area the closing swallowed whole (a tiny closed shape between two
    close lines) keeps its own vote from the unsealed labelling, so a dot or
    a sliver is never lost

`seal` defaults to 1.5 source pixels of the line art, scaled to the output:
the breaks an AI drawing has are 1-3 of its own pixels wide, while a gap
between two real lines is wider than that. Seal 0 is Method 1.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from . import palette as pl
from .fill_method1 import Fill, FillError, alignment_score, output_size
from .io_utils import read_cv2, rgb_of


def default_seal(src_width, out_width, source_px=1.5):
    """The closing radius, in output px: `source_px` of the line art's own."""
    return max(1, int(round(source_px * out_width / max(src_width, 1))))


def fill(line_path, ref_path, size=3535, max_colors=16, min_share=0.0005, line_threshold=150,
         line_color='auto', seal=None, min_align=0.55, force=False, log=print) -> Fill:
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
    pal, is_flat = pl.extract_palette(ref, max_colors, min_share, log=log)
    K = len(pal)
    ref_idx = cv2.resize(pl.map_to_palette(ref, pal), (W, H), interpolation=cv2.INTER_NEAREST)
    g = cv2.GaussianBlur(cv2.resize(line, (W, H), interpolation=cv2.INTER_LANCZOS4), (3, 3), 0)
    lines = g < line_threshold
    del g
    score = alignment_score(lines, ref_idx)
    log(f"[align] score = {score:.2f} (>= {min_align} chahiye)")
    if score < min_align and not force:
        raise FillError("line art aur reference aligned nahi lag rahe. Images check karo "
                        "(crop/shift/rotation/alag design). Jaan-boojh kar chalana ho to --force do.")

    r = default_seal(line.shape[1], W) if seal is None else int(seal)
    if r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        sealed = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    else:
        sealed = lines
    log(f"[seal] {r} px: line art ke tootne wale gap band kiye ({(sealed & ~lines).mean() * 100:.2f}% pixels)")

    def votes_of(lab, n, mask):
        return np.bincount(lab[mask].astype(np.int64) * K + ref_idx[mask],
                           minlength=(n + 1) * K).reshape(n + 1, K)

    lab, n = ndimage.label(~sealed)
    votes = votes_of(lab, n, ~sealed)
    region_color = votes.argmax(1)
    out = region_color[lab].astype(np.uint8)
    if r > 0:
        _, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
        added = sealed & ~lines
        out[added] = region_color[lab[iy[added], ix[added]]]
        # an area the closing swallowed whole keeps its own vote
        lab0, n0 = ndimage.label(~lines)
        alive = np.bincount(lab0[~sealed], minlength=n0 + 1)
        own = votes_of(lab0, n0, ~lines).argmax(1)
        swallowed = alive == 0
        swallowed[0] = False
        m = swallowed[lab0]
        out[m] = own[lab0[m]].astype(np.uint8)
        del lab0, m

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
        log(f"[warn] {len(bad)} bade regions me reference ke multiple colors mile (purity < 60%) -> debug image dekho.")
        dbg = pal[out].copy()
        dbg[np.isin(lab, [b[0] for b in bad])] = (dbg[np.isin(lab, [b[0] for b in bad])] * 0.3
                                                  + np.array([255, 0, 0]) * 0.7).astype(np.uint8)
        debug = np.asarray(Image.fromarray(dbg).resize((rw, rh), Image.NEAREST))
    del lab
    view = np.asarray(Image.fromarray(ref).resize((rw, rh), Image.NEAREST))
    return Fill(out, pal, lc, bool(is_flat), score, int(n), bad, debug, view)
