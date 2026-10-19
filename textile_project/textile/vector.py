"""Vector trace (ROADMAP Phase 5, optional): each channel's outlines traced to
polygons, then drawn back at the same size WITHOUT anti-aliasing, so the
pixel staircases along slanted edges become straight runs. Only when the user
asks for clean edges; never Gaussian smoothing.

potrace is not used (it is not in the requirements and must work offline):
OpenCV's contours (`findContours`, holes included) simplified by
`approxPolyDP` (`eps` px: the most a traced edge may move off the pixels),
filled back with `fillPoly` (LINE_8: hard edges). One ink per pixel stays
true by construction: the channels are painted in order, biggest first, each
over the ones before, starting from the design itself; a pixel no traced
shape claims keeps its own colour. Shapes smaller than `min_area` px are
kept as they are (a dot is not traced away).

An SVG of the same polygons (one <path> per channel, even-odd holes) is
written too, for the mill's vector software.
"""
from __future__ import annotations

import cv2
import numpy as np

from .io_utils import hex_of


def trace(index, pal, eps=0.8, min_area=12):
    """(new index map, svg text, share of pixels that changed)."""
    H, W = index.shape
    K = len(pal)
    order = np.argsort(-np.bincount(index.ravel(), minlength=K), kind='stable')
    out = index.copy()
    paths = []
    for k in order[1:] if K > 1 else []:             # the biggest is the ground: everything is painted over it
        mask = (index == k).astype(np.uint8)
        contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        polys = [cv2.approxPolyDP(c, eps, True) if cv2.contourArea(c) >= min_area else c for c in contours]
        drawn = np.zeros((H, W), np.uint8)
        cv2.fillPoly(drawn, polys, 1, lineType=cv2.LINE_8)      # even-odd: holes stay open
        # small shapes and 1 px lines (no area to trace) come back exactly
        small = np.zeros((H, W), np.uint8)
        cv2.fillPoly(small, [c for c in contours if cv2.contourArea(c) < min_area], 1, lineType=cv2.LINE_8)
        on = (drawn > 0) | ((small > 0) & (mask > 0))
        out[on] = k
        d = ' '.join('M' + ' L'.join(f'{p[0][0]},{p[0][1]}' for p in poly) + ' Z' for poly in polys if len(poly) > 2)
        paths.append(f'<path fill="#{hex_of(pal[k])}" fill-rule="evenodd" d="{d}"/>')
    # a pixel the traced shapes left uncovered keeps its own colour; one the
    # simplification moved goes to whichever shape now covers it
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<rect width="{W}" height="{H}" fill="#{hex_of(pal[order[0]])}"/>' + ''.join(paths) + '</svg>')
    changed = float((out != index).mean())
    return out, svg, changed
