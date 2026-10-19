"""A flat design redrawn as smooth curves: sharp at any zoom (SVG) and as a bigger flat PNG.

  crisp(rgb, scale=2.0, zoom=2)  -> (svg text, zoomed image H*zoom x W*zoom x 3, report)

A raster design shows its pixels when you zoom (steps, or a blur if the viewer smooths them). Here every
colour's outlines get the same treatment as the bold sketch (`curves`: circles and ovals made true, straight
sides straight, corners kept, every other stretch faired into one flowing curve) and are drawn back as
filled shapes: the biggest colour is the ground, the others are painted over it, biggest first (so there is
never a gap between two colours). The zoomed PNG is drawn straight from the curves with hard edges (no
anti-aliasing: still one flat colour per pixel), never an enlarged copy of the pixels.

For looking at, editing and sharing. The mill's file stays the 300 DPI pixel TIF: a redrawn shape differs
from the pixel design by a fraction of a source pixel along its edges, and the report says by how much.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from . import curves as cv
from .io_utils import hex_of, unique_rgb


def _shape(p, size, scale, circle=cv.CIRCLE_IOU, polygons=cv.CIRCLE_IOU, motifs=cv.CIRCLE_IOU, fair=cv.FAIR_SIGMA):
    """One outline as its cleanest curve (the same order the bold sketch uses)."""
    q = cv.circle_of(p, size, circle)
    if q is None:
        q = cv.polygon_of(p, size, polygons)
    if q is None:
        q = cv.oval_of(p, size, motifs)
    if q is None:
        q = cv.leaf_of(p, size, scale, motifs)
    if q is None:
        q = cv.smooth(p, scale, fair)
    return q


def crisp(rgb, scale=2.0, zoom=2, smoothing=70.0):
    H, W = rgb.shape[:2]
    pal, inv = unique_rgb(rgb, return_inverse=True)
    index = inv.reshape(H, W)
    if len(pal) > 64:
        raise ValueError(f'{len(pal)} rang: ye flat design nahi (pehle number / export se flat karo)')
    order = np.argsort(-np.bincount(index.ravel(), minlength=len(pal)), kind='stable')
    S = 16                                                          # cv2 shift 4: 1/16 px
    out = np.zeros((H * zoom, W * zoom, 3), np.uint8)
    out[:] = pal[order[0]]                                          # the ground fills the sheet
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="{hex_of(pal[order[0]]).join(["#", ""])}"/>']
    fair = smoothing / 20.0
    n_out = 0
    for k in order[1:]:
        mask = (index == k).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        polys = []
        for c in cs:
            p = c[:, 0, :].astype(np.float64)
            if len(p) < 3:
                continue
            if len(p) >= 12:
                t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
                t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
                g = p - cv.EDGE_SHIFT * np.stack([t[:, 1], -t[:, 0]], 1)     # the true edge, half a pixel out
                q = _shape(g, (H, W), scale, fair=fair)
            else:
                q = p + 0.5 - 0.5                                            # a speck: its own pixel corners
            polys.append(q)
        if not polys:
            continue
        m = np.zeros((H * zoom, W * zoom), np.uint8)
        for q in polys:                                                      # even-odd: a hole stays open
            part = np.zeros_like(m)
            cv2.fillPoly(part, [np.round(q * zoom * S).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
            m ^= part
        out[m > 0] = pal[k]
        d = ' '.join('M' + ' L'.join(f'{x:.1f} {y:.1f}' for x, y in cv2.approxPolyDP(
            q.astype(np.float32).reshape(-1, 1, 2), 0.15, True)[:, 0, :]) + ' Z' for q in polys)
        svg.append(f'<path fill="#{hex_of(pal[k]).lstrip("#")}" fill-rule="evenodd" d="{d}"/>')
        n_out += len(polys)
    svg.append('</svg>')
    back = cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST) if zoom > 1 else out
    # how far the redraw is from the pixel design: the share of pixels whose colour differs (edges only)
    diff = float((back != rgb).any(2).mean() * 100)
    return '\n'.join(svg), out, {'colours': len(pal), 'outlines': n_out, 'zoom': zoom,
                                 'pixels_different_percent': round(diff, 2)}
