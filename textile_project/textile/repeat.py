"""Repeat identify (Kaam B): reference_code/repeat_analyze.py as a module.

The logic is the script's, step for step (its numbers are the defaults):
  1. many small patches (150 px, every 90 px, skipping flat ones: std < 40)
  2. each template-matched over the whole image (TM_CCOEFF_NORMED); the 8
     best matches over 0.55 give displacement vectors (dy >= 0, dy <= 400)
  3. DBSCAN (eps 25, min 8) clusters them into the lattice vectors
  4. a horizontal vector (|dy| < 10) and a vertical one (dy > 150, |dx| small)
     give the straight block W x H; a vector near (H/2, W/2) means half-drop
     (or brick); the vertical vector's dx is the shear (jhukav)
  5. mirrored patches matching > 0.6 on average: mirror repeat

Changed from the script: it returns a dict (JSON for the user and for
`textile tile`) instead of only printing, reads with cv2.imdecode (any
Windows path; the same pixels), and says what it found when there is no
full lattice, as ROADMAP Phase 3 asks: a vertical repeat with no horizontal
one is a panel print (no tile to cut), and neither is "no repeat".
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from .io_utils import read_cv2


def analyze(path, patch=150, step=90, thr=0.55, max_dy=400, gray=None):
    if gray is None:
        im = read_cv2(path, cv2.IMREAD_COLOR)       # colour, then grey: the script's own pixels
        if im is None:
            raise ValueError('image read nahi hui - path check karo.')
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = gray
    Hh, Ww = g.shape
    vecs, mirror_scores = [], []
    for y in range(60, Hh - patch - 60, step):
        for x in range(40, Ww - patch - 40, step):
            t = g[y:y + patch, x:x + patch]
            if t.std() < 40:          # khaali patch skip
                continue
            r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
            mirror_scores.append(cv2.matchTemplate(g, cv2.flip(t, 1), cv2.TM_CCOEFF_NORMED).max())
            rr = r.copy()
            rr[max(0, y - 60):y + 60, max(0, x - 60):x + 60] = -1
            for _ in range(8):
                yy, xx = np.unravel_index(rr.argmax(), rr.shape)
                v = rr[yy, xx]
                if v < thr:
                    break
                dy, dx = yy - y, xx - x
                if dy < 0 or (dy == 0 and dx < 0):
                    dy, dx = -dy, -dx
                if dy <= max_dy:
                    vecs.append((dy, dx, v))
                rr[max(0, yy - 40):yy + 40, max(0, xx - 40):xx + 40] = -1
    out = {'size_px': [Ww, Hh], 'type': 'none', 'clusters': [],
           'mirror_score': round(float(np.mean(mirror_scores)), 2) if mirror_scores else None}
    out['mirror'] = bool(mirror_scores) and bool(np.mean(mirror_scores) > 0.6)
    if not vecs:
        return out
    V = np.array(vecs)
    lab = DBSCAN(eps=25, min_samples=8).fit(V[:, :2]).labels_
    clusters = []
    for l in sorted(set(lab) - {-1}):
        m = V[lab == l]
        clusters.append((len(m), m[:, 0].mean(), m[:, 1].mean(), m[:, 2].mean()))
    clusters.sort(key=lambda c: -c[0])
    out['clusters'] = [{'n': int(c[0]), 'dy': round(float(c[1]), 1), 'dx': round(float(c[2]), 1),
                        'score': round(float(c[3]), 2)} for c in clusters]
    if not clusters:
        return out

    horiz = [c for c in clusters if abs(c[1]) < 10]
    vert = [c for c in clusters if c[1] > 150 and abs(c[2]) < 0.25 * Ww and abs(c[2]) < c[1] * 0.3]
    if horiz and vert:
        W = abs(min(horiz, key=lambda c: abs(c[2]))[2])
        Hv = min(vert, key=lambda c: c[1])
        H, shear = Hv[1], Hv[2]
        half = [c for c in clusters if abs(c[1] - H / 2) < 30 and abs(abs(c[2] - shear / 2) - W / 2) < 40]
        out.update({'type': 'half-drop' if half else 'straight', 'W': round(float(W), 1), 'H': round(float(H), 1),
                    'shear': round(float(shear), 1),
                    'shear_deg': round(float(np.degrees(np.arctan2(shear, H))), 1)})
        if half:
            out.update({'unit': [round(float(W / 2), 1), round(float(H), 1)], 'drop': round(float(H / 2), 1)})
    elif vert:
        Hv = min(vert, key=lambda c: c[1])
        out.update({'type': 'panel', 'H': round(float(Hv[1]), 1), 'shear': round(float(Hv[2]), 1)})
    elif horiz:
        out.update({'type': 'border', 'W': round(float(abs(min(horiz, key=lambda c: abs(c[2]))[2])), 1)})
    out['avg_match'] = round(float(np.mean([c[3] for c in clusters])), 2)
    return out


def summary_hinglish(r):
    """The script's own lines, from the dict."""
    lines = []
    if r['clusters']:
        lines.append('Repeat vectors (count, dy, dx, match score):')
        lines += [f"  n={c['n']:4d}  dy={c['dy']:7.1f}  dx={c['dx']:7.1f}  score={c['score']:.2f}" for c in r['clusters']]
    if r['mirror_score'] is not None:
        lines.append(f"Mirror match: {r['mirror_score']:.2f} avg -> {'MIRROR ho sakta hai' if r['mirror'] else 'mirror nahi'}")
    t = r['type']
    if t in ('half-drop', 'straight'):
        lines.append(f"Repeat type: {'HALF-DROP (ya brick)' if t == 'half-drop' else 'STRAIGHT / FULL DROP'}")
        lines.append(f"Full straight repeat block ~ {r['W']:.0f} x {r['H']:.0f} px")
        if t == 'half-drop':
            lines.append(f"Half-drop unit ~ {r['unit'][0]:.0f} x {r['unit'][1]:.0f} px, drop {r['drop']:.0f} px")
        if abs(r['shear']) > 5:
            lines.append(f"Shear/jhukav: har {r['H']:.0f} px par {r['shear']:.0f} px (~{r['shear_deg']:.1f} deg) -> deshear karo")
    elif t == 'panel':
        lines.append(f"Repeat type: PANEL PRINT (sirf upar-neeche {r['H']:.0f} px par repeat, aar-paar nahi). "
                     'All-over nahi hai: tile mat kaato, mill ko "panel print, straight vertical repeat" bolo.')
    elif t == 'border':
        lines.append(f"Repeat type: BORDER (sirf aar-paar {r['W']:.0f} px par repeat). Tile nahi banega.")
    else:
        lines.append('Koi repeat nahi mila (ya design bahut irregular hai).')
    if r.get('avg_match') is not None and r['avg_match'] < 0.85:
        lines.append(f"Note: avg match {r['avg_match']:.2f} < 0.85 -> copies exact nahi (AI image). "
                     'Tile banate waqt seam-cut zaroori.')
    return '\n'.join(lines)
