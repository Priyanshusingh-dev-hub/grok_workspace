#!/usr/bin/env python3
"""
repeat_analyze.py - kisi all-over design me repeat type pehchaanna
(straight / half-drop / brick / mirror) + repeat vectors + jhukav (shear).

Tarika (tested on user's floral design -> half-drop mila):
  1. Image ke bahut saare chhote patches lo (150x150).
  2. Har patch ko poori image me template-match karo (TM_CCOEFF_NORMED).
  3. Strong matches (>0.55) ke displacement vectors jama karo.
  4. DBSCAN se vectors cluster karo -> lattice vectors milte hain.
  5. Lattice se repeat type decide karo:
       - sirf (0,W) aur (H,0) type vectors          -> straight / full drop
       - (H/2, W/2) wala vector bhi                 -> half-drop (ya brick)
       - mirrored patch strong match kare           -> mirror repeat
  6. Vertical vector ka dx != 0 ho to design tircha hai (shear); aage
     tile banane se pehle deshear karna padega.

Usage: python repeat_analyze.py design.png
"""
import sys
import numpy as np
import cv2
from sklearn.cluster import DBSCAN


def analyze(path, patch=150, step=90, thr=0.55, max_dy=400):
    im = cv2.imread(path)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
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
    if not vecs:
        print("Koi repeat nahi mila (ya design bahut irregular hai).")
        return
    V = np.array(vecs)
    lab = DBSCAN(eps=25, min_samples=8).fit(V[:, :2]).labels_
    clusters = []
    for l in sorted(set(lab) - {-1}):
        m = V[lab == l]
        clusters.append((len(m), m[:, 0].mean(), m[:, 1].mean(), m[:, 2].mean()))
    clusters.sort(key=lambda c: -c[0])
    print("Repeat vectors (count, dy, dx, match score):")
    for c in clusters:
        print(f"  n={c[0]:4d}  dy={c[1]:7.1f}  dx={c[2]:7.1f}  score={c[3]:.2f}")

    horiz = [c for c in clusters if abs(c[1]) < 10]
    vert = [c for c in clusters if c[1] > 150 and abs(c[2]) < 0.25 * Ww and abs(c[2]) < c[1] * 0.3]
    print(f"\nMirror match (max): {np.mean(mirror_scores):.2f} avg -> "
          f"{'MIRROR ho sakta hai' if np.mean(mirror_scores) > 0.6 else 'mirror nahi'}")
    if horiz and vert:
        W = abs(min(horiz, key=lambda c: abs(c[2]))[2])
        Hv = min(vert, key=lambda c: c[1])
        H, shear = Hv[1], Hv[2]
        half = [c for c in clusters if abs(c[1] - H / 2) < 30 and abs(abs(c[2] - shear / 2) - W / 2) < 40]
        kind = "HALF-DROP (ya brick)" if half else "STRAIGHT / FULL DROP"
        print(f"Repeat type: {kind}")
        print(f"Full straight repeat block ~ {W:.0f} x {H:.0f} px")
        if half:
            print(f"Half-drop unit ~ {W / 2:.0f} x {H:.0f} px, drop {H / 2:.0f} px")
        if abs(shear) > 5:
            ang = np.degrees(np.arctan2(shear, H))
            print(f"Shear/jhukav: har {H:.0f} px par {shear:.0f} px (~{ang:.1f} deg) -> deshear karo")
    avg = np.mean([c[3] for c in clusters])
    if avg < 0.85:
        print(f"Note: avg match {avg:.2f} < 0.85 -> copies exact nahi (AI image). "
              "Tile banate waqt seam-cut zaroori.")


if __name__ == "__main__":
    analyze(sys.argv[1])
