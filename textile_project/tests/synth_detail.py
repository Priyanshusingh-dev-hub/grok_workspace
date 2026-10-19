"""SYNTHETIC test artwork with known ground truth, and small-detail metrics.

Everything here is generated in code (labelled synthetic); it is NOT the user's reference PSDs.
Artwork is drawn at 4x and box-reduced, so edges are anti-aliased like real exported art, while
every object keeps an exact ground-truth mask (its visible pixels at 1x, coverage >= 50%).

  small_object_card(scale)  -> (rgb, objects)   tiny dots, rings, thin lines, gaps, petals, leaves, rare colours
  style_a(scale)            -> (rgb, objects)   fine-detail repeating floral, black on white, seamless tile
  style_b(scale)            -> (rgb, objects)   multitone daisies: black / dark grey / mid grey / light grey / white
  detail_metrics(objects, out_rgb)  -> dict     survival %, IoU, area / centroid error, merges, line retention
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw

SS = 4                      # supersampling factor
DE_OK = 15.0                # an output pixel is "the object's colour" within this CIEDE2000


def _lab(rgb):
    c = np.asarray(rgb, np.float64) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def de2000(lab1, lab2):
    """CIEDE2000 (same formula as textile.names.delta_e2000), broadcasting on the last axis."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cb7 / (Cb7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p, h2p = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(C1p * C2p == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) > 180, (hs + 360) / 2, hs / 2)
    hbp = np.where(C1p * C2p == 0, hs, hbp)
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dth = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


class Canvas:
    """Draw objects at SS x; remember which object owns every sub-pixel."""

    def __init__(self, w, h, ground=(255, 255, 255)):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w * SS, h * SS), ground)
        self.owner = Image.new("I", (w * SS, h * SS), 0)
        self.d = ImageDraw.Draw(self.img)
        self.o = ImageDraw.Draw(self.owner)
        self.objects = [{"id": 0, "kind": "ground", "colour": tuple(ground), "name": "ground"}]

    def _new(self, kind, colour, name):
        oid = len(self.objects)
        self.objects.append({"id": oid, "kind": kind, "colour": tuple(colour), "name": name})
        return oid

    def ellipse(self, cx, cy, rx, ry, colour, kind="dot", name="", outline_w=0):
        oid = self._new(kind, colour, name)
        box = [(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS - 1, (cy + ry) * SS - 1]
        if outline_w:
            self.d.ellipse(box, outline=colour, width=max(1, round(outline_w * SS)))
            self.o.ellipse(box, outline=oid, width=max(1, round(outline_w * SS)))
        else:
            self.d.ellipse(box, fill=colour)
            self.o.ellipse(box, fill=oid)
        return oid

    def polygon(self, pts, colour, kind="shape", name=""):
        oid = self._new(kind, colour, name)
        p = [(x * SS, y * SS) for x, y in pts]
        self.d.polygon(p, fill=colour)
        self.o.polygon(p, fill=oid)
        return oid

    def rect(self, x0, y0, x1, y1, colour, kind="shape", name=""):
        return self.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], colour, kind, name)

    def line(self, x0, y0, x1, y1, width, colour, kind="line", name=""):
        oid = self._new(kind, colour, name)
        self.d.line([(x0 * SS, y0 * SS), (x1 * SS, y1 * SS)], fill=colour, width=max(1, round(width * SS)))
        self.o.line([(x0 * SS, y0 * SS), (x1 * SS, y1 * SS)], fill=oid, width=max(1, round(width * SS)))
        return oid

    def petal(self, cx, cy, ang, length, width, colour, name="petal"):
        """An ellipse petal from (cx, cy) outward along `ang`."""
        pts = []
        for t in np.linspace(0, 2 * math.pi, 40, endpoint=False):
            u, v = (1 + math.cos(t)) / 2 * length, math.sin(t) * width / 2
            pts.append((cx + u * math.cos(ang) - v * math.sin(ang), cy + u * math.sin(ang) + v * math.cos(ang)))
        return self.polygon(pts, colour, "petal", name)

    def finish(self):
        rgb = np.asarray(self.img.reduce(SS), np.uint8).copy()
        own = np.asarray(self.owner, np.int32).reshape(self.h, SS, self.w, SS).transpose(0, 2, 1, 3).reshape(self.h, self.w, -1)
        objs = []
        for ob in self.objects:
            frac = (own == ob["id"]).mean(-1)
            m = frac >= 0.5
            if ob["id"] and not m.any():
                m = frac >= frac.max() * 0.999 if frac.max() > 0 else m   # a sub-pixel object: its best pixel
                m &= frac > 0
            objs.append({**ob, "mask": m, "area": int(m.sum())})
        return rgb, objs


BLACK, WHITE = (10, 10, 10), (255, 255, 255)


def small_object_card(scale=1.0):
    """Every small-detail case Rukh listed, at `scale` (1 = tiny, 2 = twice the pixels)."""
    s = scale
    W, H = int(260 * s), int(200 * s)
    c = Canvas(W, H)
    # row 1: tiny filled circles, three colours (black, red, rare teal), radius 1..4 px at scale 1
    for i, r in enumerate((1.0, 1.5, 2.0, 3.0, 4.0)):
        c.ellipse((14 + 18 * i) * s, 14 * s, r * s, r * s, BLACK, "dot", f"black dot r{r}")
        c.ellipse((14 + 18 * i) * s, 32 * s, r * s, r * s, (240, 120, 0), "dot", f"orange dot r{r}")
    for i, r in enumerate((2.0, 3.0)):                      # rare colour: < 0.05% of the card
        c.ellipse((110 + 14 * i) * s, 14 * s, r * s, r * s, (0, 150, 160), "rare", f"rare teal dot r{r}")
    # small outlined circles (rings)
    for i, (r, lw) in enumerate(((4, 1.0), (5, 1.5), (7, 1.0))):
        c.ellipse((150 + 22 * i) * s, 18 * s, r * s, r * s, BLACK, "ring", f"ring r{r} w{lw}", outline_w=lw * s)
    # thin lines, 1 / 1.5 / 2 px, straight and slant
    for i, lw in enumerate((1.0, 1.5, 2.0)):
        c.line(10 * s, (50 + 8 * i) * s, 120 * s, (50 + 8 * i) * s, lw * s, BLACK, "line", f"h-line w{lw}")
        c.line((130 + 14 * i) * s, 46 * s, (170 + 14 * i) * s, 80 * s, lw * s, BLACK, "line", f"slant line w{lw}")
    # narrow gaps: two black bars with a white 1 / 2 / 3 px gap between (the gap is an object of the ground ink)
    for i, gap in enumerate((1.0, 2.0, 3.0)):
        x0 = (10 + 40 * i) * s
        c.rect(x0, 84 * s, x0 + 14 * s, 104 * s, BLACK, "bar", f"bar L gap{gap}")
        c.rect(x0 + 14 * s + gap * s, 84 * s, x0 + 30 * s + gap * s, 104 * s, BLACK, "bar", f"bar R gap{gap}")
        c.rect(x0 + 14 * s, 84 * s, x0 + 14 * s + gap * s, 104 * s, WHITE, "gap", f"white gap {gap}px")
    # small flowers built from petals around a centre (pink petals, yellow centre)
    for i, L in enumerate((5.0, 7.0)):
        cx, cy = (150 + 40 * i) * s, 100 * s
        for k in range(5):
            c.petal(cx, cy, 2 * math.pi * k / 5, L * s, L * 0.55 * s, (225, 90, 150), f"petal L{L} #{k}")
        c.ellipse(cx, cy, 1.6 * s, 1.6 * s, (245, 200, 30), "centre", f"flower centre L{L}")
    # close leaves along a stem, 1 px apart
    c.line(20 * s, 150 * s, 120 * s, 150 * s, 1.5 * s, (40, 120, 50), "stem", "stem w1.5")
    for k in range(5):
        x = (30 + 18 * k) * s
        c.ellipse(x, 143 * s, 7.5 * s, 3 * s, (60, 150, 60), "leaf", f"leaf top {k}")
        c.ellipse(x + 8 * s, 157 * s, 7.5 * s, 3 * s, (60, 150, 60), "leaf", f"leaf bottom {k}")
    # adjacent similar colours (dE2000 ~ 10-13: distinct inks that touch)
    c.rect(140 * s, 130 * s, 162 * s, 160 * s, (200, 40, 40), "similar", "red A")
    c.rect(162 * s, 130 * s, 184 * s, 160 * s, (170, 40, 80), "similar", "red B (similar)")
    # small grey regions inside a dark region
    c.rect(196 * s, 126 * s, 250 * s, 190 * s, (30, 40, 100), "dark", "dark navy block")
    for i, r in enumerate((1.5, 2.5, 4.0)):
        c.ellipse((206 + 16 * i) * s, 145 * s, r * s, r * s, (128, 128, 128), "grey-in-dark", f"grey dot r{r}")
    c.line(202 * s, 175 * s, 244 * s, 175 * s, 1.0 * s, (128, 128, 128), "grey-in-dark", "grey hairline in dark")
    return c.finish()


def style_a(scale=1.0, tile=4):
    """SYNTHETIC Style A: fine-detail repeating floral, black on white, a seamless `tile` x `tile` repeat.
    Each motif: solid black leaves, an outlined flower with inner contours, a filled bud, tiny dots."""
    u = 64 * scale
    W = H = int(u * tile)
    c = Canvas(W, H)
    for ty in range(tile):
        for tx in range(tile):
            ox, oy = tx * u, ty * u
            cx, cy = ox + 0.45 * u, oy + 0.45 * u
            c.ellipse(cx, cy, 0.17 * u, 0.17 * u, BLACK, "ring", "flower outline", outline_w=max(1.0, 0.025 * u))
            c.ellipse(cx, cy, 0.08 * u, 0.08 * u, BLACK, "ring", "inner contour", outline_w=max(1.0, 0.018 * u))
            c.ellipse(cx, cy, 0.025 * u, 0.025 * u, BLACK, "dot", "flower centre")
            for k in range(2):
                a = math.pi * (0.25 + k)
                c.petal(cx + math.cos(a) * 0.19 * u, cy + math.sin(a) * 0.19 * u, a, 0.2 * u, 0.08 * u, BLACK, "solid leaf")
            c.ellipse(ox + 0.82 * u, oy + 0.2 * u, 0.06 * u, 0.06 * u, BLACK, "bud", "filled bud")
            c.line(ox + 0.82 * u, oy + 0.26 * u, ox + 0.75 * u, oy + 0.42 * u, max(1.0, 0.018 * u), BLACK, "stem", "thin stem")
            for k in range(3):
                c.ellipse(ox + (0.15 + 0.12 * k) * u, oy + 0.85 * u, max(0.8, 0.018 * u), max(0.8, 0.018 * u), BLACK, "dot", "tiny dot")
    return c.finish()


GREYS = {"black": (12, 12, 12), "dark grey": (70, 70, 70), "mid grey": (128, 128, 128), "light grey": (200, 200, 200)}


def style_b(scale=1.0):
    """SYNTHETIC Style B: scattered daisies whose petals hold black / dark / mid / light grey zones over white,
    separate leaves, overlapping petals, white negative space."""
    W, H = int(320 * scale), int(240 * scale)
    c = Canvas(W, H)
    s = scale
    tones = list(GREYS.values())
    for j, (cx, cy, R) in enumerate(((90, 90, 60), (235, 150, 50), (250, 45, 28))):
        cx, cy, R = cx * s, cy * s, R * s
        for k in range(12):                               # outer petals, alternating tones, overlapping
            a = 2 * math.pi * k / 12
            c.petal(cx, cy, a, R, R * 0.36, tones[(k + j) % 3 + 1], f"petal tone{(k + j) % 3 + 1}")
        for k in range(12):                               # inner petal zones (a second tone inside each petal)
            a = 2 * math.pi * (k + 0.5) / 12
            c.petal(cx, cy, a, R * 0.55, R * 0.18, tones[(k + j + 1) % 3 + 1], "inner petal zone")
        c.ellipse(cx, cy, R * 0.22, R * 0.22, tones[0], "centre", "black centre")
        c.ellipse(cx, cy, R * 0.09, R * 0.09, (255, 255, 255), "centre", "white eye")
        for k in range(6):                                # tiny light dots on the black centre
            a = 2 * math.pi * k / 6
            c.ellipse(cx + math.cos(a) * R * 0.15, cy + math.sin(a) * R * 0.15, max(0.9, R * 0.025), max(0.9, R * 0.025),
                      tones[3], "dot", "light dot on centre")
    for k, (x, y) in enumerate(((180, 40), (40, 200), (150, 215))):
        c.petal(x * s, y * s, 0.6 + k, 40 * s, 14 * s, tones[0], "black leaf")
        c.line(x * s, y * s, (x + 26 * math.cos(0.6 + k)) * s, (y + 26 * math.sin(0.6 + k)) * s, max(1.0, 1.2 * s), (255, 255, 255), "vein", "white leaf vein")
    return c.finish()


def detail_metrics(objects, out_rgb, kinds=None):
    """Compare an output (H x W x 3, any colours) with the ground-truth objects.

    Per object (not the ground): `ok` = output pixels within DE_OK (CIEDE2000) of the object's colour, inside
    the object's mask grown 2 px. IoU of that with the true mask; survived when IoU >= 0.5. Area ratio and
    centroid shift on the same pixels. `merged`: two same-colour objects that do not touch in the truth but
    fall in one 8-connected component of their colour. Lines / gaps / rings: recall of their true pixels."""
    from scipy import ndimage
    out_lab = _lab(out_rgb.reshape(-1, 3).astype(np.float64)).reshape(out_rgb.shape)
    k8 = np.ones((3, 3), bool)
    colours = {}
    rows = []
    truth_cols = sorted({o["colour"] for o in objects})
    tl = _lab(np.array(truth_cols, np.float64))
    tol = {}
    for i, col in enumerate(truth_cols):       # "the object's colour": within DE_OK, and nearer it than any other true ink
        others = [de2000(tl[i], tl[j]) for j in range(len(truth_cols)) if j != i]
        tol[col] = min(DE_OK, 0.5 * min(others)) if others else DE_OK
    owner = np.zeros(out_rgb.shape[:2], np.int32)
    for o in objects[1:]:
        owner[o["mask"]] = o["id"]
    for ob in objects:
        if ob["id"] == 0 or (kinds and ob["kind"] not in kinds) or ob["area"] == 0:
            continue
        col = ob["colour"]
        if col not in colours:
            ok = de2000(out_lab, _lab(np.array(col, np.float64))[None, None]) < tol[col]
            colours[col] = (ok, ndimage.label(ok, structure=k8)[0])
        ok, lab = colours[col]
        m = ob["mask"]
        near = ndimage.binary_dilation(m, k8, iterations=2)
        same = np.zeros_like(m)
        for o in objects[1:]:                     # pixels the truth gives to another object of this colour are not ours
            if o["colour"] == col and o["id"] != ob["id"]:
                same |= o["mask"]
        pred = ok & near & ~same
        inter, union = int((pred & m).sum()), int((pred | m).sum())
        iou = inter / max(union, 1)
        ys, xs = np.nonzero(m)
        py, px = np.nonzero(pred)
        shift = float(np.hypot(py.mean() - ys.mean(), px.mean() - xs.mean())) if len(py) else float("nan")
        rows.append({"id": ob["id"], "name": ob["name"], "kind": ob["kind"], "area": ob["area"],
                     "iou": round(iou, 3), "recall": round(inter / max(ob["area"], 1), 3),
                     "area_ratio": round(len(py) / max(ob["area"], 1), 3), "centroid_shift_px": round(shift, 2),
                     "survived": iou >= 0.5, "_labels": set(np.unique(lab[m & ok]).tolist()) - {0}})
    # merges: two objects of one colour that lie in different connected parts of that colour in the truth
    # but in one 8-connected component of the output
    merged = 0
    by_id = {o["id"]: o for o in objects}
    by_col = {}
    for r in rows:
        by_col.setdefault(by_id[r["id"]]["colour"], []).append(r)
    for col, items in by_col.items():
        if col == objects[0]["colour"] or len(items) < 2:
            continue
        truth = np.zeros(out_rgb.shape[:2], bool)
        for o in objects:
            if o["colour"] == col and o["id"]:
                truth |= o["mask"]
        tl = ndimage.label(truth, structure=k8)[0]
        comp = {r["id"]: set(np.unique(tl[by_id[r["id"]]["mask"]]).tolist()) - {0} for r in items}
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if comp[a["id"]] & comp[b["id"]]:
                    continue
                if a["_labels"] & b["_labels"]:
                    merged += 1
    for r in rows:
        r.pop("_labels")
    n = max(len(rows), 1)
    surv = [r for r in rows if r["survived"]]
    linear = [r for r in rows if r["kind"] in ("line", "stem", "ring", "gap", "vein")]
    by_kind = {}
    for r in rows:
        k = by_kind.setdefault(r["kind"], [0, 0])
        k[0] += r["survived"]
        k[1] += 1
    return {
        "objects": len(rows),
        "survival_percent": round(100 * len(surv) / n, 1),
        "mean_iou": round(float(np.mean([r["iou"] for r in rows])) if rows else 0.0, 3),
        "mean_abs_area_error_percent": round(100 * float(np.mean([abs(r["area_ratio"] - 1) for r in surv])) if surv else float("nan"), 1),
        "mean_centroid_shift_px": round(float(np.nanmean([r["centroid_shift_px"] for r in surv])) if surv else float("nan"), 2),
        "merged_pairs": merged,
        "line_gap_retention_percent": round(100 * float(np.mean([r["recall"] for r in linear])) if linear else float("nan"), 1),
        "survival_by_kind": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_kind.items())},
        "rows": rows,
    }


def ideal_flat(objects, shape):
    """The truth drawn flat: every pixel in its owning object's colour (what a perfect separation gives)."""
    out = np.zeros(shape[:2] + (3,), np.uint8)
    out[:] = objects[0]["colour"]
    for o in objects[1:]:
        out[o["mask"]] = o["colour"]
    return out


def reconstruction_error(src_rgb, out_rgb):
    src, out = src_rgb.astype(np.float64), out_rgb.astype(np.float64)
    mse = float(np.mean((src - out) ** 2))
    psnr = 20 * math.log10(255.0 / max(math.sqrt(mse), 1e-8))
    step = max(1, src_rgb.size // 3 // 200_000)
    a = _lab(src_rgb.reshape(-1, 3)[::step].astype(np.float64))
    b = _lab(out_rgb.reshape(-1, 3)[::step].astype(np.float64))
    return {"psnr_db": round(psnr, 2), "mean_de2000": round(float(de2000(a, b).mean()), 2)}
