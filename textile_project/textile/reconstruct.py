"""Reconstruct a textile design: patch-work redrawn, spray left to the spray solver.

Patch work (anything with a hard boundary — ground, motifs, dots, stripes)
is redrawn at a higher resolution so neighbours share one clean edge.
Circles are redrawn as circles. Thin strokes are kept and, if they would
not print, thickened. They are never deleted.

Spray and gradients are NOT redrawn as flat patches. Their ink amount comes
from ``gradients.solve_tones``, the same continuous overprint ``spray`` uses.

A quality step runs first (Lanczos upscale + a light edge-preserving denoise)
so the tracer sees a defined boundary instead of JPEG stairs and brown fringe.
``--ai-upscale`` tries Real-ESRGAN and skips cleanly when it is not installed.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import auto as am
from . import gradients as gr
from . import names as ink_names
from . import spray as sp
from .io_utils import hex_of, save_png, to_image


def _empty_log(_msg: str) -> None:
    return None


def boost_for_boundaries(
    rgb: np.ndarray,
    scale: float,
    *,
    ai: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Raise resolution so a patch edge can be traced as one boundary.

    Returns ``(boundary_rgb, colour_rgb, meta)``.

    * ``colour_rgb`` is a plain Lanczos upscale — spray tones are solved on
      this, so a denoise pass cannot change how much ink a gradient has.
    * ``boundary_rgb`` is that upscale plus a light bilateral denoise, which
      knocks JPEG dirt off edges without inventing new petals.

    Real-ESRGAN replaces the boundary image only when ``ai`` is set and the
    package (and its weights) actually import. Otherwise the classical boost
    is used and ``meta['ai_skipped']`` says why.
    """
    scale = max(1.0, float(scale))
    h, w = rgb.shape[:2]
    ow, oh = int(round(w * scale)), int(round(h * scale))
    ow, oh = max(ow, w), max(oh, h)
    colour = np.asarray(
        Image.fromarray(rgb, mode="RGB").resize((ow, oh), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    meta: dict[str, Any] = {
        "method": "lanczos_bilateral",
        "ai": False,
        "scale": scale,
        "out_px": [ow, oh],
    }
    boundary = colour
    if ai:
        ai_img, why = _try_realesrgan(rgb, scale, (ow, oh))
        if ai_img is None:
            meta["ai_skipped"] = why
        else:
            boundary = ai_img
            meta["method"] = "realesrgan"
            meta["ai"] = True
    if meta["method"] != "realesrgan":
        import cv2
        boundary = cv2.bilateralFilter(colour, d=5, sigmaColor=18, sigmaSpace=2)
    return boundary, colour, meta


def _try_realesrgan(
    rgb: np.ndarray, scale: float, size_wh: tuple[int, int]
) -> tuple[np.ndarray | None, str]:
    """Return an upscaled RGB array, or (None, reason). Never raises."""
    weights = os.environ.get("REALESRGAN_WEIGHTS", "").strip()
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except Exception as exc:
        return None, f"Real-ESRGAN not installed ({type(exc).__name__})"
    if not weights or not os.path.isfile(weights):
        return None, "Real-ESRGAN installed but REALESRGAN_WEIGHTS is not a file; classical boost used"
    try:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        up = RealESRGANer(scale=4, model_path=weights, model=model, tile=256, tile_pad=10, half=False)
        bgr = rgb[:, :, ::-1]
        out, _ = up.enhance(bgr, outscale=float(scale))
        rgb_up = np.ascontiguousarray(out[:, :, ::-1])
        ow, oh = size_wh
        if rgb_up.shape[1] != ow or rgb_up.shape[0] != oh:
            rgb_up = np.asarray(
                Image.fromarray(rgb_up, mode="RGB").resize((ow, oh), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
        return rgb_up, ""
    except Exception as exc:
        return None, f"Real-ESRGAN failed ({type(exc).__name__}: {exc}); classical boost used"


def _merge_close(pal: np.ndarray, merge_de: float) -> np.ndarray:
    """Fold palette entries closer than ``merge_de`` (CIE76) into one ink."""
    pal = np.asarray(pal, np.uint8)
    if len(pal) <= 1:
        return pal
    labs = gr.rgb_to_lab(pal.reshape(-1, 1, 3)).reshape(-1, 3)
    parent = list(range(len(pal)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pal)):
        for j in range(i + 1, len(pal)):
            if math.sqrt(float(((labs[i] - labs[j]) ** 2).sum())) < merge_de:
                parent[find(j)] = find(i)
    keep: list[int] = []
    seen: set[int] = set()
    for i in range(len(pal)):
        root = find(i)
        if root not in seen:
            seen.add(root)
            keep.append(i)
    return pal[np.array(keep)]


def _between(p: np.ndarray, a: np.ndarray, b: np.ndarray, line_de: float = 12.0) -> bool:
    """True when Lab colour ``p`` sits on the segment between ``a`` and ``b``."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-6:
        return False
    t = float(np.dot(p - a, ab) / denom)
    if not (0.18 < t < 0.82):
        return False
    dist = float(np.linalg.norm(p - (a + t * ab)))
    if dist > line_de:
        return False
    if min(float(np.linalg.norm(p - a)), float(np.linalg.norm(p - b))) < 6.0:
        return False
    return True


def _mostly_edge(mask: np.ndarray) -> bool:
    if int(mask.sum()) == 0:
        return False
    er = mask.copy()
    er[1:] &= mask[:-1]
    er[:-1] &= mask[1:]
    er[:, 1:] &= mask[:, :-1]
    er[:, :-1] &= mask[:, 1:]
    return float((mask & ~er).sum()) / float(mask.sum()) > 0.62


def assign_inks(rgb: np.ndarray, inks: np.ndarray) -> np.ndarray:
    """Nearest ink in Lab, row-blocked so a large file is not one giant matrix."""
    h, w, _ = rgb.shape
    ink_lab = gr.rgb_to_lab(inks.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float32)
    out = np.empty((h, w), np.uint8)
    step = max(1, 750_000 // max(w, 1))
    for y in range(0, h, step):
        block = gr.rgb_to_lab(rgb[y:y + step]).reshape(-1, 3).astype(np.float32)
        d = ((block[:, None, :] - ink_lab[None, :, :]) ** 2).sum(-1)
        out[y:y + step] = d.argmin(1).astype(np.uint8).reshape(rgb[y:y + step].shape[:2])
    return out


def _core_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels that are already flat — not the antialiased lip of an edge."""
    lum = np.clip(gr._luminance(rgb), 0, 255).astype(np.uint8)
    return am._local_range(lum, 3) <= 3.0


def inks_from_image(rgb: np.ndarray, colors: int, merge_de: float = 6.0) -> np.ndarray:
    """Real inks, taken from flat cores so a JPEG fringe never becomes an ink.

    Near-duplicates (dE76 < 6) merge. A thin stroke whose colour is NOT a mix
    of its neighbours (a real 1px stem) is added back — fringe mixes are not.
    """
    colors = max(1, int(colors))
    core = _core_mask(rgb)
    if float(core.mean()) >= 0.25 and int(core.sum()) >= 16:
        pix = rgb[core]
    else:
        pix = rgb.reshape(-1, 3)
    uniq, counts = np.unique(pix.reshape(-1, 3), axis=0, return_counts=True)
    if len(uniq) <= colors:
        pal = uniq[np.argsort(-counts)].astype(np.uint8)
    else:
        from sklearn.cluster import KMeans
        flat = pix.reshape(-1, 3)
        take = min(len(flat), 20_000)
        sample = flat[np.random.default_rng(0).choice(len(flat), take, replace=False)]
        # Cluster in Lab so a fringe mix is not nearer than a real ink.
        sample_lab = gr.rgb_to_lab(sample.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float32)
        n_clusters = min(colors, len(uniq))
        km = KMeans(n_clusters=n_clusters, n_init=4, random_state=0).fit(sample_lab)
        centres = []
        for i in range(n_clusters):
            mem = sample[km.labels_ == i]
            if len(mem) == 0:
                continue
            centres.append(np.median(mem, axis=0))
        pal = np.unique(np.clip(np.rint(np.stack(centres)), 0, 255).astype(np.uint8), axis=0)
    pal = _merge_close(pal, merge_de)
    pal = _drop_bridges(rgb, pal)
    pal = _reject_core_bridges(pal)
    pal = _rescue_thin_inks(rgb, pal, colors, merge_de)
    if len(pal) == 0:
        pal = np.array([[0, 0, 0]], np.uint8)
    return pal


def _reject_core_bridges(pal: np.ndarray) -> np.ndarray:
    """Drop a small palette entry that is only the mix of two bigger inks.

    This does not look at the picture, so a soft edge that happens to be flat
    enough to enter the core can never become a third ink.
    """
    if len(pal) < 3:
        return pal
    labs = gr.rgb_to_lab(pal.reshape(-1, 1, 3)).reshape(-1, 3)
    # Prefer keeping the more extreme inks (ends of the cloud, not the middle).
    spread = np.linalg.norm(labs - labs.mean(0), axis=1)
    keep = np.ones(len(pal), bool)
    for i in np.argsort(spread):
        others = [j for j in range(len(pal)) if keep[j] and j != i]
        if len(others) < 2:
            continue
        ol = labs[others]
        order = np.argsort(np.linalg.norm(ol - labs[i], axis=1))
        if _between(labs[i], ol[int(order[0])], ol[int(order[1])]):
            keep[i] = False
    if int(keep.sum()) == 0:
        return pal[:1]
    return pal[keep]


def _drop_bridges(rgb: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Remove an ink that only lives on the blend between two bigger inks."""
    changed = True
    while changed and len(pal) > 1:
        changed = False
        index = assign_inks(rgb, pal)
        cnt = np.bincount(index.ravel(), minlength=len(pal))
        labs = gr.rgb_to_lab(pal.reshape(-1, 1, 3)).reshape(-1, 3)
        for i in np.argsort(cnt):
            if cnt[i] == 0 or cnt[i] > 0.18 * max(int(cnt.sum()), 1):
                continue
            bigger = [j for j in range(len(pal)) if j != i and cnt[j] > cnt[i] * 1.4]
            bridge = False
            for a in range(len(bigger)):
                for b in range(a + 1, len(bigger)):
                    if _between(labs[i], labs[bigger[a]], labs[bigger[b]]):
                        bridge = True
                        break
                if bridge:
                    break
            if not bridge or not _mostly_edge(index == i):
                continue
            pal = np.delete(pal, int(i), axis=0)
            changed = True
            break
    return pal


def _nearest_bridge(med_lab: np.ndarray, ink_lab: np.ndarray) -> bool:
    """A fringe mix lies on the Lab segment of the two inks it is closest to.

    The morphological ring sometimes only sees one side of a soft edge, so the
    neighbour list alone is not enough — the colour itself has to fail the
    bridge test against the palette.
    """
    if len(ink_lab) < 2:
        return False
    order = np.argsort(np.linalg.norm(ink_lab - med_lab, axis=1))
    a, b = int(order[0]), int(order[1])
    return _between(med_lab, ink_lab[a], ink_lab[b])


def _rescue_thin_inks(rgb: np.ndarray, pal: np.ndarray, colors: int, merge_de: float) -> np.ndarray:
    """A real hairline has no flat core. Add its colour unless it is just a fringe mix."""
    if len(pal) >= colors:
        return pal
    index = assign_inks(rgb, pal)
    ink_lab = gr.rgb_to_lab(pal.reshape(-1, 1, 3)).reshape(-1, 3)
    pix_lab = gr.rgb_to_lab(rgb).reshape(-1, 3)
    de = np.sqrt(((pix_lab - ink_lab[index.ravel()]) ** 2).sum(1)).reshape(index.shape)
    far = de > 16.0
    if not far.any():
        return pal
    lab, n = _components(far)
    added: list[np.ndarray] = []
    from scipy import ndimage
    slices = ndimage.find_objects(lab)
    for c in range(1, n + 1):
        if slices[c - 1] is None:
            continue
        sl = slices[c - 1]
        sub = lab[sl] == c
        area = int(sub.sum())
        if area < 6:
            continue
        m = np.zeros(index.shape, bool)
        m[sl] = sub
        med = np.median(rgb[m].astype(np.float64), axis=0)
        med_u8 = np.clip(np.rint(med), 0, 255).astype(np.uint8)
        med_lab = gr.rgb_to_lab(med_u8.reshape(1, 1, 3)).reshape(3)
        if float(np.min(np.linalg.norm(ink_lab - med_lab, axis=1))) < merge_de:
            continue
        if _nearest_bridge(med_lab, ink_lab):
            continue
        ring = gr._dilate_bool(m, 1) & ~m
        nbs = np.unique(index[ring]) if ring.any() else np.array([], np.int32)
        if len(nbs) >= 2:
            bridged = False
            for a in range(len(nbs)):
                for b in range(a + 1, len(nbs)):
                    if _between(med_lab, ink_lab[int(nbs[a])], ink_lab[int(nbs[b])]):
                        bridged = True
                        break
                if bridged:
                    break
            if bridged:
                continue
        added.append(med_u8)
        if len(pal) + len(added) >= colors:
            break
    if not added:
        return pal
    pal = np.vstack([pal, np.stack(added)])
    return _merge_close(pal, merge_de)


def snap_fringe(index: np.ndarray, inks: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """An antialiased pixel between two inks joins one of those two, never a third."""
    out = index.copy()
    shifts = []
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        shifts.append(np.roll(np.roll(index, dy, 0), dx, 1))
    # Pixels whose neighbours are only one or two inks, and which disagree.
    a = shifts[0]
    same = np.ones(index.shape, bool)
    for s in shifts[1:]:
        same &= s == a
    mixed = ~same
    if not mixed.any():
        return out
    ink_lab = gr.rgb_to_lab(inks.reshape(-1, 1, 3)).reshape(-1, 3)
    pix = gr.rgb_to_lab(rgb).reshape(-1, 3)
    ys, xs = np.nonzero(mixed)
    # Work in chunks. Candidate inks = the neighbour labels at that pixel.
    nb = np.stack([s[ys, xs] for s in shifts], axis=1)
    own = index[ys, xs]
    for n in range(0, len(ys), 8000):
        sl = slice(n, n + 8000)
        rows = nb[sl]
        choose = own[sl].copy()
        lab_rows = pix[ys[sl] * index.shape[1] + xs[sl]]
        for t in range(rows.shape[0]):
            opts = np.unique(rows[t])
            if len(opts) == 1:
                choose[t] = int(opts[0])
                continue
            if len(opts) > 2:
                # More than two neighbours: still refuse a label that isn't one of them.
                if choose[t] in opts:
                    continue
                opts = opts[:2]
            d = ((ink_lab[opts] - lab_rows[t]) ** 2).sum(-1)
            choose[t] = int(opts[int(d.argmin())])
        out[ys[sl], xs[sl]] = choose
    return out


def _components(mask: np.ndarray):
    from scipy import ndimage
    return ndimage.label(mask, structure=np.ones((3, 3), np.int8))


def _boundary_cv(sub: np.ndarray) -> float:
    """How much the outline radius wanders. A circle is near 0, a box is not."""
    er = sub.copy()
    er[1:] &= sub[:-1]
    er[:-1] &= sub[1:]
    er[:, 1:] &= sub[:, :-1]
    er[:, :-1] &= sub[:, 1:]
    edge = sub & ~er
    ys, xs = np.nonzero(edge)
    if len(ys) < 6:
        return 1.0
    fill_y, fill_x = np.nonzero(sub)
    cy, cx = float(fill_y.mean()), float(fill_x.mean())
    rr = np.hypot(ys - cy, xs - cx)
    return float(rr.std()) / (float(rr.mean()) + 1e-6)


def find_dots(index: np.ndarray, *, max_frac: float = 0.40) -> list[dict[str, float]]:
    """Compact round blobs. Same detector on the source and on the redraw."""
    h, w = index.shape
    total = h * w
    dots: list[dict[str, float]] = []
    for k in range(int(index.max()) + 1):
        lab, n = _components(index == k)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        slices = None
        from scipy import ndimage
        slices = ndimage.find_objects(lab)
        for c in range(1, n + 1):
            area = int(sizes[c])
            if area < 6 or area > max_frac * total:
                continue
            sl = slices[c - 1]
            if sl is None:
                continue
            sub = lab[sl] == c
            ys, xs = np.nonzero(sub)
            bh = sl[0].stop - sl[0].start
            bw = sl[1].stop - sl[1].start
            aspect = max(bh, bw) / max(1, min(bh, bw))
            if aspect >= 1.55:
                continue
            cv = _boundary_cv(sub)
            # A 3px dot is blocky. A large box is not a dot, even if it is compact.
            limit = 0.36 if area < 48 else 0.12
            if cv > limit or aspect >= 1.40:
                continue
            dots.append({
                "ink": float(k),
                "cy": float(ys.mean() + sl[0].start),
                "cx": float(xs.mean() + sl[1].start),
                "r": float(math.sqrt(area / math.pi)),
                "area": float(area),
            })
    return dots


def _is_thin_stroke(sub: np.ndarray, longer: int) -> bool:
    """True when the shape is a long ribbon, even after we thickened the paint.

    Erosion does not thin a filled bar, so the test is the short side of the
    bounding box against the long side — a hairline we widened to ~14 px is
    still a stroke, a filled rectangle of a motif is not.
    """
    if longer < 8 or int(sub.sum()) < 8:
        return False
    bh, bw = sub.shape
    shorter = max(1, min(bh, bw))
    if longer / shorter < 4.0:
        return False
    # A stroke fills its ribbon. A diagonal scatter or a fringe does not.
    return int(sub.sum()) >= 0.45 * longer * shorter


def find_lines(index: np.ndarray) -> list[dict[str, Any]]:
    """Thin or long strokes. A thickened printable stroke still counts."""
    h, w = index.shape
    total = h * w
    lines: list[dict[str, Any]] = []
    from scipy import ndimage
    for k in range(int(index.max()) + 1):
        lab, n = _components(index == k)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        slices = ndimage.find_objects(lab)
        for c in range(1, n + 1):
            area = int(sizes[c])
            if area < 8 or area > 0.45 * total:
                continue
            sl = slices[c - 1]
            if sl is None:
                continue
            bh = sl[0].stop - sl[0].start
            bw = sl[1].stop - sl[1].start
            longer = max(bh, bw)
            shorter = max(1, min(bh, bw))
            if longer < 8 or longer / shorter < 4.0:
                continue
            sub = lab[sl] == c
            ys, xs = np.nonzero(sub)
            # A soft edge's antialias band is long, but it is not a stroke.
            # A real hairline is thin. A stroke we thickened on purpose is
            # still a stroke: its skeleton stays thin even when the paint is wide.
            skel = _is_thin_stroke(sub, longer)
            if area / longer > 4.5 and not skel:
                continue
            horizontal = bw >= bh
            if horizontal:
                keys = xs
                other = ys
            else:
                keys = ys
                other = xs
            order = np.argsort(keys)
            keys = keys[order]
            other = other[order]
            # centreline: mean of the short axis at each long-axis pixel
            uniq, start = np.unique(keys, return_index=True)
            centres = []
            widths = []
            for i, u in enumerate(uniq):
                a = start[i]
                b = start[i + 1] if i + 1 < len(start) else len(keys)
                centres.append(float(other[a:b].mean()))
                widths.append(int(b - a))
            if horizontal:
                pts = np.stack([uniq.astype(np.float64) + sl[1].start,
                                np.array(centres) + sl[0].start], 1)
            else:
                pts = np.stack([np.array(centres) + sl[1].start,
                                uniq.astype(np.float64) + sl[0].start], 1)
            lines.append({
                "ink": int(k),
                "pts": pts,
                "width": float(np.median(widths)),
                "length": float(longer),
            })
    return lines


def _smooth_open(pts: np.ndarray) -> np.ndarray:
    """Smooth a stroke but keep a sharp corner (a real tip stays a tip)."""
    if len(pts) < 5:
        return pts
    v1 = pts[1:-1] - pts[:-2]
    v2 = pts[2:] - pts[1:-1]
    n1 = np.linalg.norm(v1, axis=1) + 1e-6
    n2 = np.linalg.norm(v2, axis=1) + 1e-6
    cos = np.clip((v1 * v2).sum(1) / (n1 * n2), -1, 1)
    sharp = np.degrees(np.arccos(cos)) > 55
    sharp = np.concatenate([[True], sharp, [True]])
    out = pts.copy()
    i = 0
    while i < len(pts):
        j = i + 1
        while j < len(pts) and not sharp[j]:
            j += 1
        run = pts[i:j + 1]
        if len(run) >= 5:
            pad = np.pad(run, ((2, 2), (0, 0)), mode="edge")
            ker = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
            sm = np.stack([np.convolve(pad[:, a], ker, mode="valid") for a in (0, 1)], 1)
            # pin the ends (the corner, or the stroke end)
            sm[0] = run[0]
            sm[-1] = run[-1]
            out[i:j + 1] = sm
        i = max(j, i + 1)
    return out


def _paint_disk(index: np.ndarray, cy: float, cx: float, r: float, ink: int) -> None:
    r = max(float(r), 0.6)
    y0 = max(0, int(math.floor(cy - r - 1)))
    y1 = min(index.shape[0], int(math.ceil(cy + r + 2)))
    x0 = max(0, int(math.floor(cx - r - 1)))
    x1 = min(index.shape[1], int(math.ceil(cx + r + 2)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    index[y0:y1, x0:x1][m] = np.uint8(ink)


def _paint_stroke(index: np.ndarray, pts: np.ndarray, width: float, ink: int) -> None:
    if len(pts) < 2:
        return
    im = Image.fromarray(index, mode="L")
    dr = ImageDraw.Draw(im)
    seq = [tuple(float(v) for v in p) for p in pts]
    dr.line(seq, fill=int(ink), width=max(1, int(round(width))), joint="curve")
    index[:] = np.asarray(im, dtype=np.uint8)


def _refine_boundary(labels: np.ndarray, colour: np.ndarray, inks: np.ndarray) -> np.ndarray:
    """Move a shared edge onto the boosted image, without inventing an ink.

    Only a pixel that already touches a different ink may switch, and only to
    that neighbour. The boosted colour decides which of those two it is, so a
    JPEG fringe cannot open a third patch.
    """
    if colour.shape[:2] != labels.shape:
        colour = np.asarray(
            Image.fromarray(colour, mode="RGB").resize(
                (labels.shape[1], labels.shape[0]), Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )
    ink_lab = gr.rgb_to_lab(inks.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float32)
    out = labels.copy()
    h, w = out.shape
    step = max(1, 750_000 // max(w, 1))
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = np.roll(np.roll(out, dy, 0), dx, 1)
        border = nb != out
        if dy > 0:
            border[:dy] = False
        elif dy < 0:
            border[dy:] = False
        if dx > 0:
            border[:, :dx] = False
        elif dx < 0:
            border[:, dx:] = False
        if not border.any():
            continue
        for y in range(0, h, step):
            b = border[y:y + step]
            if not b.any():
                continue
            own = out[y:y + step]
            nbs = nb[y:y + step]
            block = gr.rgb_to_lab(colour[y:y + step]).reshape(-1, 3).astype(np.float32)
            d_own = ((block - ink_lab[own.ravel()]) ** 2).sum(1).reshape(own.shape)
            d_nb = ((block - ink_lab[nbs.ravel()]) ** 2).sum(1).reshape(own.shape)
            take = b & (d_nb < d_own)
            if take.any():
                own = own.copy()
                own[take] = nbs[take]
                out[y:y + step] = own
    return out


def redraw_patches(
    index_src: np.ndarray,
    colour_hi: np.ndarray,
    inks: np.ndarray,
    scale: float,
    dots: list[dict[str, float]],
    lines: list[dict[str, Any]],
    min_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared-boundary redraw at output resolution, then primitive restamp.

    Labels come from the source, then the SDF (one distance field per ink,
    smoothed, argmax) is evaluated at the output grid so neighbours share one
    edge. The boosted colour only slides that edge onto the real boundary —
    it is not re-clustered, so a soft upscale cannot invent islands.

    Dots and strokes measured on the source are painted back afterwards so the
    smooth step cannot eat them, and so a dot is a circle rather than stairs.
    """
    hi, _erep = am.smooth_index_sdf(index_src, scale=float(scale), sigma_px=0.85, log=_empty_log)
    if hi.shape[0] != colour_hi.shape[0] or hi.shape[1] != colour_hi.shape[1]:
        hi = np.asarray(
            Image.fromarray(hi, mode="L").resize(
                (colour_hi.shape[1], colour_hi.shape[0]), Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        )
    hi = _refine_boundary(hi, colour_hi, inks)
    before = hi.copy()
    s = float(scale)
    for ln in lines:
        pts = _smooth_open(np.asarray(ln["pts"], np.float64) * s)
        width = max(float(ln["width"]) * s, float(min_px))
        _paint_stroke(hi, pts, width, int(ln["ink"]))
    for d in dots:
        r = float(d["r"]) * s
        if 2.0 * r < min_px:
            r = min_px / 2.0
        _paint_disk(hi, float(d["cy"]) * s, float(d["cx"]) * s, r, int(d["ink"]))
    stamped = hi != before
    return hi, stamped


def shade_mask_of(src: np.ndarray, flat: np.ndarray, inks: np.ndarray) -> np.ndarray:
    """Where the source is actually sprayed / graded, not a hard patch edge.

    Uses the same FLAT / LINE / SHADE split as ``auto``. A smooth full-frame
    ramp that the flat posteriser swallowed is forced back to shade — otherwise
    a gradient design would be redrawn as stripes.
    """
    regions = am.classify_auto_regions(src, flat, inks)
    shade = regions["shade_mask"].copy()
    lum = np.clip(gr._luminance(src), 0, 255).astype(np.uint8)
    r3 = am._local_range(lum, 3)
    r15 = am._local_range(lum, 15)
    hard = (r3 >= 8.0) & (r15 <= r3 + 6.0)
    soft = (r15 >= 10.0) & (r3 <= 16.0) & ~hard
    if float(shade.mean()) < 0.08 and float(soft.mean()) > 0.45 and float(hard.mean()) < 0.05:
        shade = soft
    shade = shade & ~gr._dilate_bool(hard, 1)
    # A few soft pixels on a hard edge are a halo, not a spray region.
    if float(shade.mean()) < 0.03:
        shade = np.zeros_like(shade)
    return shade


def _solve_tiled(rgb: np.ndarray, inks: np.ndarray, band: int = 180) -> np.ndarray:
    """``solve_tones`` in row bands (the spray solver, same math, less RAM)."""
    h, w, _ = rgb.shape
    k = len(inks)
    out = np.zeros((h, w, k), np.float32)
    overlap = 8
    y = 0
    while y < h:
        y1 = min(h, y + band)
        y0 = max(0, y - overlap)
        y1b = min(h, y1 + overlap)
        part = gr.solve_tones(
            rgb[y0:y1b], inks, paper=(255, 255, 255),
            max_inks_per_pixel=k, iters=80, seal_motifs=False,
        )
        a = y - y0
        b = a + (y1 - y)
        out[y:y1] = part[a:b].astype(np.float32)
        y = y1
    return np.clip(out, 0.0, 1.0)


def _up_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    ow, oh = size_wh
    im = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    up = im.resize((ow, oh), Image.Resampling.NEAREST)
    return np.asarray(up) > 127


def _count_pair(src_index: np.ndarray, out_index: np.ndarray) -> dict[str, int]:
    return {
        "dots_source": len(find_dots(src_index)),
        "dots_output": len(find_dots(out_index)),
        "lines_source": len(find_lines(src_index)),
        "lines_output": len(find_lines(out_index)),
    }


def _flat_stats(tones: np.ndarray, labels: np.ndarray, patch: np.ndarray, composite: np.ndarray) -> dict[str, float]:
    if not patch.any():
        return {"flat_std": 0.0, "flat_stray_tone_percent": 0.0}
    # Interior of the biggest patch ink — edge pixels are still flat ink, so
    # the whole patch must be a single RGB if tones are truly solid.
    k = tones.shape[-1]
    counts = np.bincount(labels[patch].ravel(), minlength=k)
    ink = int(counts.argmax())
    region = patch & (labels == ink)
    if region.sum() < 10:
        region = patch
    region_px = composite[region].astype(np.float64).reshape(-1, 3)
    std = float(region_px.std(axis=0).max()) if len(region_px) else 0.0
    # Stray = a patch pixel whose plate is not a clean 0/1 of its own ink.
    own = tones[patch, labels[patch]]
    others_max = np.zeros(int(patch.sum()), np.float32)
    # max of the other inks
    flat_t = tones[patch]
    idx = labels[patch]
    flat_t = flat_t.copy()
    flat_t[np.arange(len(idx)), idx] = 0
    others_max = flat_t.max(1) if flat_t.size else others_max
    stray = (own < 0.98) | (others_max > 0.02)
    return {
        "flat_std": round(std, 5),
        "flat_stray_tone_percent": round(float(100.0 * stray.mean()) if stray.size else 0.0, 5),
    }


def _edge_deviation(labels: np.ndarray) -> float:
    """Max radial error, in px, of round blobs vs a true circle. 0 if none."""
    devs = []
    for d in find_dots(labels):
        ink = int(d["ink"])
        cy, cx, r = d["cy"], d["cx"], max(d["r"], 1.0)
        y0 = max(0, int(cy - r - 3))
        y1 = min(labels.shape[0], int(cy + r + 4))
        x0 = max(0, int(cx - r - 3))
        x1 = min(labels.shape[1], int(cx + r + 4))
        sub = labels[y0:y1, x0:x1] == ink
        er = sub.copy()
        er[1:] &= sub[:-1]
        er[:-1] &= sub[1:]
        er[:, 1:] &= sub[:, :-1]
        er[:, :-1] &= sub[:, 1:]
        edge = sub & ~er
        ys, xs = np.nonzero(edge)
        if len(ys) < 8:
            continue
        rr = np.hypot(ys + y0 - cy, xs + x0 - cx)
        devs.append(float(np.max(np.abs(rr - r))))
    return round(max(devs) if devs else 0.0, 4)


def run_reconstruct(
    image_path: str,
    out_dir: str,
    *,
    name: str | None = None,
    colors: int = 8,
    scale: float = 2.0,
    dpi: int = 300,
    min_feature_mm: float = 0.15,
    ai_upscale: bool = False,
    views: bool = False,
    compare_auto: bool = False,
    log=print,
) -> dict[str, Any]:
    """Full reconstruct. Output is never smaller than the source.

    Patch plates are solid 0/255. Spray plates are 8-bit, and only where the
    source is actually shaded. Channels are written darkest-first, no ink layers.
    """
    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    base = name or os.path.splitext(os.path.basename(image_path))[0]
    base = base.replace("#", "").replace(" ", "_") or "reconstruct"
    src_im = Image.open(image_path).convert("RGB")
    src = np.asarray(src_im, dtype=np.uint8)
    h, w = src.shape[:2]
    scale = max(1.0, float(scale))
    min_px = max(1.0, float(min_feature_mm) / 25.4 * float(dpi))

    log(f"[reconstruct] inks from source ({w}x{h}, colors<={colors})")
    inks = inks_from_image(src, colors)
    labels_src = snap_fringe(assign_inks(src, inks), inks, src)
    # JPEG crumbs are not patches. Compact dots and real strokes stay.
    labels_src, _crumbs = am.remove_edge_specks(labels_src, max_px=8)
    # Detail map BEFORE the quality boost. The boost is allowed to clean
    # edges; it is not allowed to forget a dot or a hairline.
    dots = find_dots(labels_src)
    lines = find_lines(labels_src)
    log(f"[reconstruct] protected dots={len(dots)} lines={len(lines)}")

    boundary, colour, boost_meta = boost_for_boundaries(src, scale, ai=ai_upscale)
    oh, ow = boundary.shape[:2]
    log(f"[reconstruct] quality boost {boost_meta['method']} -> {ow}x{oh}")
    # Patch plates are the source labels, redrawn at output size. The boosted
    # image only tells the tracer where the edge sits.
    labels, stamped = redraw_patches(labels_src, boundary, inks, scale, dots, lines, min_px)

    flat_src = am.index_to_flat(labels_src, inks)
    shade_src = shade_mask_of(src, flat_src, inks)
    shade = _up_mask(shade_src, (ow, oh))
    # The lip of a hard edge stays a patch, so the boundary stays defined.
    shade = gr._erode_bool(shade, 1) if shade.any() else shade
    shade = shade & ~stamped
    if float(shade.mean()) < 0.03:
        shade = np.zeros(shade.shape, dtype=bool)

    if shade.any():
        log(f"[reconstruct] spray solver on {float(shade.mean())*100:.1f}% (same tone math as spray)")
        solved = _solve_tiled(colour, inks)
    else:
        log("[reconstruct] no spray in this design — solid plates only")
        solved = None

    k = len(inks)
    tones = np.zeros((oh, ow, k), np.float32)
    patch = ~shade
    if solved is not None:
        tones[shade] = solved[shade]
    tones[patch] = 0.0
    if patch.any():
        tones[patch, labels[patch]] = 1.0

    # Drop inks that print nothing.
    keep = [i for i in range(k) if float(tones[..., i].max()) > 0.004]
    if not keep:
        keep = [int(np.argmax(tones.reshape(-1, k).max(0)))] if k else []
    inks_k = inks[np.array(keep)] if keep else inks
    tones_k = tones[..., keep] if keep else tones
    # Remap labels into the kept palette for the stats / preview consistency.
    remap = np.full(k, 0, np.uint8)
    for ni, oi in enumerate(keep):
        remap[oi] = ni
    labels_k = remap[labels]

    paper = (255, 255, 255)
    composite = gr.overprint_rgb(tones_k.astype(np.float64), inks_k, paper)
    # de against the Lanczos source (same pixels, no extra blur)
    de_all = float(gr.delta_e_cie76(colour, composite))
    # Flat dE is the interior. The Lanczos source is soft on the lip; the
    # redraw is a hard edge there on purpose, so the lip is not "flat error".
    interior = gr._erode_bool(patch, 2) if patch.any() else patch
    de_flat = float(gr.delta_e_cie76(colour, composite, interior)) if interior.any() else de_all
    de_shade = float(gr.delta_e_cie76(colour, composite, shade)) if shade.any() else 0.0

    counts = _count_pair(labels_src, labels_k)
    stats = _flat_stats(tones_k, labels_k, patch, composite)
    # Banding on the spray composite only. A solid design has no bands.
    if shade.any():
        band = float(gr.band_step_metric(composite))
    else:
        band = 0.0

    rdir = os.path.join(out_dir, "reconstruct")
    os.makedirs(rdir, exist_ok=True)
    tone_dir = os.path.join(rdir, "tone_channels")
    os.makedirs(tone_dir, exist_ok=True)
    plates = []
    for i, ink in enumerate(inks_k):
        hx = hex_of(ink)
        cname = ink_names.colour_name(ink)
        tone_u8 = np.clip(np.rint(tones_k[..., i] * 255.0), 0, 255).astype(np.uint8)
        tone_path = os.path.join(tone_dir, f"TONE_{i+1:02d}_{hx}.png")
        save_png(Image.fromarray(tone_u8, mode="L"), tone_path, dpi)
        plates.append({
            "name": f"INK_{i+1:02d} {cname}",
            "hex": "#" + hx,
            "ink_rgb": [int(x) for x in ink],
            "tone_path": tone_path,
            "mean_tone_percent": round(float(tones_k[..., i].mean() * 100), 3),
            "role": "tone" if int(((tone_u8 > 0) & (tone_u8 < 255)).any()) else "spot",
        })

    comp_path = os.path.join(rdir, f"{base}_reconstruct.png")
    save_png(to_image(composite), comp_path, dpi)
    pkg = sp.write_tone_package(
        out_dir, base, plates, dpi=dpi, write_psd=True,
        composite_path=comp_path, include_composite=False, use_layers=False, log=log,
    )
    # Pretty channel list, darkest first (write_tone_package already sorted the files).
    order = sorted(range(len(plates)), key=lambda i: gr._lum1(inks_k[i]))
    channels = [
        f"INK_{n+1:02d} {ink_names.colour_name(inks_k[i])} #{hex_of(inks_k[i])}"
        for n, i in enumerate(order)
    ]

    import resource
    rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports kilobytes.
    peak_mb = round(rss_kb / 1024.0, 1)

    report: dict[str, Any] = {
        "name": base,
        "mode": "reconstruct",
        "applied": True,
        "source_px": [w, h],
        "size_px": [ow, oh],
        "scale": scale,
        "dpi": dpi,
        "min_feature_mm": min_feature_mm,
        "min_feature_px": round(min_px, 3),
        "quality_boost": boost_meta,
        "n_inks": len(inks_k),
        "channels": channels,
        "channel_roles": [plates[i]["role"] for i in order],
        "spray_percent": round(float(100.0 * shade.mean()), 3),
        "tone_pixels": int(((tones_k > 0.02) & (tones_k < 0.98)).any(-1).sum()),
        "delta_e_cie76": {
            "overall_mean": round(de_all, 3),
            "flat_mean": round(de_flat, 3),
            "shade_mean": round(de_shade, 3),
        },
        "counts": counts,
        "flat_std": stats["flat_std"],
        "flat_stray_tone_percent": stats["flat_stray_tone_percent"],
        "band_step": round(band, 3),
        "edge_deviation_px": _edge_deviation(labels_k),
        "output_ge_source": bool(ow >= w and oh >= h),
        "package": {k2: pkg.get(k2) for k2 in ("psd", "spot_tif", "channels_zip", "channel_count")},
        "preview": comp_path,
        "peak_ram_mb": peak_mb,
        "runtime_s": round(time.perf_counter() - t0, 3),
    }
    if views:
        try:
            view_paths = make_views(
                src, composite, labels_k, dots, image_path, out_dir, base,
                scale=scale, compare_auto=compare_auto, log=log,
            )
            report["views"] = view_paths
        except Exception as exc:
            log(f"[reconstruct] views skipped: {type(exc).__name__}: {exc}")
            report["views_error"] = str(exc)

    rp = os.path.join(out_dir, "report.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    report["report"] = rp
    log(
        f"[reconstruct] inks={report['n_inks']} spray={report['spray_percent']}% "
        f"dE={de_all:.2f} flat_std={stats['flat_std']} "
        f"dots {counts['dots_source']}->{counts['dots_output']} "
        f"lines {counts['lines_source']}->{counts['lines_output']}"
    )
    return report


def _panel_row(images: list[np.ndarray], titles: list[str], height: int) -> Image.Image:
    resized = []
    for im in images:
        pil = Image.fromarray(im, mode="RGB")
        pil = pil.resize((max(1, int(round(pil.width * height / pil.height))), height), Image.Resampling.NEAREST)
        resized.append(pil)
    gap = 8
    bar = 22
    w = sum(p.width for p in resized) + gap * (len(resized) + 1)
    sheet = Image.new("RGB", (w, height + bar + gap), (24, 22, 20))
    dr = ImageDraw.Draw(sheet)
    x = gap
    for pil, title in zip(resized, titles):
        sheet.paste(pil, (x, bar))
        dr.text((x, 4), title, fill=(255, 246, 230))
        x += pil.width + gap
    return sheet


def make_views(
    src: np.ndarray,
    composite: np.ndarray,
    labels: np.ndarray,
    dots: list[dict[str, float]],
    image_path: str,
    out_dir: str,
    base: str,
    *,
    scale: float,
    compare_auto: bool,
    log=print,
) -> dict[str, str]:
    """compare.jpg plus 4x zooms. Auto is only run on a small crop, and only
    when ``compare_auto`` is set — it calls the existing ``auto`` pipeline
    unchanged, which is too heavy to run on every test.
    """
    h, w = src.shape[:2]
    # Side-by-side at a viewable width. Nearest on the source so stairs stay
    # visible; the reconstruct is already smooth, nearest does not add stairs.
    view_h = min(520, max(h, composite.shape[0]))
    compare = _panel_row(
        [src, composite],
        ["source", "reconstruct"],
        view_h,
    )
    compare_path = os.path.join(out_dir, "compare.jpg")
    compare.save(compare_path, quality=90)

    def crop_of(arr, y0, x0, y1, x1):
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(arr.shape[0], y1), min(arr.shape[1], x1)
        if y1 <= y0 or x1 <= x0:
            y0, x0, y1, x1 = 0, 0, min(arr.shape[0], 24), min(arr.shape[1], 24)
        return arr[y0:y1, x0:x1]

    # An edge crop: a label change near the middle.
    edge_y, edge_x = h // 2, w // 2
    ys, xs = np.nonzero(labels[:: max(1, int(scale))] != np.roll(labels[:: max(1, int(scale))], 1, 1))
    if len(ys):
        mid = len(ys) // 2
        edge_y = int(ys[mid] * max(1, int(round(scale))) / max(scale, 1) ) if False else int(ys[mid])
        edge_x = int(xs[mid])
    box = 36
    src_edge = crop_of(src, edge_y - box, edge_x - box, edge_y + box, edge_x + box)
    # Matching window in the reconstruct (scale times larger).
    s = float(scale)
    ry, rx = int(round(edge_y * s)), int(round(edge_x * s))
    rb = int(round(box * s))
    rec_edge = crop_of(composite, ry - rb, rx - rb, ry + rb, rx + rb)

    panels_e = [src_edge, rec_edge]
    titles_e = ["source 4x", "reconstruct"]
    if compare_auto:
        auto_crop = _auto_crop(image_path, edge_y, edge_x, box, log)
        if auto_crop is not None:
            panels_e = [src_edge, auto_crop, rec_edge]
            titles_e = ["source 4x", "auto", "reconstruct"]
    zoom_e = _panel_row(panels_e, titles_e, 144)
    zoom_e_path = os.path.join(out_dir, "zoom_edges.jpg")
    zoom_e.save(zoom_e_path, quality=90)

    if dots:
        d0 = dots[0]
        dy, dx = int(d0["cy"]), int(d0["cx"])
    else:
        dy, dx = edge_y, edge_x
    src_d = crop_of(src, dy - box, dx - box, dy + box, dx + box)
    ry, rx = int(round(dy * s)), int(round(dx * s))
    rec_d = crop_of(composite, ry - rb, rx - rb, ry + rb, rx + rb)
    panels_d = [src_d, rec_d]
    titles_d = ["source 4x", "reconstruct"]
    if compare_auto:
        auto_crop = _auto_crop(image_path, dy, dx, box, log)
        if auto_crop is not None:
            panels_d = [src_d, auto_crop, rec_d]
            titles_d = ["source 4x", "auto", "reconstruct"]
    zoom_d = _panel_row(panels_d, titles_d, 144)
    zoom_d_path = os.path.join(out_dir, "zoom_dots.jpg")
    zoom_d.save(zoom_d_path, quality=90)
    return {"compare": compare_path, "zoom_edges": zoom_e_path, "zoom_dots": zoom_d_path}


def _auto_crop(image_path: str, cy: int, cx: int, box: int, log) -> np.ndarray | None:
    """Run the existing auto pipeline on one small crop. Does not change auto."""
    try:
        src = np.asarray(Image.open(image_path).convert("RGB"))
    except Exception:
        return None
    y0, x0 = max(0, cy - box), max(0, cx - box)
    y1, x1 = min(src.shape[0], cy + box), min(src.shape[1], cx + box)
    crop = src[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    import tempfile
    tmp = tempfile.mkdtemp(prefix="recon_auto_")
    path = os.path.join(tmp, "crop.png")
    Image.fromarray(crop).save(path)
    try:
        r = am.run_auto(path, os.path.join(tmp, "out"), name="crop", colors=6, dpi=300, size=crop.shape[1])
        comp = r.get("composite")
        if comp and os.path.isfile(comp):
            return np.asarray(Image.open(comp).convert("RGB"))
    except Exception as exc:
        log(f"[reconstruct] auto crop skipped: {type(exc).__name__}: {exc}")
    return None
