"""SPRAY mode: pure tonal multi-ink pipeline (no lines / flat numbering / min-area).

CLI: python -m textile spray <img> --out <dir> [--colors N]

Reuses gradients.pick_inks / solve_tones / overprint_rgb. Writes:
  - greyscale tone channel per ink
  - FM/stochastic films by default (AM optional via --halftone am)
  - overprint composite preview
  - package/ colored_channels zip + PSD layers + multipage spot TIF
"""
from __future__ import annotations

import csv
import json
import os
import zipfile
from typing import Any

import numpy as np
from PIL import Image

from . import gradients as gr
from . import package_channels as pkgch
from .io_utils import hex_of, png_bytes, save_png, to_image



def _candidate_sets(src_rgb: np.ndarray, colors_limit: int) -> list[np.ndarray]:
    """Build a few ink-set candidates (hull / hue / vivid)."""
    lab = gr.rgb_to_lab(src_rgb).reshape(-1, 3)
    rgb = src_rgb.reshape(-1, 3)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma = np.sqrt(a * a + b * b)
    hue = np.arctan2(b, a)
    paper_m = (L > 96.0) & (chroma < 4.0)
    keep = ~paper_m
    sets: list[list[tuple[int, int, int]]] = []

    def _uniq(cols, lim):
        out = []
        for c in cols:
            c = tuple(int(x) for x in c)
            if gr._lum1(c) >= 252:
                continue
            if not out or min(((np.array(c) - np.array(o)) ** 2).sum() for o in out) > 28 ** 2:
                out.append(c)
            if len(out) >= lim:
                break
        return out

    # 1) hue-sector high-chroma endmembers
    hue_cols = []
    light = keep & (L > 78) & (chroma > 3) & (chroma < 28)
    if light.any():
        hue_cols.append(tuple(int(x) for x in np.median(rgb[light].astype(np.float64), 0).round()))
    n_sec = max(colors_limit + 2, 8)
    for s in range(n_sec):
        lo = -np.pi + 2 * np.pi * s / n_sec
        hi = lo + 2 * np.pi / n_sec
        m = keep & (chroma > 10) & (hue >= lo) & (hue < hi)
        if not m.any():
            continue
        score = np.where(m, chroma + (100 - L) * 0.12, -1.0)
        j = int(np.argmax(score))
        hue_cols.append(tuple(int(x) for x in rgb[j]))
    sets.append(_uniq(hue_cols, colors_limit))

    # 2) convex hull vertices in absorption space
    ppr = np.array([255.0, 255.0, 255.0])
    absb = np.clip(1.0 - src_rgb.astype(np.float64).reshape(-1, 3) / ppr, 0, 1)
    idx = np.flatnonzero(keep & (absb.sum(1) > 0.03))
    if len(idx) > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, 5000, replace=False)
    X = absb[idx]
    hull_cols = []
    try:
        from scipy.spatial import ConvexHull
        hv = np.unique(ConvexHull(X).vertices)
        for v in hv:
            hull_cols.append(tuple(int(x) for x in rgb[idx[v]]))
    except Exception:
        pass
    for axis in range(3):
        j = int(idx[int(np.argmax(X[:, axis]))])
        hull_cols.append(tuple(int(x) for x in rgb[j]))
    # max chroma
    if keep.any():
        j = int(np.flatnonzero(keep)[int(np.argmax(chroma[keep]))])
        hull_cols.append(tuple(int(x) for x in rgb[j]))
    sets.append(_uniq(hull_cols, colors_limit))

    # 3) vivid furthest-point in Lab
    vivid_idx = np.flatnonzero(keep & (chroma > 8))
    if len(vivid_idx) > 4000:
        vivid_idx = vivid_idx[:: max(1, len(vivid_idx) // 4000)]
    if len(vivid_idx):
        Xv = lab[vivid_idx]
        centers = [Xv[int(np.argmax(chroma[vivid_idx]))]]
        cols = [tuple(int(x) for x in rgb[vivid_idx[int(np.argmax(chroma[vivid_idx]))]])]
        for _ in range(colors_limit - 1):
            d = np.min([((Xv - c) ** 2).sum(1) for c in centers], axis=0)
            j = int(np.argmax(d))
            centers.append(Xv[j])
            cols.append(tuple(int(x) for x in rgb[vivid_idx[j]]))
        sets.append(_uniq(cols, colors_limit))

    out = []
    for s in sets:
        if len(s) >= 2:
            out.append(np.array(s, dtype=np.uint8))
    if not out:
        out.append(gr.pick_inks(src_rgb, colors_limit, src_rgb=src_rgb))
    return out


def pick_inks_from_source(src_rgb: np.ndarray, colors_limit: int = 8) -> np.ndarray:
    """Choose the ink set that best reconstructs the source (quick tone solve score)."""
    cands = _candidate_sets(src_rgb, colors_limit)
    paper = (255, 255, 255)
    best = None
    best_de = 1e9
    # downsample for scoring
    h, w, _ = src_rgb.shape
    step = max(1, int(round(max(h, w) / 120)))
    small = src_rgb[::step, ::step]
    for inks in cands:
        tones = gr.solve_tones(
            small, inks, paper=paper, max_inks_per_pixel=len(inks),
            iters=40, seal_motifs=False,
        )
        comp = gr.overprint_rgb(tones, inks, paper)
        de = gr.delta_e_cie76(small, comp)
        if de < best_de:
            best_de = de
            best = inks
    return best if best is not None else cands[0]


def _load_rgb(path: str, max_side: int | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def write_tone_package(
    out_dir: str,
    name: str,
    tone_plates: list[dict[str, Any]],
    *,
    dpi: int = 300,
    write_psd: bool = True,
    composite_path: str | None = None,
    include_composite: bool = False,
    use_layers: bool = False,
    log=print,
) -> dict[str, Any]:
    """Create package/ from tone plates alone (no prior numbering zip)."""
    pkg = os.path.join(out_dir, "package")
    os.makedirs(pkg, exist_ok=True)
    if not tone_plates:
        return {"merged": False, "reason": "no_tones"}
    # Darkest ink first (channel 01 / PSD bottom above Paper)
    def _plate_lum(tp):
        ink = tp.get("ink_rgb") or [0, 0, 0]
        return 0.2126 * ink[0] + 0.7152 * ink[1] + 0.0722 * ink[2]
    tone_plates = sorted(tone_plates, key=_plate_lum)
    # probe size
    t0 = np.asarray(Image.open(tone_plates[0]["tone_path"]).convert("L"))
    h, w = t0.shape
    tag = f"{w}px_{dpi}dpi"
    zpath = os.path.join(pkg, f"{name}_colored_channels_{tag}.zip")
    bw_path = os.path.join(pkg, f"{name}_bw_separations_{tag}.zip")
    names_out = []
    pages = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zc, \
         zipfile.ZipFile(bw_path, "w", zipfile.ZIP_DEFLATED) as zb:
        for i, tp in enumerate(tone_plates, 1):
            ink = tuple(int(x) for x in tp["ink_rgb"])
            hx = hex_of(ink)
            tone = np.asarray(Image.open(tp["tone_path"]).convert("L"))
            if tone.shape != (h, w):
                tone = np.asarray(Image.fromarray(tone).resize((w, h), Image.Resampling.BILINEAR))
            rgba = pkgch._tone_to_rgba(tone, ink)
            fn = pkgch._channel_filename(i, tp.get("name") or f"INK-{hx}", "tone", hx)
            names_out.append(fn)
            zc.writestr(fn, png_bytes(to_image(rgba), dpi))
            zb.writestr(fn, png_bytes(Image.fromarray((255 - tone).astype(np.uint8), mode="L"), dpi))
            pages.append(Image.fromarray(tone, mode="L"))
    from . import stack
    stack.loose_images(zpath, pkg, name, dpi)
    psd_path = None
    if write_psd:
        from . import psdout
        merged = None
        if composite_path and os.path.isfile(composite_path):
            merged = Image.open(composite_path).convert("RGB")
        psd_path = psdout.write_psd(
            zpath, os.path.join(pkg, f"{name}_layers_{w}px_{dpi}dpi.psd"), dpi,
            merged_rgb=merged, include_composite=include_composite,
            use_layers=use_layers,
        )
    spot_tif = os.path.join(pkg, f"{name}_spot_channels_{w}px_{dpi}dpi.tif")
    if pages:
        pages[0].save(
            spot_tif, format="TIFF", save_all=True, append_images=pages[1:],
            compression="tiff_lzw", dpi=(dpi, dpi),
        )
    log(f"[spray] package: {len(names_out)} tone plates -> {pkg}")
    return {
        "merged": True,
        "channel_names": names_out,
        "channel_count": len(names_out),
        "channels_zip": zpath,
        "psd": psd_path,
        "spot_tif": spot_tif if pages else None,
        "package_dir": pkg,
    }


def run_spray(
    image_path: str,
    out_dir: str,
    *,
    name: str | None = None,
    colors: int = 8,
    dpi: int = 300,
    lpi: int = 45,
    halftone: str = "fm",
    max_inks_per_pixel: int = 5,
    max_side: int | None = None,
    write_psd: bool = True,
    include_composite: bool = False,
    use_layers: bool = False,
    log=print,
) -> dict[str, Any]:
    """Pure tonal spray pipeline."""
    os.makedirs(out_dir, exist_ok=True)
    base = name or os.path.splitext(os.path.basename(image_path))[0]
    base = base.replace("#", "").replace(" ", "_") or "spray"
    src = _load_rgb(image_path, max_side=max_side)
    h, w = src.shape[:2]
    paper = (255, 255, 255)
    # Ink set from source alone (no flat numbering)
    inks = pick_inks_from_source(src, colors)
    report: dict[str, Any] = {
        "name": base,
        "mode": "spray",
        "size_px": [w, h],
        "dpi": dpi,
        "colors": colors,
        "n_inks": int(len(inks)),
        "halftone": halftone,
        "applied": False,
        "files": [],
        "tone_channels": [],
        "tone_plates": [],
    }
    if len(inks) == 0:
        log("[spray] no inks")
        return report

    log(f"[spray] solving continuous tones for {len(inks)} inks (no motif seal) …")
    tones = gr.solve_tones(
        src, inks, paper=paper, max_inks_per_pixel=max(max_inks_per_pixel, len(inks)),
        iters=160, seal_motifs=False,
    )
    tones = np.clip(tones, 0.0, 1.0)

    sdir = os.path.join(out_dir, "spray")
    film_dir = os.path.join(sdir, "film")
    tone_dir = os.path.join(sdir, "tone_channels")
    os.makedirs(film_dir, exist_ok=True)
    os.makedirs(tone_dir, exist_ok=True)

    for i, ink in enumerate(inks):
        hx = hex_of(ink)
        cov = tones[..., i]
        tone_u8 = np.clip(np.rint(cov * 255.0), 0, 255).astype(np.uint8)
        tone_path = os.path.join(tone_dir, f"TONE_{i+1:02d}_{hx}.png")
        save_png(Image.fromarray(tone_u8, mode="L"), tone_path, dpi)
        if halftone == "am":
            angle = gr._SCREEN_ANGLES[i % len(gr._SCREEN_ANGLES)]
            dots = gr.am_halftone(cov, lpi=lpi, dpi=dpi, angle_deg=angle)
            film_kind = "am"
        else:
            dots = gr.fm_halftone(cov, seed=i + 1)
            film_kind = "fm"
            angle = None
        film_path = os.path.join(film_dir, f"INK_{i+1:02d}_{hx}_film.png")
        save_png(Image.fromarray(dots, mode="L"), film_path, dpi)
        entry = {
            "name": f"INK_{i+1:02d}",
            "hex": "#" + hx,
            "ink_rgb": [int(x) for x in ink],
            "angle_deg": angle,
            "lpi": lpi if film_kind == "am" else None,
            "halftone": film_kind,
            "tone_path": tone_path,
            "film_path": film_path,
            "mean_tone_percent": round(float(cov.mean() * 100), 3),
        }
        report["tone_plates"].append(entry)
        report["tone_channels"].append(entry)
        report["files"].extend([tone_path, film_path])

    composite = gr.overprint_rgb(tones, inks, paper)
    comp_path = os.path.join(sdir, f"{base}_spray_composite.png")
    save_png(to_image(composite), comp_path, dpi)
    report["files"].append(comp_path)
    report["composite"] = comp_path

    de_all = gr.delta_e_cie76(src, composite)
    report["delta_e_cie76"] = {"overall_mean": round(de_all, 3)}
    log(f"[spray] ΔE76 overall={de_all:.2f}")

    csv_path = os.path.join(sdir, f"{base}_tone_colors.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["Plate", "HEX", "Halftone", "MeanTone%", "ToneFile", "FilmFile"])
        for e in report["tone_plates"]:
            wr.writerow([e["name"], e["hex"], e["halftone"], e["mean_tone_percent"],
                         os.path.basename(e["tone_path"]), os.path.basename(e["film_path"])])
    report["files"].append(csv_path)

    pkg_rep = write_tone_package(
        out_dir, base, report["tone_plates"], dpi=dpi, write_psd=write_psd,
        composite_path=comp_path, include_composite=include_composite,
        use_layers=use_layers, log=log,
    )
    report["package"] = pkg_rep

    report["applied"] = True
    report["inks"] = ["#" + hex_of(c) for c in inks]
    report["paper_rgb"] = list(paper)
    rp = os.path.join(sdir, f"{base}_spray_report.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    report["report"] = rp
    report["files"].append(rp)
    log(f"[spray] done -> {sdir}")
    return report
