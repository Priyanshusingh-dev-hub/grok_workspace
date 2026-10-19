"""Optional prepress steps for `textile number` (all OFF by default).

Ported / adapted from TextileAI_Hybrid_Separation_Lab_v1_3/backend/app/separation.py
and analysis.py (credit: Hybrid Lab). Licence-compatible (project MIT/Apache/BSD only).

Does not change colour separation, flat_index, detail rescue, or resize logic.
When every flag is off the caller must not invoke this module, so number outputs
stay byte-identical to today's.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .io_utils import hex_of, save_png, to_image

# ---- ported from Hybrid Lab separation.py (credit) ----

DEPOSIT_RATES_G_PER_SQM: dict[str, float] = {
    "underbase": 50.0,
    "spot": 30.0,
    "tonal": 18.0,
    "halftone": 18.0,
}


def choke_mask(mask: np.ndarray, choke_px: int) -> np.ndarray:
    """Erode binary mask by choke_px using 3x3 min filter. (Hybrid Lab)"""
    if choke_px <= 0:
        return mask
    im = Image.fromarray(mask, mode="L")
    for _ in range(choke_px):
        im = im.filter(ImageFilter.MinFilter(3))
    return np.asarray(im, dtype=np.uint8)


def spread_mask(mask: np.ndarray, spread_px: int) -> np.ndarray:
    """Dilate binary mask by spread_px using 3x3 max filter. (Hybrid Lab)"""
    if spread_px <= 0:
        return mask
    im = Image.fromarray(mask, mode="L")
    for _ in range(spread_px):
        im = im.filter(ImageFilter.MaxFilter(3))
    return np.asarray(im, dtype=np.uint8)


def apply_spot_trapping(
    spot_masks: list[np.ndarray],
    spot_colors: list[tuple[int, int, int]],
    trap_px: int = 1,
    alpha_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], int]:
    """Expand lighter spot plates slightly under darker adjacent spot plates. (Hybrid Lab)"""
    if trap_px <= 0 or len(spot_masks) < 2:
        return [m.copy() for m in spot_masks], 0
    luminances = [0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] for c in spot_colors]
    trapped_masks = [m.copy() for m in spot_masks]
    total_overlap = 0
    for i in range(len(spot_masks)):
        for j in range(i + 1, len(spot_masks)):
            if luminances[i] >= luminances[j]:
                light_idx, dark_idx = i, j
            else:
                light_idx, dark_idx = j, i
            dilated = spread_mask(spot_masks[light_idx], trap_px)
            seam_overlap = (dilated > 0) & (spot_masks[dark_idx] > 0)
            if alpha_mask is not None:
                seam_overlap &= alpha_mask
            if np.any(seam_overlap):
                trapped_masks[light_idx][seam_overlap] = 255
                total_overlap += int(np.sum(seam_overlap))
    return trapped_masks, total_overlap


def generate_underbase_mask(
    printable_masks: list[np.ndarray],
    choke_px: int = 1,
    alpha_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """White underbase plate with choke for dark-fabric printing. (Hybrid Lab)"""
    if not printable_masks:
        raise ValueError("No printable masks provided for underbase generation")
    union_mask = np.maximum.reduce(printable_masks)
    if alpha_mask is not None:
        union_mask = np.where(alpha_mask, union_mask, 0).astype(np.uint8)
    pre = int(np.sum(union_mask > 0))
    choked = choke_mask(union_mask, choke_px) if choke_px > 0 else union_mask
    if alpha_mask is not None:
        choked = np.where(alpha_mask, choked, 0).astype(np.uint8)
    post = int(np.sum(choked > 0))
    return choked, {
        "choke_px": choke_px,
        "pre_choke_pixels": pre,
        "post_choke_pixels": post,
        "eroded_pixels": pre - post,
        "detail_retained": post > 0 or pre == 0,
    }


def recommend_mesh_count(plate_type: str | None, lpi: int | None = None) -> str:
    """Polyester screen mesh guidance. (Hybrid Lab)"""
    ptype = (plate_type or "spot").lower()
    if ptype == "underbase":
        return "110-156 mesh (opaque underbase deposit)"
    if ptype in ("tonal", "halftone"):
        effective_lpi = lpi or 45
        if effective_lpi <= 45:
            return "200-230 mesh (halftone screen)"
        if effective_lpi <= 55:
            return "230-280 mesh (halftone screen)"
        return "305-355 mesh (fine halftone screen)"
    return "156-230 mesh (standard spot colour)"


def calculate_theoretical_ink_estimate(
    width_px: int,
    height_px: int,
    dpi: int,
    coverage_fraction: float,
    plate_type: str | None,
) -> float | None:
    """Theoretical wet ink grams per 100 prints. (Hybrid Lab)"""
    if plate_type is None:
        return None
    rate = DEPOSIT_RATES_G_PER_SQM.get(str(plate_type).lower())
    if rate is None:
        return None
    effective_dpi = max(72, dpi)
    w_m = (width_px / effective_dpi) * 0.0254
    h_m = (height_px / effective_dpi) * 0.0254
    cov = max(0.0, min(1.0, float(coverage_fraction)))
    return round(w_m * h_m * cov * rate * 100.0, 1)


def _film_font(size: int = 11):
    """Windows-safe font pick (no hard-coded Unix-only path)."""
    candidates = (
        "arial.ttf",
        "Arial.ttf",
        "segoeui.ttf",
        "SegoeUI.ttf",
        "DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def create_film_positive(
    mask_arr: np.ndarray,
    plate_name: str,
    plate_type: str,
    hex_color: str,
    dpi: int = 300,
    lpi: int | None = None,
    margin_px: int = 40,
) -> Image.Image:
    """Black-on-white film positive with registration targets and label.

    Adapted from Hybrid Lab create_film_positive: Lab used white ink on black;
    numbering program wants black-on-white plates as specified.
    """
    h, w = mask_arr.shape
    header_h = 32
    canvas_w = w + 2 * margin_px
    canvas_h = h + 2 * margin_px + header_h
    film = Image.new("L", (canvas_w, canvas_h), 255)
    # ink areas -> black on white
    ink = np.where(mask_arr > 0, 0, 255).astype(np.uint8)
    film.paste(Image.fromarray(ink, mode="L"), (margin_px, margin_px + header_h))
    draw = ImageDraw.Draw(film)

    def draw_target(cx: int, cy: int, r: int = 8):
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=0, width=1)
        draw.ellipse((cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2), outline=0, width=1)
        draw.line((cx - r - 6, cy, cx + r + 6, cy), fill=0, width=1)
        draw.line((cx, cy - r - 6, cx, cy + r + 6), fill=0, width=1)

    m_half = margin_px // 2
    top_y = m_half + header_h
    bot_y = canvas_h - m_half
    left_x = m_half
    right_x = canvas_w - m_half
    mid_x = canvas_w // 2
    mid_y = (canvas_h + header_h) // 2
    for cx, cy in (
        (left_x, top_y), (right_x, top_y), (left_x, bot_y), (right_x, bot_y),
        (mid_x, top_y), (mid_x, bot_y), (left_x, mid_y), (right_x, mid_y),
    ):
        draw_target(cx, cy)
    lpi_str = f" · {lpi} LPI" if lpi else ""
    slug = f"LOOMLAB FILM · {plate_name} ({plate_type.upper()}) · {hex_color} · {dpi} DPI{lpi_str}"
    draw.text((margin_px, 10), slug, fill=0, font=_film_font(11))
    return film


# ---- seam check from Hybrid Lab analysis.py (credit) ----

def seam_metrics(rgb: np.ndarray) -> dict[str, Any]:
    """Edge-border seam mismatch scores for a repeat tile. (Hybrid Lab analysis)"""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB")
    h, w, _ = rgb.shape
    arr = rgb.astype(np.float32) / 255.0
    if w > 1:
        h_diffs = np.linalg.norm(arr[:, 0, :] - arr[:, -1, :], axis=-1) / np.sqrt(3.0)
        h_mean, h_max, h_p95 = float(np.mean(h_diffs)), float(np.max(h_diffs)), float(np.percentile(h_diffs, 95))
    else:
        h_mean = h_max = h_p95 = 0.0
    if h > 1:
        v_diffs = np.linalg.norm(arr[0, :, :] - arr[-1, :, :], axis=-1) / np.sqrt(3.0)
        v_mean, v_max, v_p95 = float(np.mean(v_diffs)), float(np.max(v_diffs)), float(np.percentile(v_diffs, 95))
    else:
        v_mean = v_max = v_p95 = 0.0
    is_seamless = bool(
        h_mean < 0.035 and v_mean < 0.035
        and h_p95 < 0.08 and v_p95 < 0.08
        and h_max < 0.15 and v_max < 0.15
    )
    score = round(float(max(h_mean, v_mean, h_p95 / 2, v_p95 / 2)), 4)
    return {
        "is_seamless_repeat": is_seamless,
        "seam_mismatch_score": score,
        "horizontal_seam_delta": round(h_mean, 4),
        "vertical_seam_delta": round(v_mean, 4),
        "horizontal_seam_p95": round(h_p95, 4),
        "vertical_seam_p95": round(v_p95, 4),
        "horizontal_seam_max": round(h_max, 4),
        "vertical_seam_max": round(v_max, 4),
        "note": (
            "Edge continuity diagnostic only; not proof of visual repeat quality."
            if is_seamless
            else "Repeat seam discontinuity detected at tile borders."
        ),
    }


def tiled_preview_2x2(rgb: np.ndarray) -> np.ndarray:
    """2x2 tiled copy of the design for visual seam inspection."""
    row = np.concatenate([rgb, rgb], axis=1)
    return np.concatenate([row, row], axis=0)


def _luminance(rgb: tuple[int, int, int] | np.ndarray) -> float:
    c = np.asarray(rgb, dtype=np.float64)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def plates_from_flat(flat_rgb: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]], list[np.ndarray]]:
    """Build palette + binary masks from a flat numbered design (unique RGB)."""
    h, w, _ = flat_rgb.shape
    flat = np.ascontiguousarray(flat_rgb, dtype=np.uint8)
    # stable order: by luminance then hex
    colors = [tuple(int(x) for x in c) for c in np.unique(flat.reshape(-1, 3), axis=0)]
    colors.sort(key=lambda c: (_luminance(c), hex_of(c)))
    pal = np.array(colors, dtype=np.uint8)
    index = np.zeros((h, w), dtype=np.int32)
    masks: list[np.ndarray] = []
    for i, c in enumerate(colors):
        m = (flat[:, :, 0] == c[0]) & (flat[:, :, 1] == c[1]) & (flat[:, :, 2] == c[2])
        index[m] = i
        masks.append(np.where(m, 255, 0).astype(np.uint8))
    return index, colors, masks


def selective_underbase_masks(
    colors: list[tuple[int, int, int]],
    masks: list[np.ndarray],
    lum_max: float = 240.0,
) -> list[np.ndarray]:
    """Underbase only under inks that need it: skip near-white (L >= lum_max)."""
    out = []
    for c, m in zip(colors, masks):
        if _luminance(c) < lum_max:
            out.append(m)
    return out


def run_prepress(
    out_dir: str,
    name: str,
    *,
    underbase: bool = False,
    underbase_choke_px: int = 1,
    trap_px: int | None = None,
    film: bool = False,
    estimate: bool = False,
    seam_check: bool = False,
    dpi: int = 300,
    lpi: int | None = None,
    log=print,
) -> dict[str, Any]:
    """Run selected prepress steps into OUT/prepress/. Returns a report dict.

    Reads `{name}_flat.png` (and `{name}_rangeen.png` for seam preview when present).
    Never rewrites the main numbering outputs.
    """
    any_on = underbase or (trap_px is not None and trap_px > 0) or film or estimate or seam_check
    if not any_on:
        return {"applied": False, "reason": "all flags off"}

    flat_path = os.path.join(out_dir, f"{name}_flat.png")
    if not os.path.isfile(flat_path):
        raise FileNotFoundError(f"prepress needs {flat_path}")
    flat = np.asarray(Image.open(flat_path).convert("RGB"), dtype=np.uint8)
    h, w = flat.shape[:2]
    _, colors, masks = plates_from_flat(flat)

    pre_dir = os.path.join(out_dir, "prepress")
    os.makedirs(pre_dir, exist_ok=True)
    report: dict[str, Any] = {
        "name": name,
        "size_px": [w, h],
        "dpi": dpi,
        "n_inks": len(colors),
        "files": [],
    }

    work_masks = [m.copy() for m in masks]
    trap_stats = {"applied": False, "trap_px": 0, "overlap_pixels": 0}
    if trap_px is not None and trap_px > 0:
        work_masks, overlap = apply_spot_trapping(masks, colors, trap_px=trap_px)
        trap_stats = {"applied": True, "trap_px": int(trap_px), "overlap_pixels": int(overlap)}
        trap_dir = os.path.join(pre_dir, "trapped_plates")
        os.makedirs(trap_dir, exist_ok=True)
        for i, (c, m) in enumerate(zip(colors, work_masks), 1):
            hx = hex_of(c)
            path = os.path.join(trap_dir, f"plate_{i:02d}_{hx}.png")
            save_png(to_image(m), path, dpi)
            report["files"].append(path)
        log(f"[prepress] trap {trap_px}px: {overlap} overlap px -> {trap_dir}")
    report["trapping"] = trap_stats

    ub_stats: dict[str, Any] = {"generated": False}
    if underbase:
        sel = selective_underbase_masks(colors, work_masks)
        if not sel:
            # fallback: all plates (still ink-area only, not full rectangle)
            sel = work_masks
        ub_mask, stats = generate_underbase_mask(sel, choke_px=underbase_choke_px)
        ub_path = os.path.join(pre_dir, "UNDERBASE_WHITE.png")
        save_png(to_image(ub_mask), ub_path, dpi)
        ub_stats = {
            "generated": True,
            "path": ub_path,
            "coverage_percent": round(float((ub_mask > 0).mean() * 100), 3),
            "inks_included": len(sel),
            "inks_skipped_near_white": len(colors) - len(sel),
            **stats,
        }
        report["files"].append(ub_path)
        log(f"[prepress] underbase: {ub_stats['coverage_percent']}% cover, choke {underbase_choke_px}px")
    report["underbase"] = ub_stats

    film_files: list[str] = []
    if film:
        film_dir = os.path.join(pre_dir, "film")
        os.makedirs(film_dir, exist_ok=True)
        plates_for_film: list[tuple[str, str, str, np.ndarray]] = []
        if ub_stats.get("generated"):
            plates_for_film.append(("UNDERBASE_WHITE", "underbase", "#FFFFFF",
                                    np.asarray(Image.open(ub_stats["path"]).convert("L"))))
        for i, (c, m) in enumerate(zip(colors, work_masks), 1):
            hx = "#" + hex_of(c)
            plates_for_film.append((f"INK_{i:02d}", "spot", hx, m))
        for pname, ptype, hx, m in plates_for_film:
            im = create_film_positive(m, pname, ptype, hx, dpi=dpi, lpi=lpi)
            path = os.path.join(film_dir, f"{pname}_{ptype}_film.png")
            save_png(im, path, dpi)
            film_files.append(path)
            report["files"].append(path)
        log(f"[prepress] film: {len(film_files)} positives -> {film_dir}")
    report["film_positives"] = film_files

    estimate_path = None
    if estimate:
        rows = []
        if ub_stats.get("generated"):
            cov = float(ub_stats["coverage_percent"]) / 100.0
            rows.append({
                "plate": "UNDERBASE_WHITE",
                "type": "underbase",
                "hex": "#FFFFFF",
                "coverage_percent": ub_stats["coverage_percent"],
                "mesh": recommend_mesh_count("underbase", lpi),
                "grams_per_100_prints": calculate_theoretical_ink_estimate(w, h, dpi, cov, "underbase"),
            })
        for i, (c, m) in enumerate(zip(colors, work_masks), 1):
            cov = float((m > 0).mean())
            rows.append({
                "plate": f"INK_{i:02d}",
                "type": "spot",
                "hex": "#" + hex_of(c),
                "coverage_percent": round(cov * 100, 3),
                "mesh": recommend_mesh_count("spot", lpi),
                "grams_per_100_prints": calculate_theoretical_ink_estimate(w, h, dpi, cov, "spot"),
            })
        csv_path = os.path.join(pre_dir, f"{name}_estimate.csv")
        txt_path = os.path.join(pre_dir, f"{name}_estimate.txt")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        lines = [
            f"LoomLab mesh / ink estimate for {name} ({w}x{h} @ {dpi} DPI)",
            "Assumptions: underbase ~50 g/m2, spot ~30 g/m2 wet deposit; no setup waste.",
            "Add 30-50 g per screen for production mixing. Mesh counts are general guidance.",
            "",
        ]
        for r in rows:
            lines.append(
                f"{r['plate']} {r['hex']}  cover {r['coverage_percent']}%  "
                f"mesh {r['mesh']}  ~{r['grams_per_100_prints']} g / 100 prints"
            )
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        estimate_path = csv_path
        report["files"].extend([csv_path, txt_path])
        report["estimate_rows"] = rows
        log(f"[prepress] estimate: {csv_path}")
    report["estimate"] = estimate_path

    seam: dict[str, Any] = {"applied": False}
    if seam_check:
        src_path = os.path.join(out_dir, f"{name}_rangeen.png")
        if not os.path.isfile(src_path):
            src_path = flat_path
        src = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.uint8)
        metrics = seam_metrics(src)
        tile = tiled_preview_2x2(src)
        tile_path = os.path.join(pre_dir, f"{name}_seam_tiled_2x2.png")
        # keep tiled preview manageable for huge sheets
        th, tw = tile.shape[:2]
        if max(th, tw) > 4000:
            scale = 4000 / max(th, tw)
            nw, nh = max(1, int(tw * scale)), max(1, int(th * scale))
            tile_im = Image.fromarray(tile).resize((nw, nh), Image.NEAREST)
            save_png(tile_im, tile_path, dpi)
        else:
            save_png(to_image(tile), tile_path, dpi)
        json_path = os.path.join(pre_dir, f"{name}_seam_check.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump({**metrics, "source": os.path.basename(src_path), "tiled_preview": os.path.basename(tile_path)},
                      fh, indent=2)
        seam = {**metrics, "path": json_path, "tiled_preview": tile_path, "applied": True}
        report["files"].extend([json_path, tile_path])
        log(f"[prepress] seam-check: score={metrics['seam_mismatch_score']} "
            f"seamless={metrics['is_seamless_repeat']} -> {tile_path}")
    report["seam_check"] = seam

    report_path = os.path.join(pre_dir, f"{name}_prepress_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    report["report"] = report_path
    report["applied"] = True
    return report
