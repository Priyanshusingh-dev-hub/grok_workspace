#!/usr/bin/env python3
"""
colorfill.py  -  Line art me reference image ke hisaab se flat colors bharna (Method 1).

Rules (CLAUDE.md me detail hai):
  * Sirf reference ke flat colors (channel-wise). Koi shading / anti-aliasing / mixed pixel nahi.
  * Koi smoothing NAHI (corners, dots, patli lines original jaisi rehni chahiye).
  * Default 3535 x 3535 px @ 300 DPI (= 11.78 inch) - mill ka working format.

Usage:
  python colorfill.py --line lineart.png --ref colored.png --out output/
  python colorfill.py --line a.png --ref b.png --out out/ --size 3535 --dpi 300 --name design01
"""
import argparse, io, json, os, sys, zipfile
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- palette
def extract_palette(ref_rgb, max_colors, min_share):
    """Reference ke flat colors nikaalo. Agar reference already flat hai to exact colors,
    warna k-means se max_colors tak."""
    flat = ref_rgb.reshape(-1, 3)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    share = counts / counts.sum()
    keep = share >= min_share
    if keep.sum() <= max_colors and share[keep].sum() > 0.97:
        log(f"[palette] Reference flat hai: {keep.sum()} exact colors mile.")
        return cols[keep].astype(np.uint8), True
    log(f"[palette] Reference flat nahi hai ({len(cols)} unique colors). "
        f"k-means se {max_colors} colors bana raha hoon - result zaroor check karna.")
    from sklearn.cluster import KMeans
    sample = flat[np.random.default_rng(0).choice(len(flat), min(len(flat), 400000), replace=False)]
    km = KMeans(max_colors, n_init=4, random_state=0).fit(sample.astype(np.float32))
    pal = np.unique(np.clip(np.rint(km.cluster_centers_), 0, 255).astype(np.uint8), axis=0)
    return pal, False


def map_to_palette(img_rgb, pal):
    """Har pixel ko nearest palette index par map karo (chunked, memory safe)."""
    h, w, _ = img_rgb.shape
    out = np.empty((h, w), np.uint8)
    p = pal.astype(np.int32)
    step = max(1, 4_000_000 // (w * len(pal)))
    for y in range(0, h, step):
        blk = img_rgb[y:y + step].astype(np.int32)
        out[y:y + step] = ((blk[:, :, None, :] - p[None, None]) ** 2).sum(-1).argmin(-1)
    return out


# ---------------------------------------------------------------- alignment
def alignment_score(lines, ref_idx):
    """Reference ke color-boundaries kitne % line art ki lines ke paas padte hain.
    Aligned images me ye high hota hai; shift / alag design par low."""
    b = np.zeros_like(lines)
    b[:, 1:] |= ref_idx[:, 1:] != ref_idx[:, :-1]
    b[1:, :] |= ref_idx[1:, :] != ref_idx[:-1, :]
    near = cv2.dilate(lines.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return float((b & near).sum() / max(b.sum(), 1))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Line art color fill (Method 1, no smoothing)")
    ap.add_argument("--line", required=True, help="Black & white line art")
    ap.add_argument("--ref", required=True, help="Colored reference (same design, same alignment)")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--name", default=None, help="Output file prefix (default: line art ka naam)")
    ap.add_argument("--size", type=int, default=3535, help="Output px (square). Default 3535")
    ap.add_argument("--dpi", type=int, default=300, help="Default 300 (mill format)")
    ap.add_argument("--max-colors", type=int, default=16, help="Non-flat reference ke liye k-means colors")
    ap.add_argument("--min-share", type=float, default=0.0005, help="Isse kam share wale stray colors ignore")
    ap.add_argument("--line-threshold", type=int, default=150, help="Gray < ye = line (0-255)")
    ap.add_argument("--line-color", default="auto", help="'auto' ya hex jaise EFCE6A")
    ap.add_argument("--min-align", type=float, default=0.55, help="Isse kam alignment score par STOP")
    ap.add_argument("--force", action="store_true", help="Alignment warning ke bawajood chalao")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    name = a.name or os.path.splitext(os.path.basename(a.line))[0]
    S, DPI = a.size, a.dpi

    line = cv2.imread(a.line, cv2.IMREAD_GRAYSCALE)
    ref_bgr = cv2.imread(a.ref, cv2.IMREAD_COLOR)
    if line is None or ref_bgr is None:
        sys.exit("ERROR: image read nahi hui - path check karo.")
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    log(f"[input] line art {line.shape[1]}x{line.shape[0]}, reference {ref.shape[1]}x{ref.shape[0]}")

    ar_l = line.shape[1] / line.shape[0]
    ar_r = ref.shape[1] / ref.shape[0]
    if abs(ar_l - ar_r) > 0.01:
        sys.exit(f"ERROR: aspect ratio alag hai (line {ar_l:.3f} vs ref {ar_r:.3f}). "
                 "Dono same crop ke hone chahiye.")

    # 1. palette
    pal, is_flat = extract_palette(ref, a.max_colors, a.min_share)
    K = len(pal)

    # 2. reference -> palette index at target size (NEAREST: koi naya color na bane)
    ref_idx = map_to_palette(ref, pal)
    ref_idx = cv2.resize(ref_idx, (S, S), interpolation=cv2.INTER_NEAREST)

    # 3. line art upscale + threshold (sirf 3x3 halka blur jaggies ke liye; NO boundary smoothing)
    g = cv2.resize(line, (S, S), interpolation=cv2.INTER_LANCZOS4)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    lines = g < a.line_threshold
    del g

    # 4. alignment check
    score = alignment_score(lines, ref_idx)
    log(f"[align] score = {score:.2f} (>= {a.min_align} chahiye)")
    if score < a.min_align and not a.force:
        sys.exit("STOP: line art aur reference aligned nahi lag rahe. Images check karo "
                 "(crop/shift/rotation/alag design). Jaan-boojh kar chalana ho to --force do.")

    # 5. regions + majority vote
    lab, n = ndimage.label(~lines)  # 4-connectivity: diagonal se color leak nahi hota
    votes = np.bincount(lab.ravel().astype(np.int64) * K + ref_idx.ravel(),
                        minlength=(n + 1) * K).reshape(n + 1, K)
    region_color = votes.argmax(1)
    out = region_color[lab].astype(np.uint8)

    # 6. line color
    if a.line_color == "auto":
        lc = int(np.bincount(ref_idx[lines], minlength=K).argmax())
    else:
        hx = a.line_color.lstrip("#")
        tgt = np.array([int(hx[i:i + 2], 16) for i in (0, 2, 4)])
        lc = int(((pal.astype(int) - tgt) ** 2).sum(1).argmin())
    out[lines] = lc

    # 7. leak / doubtful regions report
    sizes = votes.sum(1)
    purity = votes.max(1) / np.maximum(sizes, 1)
    bad = [(int(i), int(sizes[i]), float(purity[i])) for i in range(1, n + 1)
           if sizes[i] > S * S * 0.002 and purity[i] < 0.6]
    if bad:
        log(f"[warn] {len(bad)} bade regions me reference ke multiple colors mile (purity < 60%). "
            "Ho sakta hai line tooti ho aur color leak hua ho -> debug image dekho.")
        dbg = pal[out].copy()
        mask = np.isin(lab, [b[0] for b in bad])
        dbg[mask] = (dbg[mask] * 0.3 + np.array([255, 0, 0]) * 0.7).astype(np.uint8)
        Image.fromarray(dbg).resize((1200, 1200), Image.NEAREST).save(
            os.path.join(a.out, f"{name}_DEBUG_doubtful_regions.png"))
    del lab

    # 8. outputs
    rgb = Image.fromarray(pal[out], "RGB")
    f_png = os.path.join(a.out, f"{name}_final_{S}px_{DPI}dpi.png")
    f_tif = os.path.join(a.out, f"{name}_final_{S}px_{DPI}dpi.tif")
    rgb.save(f_png, dpi=(DPI, DPI))
    rgb.save(f_tif, dpi=(DPI, DPI), compression="tiff_lzw")

    cnt = np.bincount(out.ravel(), minlength=K)
    order = [k for k in np.argsort(-cnt) if cnt[k] > 0]
    f_zip = os.path.join(a.out, f"{name}_colored_channels_{S}px_{DPI}dpi.zip")
    f_bw = os.path.join(a.out, f"{name}_bw_separations_{S}px_{DPI}dpi.zip")
    recon = np.zeros((S, S, 3), np.uint8)
    thumbs, channels = [], []
    with zipfile.ZipFile(f_zip, "w", zipfile.ZIP_DEFLATED) as zc, \
         zipfile.ZipFile(f_bw, "w", zipfile.ZIP_DEFLATED) as zb:
        for i, k in enumerate(order, 1):
            r, gg, b = (int(v) for v in pal[k])
            hx = f"{r:02X}{gg:02X}{b:02X}"           # NOTE: file name me '#' nahi
            m = out == k
            rgba = np.zeros((S, S, 4), np.uint8)
            rgba[m] = (r, gg, b, 255)
            recon[m] = (r, gg, b)
            buf = io.BytesIO()
            Image.fromarray(rgba, "RGBA").save(buf, "PNG", dpi=(DPI, DPI))
            fn = f"channel_{i:02d}_{hx}.png"
            zc.writestr(fn, buf.getvalue())
            buf = io.BytesIO()
            Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).convert("1").save(buf, "PNG", dpi=(DPI, DPI))
            zb.writestr(fn, buf.getvalue())
            share = cnt[k] / out.size * 100
            channels.append({"channel": i, "hex": hx, "coverage_percent": round(share, 2)})
            th = np.full((S, S, 3), 255, np.uint8)
            th[m] = (r, gg, b)
            thumbs.append((Image.fromarray(th).resize((500, 500), Image.NEAREST), f"{fn}  {share:.2f}%"))

    # preview sheet
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 540, rows * 570), "white")
    d = ImageDraw.Draw(sheet)
    for j, (t, label) in enumerate(thumbs):
        x, y = (j % cols) * 540 + 20, (j // cols) * 570 + 20
        sheet.paste(t, (x, y))
        d.rectangle([x, y, x + 499, y + 499], outline="gray")
        d.text((x, y + 508), label, fill="black")
    f_prev = os.path.join(a.out, f"{name}_channels_preview.png")
    sheet.save(f_prev)

    # 9. verification
    n_colors = len(rgb.getcolors(maxcolors=1 << 16) or [])
    match = bool((recon == np.array(rgb)).all())
    tif = Image.open(f_tif)
    report = {
        "design": name,
        "size_px": [S, S],
        "dpi": DPI,
        "print_size_inch": round(S / DPI, 2),
        "reference_was_flat": is_flat,
        "colors_in_final": n_colors,
        "channels": channels,
        "line_color_hex": "{:02X}{:02X}{:02X}".format(*(int(v) for v in pal[lc])),
        "alignment_score": round(score, 3),
        "regions": int(n),
        "doubtful_regions": len(bad),
        "channels_overlap_equals_final": match,
        "tif_dpi": [round(float(v)) for v in tif.info.get("dpi", (0, 0))],
    }
    with open(os.path.join(a.out, f"{name}_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    log(f"[done] {S}x{S}px @ {DPI} DPI = {S / DPI:.2f} inch | colors: {n_colors} | "
        f"channels match final: {match} | doubtful regions: {len(bad)}")
    for c in channels:
        log(f"        channel_{c['channel']:02d}  #{c['hex']}  {c['coverage_percent']}%")
    if not match or n_colors != len(order):
        sys.exit("ERROR: verification fail hua - output mat bhejna.")


if __name__ == "__main__":
    main()
