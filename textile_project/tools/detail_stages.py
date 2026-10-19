"""Small-detail stage trace for `textile number`'s colour work (number.flat_index, clean-design branch).

python tools/detail_stages.py [--out DIR]
SYNTHETIC inputs (tests/synth_detail.py: small-object card, Style A line+fill repeat, Style B multitone daisies) are
run through the same steps flat_index runs, in the same order, with a snapshot after each; every snapshot is scored
against the ground truth (object survival, IoU, line/gap retention, merges). Output size = source x K (K = 1 and the
2.44 that `number`'s default --size 3535 gives a 1448 px picture), keep_px = --detail normal (0.4 mm2 at 300 DPI).
"""
import argparse, json, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import cv2
import numpy as np
from textile import number as nb, palette as pl, edges as ed, names as nm
import synth_detail as sd


DBG = {}


def stages(rgb, W, H, colours=8, keep_px=None, small_px=nb.SMALL_PX, rescue=False):
    """flat_index's clean branch, step by step. impl: optional module providing the same steps (the fixed code)."""
    out = []
    snap = lambda name, pal, idx: out.append((name, pal[idx].copy()))
    scale = W / rgb.shape[1]
    keep_px = keep_px if keep_px is not None else nb.DETAIL_MM2['normal'] * (300 / 25.4) ** 2
    g = nb.grain(rgb)
    pal = nb.pt._ref_palette(nb.solid_pixels(rgb), colours, nb.CLEAN_SAME_DE)
    index = pl.map_to_palette(rgb, pal); snap('1 palette+RGB map', pal, index)
    pal, index = nb._merge_twins(rgb, pal, index); snap('2 twins', pal, index)
    share = np.bincount(index.ravel(), minlength=len(pal)) / index.size
    keep = share >= nb.MIN_INK_SHARE
    from scipy import ndimage
    for k in np.flatnonzero(~keep & (share > 0)):
        mk = index == k
        L = nm._lab(pal.astype(np.float64))
        far = min(float(nm.delta_e2000(L[k], L[j])) for j in range(len(pal)) if j != k) >= nb.SMALL_INK_DE
        if far and (ndimage.distance_transform_edt(mk)[mk] >= 2).mean() >= nb.SMALL_INK_CORE:
            keep[k] = True
        elif rescue:
            near = min(float(nm.delta_e2000(L[k], L[j])) for j in range(len(pal)) if j != k)
            if near >= nb.SMALL_KEEP_DE and nb._pure_share(rgb, mk, L[k]) >= nb.SMALL_KEEP_SOLID:
                keep[k] = True
    if not keep.all() and keep.any():
        pal = pal[keep]; index = pl.map_to_palette(rgb, pal)
    snap('3 small-ink fold', pal, index)
    src_lab = nm._lab(rgb.reshape(-1, 3).astype(np.float64)).reshape(rgb.shape).astype(np.float32)
    if rescue:
        pal, index, _ = nb.rare_rescue(rgb, pal, index, colours, src_lab)
    snap('3b rare-colour rescue', pal, index)
    pal_lab = nm._lab(pal.astype(np.float64))
    dot_only = np.zeros(len(pal), bool)
    for _ in range(3):
        blend = nb.blend_inks(index, pal, nb.RIM, islands=rescue)
        if not blend.any() or blend.all():
            break
        move = blend[index]; used = ~blend
        if rescue:
            stay = move & nb.dot_islands(index, pal, nb.DOT_MIN_SRC_PX, de_min=nb.RARE_DE); stay &= nb._true_dots(stay, index, src_lab, pal_lab); move &= ~stay
            kept = np.bincount(index[stay], minlength=len(pal)).astype(bool); dot_only |= kept & blend; used |= kept
        index = nb._to_nearest_colour(index, move, src_lab, pal_lab).astype(index.dtype)
        pal = pal[used]; pal_lab = pal_lab[used]; dot_only = dot_only[used]
        index = (np.cumsum(used) - 1)[index].astype(index.dtype)
    snap('4 blend-ink removal', pal, index)
    if g < nb.LINE_RESCUE_GRAIN:
        lk = nb.line_inks(rgb, pal, index, dot_only if rescue else None); index = np.where(lk >= 0, lk, index).astype(index.dtype)
    snap('5 line rescue', pal, index)
    dots = max(nb.DOT_MIN_SRC_PX, nb.DOT_MIN_MM2 * keep_px / nb.DETAIL_MM2['normal'] / scale ** 2) if rescue else 0.0
    kp = max(2.0, keep_px / scale ** 2)
    index = nb.shade_rims(index, pal, nb.SHADE_THIN); snap('6 shade rims', pal, index); DBG.update(index=index.copy(), pal=pal.copy(), kp=kp, dots=dots)
    _, _, index = nb.patches(index, kp, thin=0, enclosed_max=0, rim=nb.RIM, rim_area=30, src_lab=src_lab, pal=pal,
                             lines=nb.LINE_RESCUE if g < nb.LINE_RESCUE_GRAIN else 0, keep_dots=min(dots, kp))
    snap(f'7 melt (min-area {kp:.1f}px, rim)', pal, index)
    src_index = index
    if scale != 1:
        index, _ = ed.clean(index.astype(np.uint8), scale, 2)
    if index.shape != (H, W):
        index = cv2.resize(index.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    snap('8 edges smoothing', pal, index)
    if small_px and scale > 1:
        index = nb.draw_small(rgb, src_index.astype(np.uint8), pal, index.astype(np.uint8), small_px)
    snap('9 small-px redraw', pal, index)
    index, _ = nb.merge_shade_inks(index, pal, 300 / 25.4); snap('10 merge-shades', pal, index)
    return out


def score(objs_big, img):
    d = sd.detail_metrics(objs_big, img)
    return {k: d[k] for k in ('survival_percent', 'mean_iou', 'line_gap_retention_percent', 'merged_pairs',
                              'mean_abs_area_error_percent', 'survival_by_kind')}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='/workspace/hybrid-upgrade-out/numbering'); ap.add_argument('--rescue', action='store_true'); ap.add_argument('--only', default=''); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gens = {'card': (sd.small_object_card, 12), 'styleA': (sd.style_a, 8), 'styleB': (sd.style_b, 8)}
    res = {}
    for name, (fn, cols) in gens.items():
        if a.only and name != a.only: continue
        for s, K in (((1.0, 1),) if a.only else ((1.0, 1), (1.0, 2.44), (2.0, 1))):
            rgb, _ = fn(s)
            H, W = rgb.shape[:2]
            Wk, Hk = round(W * K), round(H * K)
            _, objs_big = fn(s * K)
            if objs_big[0]['mask'].shape != (Hk, Wk):
                Hk, Wk = objs_big[0]['mask'].shape
            st = stages(rgb, Wk, Hk, cols, rescue=a.rescue)
            key = f'{name} src x{s:g} out x{K:g}'
            res[key] = []
            for nm_, img in st:
                if img.shape[:2] != (Hk, Wk):
                    img = cv2.resize(img, (Wk, Hk), interpolation=cv2.INTER_NEAREST)
                res[key].append((nm_, score(objs_big, img)))
            print(key); [print('   ', n, v['survival_percent'], v['line_gap_retention_percent'], v['merged_pairs'], v['survival_by_kind']) for n, v in res[key]]
    json.dump(res, open(f"{a.out}/stage_trace_{'after' if a.rescue else 'before'}{a.only}.json", 'w'), indent=1)
