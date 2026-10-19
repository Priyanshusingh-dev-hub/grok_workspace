"""A design given in parts, put back together; and one design cut into parts to work on.

  split(image, out_dir, grid=(3, 3), overlap=0.12)  -> part files NAME_r1c1.png ... (each part overlaps its
      neighbours by `overlap` of a part, so after each is enlarged or redone on its own they can be lined up)
  stitch(parts, grid=None)  -> (H x W x 3 image, report)

Stitching: every part is brought to the common part size, then each neighbour pair is lined up by finding a
strip of the next part inside the previous one (template match: how much they overlap and how far one is
shifted). A pair whose strip is not found (parts cut edge to edge, no shared area) is joined edge to edge. In
an overlap the two parts are joined along the line where they differ least (the seam cut `tile` uses), never
blended: a blend would invent mixed colours. The report gives, per joint, the overlap, the shift and how well
the two sides agreed, so a joint that does not match (parts drawn separately by an AI) is named, not hidden.
"""
from __future__ import annotations

import os
import re

import cv2
import numpy as np

from .io_utils import read_cv2
from .tile import seam_path

STRIP = 0.06          # the strip of the next part looked for in the previous one, share of the part
MAX_OVERLAP = 0.35    # overlap searched up to this share of the part
MAX_SHIFT = 0.05      # sideways shift searched up to this share of the part
MATCH_MIN = 0.80      # normalised correlation a found strip needs; below it the parts are joined edge to edge
TEXTURE_MIN = 6.0     # a strip with less contrast (plain ground) cannot be matched reliably
OVERLAP_DIFF = 12.0   # a found overlap must agree to within this mean grey difference over its whole area
SEAM_BAD = 18.0       # mean colour difference along a joint above this is reported as a visible joint

IMAGES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp')


def _rc(name):
    """(row, col) from a part's file name: r2c3 / R2C3 / _2_3 / -2x3, else None."""
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r'[_\-\s]\d{3,}x\d{3,}$', '', stem)          # puzzle_piece_3_2_1536x1536: the size is not the place
    m = re.search(r'r(\d+)\s*[_-]?\s*c(\d+)', stem, re.I) or re.search(r'(?:^|[_\-\s])(\d+)[_\-x](\d+)$', stem)
    return (int(m.group(1)), int(m.group(2))) if m else None


def order_parts(paths, grid=None):
    """The parts as a grid [[path]] (rows top to bottom). Names with row/col win; else the files are taken in
    name order (numbers compared as numbers), row by row, the grid from `grid` or a square count (9 -> 3x3)."""
    paths = [p for p in paths if os.path.splitext(p)[1].lower() in IMAGES]
    if not paths:
        raise ValueError('koi part image nahi mili')
    rcs = [_rc(p) for p in paths]
    if all(rcs):
        R, C = max(r for r, _ in rcs), max(c for _, c in rcs)
        g = [[None] * C for _ in range(R)]
        for p, (r, c) in zip(paths, rcs):
            g[r - 1][c - 1] = p
        if any(x is None for row in g for x in row):
            raise ValueError(f'grid {R}x{C} me kuch part missing hain (naam r1c1 ... r{R}c{C} hone chahiye)')
        return g
    key = lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(p))]
    paths = sorted(paths, key=key)
    n = len(paths)
    if grid is None:
        k = int(round(n ** 0.5))
        if k * k != n:
            raise ValueError(f'{n} parts: grid khud nahi pata (9 = 3x3, 4 = 2x2). --grid ROWSxCOLS do, ya naam r1c1, r1c2 ... rakho')
        grid = (k, k)
    R, C = grid
    if R * C != n:
        raise ValueError(f'grid {R}x{C} = {R * C} parts chahiye, {n} mile')
    return [paths[r * C:(r + 1) * C] for r in range(R)]


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _match_lr(a, b):
    """Line up b to the right of a. Returns (overlap px, shift of b down in px, score); (0, 0, score) when no
    shared strip is found. A found overlap is checked over its whole area (both parts must agree there, and
    agree better than the plain edge-to-edge join would): a repeating motif can fool the strip alone."""
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h, w = min(ha, hb), min(wa, wb)
    ga, gb = _gray(a), _gray(b)
    t, mo, ms = max(8, int(w * STRIP)), max(10, int(w * MAX_OVERLAP)), max(2, int(h * MAX_SHIFT))
    templ = gb[ms:hb - ms, :t]
    if templ.std() < TEXTURE_MIN or templ.shape[0] > ha:
        return 0, 0, 0.0
    area = ga[:, wa - mo:]
    res = cv2.matchTemplate(area, templ, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(res)
    ov, d = mo - x, y - ms
    score = float(score) if np.isfinite(score) else 0.0
    if score < MATCH_MIN or ov < t:
        return 0, 0, score
    y0, y1 = max(0, d), min(ha, hb + d)                 # rows of a that b covers
    if y1 - y0 < h // 2:
        return 0, 0, score
    sa = ga[y0:y1, wa - ov:]
    sb = gb[y0 - d:y1 - d, :ov]
    over = float(np.abs(sa - sb).mean())
    edge = float(np.abs(ga[:min(ha, hb), -1] - gb[:min(ha, hb), 0]).mean())
    if over > OVERLAP_DIFF or over > edge:
        return 0, 0, score
    return int(ov), int(d), score


def _match_tb(a, b):
    """Line up b below a: (overlap, shift of b right, score)."""
    ov, d, s = _match_lr(np.ascontiguousarray(a.transpose(1, 0, 2)), np.ascontiguousarray(b.transpose(1, 0, 2)))
    return ov, d, s


def _paste(canvas, valid, tile, x, y, left_ov, top_ov):
    """Put a part on the canvas at (x, y). Where it lies over parts already placed the joint follows the line of
    least difference (over the left neighbour: a seam top to bottom; over the one above: left to right).
    Returns the joint lines {'left': (n, 2) x,y, 'top': ...} in canvas coordinates."""
    h, w = tile.shape[:2]
    reg = canvas[y:y + h, x:x + w]
    old = valid[y:y + h, x:x + w].copy()
    take = np.ones((h, w), bool)
    diff = np.abs(reg.astype(np.int32) - tile.astype(np.int32)).sum(-1).astype(np.float64)
    diff[~old] = 0
    lines = {}
    if left_ov > 0:
        p, _ = seam_path(diff[:, :min(left_ov, w)] + 1e-3)
        take &= ~((np.arange(w)[None, :] < p[:, None]) & old)
        lines['left'] = np.c_[p + x, np.arange(h) + y]
    else:
        lines['left'] = np.c_[np.full(h, x), np.arange(h) + y]
    if top_ov > 0:
        q, _ = seam_path(diff[:min(top_ov, h), :].T + 1e-3)
        take &= ~((np.arange(h)[:, None] < q[None, :]) & old)
        lines['top'] = np.c_[np.arange(w) + x, q + y]
    else:
        lines['top'] = np.c_[np.arange(w) + x, np.full(w, y)]
    reg[take] = tile[take]
    valid[y:y + h, x:x + w] |= take
    return lines


def _jump(img, line, vertical, half=2):
    """Mean colour jump across a joint line (0-255 per channel): the pixels `half` px either side compared."""
    H, W = img.shape[:2]
    f = img.astype(np.float32)
    x, y = line[:, 0], line[:, 1]
    if vertical:
        a = f[np.clip(y, 0, H - 1), np.clip(x - half, 0, W - 1)]
        b = f[np.clip(y, 0, H - 1), np.clip(x + half - 1, 0, W - 1)]
    else:
        a = f[np.clip(y - half, 0, H - 1), np.clip(x, 0, W - 1)]
        b = f[np.clip(y + half - 1, 0, H - 1), np.clip(x, 0, W - 1)]
    return float(np.abs(a - b).mean())


def stitch(paths, grid=None, log=print):
    """Join parts into one image. Returns (image H x W x 3 uint8 RGB, report dict)."""
    g = order_parts(paths, grid)
    R, C = len(g), len(g[0])
    imgs = [[cv2.cvtColor(read_cv2(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in row] for row in g]
    # parts are used at their own size (parts cut with overlap differ: an edge part has one neighbour less); only a
    # part enlarged differently from the rest (a side 30%+ off the median in both directions) is scaled back
    hs = sorted(im.shape[0] for row in imgs for im in row)
    ws = sorted(im.shape[1] for row in imgs for im in row)
    mh, mw = hs[len(hs) // 2], ws[len(ws) // 2]
    resized = 0
    for r in range(R):
        for c in range(C):
            ih, iw = imgs[r][c].shape[:2]
            k = ((mh / ih) + (mw / iw)) / 2
            if abs(mh / ih - 1) > 0.3 and abs(mw / iw - 1) > 0.3:
                resized += 1
                imgs[r][c] = cv2.resize(imgs[r][c], (max(1, round(iw * k)), max(1, round(ih * k))),
                                        interpolation=cv2.INTER_LANCZOS4)
    size = [[im.shape[:2] for im in row] for row in imgs]
    h, w = mh, mw
    pos = [[(0, 0)] * C for _ in range(R)]
    joints = []
    for r in range(R):
        for c in range(C):
            if c > 0:
                ov, d, s = _match_lr(imgs[r][c - 1], imgs[r][c])
                px, py = pos[r][c - 1]
                pos[r][c] = (px + size[r][c - 1][1] - ov, py + d)
                joints.append({'between': f'r{r + 1}c{c} | r{r + 1}c{c + 1}', 'overlap_px': ov, 'shift_px': d,
                               'match': round(s, 3), 'side': 'left'})
            elif r > 0:
                ov, d, s = _match_tb(imgs[r - 1][0], imgs[r][0])
                px, py = pos[r - 1][0]
                pos[r][0] = (px + d, py + size[r - 1][0][0] - ov)
                joints.append({'between': f'r{r}c1 / r{r + 1}c1', 'overlap_px': ov, 'shift_px': d,
                               'match': round(s, 3), 'side': 'top'})
    x0 = min(pos[r][c][0] for r in range(R) for c in range(C))
    y0 = min(pos[r][c][1] for r in range(R) for c in range(C))
    W = max(pos[r][c][0] + size[r][c][1] for r in range(R) for c in range(C)) - x0
    H = max(pos[r][c][1] + size[r][c][0] for r in range(R) for c in range(C)) - y0
    canvas = np.full((H, W, 3), 255, np.uint8)
    valid = np.zeros((H, W), bool)
    lines = {}
    for r in range(R):
        for c in range(C):
            x, y = pos[r][c][0] - x0, pos[r][c][1] - y0
            left = (pos[r][c - 1][0] - x0 + size[r][c - 1][1] - x) if c > 0 else 0       # how far this part runs over its left neighbour
            top = (pos[r - 1][c][1] - y0 + size[r - 1][c][0] - y) if r > 0 else 0        # ...and over the one above
            ln = _paste(canvas, valid, imgs[r][c], x, y, max(0, left), max(0, top))
            if c > 0:
                lines[('h', r, c)] = ln['left']
            if r > 0:
                lines[('v', r, c)] = ln['top']
    # how visible each joint is: the colour jump across it, against the same jump measured inside the parts
    base = []
    for r in range(R):
        for c in range(C):
            x, y = pos[r][c][0] - x0, pos[r][c][1] - y0
            h, w = size[r][c]
            base.append(_jump(canvas, np.c_[np.full(h // 2, x + w // 2), np.arange(h // 4, h // 4 + h // 2) + y], True))
            base.append(_jump(canvas, np.c_[np.arange(w // 4, w // 4 + w // 2) + x, np.full(w // 2, y + h // 2)], False))
    inside = float(np.median(base))
    joints_all = []
    for (kind, r, c), ln in sorted(lines.items()):
        jmp = _jump(canvas, ln, kind == 'h')
        joints_all.append({'between': (f'r{r + 1}c{c} | r{r + 1}c{c + 1}' if kind == 'h' else f'r{r}c{c + 1} / r{r + 1}c{c + 1}'),
                           'jump': round(jmp, 1), 'visible': bool(jmp > max(SEAM_BAD, 2.5 * inside))})
    for j in joints:                                     # the line-up found for each chain joint
        for k in joints_all:
            if k['between'] == j['between']:
                k.update({kk: j[kk] for kk in ('overlap_px', 'shift_px', 'match')})
    gaps = int((~valid).sum())
    report = {'grid': [R, C], 'part_size': [w, h], 'parts_resized': resized, 'size': [W, H], 'joints': joints_all,
              'inside_jump': round(inside, 1), 'gaps_px': gaps, 'parts': [[os.path.basename(p) for p in row] for row in g]}
    overl = [j for j in joints if j['overlap_px'] > 0]
    vis = [j['between'] for j in joints_all if j['visible']]
    log(f'[stitch] {R}x{C} parts ({w}x{h} px har ek) -> {W}x{H} px; {len(overl)}/{len(joints)} jod overlap se mile'
        + (f'; {resized} part ka size alag tha, barabar kiya' if resized else '')
        + (f'; DIKHNE WALE JOD: {", ".join(vis)}' if vis else '; koi jod dikhta nahi'))
    return canvas, report


def _geometry(W, H, grid, overlap):
    """The crop boxes (x0, y0, w, h) of a split: every part one size, a part plus `overlap` of it, edge parts reaching inwards."""
    R, C = grid
    pw, ph = W / C, H / R
    fw, fh = min(W, int(round(pw * (1 + overlap)))), min(H, int(round(ph * (1 + overlap))))
    boxes = {}
    for r in range(R):
        for c in range(C):
            x0 = int(np.clip(round((c + 0.5) * pw - fw / 2), 0, W - fw))
            y0 = int(np.clip(round((r + 0.5) * ph - fh / 2), 0, H - fh))
            boxes[(r + 1, c + 1)] = (x0, y0, fw, fh)
    return boxes


def split(image_path, out_dir, name=None, grid=(3, 3), overlap=0.15, target=None):
    """Cut a design into grid parts of one size that overlap their neighbours (by about `overlap` of a part), named
    NAME_r1c1.png ... so each can be enlarged / redone on its own and stitched back. Also writes NAME_parts_plan.json
    (the geometry, so `partscheck` can lay the same crops over the reference) and, for a wanted final width `target`,
    the part size to ask an AI for. Returns (paths, plan)."""
    import json
    img = read_cv2(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('image read nahi hui')
    H, W = img.shape[:2]
    R, C = grid
    name = name or os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    boxes = _geometry(W, H, grid, overlap)
    out = []
    for (r, c), (x0, y0, fw, fh) in boxes.items():
        p = os.path.join(out_dir, f'{name}_r{r}c{c}.png')
        cv2.imwrite(p, img[y0:y0 + fh, x0:x0 + fw])
        out.append(p)
    fw, fh = boxes[(1, 1)][2:]
    plan = {'reference': os.path.basename(image_path), 'size': [W, H], 'grid': [R, C], 'overlap': overlap,
            'part_size': [fw, fh], 'boxes': {f'r{r}c{c}': list(b) for (r, c), b in boxes.items()}}
    if target:
        # parts of size p joined with an overlap of o = overlap * (p / (1 + overlap)) lose (C-1) * o: total = C*p - (C-1)*o
        k = C - (C - 1) * overlap / (1 + overlap)
        plan['target_width'] = target
        plan['ask_ai_part_px'] = int(round(target / k))
    with open(os.path.join(out_dir, f'{name}_parts_plan.json'), 'w', encoding='utf-8') as fh:
        json.dump(plan, fh, indent=1)
    return out, plan


# ---- checking what an AI gave back against the reference -------------------------------------------------------

SHIFT_MAX = 0.012      # a motif moved by more than this share of the part's width is a moved motif
NEW_COLOUR_MAX = 6.0   # % of pixels farther than DE_NEW from every colour of the reference tile
DE_NEW = 22.0          # Lab distance that makes a colour 'new'
EDGE_LOST = 0.80       # edge length of the redrawn tile / the reference's: below = detail lost
EDGE_ADDED = 1.35      # above = detail invented


def _lab(a):
    return cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def _ref_palette(ref, share=0.004, bits=3):
    """Colours of the reference tile that matter: coarse-quantised, those above `share` of the pixels, averaged back."""
    q = (ref >> (8 - bits)).astype(np.int32)
    key = (q[..., 0] << (2 * bits)) | (q[..., 1] << bits) | q[..., 2]
    ks, inv, cnt = np.unique(key.ravel(), return_inverse=True, return_counts=True)
    keep = cnt >= share * key.size
    pal = np.stack([np.bincount(inv, weights=ref.reshape(-1, 3)[:, ch], minlength=len(ks)) / cnt for ch in range(3)], 1)
    return pal[keep].astype(np.uint8)


def check_tile(ref_crop, ai_tile):
    """Compare one redrawn tile (any size) with the reference crop it came from. Returns a dict of measurements
    and 'redo': the reasons the tile should be asked for again (empty = fine)."""
    h, w = ref_crop.shape[:2]
    ai = cv2.resize(ai_tile, (w, h), interpolation=cv2.INTER_AREA)       # at the reference's scale
    ga, gr = _gray(ai), _gray(ref_crop)
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(gr, ga, win)
    shift = float(np.hypot(dx, dy)) / w
    pal = _lab(_ref_palette(ref_crop)[None])[0]
    lab_ai = _lab(ai).reshape(-1, 3)
    d = np.full(len(lab_ai), np.inf, np.float32)
    for c in pal:
        d = np.minimum(d, np.linalg.norm(lab_ai - c, axis=1))
    new_pct = float((d > DE_NEW).mean() * 100)
    def edge_len(g):
        return float((cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.2).astype(np.uint8), 40, 120) > 0).sum())
    er, ea = edge_len(gr), edge_len(ga)
    ratio = ea / er if er else 1.0
    redo = []
    if shift > SHIFT_MAX:
        redo.append(f'motif khisak gaye ({shift * 100:.1f}% chaudai)')
    if new_pct > NEW_COLOUR_MAX:
        redo.append(f'naye rang aaye ({new_pct:.0f}% pixel)')
    if ratio < EDGE_LOST:
        redo.append(f'detail kam hui (kinare {ratio * 100:.0f}% reh gaye)')
    if ratio > EDGE_ADDED:
        redo.append(f'nayi detail jodi gayi (kinare {ratio * 100:.0f}%)')
    return {'shift_percent': round(shift * 100, 2), 'new_colour_percent': round(new_pct, 1),
            'edge_ratio': round(ratio, 2), 'redo': redo}


def parts_check(reference, tiles, plan=None, grid=(3, 3), overlap=0.15):
    """Check every redrawn tile against the matching crop of the reference. `tiles`: files named r1c1... (any size).
    Returns {'tiles': {name: result}, 'redo': [names]}."""
    import json
    ref = cv2.cvtColor(read_cv2(reference, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = ref.shape[:2]
    if plan:
        with open(plan, encoding='utf-8') as fh:
            pl = json.load(fh)
        boxes = {tuple(int(v) for v in k[1:].split('c')): tuple(b) for k, b in pl['boxes'].items()}
        if pl['size'] != [W, H]:
            raise ValueError(f"plan {pl['size']} px ke reference ke liye hai, ye image {W}x{H} hai")
    else:
        boxes = _geometry(W, H, grid, overlap)
    res = {}
    for t in tiles:
        rc = _rc(t)
        if rc not in boxes:
            continue
        x0, y0, fw, fh = boxes[rc]
        ai = cv2.cvtColor(read_cv2(t, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        res[f'r{rc[0]}c{rc[1]}'] = check_tile(ref[y0:y0 + fh, x0:x0 + fw], ai)
    return {'tiles': res, 'redo': [k for k, v in res.items() if v['redo']]}
