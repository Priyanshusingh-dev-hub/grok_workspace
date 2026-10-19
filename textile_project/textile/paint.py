"""Paint a sketch: line art in, the colours the user names out, one channel per colour.

No reference image and no guessing: the user says which part takes which
colour. Two steps, the same command:

  1. `textile paint sketch.png --out o/`: the sketch's closed areas are found
     (Method 4's way: Lanczos up, the 3x3 blur, threshold, the line art's small
     gaps sealed) and put into GROUPS of look-alike shapes (same size and
     shape, turned or mirrored: every petal of one kind is one group), lettered
     A, B, C... biggest first. NAME_map.png shows the letters, NAME_numbers.png
     each area's own number, NAME_groups.txt the list.
  2. `textile paint sketch.png --out o/ --colors "A=cream, B=laal, C=hara, lines=coffee"`:
     every area takes its colour, drawn hard-edged at --size, then the usual
     package (TIF for the mill, PNG, one channel per colour, B/W, report),
     checked by verify like every other command.

What can be named (later wins over earlier, a number over its group):
  A, B, ... a group       12, 40-45  areas by number      ground  the biggest area's group
  lines   the sketch's own lines (a colour, or `fill`: each line pixel takes the
          nearest area's colour, so the design has no outline)
  tiny    areas too small to letter, not named by number
  rest    every area not named
          Both default to `fill`: each pixel takes the colour of the nearest area
          that WAS named, so nothing is left white or guessed (the user's rule: a
          white the user named stays white, an unnamed gap never becomes white).
          A colour given here is used instead.
A colour is a hex (F2E8CF), a word in English or Hinglish (laal, halka neela,
mehendi; backend/app/colour_words.py when the repo has it, else names.py's
list), or another key (C=A: the same colour as A). Two keys with the same
colour are one channel.

Mill rules hold: every pixel exactly one colour, no anti-aliasing (the index
is drawn, never resampled), no '#' in names. Nothing here picks a colour the
user did not name except `rest` and `tiny`, and the report lists both.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from . import names as nm
from .fill_method1 import FillError, output_size
from .fill_method4 import default_seal
from .io_utils import hex_of, read_cv2, rgb_of

PAINT_SEAL = 2.0          # gaps closed up to 2 of the sketch's own px (fill uses 1.5): the user's flower sketch
                          # leaked its big leaf into the ground at 1.5, not at 2 (1256 px -> seal 6 at 3535)
SEP_GREY_MIN = 40         # with separators=fill: a line pixel this light or lighter is a separator (drawn 80 grey)
TINY_SHARE = 0.00002      # an area under 0.002% of the design (250 px at 3535) gets no letter
AREA_TOL, SHAPE_TOL = 0.25, 0.12   # one group: log area within 0.25 (~25%), shape numbers within 0.12
JAALI_MAX = 0.6          # 'A*': the lines crossed must be under 60% of the sketch's (2218 35%, no-jaali sketches 85%+)
NUMBERS_WIDTH = 2600     # the numbers sheet is bigger: small areas' numbers must still be readable on zoom
MAP_WIDTH = 1800          # the maps are for a phone screen, not for print


@dataclass
class Regions:
    lab: np.ndarray            # H x W int32, area number; 0 = line or seal
    lines: np.ndarray          # H x W bool, the sketch's own lines
    area: np.ndarray           # n + 1, px per area (0 unused)
    group: np.ndarray          # n + 1, group number; -1 = tiny (and area 0)
    letters: list              # group number -> letter
    ground: int                # the group of the biggest area
    seal: int
    sketch_hash: str
    groups: list = field(default_factory=list)   # [{letter, areas, share}]
    shape: np.ndarray | None = None              # n + 1 x 5, each area's shape numbers (tiny: nan)
    rep_shape: np.ndarray | None = None          # G x 5, each group's first (biggest) member's
    tolerance: float = 1.0
    raw: np.ndarray | None = None                # the sketch's grey at --size, before the blur (separators)
    sep: str | None = None                       # 'fill': grey (separator) lines take the nearest area's colour

    @property
    def n(self):
        return len(self.area) - 1


def letter(i: int) -> str:
    """0 -> A ... 25 -> Z, 26 -> AA, 27 -> AB ..."""
    s = ''
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


# --- 1. areas and groups ------------------------------------------------------------------------

def find_regions(sketch_path, size=3535, line_threshold=150, seal=None, tolerance=1.0, log=print) -> Regions:
    line = read_cv2(sketch_path, cv2.IMREAD_GRAYSCALE)
    if line is None:
        raise FillError('sketch read nahi hua - path check karo.')
    W, H = output_size(line.shape[1], line.shape[0], size)
    raw = line if line.shape[::-1] == (W, H) else cv2.resize(line, (W, H), interpolation=cv2.INTER_LANCZOS4)
    g = cv2.GaussianBlur(raw, (3, 3), 0)
    lines = g < line_threshold
    del g
    if lines.mean() > 0.6 or lines.mean() < 0.001:
        raise FillError(f'ye sketch nahi lagta ({lines.mean() * 100:.0f}% pixel line hain). Safed kagaz par kaali '
                        'lines wala line art do.')
    r = default_seal(line.shape[1], W, PAINT_SEAL) if seal is None else int(seal)
    if r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        sealed = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    else:
        sealed = lines
    lab, n = ndimage.label(~sealed)           # 4-connected: no leak across a diagonal
    if r > 0:
        # an area the seal swallowed whole (a dot between two close lines) stays an area of its own
        lab0, n0 = ndimage.label(~lines)
        alive = np.bincount(lab0[~sealed], minlength=n0 + 1)
        lost = np.flatnonzero(alive == 0)
        lost = lost[lost > 0]
        if len(lost):
            new = np.zeros(n0 + 1, np.int32)
            new[lost] = np.arange(n + 1, n + 1 + len(lost))
            m = (new[lab0] > 0) & (lab == 0)
            lab[m] = new[lab0[m]]
            n += len(lost)
        del lab0
    lab = lab.astype(np.int32)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    area[0] = 0
    log(f'[sketch] {W}x{H} px, seal {r} px, {n} band hisse')
    group, letters, ground, groups, shape, rep_shape = _group(lab, area, W * H, tolerance)
    log(f'[groups] {len(letters)} group (ek jaise hisse ek letter), {int((group[1:] < 0).sum())} bahut chhote hisse (tiny)')
    digest = hashlib.sha256(np.packbits(lines).tobytes()).hexdigest()[:12]
    return Regions(lab, lines, area, group, letters, ground, r, digest, groups, shape, rep_shape, tolerance, raw)


def _shape(mask):
    """Size-free shape numbers that do not change when the shape is turned or
    mirrored, each roughly 0-1 and steady (a log of a Hu moment near zero is
    not: a circle read 7.5 at one size and 0 at another):
      spread       -log10 of the first Hu moment (a disc 0.80, a thin leaf less)
      lopsided     sqrt|hu3| / hu1^1.5 (a disc or square 0, a triangle 0.8)
      stretched    sqrt(hu2) / hu1 (a disc 0, a 3:1 ellipse 0.8)
      box fill     area / the smallest turned box around it (a disc 0.79, a square 1, a triangle 0.5)
      holes        how many (other areas inside it): a ring is never a disc, the ground never a petal"""
    m8 = mask.astype(np.uint8)
    m = cv2.moments(m8, binaryImage=True)
    hu = cv2.HuMoments(m).ravel()
    edge, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # the box only needs the outline
    (_, _), (bw, bh), _ = cv2.minAreaRect(np.vstack(edge))         # (all pixels: 9 s a run, same box to 1e-6)
    fill = m['m00'] / max((bw + 1) * (bh + 1), 1.0)
    _, k = ndimage.label(np.pad(m8 == 0, 1, constant_values=True))
    return np.array([-np.log10(hu[0]), np.sqrt(abs(hu[2])) / hu[0] ** 1.5, np.sqrt(abs(hu[1])) / hu[0], fill,
                     min(k - 1, 50)])


def _group(lab, area, total, tolerance):
    """Areas of the same size and shape -> one group. Greedy, biggest first:
    an area joins the first group whose first member is within 25% in area
    and close in shape; tiny areas get group -1."""
    n = len(area) - 1
    group = np.full(n + 1, -1, np.int64)
    tiny = TINY_SHARE * total
    boxes = ndimage.find_objects(lab)
    reps = []                               # (log area, shape, group)
    shape = np.full((n + 1, 5), np.nan)
    order = sorted(range(1, n + 1), key=lambda i: -area[i])
    for i in order:
        if area[i] < tiny or boxes[i - 1] is None:
            continue
        sl = boxes[i - 1]
        f = shape[i] = _shape(lab[sl] == i)
        la = np.log(area[i])
        hit = -1
        for ra, rf, gi in reps:
            if abs(la - ra) < AREA_TOL * tolerance and np.abs(f - rf).max() < SHAPE_TOL * tolerance:
                hit = gi
                break
        if hit < 0:
            hit = len(reps)
            reps.append((la, f, hit))
        group[i] = hit
    # letters by the group's total area, biggest first
    G = len(reps)
    tot = np.bincount(group[group >= 0], weights=area[group >= 0], minlength=G)
    rank = np.argsort(-tot, kind='stable')
    new = np.empty(G, np.int64)
    new[rank] = np.arange(G)
    group[group >= 0] = new[group[group >= 0]]
    letters = [letter(i) for i in range(G)]
    ground = int(group[int(np.argmax(area))]) if G else -1
    groups = []
    for gi in range(G):
        ids = np.flatnonzero(group == gi)
        groups.append({'letter': letters[gi], 'areas': ids.tolist(), 'share': float(area[ids].sum() / total)})
    rep_shape = np.array([reps[j][1] for j in rank]).reshape(G, 5)
    return group, letters, ground, groups, shape, rep_shape


def across_thin(reg: Regions, gi: int, width: int | None = None):
    """'A*': group `gi` and every area reached from it across THIN lines only
    (a lattice / jaali over the ground), never across a motif's thicker outline.
    Lines thinner than `width` px are opened away (the seal closes their gaps
    as for the areas), what is left is labelled, and every area inside a part
    that holds one of `gi`'s areas is chosen. No width given: the one after the
    biggest jump in what gets chosen, from 3 to 14 px, if the lines it takes
    away are under JAALI_MAX of the sketch's lines (a jaali is a thinner part
    of the drawing, not most of it): 2218's lattice goes at 6 px, 35% of its
    lines, and the ground jumps 2% -> 60%. Without a jaali the jump comes only
    when the motifs' own outlines go (floral at 8 px: 85% of its lines, star
    93%, tree and an even-lined sketch ~100%): then nothing is added and the
    note says so. Returns (areas, width or None, share by width)."""
    seed = np.flatnonzero(reg.group == gi)
    total = reg.lab.size
    k_seal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reg.seal + 1,) * 2) if reg.seal else None
    lines8 = reg.lines.astype(np.uint8)

    def chosen(t):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
        thick = cv2.morphologyEx(lines8, cv2.MORPH_OPEN, k)
        if k_seal is not None:
            thick = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, k_seal)
        co, _ = ndimage.label(thick == 0)
        cid = np.zeros(reg.n + 1, np.int64)
        m = reg.lab > 0
        cid[reg.lab[m]] = co[m]                        # an area lies in one part (thick lines are a subset)
        parts = np.setdiff1d(np.unique(cid[seed]), [0])
        hit = np.isin(cid, parts)
        hit[0] = False
        hit[seed] = True
        return np.flatnonzero(hit)

    if width is not None:
        ids = chosen(int(width))
        return ids, int(width), {int(width): round(float(reg.area[ids].sum() / total * 100), 1)}
    shares, picks = {}, {}
    for t in range(3, 15):
        picks[t] = chosen(t)
        shares[t] = round(float(reg.area[picks[t]].sum() / total * 100), 1)
    ts = sorted(shares)
    jumps = [(shares[b] - shares[a], b) for a, b in zip(ts, ts[1:])]
    jump, best = max(jumps)
    if jump < 5 or float((stroke_widths(reg) <= best).mean()) > JAALI_MAX:
        return seed, None, shares
    return picks[best], best, shares


def stroke_widths(reg: Regions) -> np.ndarray:
    """The width (px) all along the sketch's lines: twice the distance to paper on each line's middle."""
    dt = ndimage.distance_transform_edt(reg.lines)
    ridge = reg.lines & (dt >= ndimage.maximum_filter(dt, 3) - 1e-6)
    return 2 * dt[ridge]


def alike(reg: Regions, gi: int) -> np.ndarray:
    """Areas shaped like group `gi`, of any size ('A+'): its own and every
    lettered area whose shape numbers are as close as a group's must be."""
    d = np.abs(reg.shape - reg.rep_shape[gi]).max(1)
    hit = (d < SHAPE_TOL * reg.tolerance) | (reg.group == gi)
    hit[0] = False
    return np.flatnonzero(hit)


# --- 2. what the user said ----------------------------------------------------------------------

def _colour_words():
    """backend/app/colour_words.py (Hinglish names) when this repo has it; it
    is standard library only, so it is loaded straight from its file."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, '..', '..', 'backend', 'app', 'colour_words.py')
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location('_loomlab_colour_words', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CW = _colour_words()
_HEX = re.compile(r'^#?([0-9a-fA-F]{6})$')


def colour_of(word: str) -> str | None:
    """'F2E8CF', '#f2e8cf', 'laal', 'halka neela', 'cream' -> 'F2E8CF' (or None)."""
    w = word.strip()
    m = _HEX.match(w)
    if m:
        return m.group(1).upper()
    if _CW is not None:
        hx = _CW.lookup(w.lower())
        if hx:
            return hx.lstrip('#').upper()
    key = re.sub(r'[\s_-]+', '', w.lower())
    if key in nm.COLOURS:
        return hex_of(nm.COLOURS[key])
    return None


_SPECIAL = {'lines': 'lines', 'line': 'lines', 'outline': 'lines', 'lakeer': 'lines', 'rekha': 'lines',
            'ground': 'ground', 'zameen': 'ground', 'background': 'ground', 'bg': 'ground',
            'tiny': 'tiny', 'chhote': 'tiny', 'chote': 'tiny',
            'separators': 'sep', 'separator': 'sep', 'grey': 'sep', 'gray': 'sep',
            'rest': 'rest', 'baaki': 'rest', 'baki': 'rest', 'others': 'rest'}
_CLAUSE = re.compile(r'^(.+?)\s*(?:=|:|->|→|\bko\b)\s*(.+)$', re.I)


def parse_colors(text: str):
    """'A=cream, B C=laal; 12 40-45 = hara; lines=coffee' -> [(keys, colour word)], in order."""
    out = []
    text = '\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('#'))
    text = re.sub(r'\([^)]*\)', ' ', text)            # the template's notes: '(6 hisse, 1.6%)'
    for clause in re.split(r'[;,\n]+', text):
        clause = clause.strip()
        if not clause:
            continue
        m = _CLAUSE.match(clause)
        if not m:
            raise FillError(f"'{clause}' samajh nahi aaya. Aise likho: A=cream, B=laal, 12=hara, lines=coffee")
        keys = []
        for k in re.split(r'[\s&]+|\baur\b', re.sub(r'\s+\+', '+', m.group(1).strip())):
            k = k.strip()
            if k and k != '+':
                keys.append(k)
        out.append((keys, m.group(2).strip()))
    return out


def plan(reg: Regions, said):
    """[(keys, colour)] -> (colour per area (n + 1, hex), line colour hex or 'fill',
    tiny colour hex or 'fill', notes). Said in order, later over earlier, area by
    area. Raises FillError on a key or colour it cannot read."""
    G = len(reg.letters)
    by_letter = {l: i for i, l in enumerate(reg.letters)}
    area_col = np.full(reg.n + 1, None, object)
    special = {}
    resolved = {}                               # key -> hex, for 'C=A'
    notes_early = []

    def read_colour(word):
        w = word.strip()
        up = w.upper()
        if up in resolved:
            return resolved[up]
        if w.lower() in ('fill', 'none', 'bharo', 'padosi'):
            return 'fill'
        hx = colour_of(w)
        if hx is None:
            raise FillError(f"rang '{word}' nahi pehchana. Hex do (jaise F2E8CF) ya naam (laal, hara, cream, navy...)")
        return hx

    for keys, word in said:
        hx = read_colour(word)
        for k in keys:
            low, up = k.lower(), k.upper()
            kind = _SPECIAL.get(low)
            if kind == 'sep':                           # 'separators=fill': grey lines are only where colours meet
                if hx != 'fill':
                    raise FillError("separators ke liye sirf 'fill' (grey line = do rangon ki seema, paas ka rang)")
                reg.sep = 'fill'
                continue
            if kind in ('lines', 'tiny', 'rest'):
                special[kind] = hx
                resolved[up] = hx
                if kind == 'lines':
                    resolved['LINE'] = resolved['OUTLINE'] = hx
                continue
            if hx == 'fill':
                raise FillError(f"'{k}' ko 'fill' nahi, ek rang do (fill sirf lines aur tiny ke liye)")
            if kind == 'ground':
                if reg.ground < 0:
                    raise FillError('is sketch me ground nahi mila')
                ids = np.flatnonzero(reg.group == reg.ground)
                resolved['GROUND'] = hx
            elif re.fullmatch(r'\d+(-\d+)?', k):
                a, _, b = k.partition('-')
                lo, hi = int(a), int(b or a)
                if lo < 1 or hi > reg.n or lo > hi:
                    raise FillError(f'hissa {k} nahi hai (1 se {reg.n} tak hain, NAME_numbers.png dekho)')
                ids = np.arange(lo, hi + 1)
            elif up in by_letter:
                ids = np.flatnonzero(reg.group == by_letter[up])
                resolved[up] = hx
            elif re.fullmatch(r'[A-Z]+\*\d*', up) and up.split('*')[0] in by_letter:
                L, _, w = up.partition('*')
                ids, used, _ = across_thin(reg, by_letter[L], int(w) if w else None)
                if used is None:
                    notes_early.append(f'{k}: is sketch me motif se patli jaali nahi mili, sirf {L} bhara. Jaali ho to '
                                       f'motai khud do, jaise {L}*6 (px).')
                else:
                    notes_early.append(f'{k}: {L} aur {used} px se patli lines (jaali) ke paar ke {len(ids)} hisse. '
                                       f'Motif me rang chala jaaye to {L}*{max(used - 1, 3)} likho, kam pakde to {L}*{used + 1}.')
                resolved[up] = hx
            elif up.endswith('+') and up[:-1] in by_letter:
                ids = alike(reg, by_letter[up[:-1]])
                resolved[up] = hx
            else:
                raise FillError(f"'{k}' na group hai na hissa (groups A se {reg.letters[-1] if G else '-'} tak, "
                                "A+ = A jaise saare, A* = A + jaali ke paar, ya lines / ground / tiny / rest)")
            area_col[ids] = hx
    notes = notes_early
    if not any(c is not None for c in area_col[1:]):
        raise FillError('koi rang nahi bataya. Jaise: --colors "A=cream, B=laal, lines=coffee"')
    # an area nobody named is never guessed white or ground: it takes the colour of the
    # nearest NAMED area (named colours, white included, are never touched)
    open_ = np.flatnonzero((area_col == None) & (reg.group >= 0))  # noqa: E711 (object array)
    if len(open_):
        rest = special.get('rest', 'fill')
        if 'rest' not in special:
            notes.append(f"{len(open_)} hisson ka rang nahi bataya ({_short([str(i) for i in open_])}): "
                         "unhe sabse paas wale bataye hue rang se bhara. Badalna ho to unka number aur rang do.")
        area_col[open_] = rest
    line = special.get('lines')
    if line is None:
        line = _sketch_ink(reg)
        notes.append(f'lines ka rang nahi bataya: sketch ki lines ka apna rang {line} rakha.')
    tiny = special.get('tiny', 'fill')
    area_col[area_col == None] = tiny  # noqa: E711
    area_col[0] = None
    return list(area_col), line, tiny, notes


def filled(path) -> bool:
    """A colours file with at least one line that is not a comment."""
    if not os.path.exists(path):
        return False
    with open(path, encoding='utf-8') as fh:
        return any(l.strip() and not l.lstrip().startswith('#') for l in fh)


def csv_colours(path) -> str:
    """A table (CSV/Excel export) of part -> colour, read as 'key=colour' lines.
    Either a row per part, or a row per colour with a Numbers column ("1, 33, 65").
    The part column: Number / No / Hissa / Key / Letter / Group (or the first);
    the colour: HEX / Hex / Colour / Color, else R,G,B. Other columns (a name
    such as 'Warm Ivory') are ignored: the hex is the colour."""
    import csv
    with open(path, encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise FillError(f'{os.path.basename(path)} khaali hai')
    cols = {c.strip().lower(): c for c in rows[0] if c}
    many = next((cols[c] for c in ('numbers', 'parts', 'hisse', 'areas', 'keys') if c in cols), None)
    key = many or next((cols[c] for c in ('number', 'no', 'num', 'hissa', 'key', 'letter', 'group', 'area') if c in cols),
               list(rows[0])[0])
    hexcol = (next((cols[c] for c in cols if 'hex' in c), None)          # 'HEX', 'RGB/HEX': the code, not the name
              or next((cols[c] for c in ('colour', 'color', 'rang') if c in cols), None))
    out = []
    for n, r in enumerate(rows, 2):
        k = (r.get(key) or '').strip()
        if not k:
            continue
        if hexcol and (r.get(hexcol) or '').strip():
            c = r[hexcol].strip()
            m = re.search(r'#([0-9a-fA-F]{6})\b', c)       # 'Ivory / Warm Cream #FFF4D6' -> FFF4D6
            if m:
                c = m.group(1)
        elif all(r.get(cols.get(x, x)) not in (None, '') for x in ('r', 'g', 'b')):
            c = ''.join(f"{int(float(r[cols[x]])):02X}" for x in ('r', 'g', 'b'))
        else:
            raise FillError(f'{os.path.basename(path)} line {n}: {k} ka rang nahi mila (HEX ya R,G,B column chahiye)')
        if many:                                    # one row per colour: 'Numbers' = "1, 33, 65"
            k = ' '.join(t for t in re.split(r'[\s,;]+', k) if t)
        out.append(f'{k}={c.lstrip("#")}')
    return '\n'.join(out)


def stamp(reg: Regions) -> str:
    """The first line of a saved colours file: letters and numbers belong to
    this sketch at this size and seal only."""
    H, W = reg.lab.shape
    return f'# textile paint: {W}x{H} seal {reg.seal} sketch {reg.sketch_hash}'


def stamp_mismatch(reg: Regions, text: str):
    """The saved stamp in `text` if it is not this run's, else None."""
    for l in text.splitlines():
        if l.startswith('# textile paint:'):
            return l if l.strip() != stamp(reg) else None
    return None


def _short(items, k=12):
    return ', '.join(items[:k]) + (f' ... (+{len(items) - k})' if len(items) > k else '')


def _sketch_ink(reg):
    """The sketch lines' colour when none is named: black, the ink a sketch is drawn in."""
    return '141414'


# --- 3. the coloured design ---------------------------------------------------------------------

def paint(reg: Regions, area_col, line, tiny):
    """-> (index H x W, uint8 or uint16 past 255 colours; palette K x 3 uint8; line palette index or None)."""
    cols = [c for c in area_col[1:] if c and c != 'fill']
    if line != 'fill':
        cols.append(line)
    hexes = list(dict.fromkeys(cols))
    if len(hexes) > 65535:
        raise FillError(f'{len(hexes)} alag rang bataye hain: 65535 se zyada nahi ban sakte.')
    dt = np.uint8 if len(hexes) <= 256 else np.uint16      # one byte per pixel when it fits (as before)
    pos = {h: i for i, h in enumerate(hexes)}
    lut = np.zeros(reg.n + 1, np.int64)
    fill_area = np.zeros(reg.n + 1, bool)
    for i in range(1, reg.n + 1):
        c = area_col[i]
        if c == 'fill':
            fill_area[i] = True
        else:
            lut[i] = pos[c]
    out = lut[reg.lab].astype(dt)
    # the seal (a closed gap, not a line) and 'fill' pixels take the nearest real area's colour
    solid = (reg.lab > 0) & ~fill_area[reg.lab]
    todo = ~solid & ~reg.lines
    inked = reg.lines
    if reg.sep == 'fill' and reg.raw is not None:
        # a grey line (textile number draws one where two colours meet without an outline) is no ink of its own
        sep = reg.lines & (reg.raw >= SEP_GREY_MIN)
        todo |= sep
        inked = reg.lines & ~sep
    if line == 'fill':
        todo |= reg.lines
    if todo.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~solid, return_indices=True)
        out[todo] = out[iy[todo], ix[todo]]
        del iy, ix
        # an unnamed area is ONE colour, the one most of it lies nearest to: split pixel by
        # pixel it would take two colours along a line the sketch never drew
        m = fill_area[reg.lab] & (reg.lab > 0)
        if m.any():
            pairs = reg.lab[m].astype(np.int64) * len(hexes) + out[m]
            uniq, cnt = np.unique(pairs, return_counts=True)
            area_of, col_of = uniq // len(hexes), uniq % len(hexes)
            order = np.lexsort((-cnt, area_of))               # per area, the most pixels first
            first = np.ones(len(order), bool)
            first[1:] = area_of[order][1:] != area_of[order][:-1]
            pick = np.zeros(reg.n + 1, out.dtype)
            pick[area_of[order][first]] = col_of[order][first]
            out[m] = pick[reg.lab[m]]
    li = None
    if line != 'fill':
        li = pos[line]
        out[inked] = li
    pal = np.array([[int(h[j:j + 2], 16) for j in (0, 2, 4)] for h in hexes], np.uint8)
    used = np.unique(out)
    if len(used) < len(pal):                       # a colour every pixel of which went elsewhere
        remap = np.zeros(len(pal), dt)
        remap[used] = np.arange(len(used))
        out, pal = remap[out], pal[used]
        li = int(remap[li]) if li is not None and li in used else None
    return out, pal, li


# --- maps for the user --------------------------------------------------------------------------

def _font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:                               # Pillow < 10.1: one small bitmap font
        return ImageFont.load_default()


def _tints(G, seed=7):
    rng = np.random.default_rng(seed)
    h = (np.arange(G) * 0.618034) % 1.0
    s = 0.35 + 0.25 * rng.random(G)
    v = 0.86 + 0.12 * rng.random(G)
    i = (h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    rgb = np.select([i[:, None] == k for k in range(6)],
                    [np.stack(c, 1) for c in ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))])
    return (rgb * 255).astype(np.uint8)


_POINTS = []      # the last few (lab, scale, answer): one design's sheets ask for the same map again and again


def _label_points(reg, scale):
    """Per area, the map point deepest inside it, and how deep (map px). Remembered for the same area map (the
    very array) and scale: `number` draws three numbers sheets and a map from one, and each measured it afresh
    (~2 s a time at 3535 px). The map comes back read-only, so no caller can change what the next one gets."""
    for lab, sc, ans in _POINTS:
        if lab is reg.lab and sc == scale:
            small, best, depth = ans
            return small, list(best), depth.copy()
    ans = _measure_points(reg, scale)
    ans[0].flags.writeable = False
    _POINTS.insert(0, (reg.lab, scale, ans))
    del _POINTS[2:]
    small, best, depth = ans
    return small, list(best), depth.copy()


def _measure_points(reg, scale):
    small = cv2.resize(reg.lab, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    inner = small.copy()
    # an area's own inside: not at a boundary with another label
    edge = np.zeros(small.shape, bool)
    edge[:, 1:] |= small[:, 1:] != small[:, :-1]
    edge[:, :-1] |= small[:, 1:] != small[:, :-1]
    edge[1:, :] |= small[1:, :] != small[:-1, :]
    edge[:-1, :] |= small[1:, :] != small[:-1, :]
    inner[edge] = 0
    dist = ndimage.distance_transform_edt(np.pad(inner > 0, 1))[1:-1, 1:-1]   # the image's edge is an edge too
    best, depth = _deepest(dist, inner, reg.n)
    return small, best, depth


def _deepest(dist, inner, n):
    """Per area 1..n: its deepest point (row, col) and that depth (0 for an area with no inside). Of the points
    equally deep, the middle one in reading order, so a number sits mid-stripe, not at an end. The deepest point
    is a local maximum of `dist` (areas are parted by a ring of zeros), so only those few are sorted: scipy's
    maximum_position sorted every pixel per call (~30 s of a run at 3535 px, 10x less now)."""
    cand = (inner > 0) & (dist >= ndimage.grey_dilation(dist, size=(3, 3)))
    idx = np.flatnonzero(cand)
    v = dist.ravel()[idx]
    lab = inner.ravel()[idx].astype(np.int64)
    order = np.lexsort((idx, -v, lab))                    # by area, deepest first, then reading order
    idx, v, lab = idx[order], v[order], lab[order]
    depth = np.zeros(n + 1)
    start = np.full(n + 1, -1, np.int64)
    head = np.r_[True, lab[1:] != lab[:-1]]
    start[lab[head]] = np.flatnonzero(head)
    depth[lab[head]] = v[head]
    ties = np.bincount(lab[v == depth[lab]], minlength=n + 1)
    pos = np.zeros(n + 1, np.int64)
    has = start >= 0
    pos[has] = idx[start[has] + ties[has] // 2]
    W = dist.shape[1]
    return [(int(p // W), int(p % W)) for p in pos[1:]], depth[1:]


def _numbers(reg: Regions, out_dir, name, colours=None, kind='numbers', outline=False, full=False, ink=None):
    """The numbers sheet, made bigger (up to the design's own size) while some
    number finds no free place: a dense design needs more room, not smaller text.
    `full`: drawn at the design's own size straight away (textile number: every
    number sharp when zoomed in, small areas numbered inside)."""
    W = reg.lab.shape[1]
    for width in ((W,) if full else (NUMBERS_WIDTH, 4000, 5400, W)):
        path, missed = _numbers_at(reg, out_dir, name, min(width, W), colours, kind, outline, full, ink)
        if missed == 0 or width >= W:
            return path, missed
    return path, missed


def _numbers_at(reg: Regions, out_dir, name, width, colours=None, kind='numbers', outline=False, full=False, ink=None):
    """NAME_numbers.png like a colouring book: white areas, the sketch's lines
    and EVERY area's number. A number goes inside its area when it fits (as big
    as fits, down to 9 px on a NUMBERS_WIDTH sheet); an area too small for that
    gets a dot, and its number (blue) sits in the nearest free white space with
    a thin line to the dot, never on a line or another number. Returns (path,
    how many areas found no free place at all: normally 0)."""
    H, W = reg.lab.shape
    small, best, depth = _label_points(reg, min(1.0, width / W))
    h, w = small.shape
    best = [tuple(p) for p in best]
    boxes = None
    for i in np.flatnonzero(np.asarray(depth) <= 0) + 1:   # gone in the shrink: place it from the full size
        if boxes is None:
            boxes = ndimage.find_objects(reg.lab)
        sl = boxes[i - 1]
        if sl is None:
            continue
        ys, xs = np.nonzero(reg.lab[sl] == i)
        k = len(ys) // 2
        best[i - 1] = ((sl[0].start + ys[k]) * h / H, (sl[1].start + xs[k]) * w / W)
    lines_small = cv2.resize(reg.lines.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0
    if ink is not None:                                    # numbers over a drawn sketch: its dark px are the lines
        lines_small = cv2.resize((ink < 128).astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0
    if colours is None:
        base = np.full((h, w, 3), 255, np.uint8)
        base[lines_small] = (40, 40, 40)
    else:                                                  # the check sheet: the painted design under the numbers
        base = np.ascontiguousarray(cv2.resize(colours, (w, h), interpolation=cv2.INTER_NEAREST))
        if outline:                                        # the areas' edges drawn (a colour design has no lines):
            light = base.mean(-1) > 110                    # dark on light colours, white on dark ones
            base[lines_small & light] = (20, 20, 20)
            base[lines_small & ~light] = (245, 245, 245)
    k = max(1.0, w / NUMBERS_WIDTH)                       # sizes grow with the sheet, so a big one reads the same
    halo = {} if colours is None else {'stroke_width': max(2, round(2 * k)), 'stroke_fill': (255, 255, 255)}
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    kk = max(5, round(5 * k))
    taken = cv2.dilate(lines_small.astype(np.uint8), np.ones((kk, kk), np.uint8)) > 0

    def free(box):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            return False
        return not taken[y0:y1 + 1, x0:x1 + 1].any()

    def take(box, pad=3):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        taken[max(y0 - pad, 0):y1 + pad + 1, max(x0 - pad, 0):x1 + pad + 1] = True

    later = []
    for i in range(1, reg.n + 1):
        text, dep = str(i), depth[i - 1]
        px = int(min(dep * 1.2, 48 * k, dep * 2.4 / (len(text) * 0.6)))
        y, x = best[i - 1]
        if px < (10 if k > 1 else 9):
            later.append(i)
            continue
        f = _font(px)
        d.text((x, y), text, fill=(200, 20, 40), font=f, anchor='mm', **halo)
        take(d.textbbox((x, y), text, font=f, anchor='mm'), 1)
    f = _font(round(15 * k))
    dot = max(3, round(3 * k))
    missed = 0
    for i in later:                                   # dots first, so no label covers another's dot
        y, x = best[i - 1]
        take((x - dot, y - dot, x + dot, y + dot), 1)
    for i in later:
        y, x = best[i - 1]
        text = str(i)
        spot = None
        for r in range(round(22 * k), round((700 if full else 260) * k), max(10, round(10 * k))):
            for k in range(24):
                a = 2 * np.pi * k / 24
                cx, cy = x + r * np.cos(a), y + r * np.sin(a)
                box = d.textbbox((cx, cy), text, font=f, anchor='mm')
                if free((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2)):
                    spot = (cx, cy, box)
                    break
            if spot:
                break
        d.ellipse([x - dot, y - dot, x + dot, y + dot], fill=(30, 80, 220))
        if spot is None and full:
            # no free place even far off: written beside the dot anyway, on a white halo over the lines,
            # so every area has its number (the full-size sheet cannot grow for more room)
            cx, cy = x + 2 * dot + 4 * k, y - 2 * dot - 4 * k
            spot = (cx, cy, d.textbbox((cx, cy), text, font=f, anchor='mm'))
            halo_here = {'stroke_width': max(2, round(2 * k)), 'stroke_fill': (255, 255, 255)}
        else:
            halo_here = halo
        if spot is None:
            missed += 1
            continue
        cx, cy, box = spot
        # the leader stops at the label's edge
        ex = min(max(x, box[0]), box[2])
        ey = min(max(y, box[1]), box[3])
        d.line([(x, y), (ex, ey)], fill=(30, 80, 220), width=max(1, round(k)))
        d.text((cx, cy), text, fill=(30, 80, 220), font=f, anchor='mm', **halo_here)
        take(box)
    path = os.path.join(out_dir, f'{name}_{kind}.png')
    img.save(path, dpi=(300, 300))
    return path, missed


REF_SAME_DE = 30     # two reference shades closer than this (CIELAB) are one ink. The woven jaal: its real inks
                     # (black, tan, pink, yellow, white) are 40-83 apart, the weave's dark shade 20-24 from black/tan


def _ref_palette(rgb, colours, same_de=None):
    """The reference's inks: k-means to 16 shades in CIELAB, then groups of shades
    merged while EVERY pair inside a group is closer than REF_SAME_DE (complete
    linkage), or while more than `colours` groups are left; a group's colour is
    its pixels' mean. Plain k-means to 5 on the user's woven jaal gave five
    browns and lost the pink and yellow flowers (small shares, far hues); merging
    by the nearest pair alone let white chain into tan through a light-tan shade
    and the white flowers came out yellow."""
    from sklearn.cluster import KMeans
    flat = rgb.reshape(-1, 3)
    sample = flat[np.random.default_rng(0).choice(len(flat), min(len(flat), 200000), replace=False)]
    lab = nm._lab(sample.astype(np.float64))
    km = KMeans(16, n_init=4, random_state=0).fit(lab)
    C = km.cluster_centers_
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    groups = [[j] for j in range(16)]
    while len(groups) > 1:
        best = (1e9, 0, 0)
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                d = D[np.ix_(groups[a], groups[b])].max()
                if d < best[0]:
                    best = (d, a, b)
        d, a, b = best
        if d >= (REF_SAME_DE if same_de is None else same_de) and len(groups) <= colours:
            break
        groups[a] += groups.pop(b)
    out = []
    for g in groups:
        m = np.isin(km.labels_, g)
        out.append(sample[m].mean(0))
    return np.clip(np.rint(np.array(out)), 0, 255).astype(np.uint8)


def from_reference(reg: Regions, ref_path, colours=8, out_dir=None, name='design', log=print):
    """The colours read off a coloured version of the same design (`--ref`): the
    reference is laid on the sketch's grid (same crop, any size), cut to `colours`
    flat colours (k-means, as Method 1 does for a reference that is not flat), and
    each area takes the colour most of its INSIDE shows (3 px clear of the lines,
    where a woven or printed edge blurs); the lines take the colour most common
    under them. Writes NAME_ref_colors.csv (Number, HEX, Colour) and
    NAME_ref_numbers.png (the coloured design with every number on it). Returns
    the 'N=HEX' text, ready for plan(), and the line colour."""
    import csv
    from . import palette as pl
    ref = read_cv2(ref_path, cv2.IMREAD_COLOR)
    if ref is None:
        raise FillError('rangeen design (ref) read nahi hua - path check karo.')
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    H, W = reg.lab.shape
    ar_s, ar_r = W / H, ref.shape[1] / ref.shape[0]
    if abs(ar_s - ar_r) > 0.02:
        raise FillError(f'rangeen design ka naap sketch se alag hai ({ar_r:.3f} vs {ar_s:.3f}): dono ek hi crop ke hone chahiye.')
    ref = cv2.medianBlur(ref, 5)                         # weave / print grain at the photo's own scale, not a colour
    big = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA if ref.shape[1] > W else cv2.INTER_LANCZOS4)
    small = cv2.resize(big, (min(W, 1000), min(H, round(1000 * H / W))), interpolation=cv2.INTER_AREA)
    pal = _ref_palette(small, colours)
    K = len(pal)
    idx = pl.map_to_palette(big, pal)
    inside = (ndimage.distance_transform_edt(~reg.lines) > 3) & (reg.lab > 0)
    votes = np.bincount(reg.lab[inside].astype(np.int64) * K + idx[inside], minlength=(reg.n + 1) * K).reshape(-1, K)
    anyv = np.bincount(reg.lab[reg.lab > 0].astype(np.int64) * K + idx[reg.lab > 0], minlength=(reg.n + 1) * K).reshape(-1, K)
    votes = np.where(votes.sum(1, keepdims=True) > 0, votes, anyv)   # a sliver all edge: all its pixels vote
    col = votes.argmax(1)
    line_col = hex_of(pal[int(np.bincount(idx[reg.lines], minlength=K).argmax())])
    hexes = [hex_of(c) for c in pal]
    rows = [(i, hexes[col[i]], nm.colour_name(pal[col[i]])) for i in range(1, reg.n + 1)]
    if out_dir:
        with open(os.path.join(out_dir, f'{name}_ref_colors.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['Number', 'HEX', 'Colour'])
            for i, hx, cname in rows:
                w.writerow([i, '#' + hx, cname])
            w.writerow(['lines', '#' + line_col, nm.colour_name(rgb_of(line_col))])
        _numbers(reg, out_dir, name, big, kind='ref_numbers')
    log(f'[ref] rangeen design se {K} rang: ' + ', '.join(f'{h} {nm.colour_name(c)}' for h, c in zip(hexes, pal))
        + f'; lines {line_col}')
    text = '\n'.join(f'{i}={hx}' for i, hx, _ in rows) + f'\nlines={line_col}'
    return text, big


def check(reg: Regions, index, pal, out_dir, name, top=12):
    """After painting: NAME_check.png (the painted design with every area's number
    on it, to see which number got which colour) and the biggest areas with their
    colour: a wrong colour on a big area is what shows as colour 'spreading'."""
    path, _ = _numbers(reg, out_dir, name, pal[index], kind='check')
    H, W = reg.lab.shape
    big = np.argsort(-reg.area)[:top]
    rows = []
    for i in big:
        if i == 0 or reg.area[i] == 0:
            continue
        ys, xs = np.nonzero(reg.lab == i) if reg.area[i] < 50000 else np.nonzero(reg.lab[::4, ::4] == i)
        k = len(ys) // 2
        y, x = (ys[k], xs[k]) if reg.area[i] < 50000 else (ys[k] * 4, xs[k] * 4)
        rgb = pal[index[y, x]]
        rows.append((int(i), round(float(reg.area[i] / (H * W) * 100), 2), hex_of(rgb), nm.colour_name(rgb)))
    return path, rows


def maps(reg: Regions, out_dir, name, numbers=True, template=True):
    """NAME_map.png (groups: a tint and a letter each), NAME_numbers.png (every
    area's number), NAME_groups.txt. Returns their paths."""
    H, W = reg.lab.shape
    scale = min(1.0, MAP_WIDTH / W)
    small, best, depth = _label_points(reg, scale)
    lines_small = cv2.resize(reg.lines.astype(np.uint8), small.shape[::-1], interpolation=cv2.INTER_AREA) > 0
    G = len(reg.letters)
    tint = np.vstack([_tints(G), [[225, 225, 225]]]).astype(np.uint8)
    gidx = np.where(reg.group >= 0, reg.group, G)
    base = tint[gidx[small]]
    base[small == 0] = (255, 255, 255)
    base[lines_small] = (30, 30, 30)
    paths = {}
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    for i in range(1, reg.n + 1):
        gi, dep = reg.group[i], depth[i - 1]
        if gi < 0 or dep < 5:
            continue
        px = int(min(max(dep * 1.1, 14), 44))
        y, x = best[i - 1]
        d.text((x, y), reg.letters[gi], fill=(0, 0, 0), font=_font(px), anchor='mm', stroke_width=max(1, px // 8),
               stroke_fill=(255, 255, 255))
    paths['map'] = os.path.join(out_dir, f'{name}_map.png')
    img.save(paths['map'])
    paths['numbers'], unnumbered = _numbers(reg, out_dir, name) if numbers else (None, 0)
    lines = [f'{name}: {reg.n} band hisse, {G} group. Sketch {reg.sketch_hash}, seal {reg.seal} px.',
             'NAME_numbers.png: laal number = hisse ke andar; neela number + line = chhota hissa, line ke neele dot wala.',
             'Letter = ek jaise hisse (same size + shape, ghooma ya ulta bhi). A sabse bada.', '']
    for g in reg.groups:
        ids = g['areas']
        lines.append(f"{g['letter']:>3}: {len(ids):4d} hisse, {g['share'] * 100:6.2f}%  "
                     f"(hisse {_short([str(i) for i in ids], 8)})" + ('   <- ground' if reg.letters.index(g['letter']) == reg.ground else ''))
    ntiny = int((reg.group[1:] < 0).sum())
    if unnumbered:
        lines.append(f'{unnumbered} hisson ke number ke liye jagah nahi mili (sirf neela dot hai).')
    lines += ['', f'tiny: {ntiny} bahut chhote hisse (letter nahi), default lines ka rang.', '',
              'Rang aise batao:  --colors "A=cream, B=laal, C D=hara, 12=gold, lines=coffee"',
              'Keys: A,B.. group | 12 ya 40-45 hissa | ground | lines (ya lines=fill) | tiny | rest (baaki sab)']
    paths['groups'] = os.path.join(out_dir, f'{name}_groups.txt')
    with open(paths['groups'], 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    # a file to fill in (never over one already filled): remove the '# ' and write the colour
    paths['colors'] = os.path.join(out_dir, f'{name}_colors.txt')
    if template and not os.path.exists(paths['colors']):
        rows = [stamp(reg), '# Har line se "# " hatao aur rang likho (hex, ya laal / hara / cream / navy...).',
                '# lines = coffee          (ya lines = fill: koi outline nahi)', '# rest = cream           (jo letter na likha)']
        rows += [f"# {g['letter']} = ?        ({len(g['areas'])} hisse, {g['share'] * 100:.2f}%"
                 + (', ground)' if i == reg.ground else ')') for i, g in enumerate(reg.groups[:60])]
        with open(paths['colors'], 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(rows) + '\n')
    return paths
