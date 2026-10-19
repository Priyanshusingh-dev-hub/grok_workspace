"""Original designs (Kaam D, ROADMAP Phase 4): a seamless repeat or a panel
drawn from a JSON config, in flat colours only.

No reference code exists for this; it follows CLAUDE.md's rules:
  - drawn with PIL on a palette-index canvas: polygons, ellipses and lines
    are hard-edged (no anti-aliasing), so every pixel is one colour
  - seamless: every shape is drawn 9 times, at +-W and +-H (a vertical
    panel only at +-H), so whatever crosses an edge comes back on the other
  - flat palette (the config's colours, 2-12), sizes in mm at the given DPI

Config (all sizes in mm unless said; see examples/):
  {"name": "indigo_buti", "size_inch": [11.78, 11.78], "dpi": 300, "seed": 7,
   "palette": {"ground": "1B2A4A", "cream": "F2E8CF", ...},
   "ground": "ground",
   "layers": [
     {"layout": "half-drop", "cols": 4, "rows": 4,
      "motif": "buti", "size": 38, "colors": ["cream", "gold", "sky"], "rotate": 0},
     {"layout": "scatter", "per_sq_inch": 1.5, "min_gap": 12,
      "motif": "dots", "size": 3, "colors": ["gold"]},
     {"layout": "vine", "motif": "bel", ...}]}

Layouts: straight | half-drop (cols x rows), scatter (per_sq_inch, min_gap,
random rotation), panel-centre (one motif per `rows`, centre column), and the
bands of a panel: side-bel (a vine up both sides) and side-border (zigzag
bands). "repeat": "panel" in the config makes the design repeat only
up and down (side bands touch the edges, so nothing wraps sideways).
"""
from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image, ImageDraw

from .io_utils import rgb_of

MOTIFS = ('phool', 'buti', 'patti', 'sprig', 'dots', 'haathi')
LAYOUTS = ('straight', 'half-drop', 'scatter', 'panel-centre', 'side-bel', 'side-border', 'bel')


class MakeError(ValueError):
    """A config the tool cannot draw, in the user's words."""


class Canvas:
    """An index canvas that draws every shape wrapped round its edges."""

    def __init__(self, w, h, ground, wrap_x=True, wrap_y=True):
        self.w, self.h = w, h
        self.img = Image.new('L', (w, h), ground)
        self.d = ImageDraw.Draw(self.img)
        xs = (-w, 0, w) if wrap_x else (0,)
        ys = (-h, 0, h) if wrap_y else (0,)
        self.offsets = [(dx, dy) for dx in xs for dy in ys]
        self.taken = []                  # (x, y, r) of the motifs placed: scatter keeps clear of them

    def _near(self, pts, dx, dy):
        xs, ys = pts[:, 0] + dx, pts[:, 1] + dy
        return xs.max() >= -2 and xs.min() <= self.w + 2 and ys.max() >= -2 and ys.min() <= self.h + 2

    def polygon(self, pts, c):
        pts = np.asarray(pts, float)
        for dx, dy in self.offsets:
            if self._near(pts, dx, dy):
                self.d.polygon([(x + dx, y + dy) for x, y in pts], fill=int(c))

    def line(self, pts, c, width):
        pts = np.asarray(pts, float)
        width = max(1, int(round(width)))
        for dx, dy in self.offsets:
            if self._near(pts, dx - width, dy - width) or self._near(pts, dx + width, dy + width):
                q = [(x + dx, y + dy) for x, y in pts]
                self.d.line(q, fill=int(c), width=width, joint='curve')
                r = width / 2                                   # round caps, no AA
                for x, y in (q[0], q[-1]):
                    self.d.ellipse((x - r, y - r, x + r, y + r), fill=int(c))

    def circle(self, x, y, r, c):
        r = max(0.5, r)
        self.polygon(_ellipse_pts(x, y, r, r, 0, max(12, int(r * 1.5))), c)

    def ellipse(self, x, y, rx, ry, angle, c):
        self.polygon(_ellipse_pts(x, y, rx, ry, angle, max(16, int(max(rx, ry) * 1.2))), c)

    def index(self):
        return np.asarray(self.img)


# --- geometry -------------------------------------------------------------

def _rot(pts, angle, x, y):
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    p = np.asarray(pts, float)
    return np.stack([x + p[:, 0] * c - p[:, 1] * s, y + p[:, 0] * s + p[:, 1] * c], 1)


def _ellipse_pts(x, y, rx, ry, angle, n):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return _rot(np.stack([rx * np.cos(t), ry * np.sin(t)], 1), angle, x, y)


def _petal(length, width, n=24, point=0.6):
    """A petal along +x from the origin: round base, pointed tip (point 0..1)."""
    t = np.linspace(0, 1, n)
    half = width / 2 * np.sin(np.pi * t) ** 0.7 * (1 - point * t ** 3)
    return np.concatenate([np.stack([t * length, half], 1), np.stack([t[::-1] * length, -half[::-1]], 1)])


def _bezier(p0, p1, p2, p3, n=40):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _teardrop(r, n=60, tail=1.6):
    """A paisley-like drop: round at the origin side, a curled tip."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = r * np.cos(t)
    y = r * np.sin(t) * np.sin(t / 2) ** tail
    pts = np.stack([y, -x], 1)                       # tip up
    bend = (pts[:, 1] / r + 1) / 2                   # 0 at the base, 1 at the tip
    pts[:, 0] += r * 0.45 * bend ** 2                # the tip curls to one side
    return pts


# --- motifs: draw(canvas, x, y, size px, angle, colour indices, rng, line px) --

def phool(cv, x, y, s, a, c, rng, lw, point=0.15):
    """A layered flower: outer petals, inner petals, a centre and a dot.
    `point` 0 = round petals, 0.9 = pointed (a star-like flower)."""
    n = int(rng.integers(6, 9))
    outer, inner, centre, dot = (c * 4)[:4]
    for k in range(n):
        cv.polygon(_rot(_petal(s * 0.5, s * 0.42, point=point), a + 360 * k / n, x, y), outer)
    for k in range(n):
        cv.polygon(_rot(_petal(s * 0.32, s * 0.26, point=point), a + 360 * (k + 0.5) / n, x, y), inner)
    cv.circle(x, y, s * 0.13, centre)
    cv.circle(x, y, s * 0.05, dot)


def buti(cv, x, y, s, a, c, rng, lw):
    """A small paisley buti: drop, inner drop, a row of dots."""
    outer, inner, dots = (c * 3)[:3]
    cv.polygon(_rot(_teardrop(s * 0.42), a, x, y), outer)
    cv.polygon(_rot(_teardrop(s * 0.26), a, x, y + 0 * s), inner)
    for k in range(3):
        cv.circle(*_rot([[0, s * (0.18 - 0.14 * k)]], a, x, y)[0], max(lw * 0.6, s * 0.04), dots)


def patti(cv, x, y, s, a, c, rng, lw):
    """A leaf with a midrib."""
    leaf, rib = (c * 2)[:2]
    cv.polygon(_rot(_petal(s, s * 0.42, point=0.9) - [s / 2, 0], a, x, y), leaf)
    cv.line(_rot([[-s * 0.42, 0], [s * 0.38, 0]], a, x, y), rib, lw)


def sprig(cv, x, y, s, a, c, rng, lw):
    """A curved stem with leaves and a bud."""
    stem, leaf, bud = (c * 3)[:3]
    path = _bezier((-s / 2, s * 0.1), (-s * 0.15, -s * 0.25), (s * 0.15, s * 0.25), (s / 2, -s * 0.1))
    cv.line(_rot(path, a, x, y), stem, max(lw, s * 0.04))
    for k, t in enumerate((0.25, 0.5, 0.75)):
        px, py = path[int(t * (len(path) - 1))]
        side = 1 if k % 2 else -1
        cv.polygon(_rot(_petal(s * 0.3, s * 0.13, point=0.9), a + side * 60 + 10, *_rot([[px, py]], a, x, y)[0]), leaf)
    bx, by = _rot([path[-1]], a, x, y)[0]
    cv.ellipse(bx, by, s * 0.1, s * 0.07, a, bud)


def dots(cv, x, y, s, a, c, rng, lw):
    cv.circle(x, y, s / 2, c[0])


def haathi(cv, x, y, s, a, c, rng, lw, facing=1):
    """A stylised elephant (side view) with a saddle cloth. `s` = body length."""
    body, cloth, trim, light = (c * 4)[:4]
    f = facing
    P = lambda dx, dy: (x + f * dx * s, y + dy * s)
    for lx in (-0.32, -0.15, 0.12, 0.28):                       # legs
        x0, y0 = P(lx, 0.1)
        cv.polygon([(x0 - s * 0.07, y0), (x0 + s * 0.07, y0), (x0 + s * 0.07, y0 + s * 0.38),
                    (x0 - s * 0.07, y0 + s * 0.38)], body)
        cv.ellipse(x0, y0 + s * 0.38, s * 0.08, s * 0.035, 0, light)       # toe nails
    cv.ellipse(*P(0, 0), s * 0.48, s * 0.3, 0, body)                      # body
    cv.circle(*P(0.48, -0.16), s * 0.22, body)                             # head
    trunk = _bezier((0.6, -0.1), (0.75, 0.15), (0.62, 0.42), (0.74, 0.5))
    cv.line([P(px, py) for px, py in trunk], body, s * 0.1)
    cv.line([P(px, py) for px, py in _bezier((0.66, 0.02), (0.78, 0.08), (0.82, 0.02), (0.84, -0.04), 12)],
            light, s * 0.035)                                              # tusk
    cv.ellipse(*P(0.38, -0.12), s * 0.12, s * 0.17, 0, trim)               # ear
    cv.circle(*P(0.55, -0.22), s * 0.025, light)                           # eye
    cv.line([P(-0.46, -0.05), P(-0.56, 0.12)], body, max(lw, s * 0.025))   # tail
    cloth_pts = [P(-0.28, -0.29), P(0.22, -0.29), P(0.26, 0.08), P(-0.32, 0.08)]
    cv.polygon(cloth_pts, cloth)                                           # saddle cloth
    for k in range(7):                                                     # its border dots
        cv.circle(*P(-0.3 + 0.093 * k, 0.05), s * 0.02, trim)


DRAW = {'phool': phool, 'buti': buti, 'patti': patti, 'sprig': sprig, 'dots': dots, 'haathi': haathi}


# --- layouts --------------------------------------------------------------

def _grid(cv, L, mm, rng, cols, rows, half, shift=(0, 0)):
    """Cell centres; `shift` moves the whole grid by a fraction of a cell
    (0.5, 0.5 puts a second motif between the first ones)."""
    cw, rh = cv.w / cols, cv.h / rows
    for i in range(cols):
        for j in range(rows):
            yield ((i + 0.5 + shift[0]) * cw) % cv.w, \
                  ((j + 0.5 + shift[1]) * rh + (rh / 2 if half and i % 2 else 0)) % cv.h


def _scatter(cv, L, mm, rng, per_sq_inch, min_gap, avoid=True):
    """Dart throwing with a minimum distance, measured round the wrap."""
    dpi = mm * 25.4
    n = max(1, int(round(per_sq_inch * cv.w * cv.h / dpi ** 2)))
    gap = min_gap * mm
    pts = np.zeros((0, 2))
    taken = np.array([t for t in cv.taken], float).reshape(-1, 3) if avoid else np.zeros((0, 3))
    size = np.array([cv.w, cv.h], float)

    def far(q, centres, r):
        d = np.abs(centres - q)
        d = np.minimum(d, size - d)                      # measured round the wrap
        return not len(centres) or ((d ** 2).sum(1) >= r ** 2).all()
    tries = 0
    while len(pts) < n and tries < n * 200:
        tries += 1
        p = rng.random(2) * (cv.w, cv.h)
        if far(p, pts, gap) and far(p, taken[:, :2], taken[:, 2] + gap / 2):
            pts = np.vstack([pts, p])
    return [tuple(q) for q in pts]


def _vine(cv, colors, mm, rng, lw, x_of_y=None, amplitude=8, waves=3, leaf=14, along='x', at=None):
    """A sine stem with leaves and curls. along 'x': across the width (at y);
    along 'y': up and down (at x), for a panel's side bands."""
    stem, leafc, curl, flower = (colors * 4)[:4]
    L = cv.w if along == 'x' else cv.h
    period = L / waves
    t = np.linspace(0, L, int(L / 3) + 2)
    off = amplitude * mm * np.sin(2 * np.pi * t / period)
    pts = np.stack([t, at + off], 1) if along == 'x' else np.stack([at + off, t], 1)
    cv.line(pts, stem, lw * 1.4)
    for k in range(waves * 4):
        u = (k + 0.5) * period / 4
        o = amplitude * mm * math.sin(2 * math.pi * u / period)
        slope = math.degrees(math.atan(amplitude * mm * 2 * math.pi / period * math.cos(2 * math.pi * u / period)))
        x, y = (u, at + o) if along == 'x' else (at + o, u)
        base = slope if along == 'x' else 90 - slope
        side = 1 if k % 2 else -1
        if k % 4 == 3:                                     # a curl (spiral tendril)
            th = np.linspace(0, 3.5 * np.pi, 50)
            r = leaf * mm * 0.45 * (1 - th / (4 * np.pi))
            sp = np.stack([r * np.cos(th), side * r * np.sin(th)], 1) - [leaf * mm * 0.45, 0]
            cv.line(_rot(sp, base + side * 50, x, y), curl, lw)
        elif k % 4 == 1:
            phool(cv, *_rot([[0, side * leaf * mm * 0.6]], base, x, y)[0], leaf * mm * 0.9, base,
                  [flower, curl, leafc, stem], rng, lw)
        else:
            cv.polygon(_rot(_petal(leaf * mm, leaf * mm * 0.42, point=0.9), base + side * 55, x, y), leafc)


def _zigzag(cv, colors, x0, x1, mm, teeth):
    """A border band from x0 to x1: two rules and a row of triangles."""
    band, tooth = (colors * 2)[:2]
    w = x1 - x0
    rule = max(2, 0.06 * w)
    cv.polygon([(x0, 0), (x0 + rule, 0), (x0 + rule, cv.h), (x0, cv.h)], band)
    cv.polygon([(x1 - rule, 0), (x1, 0), (x1, cv.h), (x1 - rule, cv.h)], band)
    step = cv.h / teeth
    for k in range(teeth):
        y = k * step
        cv.polygon([(x0 + rule * 1.6, y), (x1 - rule * 1.6, y + step / 2), (x0 + rule * 1.6, y + step)], tooth)


# --- the design -------------------------------------------------------------

def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def check(cfg):
    """The config's mistakes, in the user's words (empty when it is fine)."""
    errs = []
    pal = cfg.get('palette') or {}
    if not 2 <= len(pal) <= 12:
        errs.append('palette me 2 se 12 rang do (naam: hex).')
    for k, v in pal.items():
        try:
            rgb_of(v)
        except ValueError as e:
            errs.append(f'rang "{k}": {e}')
    if cfg.get('ground') not in pal:
        errs.append('"ground" palette ke kisi rang ka naam ho.')
    if cfg.get('repeat', 'all-over') not in ('all-over', 'panel'):
        errs.append('"repeat": "all-over" ya "panel".')
    for i, L in enumerate(cfg.get('layers') or [], 1):
        lay = L.get('layout')
        if lay not in LAYOUTS:
            errs.append(f'layer {i}: layout "{lay}" nahi pata ({", ".join(LAYOUTS)}).')
        if lay in ('straight', 'half-drop', 'scatter', 'panel-centre') and L.get('motif') not in MOTIFS:
            errs.append(f'layer {i}: motif "{L.get("motif")}" nahi pata ({", ".join(MOTIFS)}).')
        for c in L.get('colors') or []:
            if c not in pal:
                errs.append(f'layer {i}: rang "{c}" palette me nahi hai.')
        if not L.get('colors'):
            errs.append(f'layer {i}: "colors" do.')
    if not cfg.get('layers'):
        errs.append('kam se kam ek layer do.')
    return errs


def make(cfg, log=print):
    """(index map, palette uint8 K x 3, names) for the config."""
    errs = check(cfg)
    if errs:
        raise MakeError('config theek karo: ' + ' | '.join(errs))
    dpi = int(cfg.get('dpi', 300))
    mm = dpi / 25.4                                     # px per mm
    if cfg.get('size_px'):
        W, H = (int(v) for v in cfg['size_px'])
    else:
        w_in, h_in = cfg.get('size_inch', [11.78, 11.78])
        W, H = int(round(w_in * dpi)), int(round(h_in * dpi))
    names = list(cfg['palette'])
    idx = {n: i for i, n in enumerate(names)}
    panel = cfg.get('repeat', 'all-over') == 'panel'
    cv = Canvas(W, H, idx[cfg['ground']], wrap_x=not panel, wrap_y=True)
    rng = np.random.default_rng(int(cfg.get('seed', 1)))
    lw = float(cfg.get('line_mm', 0.6)) * mm            # thinnest line: 0.6 mm unless said
    for L in cfg['layers']:
        lay = L['layout']
        cols = [idx[c] for c in L['colors']]
        size = float(L.get('size', 30)) * mm
        draw = DRAW.get(L.get('motif'))
        rot = float(L.get('rotate', 0))
        if lay in ('straight', 'half-drop'):
            spots = list(_grid(cv, L, mm, rng, int(L.get('cols', 4)), int(L.get('rows', 4)), lay == 'half-drop',
                               tuple(L.get('shift', (0, 0)))))
        elif lay == 'scatter':
            spots = _scatter(cv, L, mm, rng, float(L.get('per_sq_inch', 1)), float(L.get('min_gap', 20)),
                             bool(L.get('avoid', True)))
        elif lay == 'panel-centre':
            rows = int(L.get('rows', 1))
            spots = [(W / 2, (j + 0.5) * H / rows) for j in range(rows)]
        else:
            spots = []
        for k, (x, y) in enumerate(spots):
            a = rot + (float(rng.uniform(0, 360)) if lay == 'scatter' and L.get('spin', True) else 0)
            if lay != 'scatter':
                cv.taken.append((x, y, size * (0.75 if draw is haathi else 0.5)))
            if draw is haathi:
                haathi(cv, x, y, size, a, cols, rng, lw, facing=-1 if L.get('alternate') and k % 2 else 1)
            elif draw is phool:
                phool(cv, x, y, size, a, cols, rng, lw, point=float(L.get('point', 0.15)))
            else:
                draw(cv, x, y, size * float(rng.uniform(0.85, 1.15)) if lay == 'scatter' else size, a, cols, rng, lw)
        if lay == 'bel':
            for at in L.get('at_mm', [H / mm / 2]):
                _vine(cv, cols, mm, rng, lw, amplitude=float(L.get('amplitude', 8)), waves=int(L.get('waves', 3)),
                      leaf=float(L.get('leaf', 14)), along='x', at=float(at) * mm)
        elif lay == 'side-bel':
            band = float(L.get('band', 40)) * mm
            for at in (band / 2, W - band / 2):
                _vine(cv, cols, mm, rng, lw, amplitude=float(L.get('amplitude', 6)), waves=int(L.get('waves', 3)),
                      leaf=float(L.get('leaf', 12)), along='y', at=at)
        elif lay == 'side-border':
            band = float(L.get('band', 20)) * mm
            start = float(L.get('from', 0)) * mm
            teeth = int(L.get('teeth', 12))
            _zigzag(cv, cols, start, start + band, mm, teeth)
            _zigzag(cv, cols, W - start - band, W - start, mm, teeth)
        log(f'[make] layer {lay}' + (f' {L.get("motif")} x{len(spots)}' if spots else ''))
    index = cv.index()
    pal = np.array([rgb_of(cfg['palette'][n]) for n in names], np.uint8)
    used = np.unique(index)
    if len(used) < len(names):                           # a colour no layer drew: no empty channel
        remap = np.zeros(len(names), np.uint8); remap[used] = np.arange(len(used))
        index, pal, names = remap[index], pal[used], [names[u] for u in used]
    return index, pal, names
