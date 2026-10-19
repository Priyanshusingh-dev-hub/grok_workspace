"""textile number: a coloured design, every patch of one colour numbered."""
import csv

import numpy as np
from PIL import Image, ImageDraw

from textile.cli import main


def test_every_colour_patch_gets_a_number_and_its_colour(tmp_path):
    im = Image.new('RGB', (600, 600), (20, 15, 12))                       # black ground
    d = ImageDraw.Draw(im)
    d.ellipse([60, 60, 240, 240], fill=(170, 55, 70))                     # pink flower
    d.ellipse([360, 60, 540, 240], fill=(170, 55, 70))                    # another one
    d.rectangle([100, 360, 500, 480], fill=(214, 189, 95))                # yellow band
    d.point([(300, 300)], fill=(255, 255, 255))                           # one speck of grain
    im.save(tmp_path / 'd.png')
    out = tmp_path / 'o'
    assert main(['number', str(tmp_path / 'd.png'), '--out', str(out), '--size', '600']) == 0
    rows = [r for r in csv.DictReader(open(out / 'd_colors.csv')) if r['Number'] not in ('lines', 'separators')]
    assert len(rows) == 4                                                  # ground, 2 flowers, band; the speck melted
    hexes = sorted(r['HEX'] for r in rows)
    assert hexes.count('#AA3746') == 2 or sum(h[1:3] in ('A9', 'AA', 'AB') for h in hexes) == 2
    for f in ('d_numbers.png', 'd_flat.png'):
        assert (out / f).exists()
    flat = np.asarray(Image.open(out / 'd_flat.png').convert('RGB'))
    assert len(np.unique(flat.reshape(-1, 3), axis=0)) == 3                # three inks


def test_a_clean_design_keeps_its_thin_outline_and_drops_the_edge_blend(tmp_path):
    from textile import number as nb
    im = Image.new('RGB', (300, 300), (242, 232, 204))                    # cream
    d = ImageDraw.Draw(im)
    d.ellipse([60, 60, 240, 240], fill=(140, 55, 70), outline=(15, 10, 10), width=2)   # maroon, thin black outline
    for k in range(6):                                                    # small cream dots inside: design
        d.ellipse([100 + 18 * k, 145, 106 + 18 * k, 151], fill=(242, 232, 204))
    im = im.resize((900, 900), Image.BICUBIC)                             # soft edges: a blend between black and cream
    im.save(tmp_path / 'c.png')
    rgb = np.asarray(im)
    assert nb.grain(rgb) < nb.WOVEN_GRAIN
    out = tmp_path / 'o'
    assert main(['number', str(tmp_path / 'c.png'), '--out', str(out), '--size', '900']) == 0
    flat = np.asarray(Image.open(out / 'c_flat.png').convert('RGB'))
    inks = np.unique(flat.reshape(-1, 3), axis=0)
    assert len(inks) == 3                                                 # cream, maroon, black: no blend ink
    assert (flat[450, 180:200] < 60).all(axis=1).any()                    # the black outline is still there (left edge)
    rows = [r for r in csv.DictReader(open(out / 'c_colors.csv')) if r['Number'] not in ('lines', 'separators')]
    assert len(rows) >= 1 + 1 + 1 + 6                                     # ground, disc, outline, the six dots


def test_the_sketch_paints_back_to_the_design_with_its_own_csv(tmp_path):
    im = Image.new('RGB', (600, 600), (242, 232, 204))
    d = ImageDraw.Draw(im)
    d.ellipse([80, 80, 300, 300], fill=(140, 55, 70), outline=(15, 10, 10), width=6)   # maroon, black outline
    d.rectangle([350, 350, 550, 500], fill=(27, 45, 72))                                # navy block
    im.save(tmp_path / 'p.png')
    out = tmp_path / 'o'
    assert main(['number', str(tmp_path / 'p.png'), '--out', str(out), '--size', '600']) == 0
    sk = out / 'p_sketch_seal0.png'
    assert sk.exists() and (out / 'p_sketch_numbers.png').exists()
    g = np.asarray(Image.open(sk).convert('L'))
    assert set(np.unique(g)) <= {0, 60, 255}                              # black outlines, grey separators, white
    kb = np.asarray(Image.open(out / 'p_sketch_black.png').convert('L'))
    assert set(np.unique(kb)) <= {0, 255}                                 # the one to show: all black on white
    back = tmp_path / 'b'
    assert main(['paint', str(sk), '--colors', str(out / 'p_colors.csv'), '--out', str(back), '--size', '600']) == 0
    a = np.asarray(Image.open(out / 'p_flat.png').convert('RGB')).astype(int)
    b = np.asarray(Image.open(back / 'p_sketch_seal0_final_600px_300dpi.png').convert('RGB')).astype(int)
    assert (np.abs(a - b).sum(-1) < 30).mean() > 0.97                     # the same design, area for area


def test_detail_and_the_round_trip_score(tmp_path):
    from textile import number as nb
    im = Image.new('RGB', (400, 400), (242, 232, 204))
    d = ImageDraw.Draw(im)
    d.ellipse([50, 50, 350, 350], fill=(27, 45, 72))                      # a navy disc, no outline
    for k in range(5):
        d.ellipse([100 + 40 * k, 195, 104 + 40 * k, 199], fill=(242, 232, 204))   # five 5 px cream specks
    im.save(tmp_path / 'd.png')
    r_hi = nb.number(str(tmp_path / 'd.png'), str(tmp_path / 'hi'), size=1200, detail='zyada', log=lambda m: None)
    r_lo = nb.number(str(tmp_path / 'd.png'), str(tmp_path / 'lo'), size=1200, detail='kam', log=lambda m: None)
    assert r_lo['areas'] < r_hi['areas']                                  # 'kam' melts the specks, 'zyada' keeps them
    assert r_hi['match'] > 97 and r_lo['match'] > 97                      # the sketch paints back to the design


def test_hd_doubles_the_pixels_at_300_dpi(tmp_path):
    im = Image.new('RGB', (200, 200), (242, 232, 204))
    ImageDraw.Draw(im).ellipse([40, 40, 160, 160], fill=(27, 45, 72))
    im.save(tmp_path / 'r_hd.png')
    out = tmp_path / 'o'
    assert main(['number', str(tmp_path / 'r_hd.png'), '--out', str(out), '--size', '400']) == 0
    with Image.open(out / 'r_hd_sketch_black.png') as sk:
        assert sk.size == (800, 800) and round(sk.info['dpi'][0]) == 300
    with Image.open(out / 'r_hd_sketch_numbers.png') as nu:
        assert nu.size == (800, 800)                                       # numbers at full size too


def test_thick_lines_never_swallow_a_thin_part(tmp_path):
    from textile import number as nb
    im = Image.new('RGB', (600, 600), (242, 232, 204))
    d = ImageDraw.Draw(im)
    d.rectangle([100, 100, 500, 500], fill=(27, 45, 72))                  # a big navy block
    d.rectangle([100, 296, 500, 303], fill=(140, 55, 70))                 # an 8 px maroon stripe across it
    im.save(tmp_path / 's.png')
    r = nb.number(str(tmp_path / 's.png'), str(tmp_path / 'o'), size=600, line_mm=0.6, log=lambda m: None)
    sk = np.asarray(Image.open(r['sketch']).convert('L'))
    assert (sk[200, 150:450] == 255).all()                                # the block's middle stays white
    assert (sk[286:300, 300] < 255).sum() >= 4                            # a thick line along the stripe's side
    assert (sk[299:301, 150:450] == 255).any(axis=0).all()                # ...but the stripe keeps its middle
    assert r['match'] > 97


def test_auto_line_width_tries_three_and_keeps_the_closest(tmp_path):
    from textile import number as nb
    im = Image.new('RGB', (400, 400), (242, 232, 204))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 60, 340, 340], fill=(27, 45, 72))
    d.rectangle([180, 100, 220, 300], fill=(140, 55, 70))
    im.save(tmp_path / 'a.png')
    r = nb.number(str(tmp_path / 'a.png'), str(tmp_path / 'o'), size=800, log=lambda m: None)
    assert [t[0] for t in r['tried']] == list(nb.LINE_AUTO)
    best = max(m for _, m, _ in r['tried'])
    assert r['line_mm'] in nb.LINE_AUTO and dict((t[0], t[1]) for t in r['tried'])[r['line_mm']] >= best - 0.05
    assert (tmp_path / 'o' / 'a_rangeen.png').exists()
    one = nb.number(str(tmp_path / 'a.png'), str(tmp_path / 'p'), size=800, line_mm=0.35, log=lambda m: None)
    assert [t[0] for t in one['tried']] == [0.35]                         # a width given: only that one


def test_bold_sketch_keeps_circles_round_and_corners_sharp_and_writes_an_svg(tmp_path):
    im = Image.new('RGB', (300, 300), (242, 232, 204))
    d = ImageDraw.Draw(im)
    d.ellipse([40, 40, 140, 140], fill=(140, 55, 70))                     # a circle
    d.rectangle([170, 170, 260, 260], fill=(27, 45, 72))                  # a square
    im.save(tmp_path / 'b.png')
    out = tmp_path / 'o'
    assert main(['number', str(tmp_path / 'b.png'), '--out', str(out), '--size', '900', '--line-mm', '0.17',
                 '--bold-scale', '2']) == 0
    bold = np.asarray(Image.open(out / 'b_sketch_bold.png').convert('L'))
    assert bold.shape == (1800, 1800) and (bold < 128).mean() > 0.01      # --bold-scale 2: a 2x canvas, lines there
    with Image.open(out / 'b_sketch_bold.png') as bi:
        assert round(bi.info['dpi'][0]) == 600                              # same inches, twice the pixels
    # the circle's stroke stays on a circle: every dark px within 12 px of the true radius (150 px at this size)
    ys, xs = np.nonzero(bold[120:960, 120:960] < 128)
    r = np.hypot(xs + 120 - 540, ys + 120 - 540)
    r = r[r < 460]                                                        # the circle's own stroke, not the sheet edge
    assert len(r) > 500 and np.percentile(np.abs(r - 300), 95) < 24
    # the square's corner is still a corner: its outer corner point (780,780) is inked, not rounded away
    assert bold[1540:1572, 1540:1572].min() < 128
    svg = (out / 'b_sketch_bold.svg').read_text()
    assert svg.startswith('<svg') and '<path' in svg
    # the default is the mill's own size: 3535-style grid at 300 DPI, numbers on it, nothing bigger
    out1 = tmp_path / 'o1'
    assert main(['number', str(tmp_path / 'b.png'), '--out', str(out1), '--size', '900', '--line-mm', '0.17']) == 0
    with Image.open(out1 / 'b_sketch_bold.png') as b1, Image.open(out1 / 'b_sketch_bold_numbers.png') as n1:
        assert b1.size == (900, 900) == n1.size and round(b1.info['dpi'][0]) == 300


def test_colour_fill_keeps_only_the_given_colours_and_rounds_edges():
    import numpy as np
    from textile import curves as cv
    rgb = np.zeros((60, 60, 3), np.uint8)
    rgb[:] = (240, 230, 200)
    rgb[20:40, 20:40] = (140, 50, 70)
    big = cv.colour_fill(rgb, 3)
    assert big.shape == (180, 180, 3)
    got = {tuple(c) for c in big.reshape(-1, 3)}
    assert got == {(240, 230, 200), (140, 50, 70)}          # no blended colour, one colour per pixel
    assert tuple(big[90, 90]) == (140, 50, 70) and tuple(big[5, 5]) == (240, 230, 200)
    over = cv.colour_sketch(big, np.full((180, 180), 255, np.uint8))
    assert (over == big).all()                              # white line = colour untouched


def test_a_nearly_round_outline_becomes_a_true_circle_but_teeth_and_squares_stay():
    import numpy as np
    from textile import curves as cv
    t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    wobbly = np.stack([100 + 30 * (1 + 0.05 * np.sin(5 * t)) * np.cos(t), 100 + 30 * (1 + 0.05 * np.sin(5 * t)) * np.sin(t)], 1)
    c = cv.circle_of(wobbly, (200, 200))
    assert c is not None and np.ptp(np.hypot(c[:, 0] - c[:, 0].mean(), c[:, 1] - c[:, 1].mean())) < 0.5   # perfectly round
    teeth = np.stack([100 + 30 * (1 + 0.25 * (np.sin(16 * t) > 0)) * np.cos(t),
                      100 + 30 * (1 + 0.25 * (np.sin(16 * t) > 0)) * np.sin(t)], 1)
    assert cv.circle_of(teeth, (200, 200)) is None                         # a toothed ring keeps its teeth
    sq = np.array([[70, 70], [130, 70], [130, 130], [70, 130]], float)
    sq = np.vstack([np.linspace(sq[i], sq[(i + 1) % 4], 20, endpoint=False) for i in range(4)])
    assert cv.circle_of(sq, (200, 200)) is None
    ell = np.stack([100 + 40 * np.cos(t), 100 + 20 * np.sin(t)], 1)
    assert cv.circle_of(ell, (200, 200)) is None
    assert cv.circle_of(wobbly, (200, 200), iou=0) is None                  # --circle 0 = off
    assert cv.circle_of(wobbly + 80, (200, 200)) is None                    # cut by the sheet's edge: never a circle


def test_a_wobbly_triangle_or_rectangle_becomes_clean_but_curved_shapes_are_left_alone():
    import numpy as np
    from textile import curves as cv
    rng = np.random.default_rng(1)

    def dense(v, n=40):
        v = np.array(v, float)
        return np.vstack([np.linspace(v[i], v[(i + 1) % len(v)], n, endpoint=False) for i in range(len(v))])

    def wob(p):
        return p + rng.normal(0, 0.8, p.shape)

    size = (200, 200)
    tri = cv.polygon_of(wob(dense([(60, 150), (140, 150), (100, 80)])), size)
    assert tri is not None and len(tri) == 3
    rect = cv.polygon_of(wob(dense([(60, 60), (150, 60), (150, 110), (60, 110)])), size)
    assert rect is not None and len(rect) == 4
    assert abs(rect[:, 0].max() - rect[:, 0].min() - 90) < 3                # its size is kept
    assert cv.polygon_of(wob(dense([(100, 40), (150, 80), (130, 150), (70, 150), (50, 80)])), size) is not None
    t = np.linspace(0, np.pi, 100)
    assert cv.polygon_of(np.stack([100 + 70 * np.cos(t), 100 - 70 * np.sin(t)], 1), size) is None   # a half circle
    tt = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    assert cv.polygon_of(np.stack([100 + 50 * np.cos(tt), 100 + 25 * np.sin(tt) * np.abs(np.cos(tt)) ** 0.2], 1), size) is None
    bulgy = dense([(60, 150), (140, 150), (100, 80)])
    bulgy[40:80, 1] += 10 * np.sin(np.linspace(0, np.pi, 40))              # one side bowed out: a curve, not a side
    assert cv.polygon_of(bulgy, size) is None
    assert cv.polygon_of(wob(dense([(60, 150), (140, 150), (100, 80)])), size, iou=0) is None   # --polygons 0 = off


def _lens_mask(L=120, W=44, asym=1.0):
    import cv2
    import numpy as np
    t = np.linspace(0, 1, 400)
    x = 40 + t * L
    up = 100 - W / 2 * np.sin(np.pi * t) ** 0.9
    dn = 100 + W / 2 * asym * np.sin(np.pi * t) ** 0.9
    pts = np.vstack([np.stack([x, up], 1), np.stack([x[::-1], dn[::-1]], 1)])
    m = np.zeros((240, 240), np.uint8)
    cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 1, shift=4)
    return m


def test_a_leaf_or_petal_gets_two_clean_arcs_and_an_oval_a_true_ellipse_but_other_shapes_stay():
    import cv2
    import numpy as np
    from textile import curves as cv
    size = (240, 240)
    sym = cv.leaf_of(cv._outlines(_lens_mask())[0], size, 1.0)
    assert sym is not None                                                   # a petal: pointed tips, round sides
    xs, ys = sym[:, 0], sym[:, 1]
    assert abs((xs.max() - xs.min()) - 120) < 3 and abs((ys.max() - ys.min()) - 44) < 3     # its size is kept
    assert abs((ys.max() - 100) - (100 - ys.min())) < 1.5                    # both sides bow out the same: symmetric
    assert cv.leaf_of(cv._outlines(_lens_mask(asym=0.6))[0], size, 1.0) is not None   # a lopsided leaf is still a leaf
    m = np.zeros((240, 240), np.uint8)
    cv2.ellipse(m, (120, 120), (60, 25), 30, 0, 360, 1, -1)
    q = cv._outlines(m)[0]
    assert cv.oval_of(q, size) is not None and cv.leaf_of(q, size, 1.0) is None   # an oval has no tips: not a leaf
    r = np.zeros((240, 240), np.uint8)
    r[60:110, 60:150] = 1
    assert cv.oval_of(cv._outlines(r)[0], size) is None and cv.leaf_of(cv._outlines(r)[0], size, 1.0) is None
    assert cv.leaf_of(cv._outlines(_lens_mask())[0], size, 1.0, iou=0) is None        # --motifs 0 = off


def test_advice_names_each_problem_with_its_fix_and_the_log_remembers_runs_and_feedback(tmp_path, monkeypatch):
    from textile import learn
    monkeypatch.setenv('TEXTILE_LEARN_LOG', str(tmp_path / 'log.jsonl'))
    good = {'name': 'a', 'size_px': [3535, 3535], 'areas': 300, 'missed': 0, 'tiny': 0, 'match': 99.9, 'line_mm': 0.17,
            'tried': [(0.17, 99.9, 300)], 'inks': [('F3E8CA', 'cream', 70.0), ('1A2F4C', 'navy', 30.0)],
            'grain': 0.6, 'woven': False, 'colours_limit': 8, 'detail': 'normal', 'snapped': {}}
    assert learn.advise(good) == []
    bad = dict(good, name='b', areas=2000, missed=3, tiny=40, match=97.0, line_mm=0.5, woven=False, grain=2.5,
               tried=[(0.17, 96.0, 2000), (0.5, 97.0, 1990)], colours_limit=3,
               inks=[('F3E8CA', 'cream', 60.0), ('1A2F4C', 'navy', 30.0), ('172A53', 'navy', 0.08)])
    codes = [c for c, _ in learn.advise(bad)]
    assert {'similar_inks', 'tiny_parts', 'missed_numbers', 'match_low', 'colours_at_limit', 'busy', 'grainy',
            'thick_line_won'} <= set(codes)
    assert all(m for _, m in learn.advise(bad))                      # every problem comes with words (its fix)
    sim = learn.similar_inks(bad['inks'])
    assert len(sim) == 1 and sim[0][0][0] == '172A53' and sim[0][1][0] == '1A2F4C'   # the stray navy next to the real one
    # the log: a run, a verdict on it, a lookalike found, a summary that mentions both
    assert learn.record(good, str(tmp_path / 'nope.png'), {'detail': 'normal'}, [])
    assert learn.record(bad, str(tmp_path / 'nope.png'), {'detail': 'normal'}, learn.advise(bad))
    assert learn.feedback('a', 'good', 'bahut achha') is not None and learn.feedback('zzz', 'bad') is None
    near = learn.similar(dict(good, name='c'), k=2)
    assert near and near[0][1]['name'] == 'a' and near[0][2] == 'good'    # a is the closest, and the user liked it
    text = learn.summary()
    assert '2 run' in text and 'good' in text and 'similar_inks' in text
    with open(tmp_path / 'log.jsonl', 'a') as fh:
        fh.write('{"half a line')                                      # a crash mid-write
    assert len(learn.similar(dict(good, name='c'), k=2)) == 2           # the rest of the log still reads


def test_number_also_makes_the_colour_plates_unless_told_not_to(tmp_path):
    im = Image.new('RGB', (300, 300), (242, 232, 204))
    d = ImageDraw.Draw(im)
    d.ellipse([40, 40, 140, 140], fill=(140, 55, 70))
    d.rectangle([170, 170, 260, 260], fill=(27, 45, 72))
    im.save(tmp_path / 'p.png')
    out = tmp_path / 'o'
    assert main(['number', str(tmp_path / 'p.png'), '--out', str(out), '--size', '900', '--line-mm', '0.17']) == 0
    pkg = out / 'package'
    assert list(pkg.glob('p_colored_channels_*.zip')) and list(pkg.glob('p_final_*.tif'))     # one channel per colour + mill TIF
    import zipfile
    with zipfile.ZipFile(next(pkg.glob('p_colored_channels_*.zip'))) as z:
        assert len([n for n in z.namelist() if n.lower().endswith('.png')]) == 3              # cream, maroon, navy
    out2 = tmp_path / 'o2'
    assert main(['number', str(tmp_path / 'p.png'), '--out', str(out2), '--size', '900', '--line-mm', '0.17',
                 '--no-package']) == 0
    assert not (out2 / 'package').exists()


def test_loose_channel_images_and_stacked_steps_are_full_size_and_the_last_step_is_the_design(tmp_path):
    import numpy as np
    from textile import stack
    im = Image.new('RGB', (300, 300), (242, 232, 204))
    d = ImageDraw.Draw(im)
    d.ellipse([40, 40, 140, 140], fill=(140, 55, 70))
    d.rectangle([170, 170, 260, 260], fill=(27, 45, 72))
    im.save(tmp_path / 'q.png')
    out = tmp_path / 'o'
    assert main(['number', str(tmp_path / 'q.png'), '--out', str(out), '--size', '900', '--line-mm', '0.17']) == 0
    pkg = out / 'package'
    chans = sorted((pkg / 'channels').glob('*.png'))
    steps = sorted((pkg / 'stacked').glob('*.png'))
    assert len(chans) == 3 and len(steps) == 3
    for f in chans + steps:
        with Image.open(f) as x:
            assert x.size == (900, 900) and round(x.info['dpi'][0]) == 300      # each one a full-size image, not a sheet
    final = np.asarray(Image.open(next(pkg.glob('q_final_*.png'))).convert('RGB'))
    assert (np.asarray(Image.open(steps[-1]).convert('RGB')) == final).all()    # all plates stacked = the design
    assert not (np.asarray(Image.open(steps[0]).convert('RGB')) == final).all()


def test_number_writes_a_channels_photoshop_file_spot_alphas_darkest_first(tmp_path):
    import numpy as np
    from psd_tools import PSDImage
    from psd_tools.constants import Resource
    im = Image.new('RGB', (300, 300), (242, 232, 204))
    d = ImageDraw.Draw(im)
    d.ellipse([40, 40, 140, 140], fill=(140, 55, 70))
    d.rectangle([170, 170, 260, 260], fill=(27, 45, 72))
    im.save(tmp_path / 'r.png')
    out = tmp_path / 'o'
    assert main(['number', str(tmp_path / 'r.png'), '--out', str(out), '--size', '900', '--line-mm', '0.17']) == 0
    f = next((out / 'package').glob('r_layers_*.psd'))
    psd = PSDImage.open(f)
    assert psd.size == (900, 900)
    assert list(psd) == []                                    # no ink layers by default
    assert psd.channels == 3 + 3                              # RGB + 3 spot channels
    names = list(psd.image_resources[Resource.ALPHA_NAMES_UNICODE].data)
    assert len(names) == 3 and all('#' not in n for n in names)
    def lum(nm):
        hx = nm.strip().split()[-1]
        r0, g0, b0 = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        return 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0
    assert [lum(n) for n in names] == sorted(lum(n) for n in names)
    r = psd.image_resources[1005].data
    assert round(r.horizontal / 65536) == 300 and round(r.vertical / 65536) == 300
    final = np.asarray(Image.open(next((out / 'package').glob('r_final_*.png'))).convert('RGB'))
    assert np.array_equal(np.asarray(psd.composite().convert('RGB')), final)     # merged preview = final
    # opt-in layered PSD still available
    out_l = tmp_path / 'ol'
    assert main(['number', str(tmp_path / 'r.png'), '--out', str(out_l), '--size', '900',
                 '--line-mm', '0.17', '--psd-layers']) == 0
    psd_l = PSDImage.open(next((out_l / 'package').glob('r_layers_*.psd')))
    assert len(list(psd_l)) == 4                              # 3 colour plates + hidden guide
    out2 = tmp_path / 'o2'
    assert main(['number', str(tmp_path / 'r.png'), '--out', str(out2), '--size', '900', '--line-mm', '0.17', '--no-psd']) == 0
    assert not list((out2 / 'package').glob('*.psd'))


def test_merge_similar_folds_a_small_near_duplicate_ink_into_the_big_one():
    import numpy as np
    from textile import number as nb
    pal = np.array([[26, 47, 76], [23, 42, 83], [243, 232, 202]], np.uint8)     # navy, a near-navy, cream
    index = np.full((100, 100), 2, np.uint8)
    index[:50] = 0
    index[10:12, 10:30] = 1                                                      # 40 px = 0.4% of the sheet: a stray shade
    out, n, moved = nb.merge_similar_inks(index, pal)
    assert n == 1 and moved == [('172A53', '1A2F4C')] and not (out == 1).any()
    index2 = index.copy()
    index2[60:80] = 1                                                            # now 20% of the sheet: a real second ink
    assert nb.merge_similar_inks(index2, pal)[1] == 0


def test_fairing_eases_a_jog_but_keeps_ends_and_corners_and_a_shared_edge_is_drawn_once():
    import cv2
    import numpy as np
    from textile import curves as cv
    # an open stretch with a small jog: after fairing the jog is gentler, the two ends do not move
    x = np.linspace(0, 200, 201)
    y = np.where(x < 100, 0.0, 6.0)                                          # a 6 px step in the middle
    seg = np.stack([x, y], 1)
    f = cv.fair(seg, 5.0, False)
    assert np.allclose(f[0], seg[0]) and np.allclose(f[-1], seg[-1])
    assert np.abs(np.diff(f[:, 1])).max() < 0.5 * 6                          # the jump is spread out, not 6 px at once
    # a corner survives: a right angle (turning 90 degrees) is above CORNER_DEG
    sq = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], float)
    sq = np.vstack([np.linspace(sq[i], sq[(i + 1) % 4], 100, endpoint=False) for i in range(4)])
    out = cv.smooth(sq, 1.0)
    assert np.linalg.norm(out - np.array([110, 10]), axis=1).min() < 2.0     # still a sharp corner at (110, 10)
    # two parts sharing one edge: that edge is stroked once (a ground with a disc: the disc's rim is not doubled)
    lab = np.ones((200, 200), np.int32)
    cv2.circle(lab, (100, 100), 40, 2, -1)
    on, _ = cv.bold(lab, np.zeros(3, bool), 1.0, 6, dedup=True)
    off, _ = cv.bold(lab, np.zeros(3, bool), 1.0, 6, dedup=False)
    assert (on < 128).sum() <= (off < 128).sum()
    ring = np.hypot(*np.meshgrid(np.arange(200) - 100.0, np.arange(200) - 100.0))
    assert ((on < 128) & (np.abs(ring - 40) < 5)).sum() > 0.8 * ((off < 128) & (np.abs(ring - 40) < 5)).sum()   # still drawn


def test_smoothing_percent_is_the_photoshop_style_slider_and_more_of_it_is_smoother(tmp_path):
    im = Image.new('RGB', (300, 300), (242, 232, 204))
    d = ImageDraw.Draw(im)
    t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    r = 80 * (1 + 0.14 * np.sin(7 * t) + 0.04 * np.sin(23 * t))                 # a wavy blob: not a circle, no corners
    d.polygon([(150 + a * np.cos(b), 150 + a * np.sin(b)) for a, b in zip(r, t)], fill=(140, 55, 70))
    im.save(tmp_path / 's.png')
    outs = {}
    for pct in (0, 100):
        out = tmp_path / f'o{pct}'
        assert main(['number', str(tmp_path / 's.png'), '--out', str(out), '--size', '900', '--line-mm', '0.17',
                     '--smoothing', str(pct), '--no-package', '--no-psd']) == 0
        outs[pct] = np.asarray(Image.open(out / 's_sketch_bold.png').convert('L'))
    assert outs[0].shape == outs[100].shape == (900, 900)
    assert not np.array_equal(outs[0], outs[100])                           # the slider does something
    for o in outs.values():
        assert (o < 128).sum() > 500                                        # and the sketch is still drawn


def test_svg_path_survives_a_degenerate_piece(tmp_path):
    import numpy as np
    from textile import curves as cv
    lab = np.zeros((60, 60), np.int32)
    lab[10:12, 10:50] = 1                                                      # a 2 px sliver: the odd outline
    lab[30:50, 30:50] = 2
    img, svg = cv.bold(lab, np.zeros(3, bool), 1.0, 2)
    assert svg.startswith('<svg') and '<path' in svg


def test_fairing_does_not_shrink_a_small_closed_shape():
    import numpy as np
    from textile import curves as cv
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    dot = np.stack([100 + 12 * np.cos(t), 100 + 12 * np.sin(t)], 1)            # a dot of radius 12 px
    f = cv.fair(dot, 8.0, True)                                                  # a heavy fairing asked for
    r = np.hypot(f[:, 0] - 100, f[:, 1] - 100)
    assert r.mean() > 11.0                                                       # the dot keeps its size (was ~8)


def test_a_slant_one_pixel_line_inside_one_ink_stays_but_an_edge_fringe_melts():
    """`patches` with `lines`: a 1 px line drawn slant is a chain of 4-connected bits (each one a 'speck'); joined
    8-connected and lying inside ONE other ink (a dark slit in a cream leaf) it is kept. A fringe of the same
    width running between two DIFFERENT inks (an edge's anti-alias) still melts."""
    from textile import names as nm
    from textile import number as nb
    pal = np.array([[74, 23, 60], [249, 230, 201], [133, 38, 33]], np.uint8)     # purple, cream, rust
    idx = np.ones((80, 80), np.uint8)                                            # a cream leaf...
    for i in range(10, 40):
        idx[i, i + 5] = 0                                                       # ...with a slant 1 px purple slit
    yy, xx = np.mgrid[:80, :80]
    idx[xx > yy + 40] = 0                                                       # purple ground beyond a slant edge
    fringe = xx == yy + 40
    idx[fringe] = 2                                                             # a 1 px rust fringe along that edge
    src = pal[idx].astype(np.float64)
    src[fringe] = (pal[1].astype(float) + pal[0]) / 2                            # its pixels LOOK half cream half purple
    src_lab = nm._lab(src.reshape(-1, 3)).reshape(80, 80, 3).astype(np.float32)
    _, _, out = nb.patches(idx.copy(), 9, thin=0, enclosed_max=0, rim=nb.RIM, rim_area=30,
                           src_lab=src_lab, pal=pal, lines=nb.LINE_RESCUE)
    assert all(out[i, i + 5] == 0 for i in range(10, 40))                      # the slit is whole
    assert not (out == 2).any()                                                  # the fringe melted, into cream or purple
    _, _, old_way = nb.patches(idx.copy(), 9, thin=0, enclosed_max=0, rim=nb.RIM, rim_area=30)
    assert sum(old_way[i, i + 5] == 0 for i in range(10, 40)) < 10             # without it the slit was melted away


def test_a_melted_pixel_takes_the_touching_ink_it_looks_like():
    from textile import names as nm
    from textile import number as nb
    pal = np.array([[0, 0, 0], [255, 255, 255], [200, 30, 30]], np.uint8)
    idx = np.zeros((5, 9), np.uint8)
    idx[:, 5:] = 1                                                               # black | white
    idx[2, 4] = idx[2, 5] = 2                                                    # two red bits on the edge, to go
    src = pal[idx].astype(np.float64)
    src[2, 4], src[2, 5] = (60, 60, 60), (220, 220, 220)                         # one looks dark, one light
    lab = nm._lab(src.reshape(-1, 3)).reshape(5, 9, 3).astype(np.float32)
    out = nb._to_nearest_colour(idx, idx == 2, lab, nm._lab(pal.astype(np.float64)))
    assert out[2, 4] == 0 and out[2, 5] == 1


def test_an_all_rim_exact_mix_is_no_ink_even_over_3_percent(tmp_path):
    """Soft AI edges make a WIDE blend: the black lines' grey rim here is 6.5% of the design, over the 3% cap that
    keeps a busy design's real thin inks; all rim, no core and an exact mix of black and cream, it is no screen."""
    import cv2
    from textile import number as nb
    from textile import palette as pl
    from textile import paint as pt
    k = 3
    big = np.full((300 * k, 300 * k, 3), (242, 232, 204), np.uint8)
    rng = np.random.default_rng(1)
    for _ in range(60):
        cv2.line(big, tuple(int(v) for v in rng.integers(0, 300 * k, 2)), tuple(int(v) for v in rng.integers(0, 300 * k, 2)),
                 (20, 20, 25), 4)
    cv2.circle(big, (450, 450), 200, (20, 20, 25), -1)
    rgb = cv2.resize(big, (300, 300), interpolation=cv2.INTER_AREA)
    pal = pt._ref_palette(nb.solid_pixels(rgb), 8, nb.CLEAN_SAME_DE)
    index = pl.map_to_palette(rgb, pal)
    share = np.bincount(index.ravel(), minlength=len(pal)) / index.size
    blend = nb.blend_inks(index, pal, nb.RIM)
    assert len(pal) == 3 and blend.sum() == 1 and share[blend][0] > nb.BLEND_MAX_SHARE
    pal2, idx2 = nb.flat_index(rgb, 300, 300, 8, 9, False, smooth=False)
    assert len(pal2) == 2 and len(np.unique(idx2)) == 2                          # cream and black, no grey screen


def test_design_match_says_how_much_of_the_picture_the_flat_inks_keep():
    import cv2
    from textile import number as nb
    flat = np.zeros((200, 200, 3), np.uint8)
    flat[:] = (240, 230, 210)
    cv2.circle(flat, (100, 100), 60, (120, 30, 40), -1)
    assert nb.design_match(flat, flat) == 100.0
    soft = cv2.GaussianBlur(flat, (0, 0), 2)                                     # a picture with soft edges
    m = nb.design_match(soft, flat)
    assert 80 < m < 100
    big = cv2.resize(flat, (600, 600), interpolation=cv2.INTER_NEAREST)           # the flat drawn at print size
    assert nb.design_match(flat, big) == 100.0


def test_the_fast_measures_give_the_old_answers():
    from scipy import ndimage
    from textile import curves as cv
    from textile import number as nb
    from textile import paint as pt
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 40, (60, 70))
    vals = rng.random((60, 70)) * 9
    assert np.array_equal(nb._label_max(vals, lab, 39), np.asarray(ndimage.maximum(vals, lab, np.arange(1, 40))))
    inner = np.kron(rng.integers(0, 6, (6, 7)), np.ones((10, 10), int))
    inner[::10, :] = 0
    inner[:, ::10] = 0
    dist = ndimage.distance_transform_edt(inner > 0)
    best, depth = pt._deepest(dist, inner, 5)
    assert np.array_equal(depth, [float(ndimage.maximum(dist, inner, i)) if (inner == i).any() else 0 for i in range(1, 6)])
    for i, (y, x) in enumerate(best, 1):
        if depth[i - 1] > 0:
            assert inner[y, x] == i and dist[y, x] == depth[i - 1]
    for _ in range(200):                                                         # outline strays: same yes / no
        m = int(rng.integers(3, 400))
        th = np.sort(rng.uniform(0, 2 * np.pi, m))
        poly = np.stack([80 * np.cos(th), 50 * np.sin(th)], 1) + rng.normal(0, 1, (m, 2))
        pts = poly[rng.integers(0, m, 300)] + rng.normal(0, 3, (300, 2))
        tol = float(rng.uniform(1, 8))
        assert cv._strays(pts, poly, tol) == bool(cv._seg_dist(pts, poly).max() > tol)


def test_a_speck_sized_outline_is_never_smoothed_to_nothing():
    from textile import curves as cv
    p = np.array([[0.65, 0.65], [0.5, 2.0], [0.65, 3.35], [2.0, 3.5], [3.35, 3.35], [3.5, 2.0], [3.35, 0.65], [2.0, 0.5]])
    q = cv.smooth(p, 2.5, 3.5)                       # a 3 x 3 px part at a big scale: the spline used to come back empty
    assert len(q) >= 3


def test_a_hairline_blurred_into_the_ground_gets_its_own_ink_back():
    """A 1 px cream line on a dark ground, blurred like an AI picture: its pixels are nearer the ground's colour,
    so nearest-ink loses the line; `line_inks` sees a ridge (both sides alike, the pixel lighter) and gives the
    cream. A soft EDGE (two different sides) is left to nearest-ink."""
    import cv2
    from textile import number as nb
    pal = np.array([[40, 44, 36], [235, 225, 200]], np.uint8)                   # black-green, cream
    big = np.zeros((120, 480, 3), np.uint8)
    big[:] = pal[0]
    cv2.line(big, (0, 60), (479, 60), pal[1].tolist(), 3)                          # a line 3 px at 4x
    big[:, 400:] = pal[1]                                                          # and a plain edge
    rgb = cv2.GaussianBlur(cv2.resize(big, (120, 30), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 0.9)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    index = np.argmin(((rgb[:, :, None, :].astype(int) - pal[None, None].astype(int)) ** 2).sum(-1), 2)
    assert not (index[13:18, 5:90] == 1).any()                                    # nearest ink loses the line
    lk = nb.line_inks(rgb, pal, index)
    fixed = np.where(lk >= 0, lk, index)
    assert (fixed[15, 5:90] == 1).all()                                           # the line is back, 1 px
    assert not (fixed[[13, 17], 5:90] == 1).any()                                 # and not wider
    assert (lk[:10, 90:110] < 0).all() and (lk[20:, 90:110] < 0).all()            # the edge is no line


def test_a_small_solid_distinct_ink_stays_and_twin_inks_become_one():
    import cv2
    from textile import number as nb
    rgb = np.zeros((300, 300, 3), np.uint8)
    rgb[:] = (240, 232, 214)                                                      # cream ground
    rgb[:, 150:] = (238, 231, 215)                                                # its twin, dE2000 < 1: one ink
    cv2.rectangle(rgb, (20, 20), (120, 120), (40, 90, 60), -1)                     # green block
    for k in range(3):                                                            # three maroon dots: 0.25%
        cv2.circle(rgb, (60 + 80 * k, 220), 5, (110, 30, 45), -1)
    pal = np.array([[240, 232, 214], [238, 231, 215], [40, 90, 60], [110, 30, 45]], np.uint8)
    index = np.argmin(((rgb[:, :, None, :].astype(int) - pal[None, None].astype(int)) ** 2).sum(-1), 2).astype(np.uint8)
    p2, i2 = nb._merge_twins(rgb, pal, index)
    assert len(p2) == 3
    pal3, idx3 = nb.flat_index(rgb, 300, 300, 8, 4, False, smooth=False)
    hexes = {tuple(c) for c in pal3[np.unique(idx3)]}
    assert any(abs(int(c[0]) - 110) < 15 and int(c[1]) < 60 for c in hexes)        # the maroon dots kept their ink


def test_an_all_rim_shade_near_one_end_of_a_pair_is_no_blend():
    """The loose all-rim test (dE 10-20 off the mix line) only counts well inside the pair: a thin dark-green stem
    sits 0.9 of the way from rose to olive, i.e. it is a darker olive, a real ink."""
    from textile import names as nm
    from textile import number as nb
    pal = np.array([[199, 71, 106], [132, 157, 90], [93, 125, 89], [243, 237, 224]], np.uint8)   # rose, olive, stem, cream
    index = np.full((60, 60), 3, np.uint8)
    index[10:50, 10:20] = 1                                                       # an olive leaf
    index[10:50, 20] = 2                                                          # the stem, 1 px: all rim
    index[10:50, 40:50] = 0                                                       # a rose petal
    assert not nb.blend_inks(index, pal, nb.RIM)[2]


def test_merge_shades_puts_a_colours_shading_on_its_screen_but_keeps_lines_and_separate_motifs():
    """--merge-shades: a darker red patch inside a red petal (the AI picture's shading, dE2000 ~7, all its border
    on the red) becomes red; a cream lattice line on the beige ground (dE ~9, all border on the ground, but a
    line) and a second red-ish motif that never touches the red (close colour, no shared border) stay inks."""
    from textile import number as nb
    pal = np.array([[240, 222, 180],    # 0 beige ground
                    [215, 23, 40],      # 1 red petal
                    [180, 17, 31],      # 2 darker red shading inside it
                    [250, 240, 214],    # 3 cream lattice line on the ground
                    [196, 20, 60],      # 4 crimson motif of its own, apart from the petal
                    [20, 22, 28]], np.uint8)
    idx = np.zeros((600, 600), np.uint8)
    yy, xx = np.mgrid[:600, :600]
    idx[(yy - 150) ** 2 + (xx - 150) ** 2 < 110 ** 2] = 1
    idx[(yy - 170) ** 2 + (xx - 140) ** 2 < 45 ** 2] = 2
    idx[:, 400:403] = 3
    idx[450:530, 80:200] = 4
    idx[300:360, 450:560] = 5
    out, moved = nb.merge_shade_inks(idx, pal, 300 / 25.4)
    assert moved == [('B4111F', 'D71728')]
    assert (out[idx == 2] == 1).all()
    assert (out[idx == 3] == 3).all() and (out[idx == 4] == 4).all()
    assert (out[idx != 2] == idx[idx != 2]).all()

def test_floral_orange_pair_E9793B_E56824_merges_and_orange_vs_beige_does_not():
    """floral.png buds: E9793B + E56824 (dE ~4) merge; orange vs beige (dE ~30) never does."""
    import numpy as np
    from textile import number as nb

    pal = np.array([[240, 222, 180],   # beige ground
                    [233, 121, 59],    # E9793B
                    [229, 104, 36],    # E56824
                    [46, 107, 79]], np.uint8)
    idx = np.zeros((400, 400), np.uint8)
    yy, xx = np.mgrid[:400, :400]
    idx[(yy - 150) ** 2 + (xx - 150) ** 2 < 90 ** 2] = 1
    idx[(yy - 160) ** 2 + (xx - 145) ** 2 < 40 ** 2] = 2
    idx[300:360, 50:200] = 3
    out, moved = nb.merge_shade_inks(idx, pal, 300 / 25.4)
    assert ('E56824', 'E9793B') in moved
    assert (out[idx == 2] == 1).all()

    pal2 = np.array([[240, 222, 180], [233, 121, 59]], np.uint8)
    idx2 = np.zeros((300, 300), np.uint8)
    idx2[80:220, 80:220] = 1
    out2, moved2 = nb.merge_shade_inks(idx2, pal2, 300 / 25.4)
    assert moved2 == []
    assert (out2 == idx2).all()


def test_number_merges_shades_by_default_and_no_merge_shades_keeps_them(tmp_path):
    """textile number: shade merge ON by default; --no-merge-shades keeps the darker patch as its own ink."""
    from PIL import Image, ImageDraw, ImageFilter
    from textile.cli import main
    import json

    beige, red, dark_red, black = (240, 222, 180), (215, 23, 40), (180, 17, 31), (24, 26, 31)
    img = Image.new('RGB', (480, 320), beige)
    d = ImageDraw.Draw(img)
    for cx, cy in ((110, 100), (300, 90), (200, 230), (400, 240)):
        d.line([(cx, cy + 40), (cx - 30, cy + 90)], fill=black, width=7)
        d.ellipse([cx - 55, cy - 40, cx + 55, cy + 40], fill=red)
        d.ellipse([cx - 40, cy - 5, cx + 5, cy + 30], fill=dark_red)
    src = tmp_path / 'petals.png'
    img.filter(ImageFilter.GaussianBlur(0.8)).save(src)

    out_on = tmp_path / 'on'
    assert main(['number', str(src), '--out', str(out_on), '--size', '900', '--colors', '8',
                 '--no-package']) == 0
    report_on = json.loads((out_on / 'petals_number.json').read_text(encoding='utf-8'))
    assert report_on['shades_merged'], report_on['shades_merged']
    assert len(report_on['inks']) == 3

    out_off = tmp_path / 'off'
    assert main(['number', str(src), '--out', str(out_off), '--size', '900', '--colors', '8',
                 '--no-package', '--no-merge-shades']) == 0
    report_off = json.loads((out_off / 'petals_number.json').read_text(encoding='utf-8'))
    assert report_off['shades_merged'] == []
    assert len(report_off['inks']) == 4

    out_explicit = tmp_path / 'explicit'
    assert main(['number', str(src), '--out', str(out_explicit), '--size', '900', '--colors', '8',
                 '--no-package', '--merge-shades']) == 0
    report_ex = json.loads((out_explicit / 'petals_number.json').read_text(encoding='utf-8'))
    assert report_ex['shades_merged']


def _soft(idx, pal, small_w, blur=0.5):
    """A truth drawn hard, then made like an AI picture: area-shrunk (anti-aliased) and a little soft."""
    import cv2
    img = cv2.resize(pal[idx], (small_w, round(small_w * idx.shape[0] / idx.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(img, (0, 0), blur) if blur else img


def test_small_objects_keep_their_shape_and_big_ones_stay_exactly_as_before():
    """Stars and dots a few picture px across, drawn 5x bigger: from the picture's anti-aliasing (small_px) they keep
    their points and size, where the edges' outline smoothing rounded them into blobs. The big shape on the other
    half is the same, pixel for pixel, and no ink is lost."""
    import cv2
    from textile import number as nb
    pal = np.array([[240, 222, 180], [215, 23, 40], [46, 107, 79]], np.uint8)
    W = 1000
    idx = np.zeros((W, W), np.uint8)
    cv2.ellipse(idx, (750, 500), (180, 320), 0, 0, 360, 2, -1)            # one big shape on the right half
    stars = []
    rng = np.random.default_rng(5)
    for cy in range(60, 960, 110):
        for cx in range(60, 450, 110):
            r = int(rng.integers(18, 34))
            ang = rng.uniform(0, 2 * np.pi) + np.arange(10) * np.pi / 5
            rad = np.where(np.arange(10) % 2 == 0, r, 0.45 * r)
            pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
            one = np.zeros_like(idx)
            cv2.fillPoly(one, [pts], 1)
            idx[one > 0] = 1
            stars.append(one > 0)
    src = _soft(idx, pal, 200)
    g = nb.grain(src)
    out = {}
    for sp in (0, 20):
        p, ix = nb.flat_index(src, W, W, 8, 30.0, False, True, g, sp)
        lab = np.array([np.argmin(np.abs(pal.astype(int) - c.astype(int)).sum(1)) for c in p])[ix]
        out[sp] = (len(p), lab)
    iou = {sp: np.mean([((lab == 1) & m).sum() / ((lab == 1) & (cv2.dilate(m.astype(np.uint8), np.ones((9, 9))) > 0) | m).sum()
                        for m in stars]) for sp, (_, lab) in out.items()}
    assert out[20][0] == out[0][0] == 3
    assert iou[20] > iou[0] + 0.05, iou
    assert (out[20][1][:, 560:] == out[0][1][:, 560:]).all()              # the big shape: not one px changed


def test_an_ink_lying_between_two_others_is_not_read_as_their_mix():
    """A sage band (between navy and cream in colour) 3 picture px wide is a small object, but it is a real ink:
    it stays sage, not navy-and-cream, however close its colour lies to their mix."""
    import cv2
    from textile import number as nb
    pal = np.array([[11, 70, 94], [235, 214, 173], [110, 140, 125]], np.uint8)      # navy, cream, sage (between)
    W = 600
    idx = np.zeros((W, W), np.uint8)
    idx[:, 330:] = 1
    for x0 in (60, 150, 240):
        idx[40:560, x0:x0 + 15] = 2                                          # sage bands 3 px at the picture's size
    src = _soft(idx, pal, 120, 0.4)
    p, ix = nb.flat_index(src, W, W, 8, 20.0, False, True, nb.grain(src), 20)
    lab = np.array([np.argmin(np.abs(pal.astype(int) - c.astype(int)).sum(1)) for c in p])[ix]
    band = idx == 2
    assert (lab[band] == 2).mean() > 0.8
    assert (lab[~band & (idx == 0)] == 2).mean() < 0.02
