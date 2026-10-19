"""textile paint: a sketch coloured as the user says, one channel per colour."""
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from textile import paint as pt
from textile.cli import main
from textile.fill_method1 import FillError


def sketch(path, size=600):
    """Four equal circles, one triangle, a small square; 4 px black lines on white."""
    im = Image.new('L', (size, size), 255)
    d = ImageDraw.Draw(im)
    for cx, cy in ((150, 150), (450, 150), (150, 450), (450, 450)):
        d.ellipse([cx - 90, cy - 90, cx + 90, cy + 90], outline=0, width=4)
    d.polygon([(300, 230), (360, 350), (240, 350)], outline=0, width=4)
    d.rectangle([290, 500, 310, 520], outline=0, width=4)
    im.save(path)
    return path


@pytest.fixture
def reg(tmp_path):
    return pt.find_regions(sketch(tmp_path / 's.png'), size=600, log=lambda m: None)


def test_like_shapes_share_a_letter_and_the_ground_is_A(reg):
    lab = reg.lab
    circles = {int(reg.group[lab[y, x]]) for x, y in ((150, 150), (450, 150), (150, 450), (450, 450))}
    assert len(circles) == 1
    tri = int(reg.group[lab[320, 300]])
    assert tri not in circles
    assert reg.ground == int(reg.group[lab[5, 5]]) == 0 and reg.letters[0] == 'A'


def test_letters_run_past_Z():
    assert [pt.letter(i) for i in (0, 25, 26, 27, 51, 52)] == ['A', 'Z', 'AA', 'AB', 'AZ', 'BA']


def test_colours_go_where_named_and_nowhere_else(reg):
    C = reg.letters[reg.group[reg.lab[150, 150]]]
    T = reg.letters[reg.group[reg.lab[320, 300]]]
    num = int(reg.lab[450, 450])               # one circle by its number, over its group
    said = pt.parse_colors(f'A=cream; {C}=laal, {T} ko hara; {num}=0000FF; lines=coffee')
    area_col, line, tiny, notes = pt.plan(reg, said)
    index, pal, li = pt.paint(reg, area_col, line, tiny)
    rgb = pal[index]
    hexes = lambda y, x: '%02X%02X%02X' % tuple(rgb[y, x])
    assert hexes(5, 5) == 'F2E8CF'                       # cream (colour words)
    assert hexes(150, 150) == hexes(150, 450) == 'C8102E'
    assert hexes(450, 450) == '0000FF'
    assert hexes(320, 300) == '2E8B3A'
    assert hexes(150, 60 + 2) == '4A2E1E'                # on the circle's line: coffee
    assert li is not None and len(pal) == 5              # cream, red, blue, green, coffee
    assert hexes(510, 300) == 'F2E8CF'                   # the square was not named: its nearest named area (ground)...
    assert any('sabse paas' in n for n in notes)         # ...and said so


def test_lines_fill_leaves_no_outline_and_same_colour_is_one_channel(reg):
    C = reg.letters[reg.group[reg.lab[150, 150]]]
    others = ' '.join(l for l in reg.letters if l not in ('A', C))
    area_col, line, tiny, _ = pt.plan(reg, pt.parse_colors(f'A=navy, {others}=1b2a4a, {C}=gold, lines=fill'))
    index, pal, li = pt.paint(reg, area_col, line, tiny)
    assert li is None
    assert {'%02X%02X%02X' % tuple(c) for c in pal} <= {'1B2A4A', 'C9A43A'}
    assert len(np.unique(index)) == len(pal)


def test_a_key_or_colour_it_cannot_read_stops_with_the_reason(reg):
    with pytest.raises(FillError, match='na group'):
        pt.plan(reg, pt.parse_colors('ZZZ=laal'))
    with pytest.raises(FillError, match='nahi pehchana'):
        pt.plan(reg, pt.parse_colors('A=sunset glow'))
    with pytest.raises(FillError, match='nahi hai'):
        pt.plan(reg, pt.parse_colors(f'{reg.n + 5}=laal'))
    with pytest.raises(FillError, match='samajh'):
        pt.parse_colors('A laal')


def test_the_same_key_can_borrow_a_colour(reg):
    area_col, *_ = pt.plan(reg, pt.parse_colors('A=F2E8CF; B=A'))
    b = reg.groups[1]['areas'][0]
    assert area_col[b] == 'F2E8CF'


def test_cli_map_then_paint_package_verifies(tmp_path):
    s = sketch(tmp_path / 'buta.png')
    out = str(tmp_path / 'o')
    assert main(['paint', str(s), '--out', out, '--size', '600']) == 0
    for f in ('buta_map.png', 'buta_numbers.png', 'buta_groups.txt', 'buta_colors.txt'):
        assert os.path.exists(os.path.join(out, f))
    template = open(os.path.join(out, 'buta_colors.txt')).read()
    assert template.startswith('# textile paint: 600x600')
    filled = template.replace('# A = ?', 'A = cream').replace('# B = ?', 'B = mehroon')
    open(os.path.join(out, 'buta_colors.txt'), 'w').write(filled)
    code = main(['paint', str(s), '--out', out, '--size', '600'])     # the filled file is picked up by itself
    assert code == 0
    assert os.path.exists(os.path.join(out, 'buta_final_600px_300dpi.tif'))
    with Image.open(os.path.join(out, 'buta_final_600px_300dpi.png')) as im:
        rgb = np.asarray(im.convert('RGB'))
    assert tuple(rgb[5, 5]) == (0xF2, 0xE8, 0xCF)
    # the saved colours belong to 600 px: a run at another size is stopped, not painted wrong
    assert main(['paint', str(s), '--out', out, '--size', '800', '--colors', os.path.join(out, 'buta_colors.txt')]) == 1


def jaali(path, size=800):
    """A thin diagonal lattice (3 px) over the ground, with motifs drawn thicker (9 px) on top,
    each with a thin line inside; the lattice is the smaller part of the lines, as on 2218."""
    im = Image.new('L', (size, size), 255)
    d = ImageDraw.Draw(im)
    for k in range(-size, 2 * size, 240):
        d.line([(k, 0), (k + size, size)], fill=0, width=3)
        d.line([(k, 0), (k - size, size)], fill=0, width=3)
    for cx, cy in ((200, 200), (600, 600), (600, 200), (200, 600)):
        d.ellipse([cx - 110, cy - 110, cx + 110, cy + 110], fill=255, outline=0, width=9)
        for r in (80, 50):
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=9)
        d.line([(cx - 110, cy + 95), (cx + 110, cy + 95)], fill=0, width=3)  # a thin line inside the motif
    im.save(path)
    return path


def test_star_takes_the_whole_ground_across_a_thin_jaali_but_not_into_motifs(tmp_path):
    reg = pt.find_regions(jaali(tmp_path / 'j.png'), size=800, log=lambda m: None)
    gi = int(reg.group[reg.lab[400, 40]])
    ids, width, _ = pt.across_thin(reg, gi)
    assert width is not None
    chosen = set(ids.tolist())
    ground_cells = {int(reg.lab[y, x]) for x, y in ((40, 400), (400, 40), (760, 400), (400, 760), (440, 400))}
    assert 0 not in ground_cells and len(ground_cells) > 1 and ground_cells <= chosen
    inside = {int(reg.lab[y, x]) for x, y in ((200, 200), (600, 600), (200, 135), (600, 535), (200, 300))}
    assert 0 not in inside and not inside & chosen
    area_col, line, tiny, notes = pt.plan(reg, pt.parse_colors(f'{reg.letters[gi]}* = cream, rest = laal'))
    assert area_col[int(reg.lab[400, 440])] == 'F2E8CF' and area_col[int(reg.lab[200, 200])] == 'C8102E'
    assert any('jaali' in n for n in notes)


def test_star_adds_nothing_when_there_is_no_thinner_jaali(reg):
    ids, width, _ = pt.across_thin(reg, reg.ground)
    assert width is None and set(ids.tolist()) == set(np.flatnonzero(reg.group == reg.ground).tolist())


def test_plus_takes_the_same_shape_at_any_size(tmp_path):
    im = Image.new('L', (600, 600), 255)
    d = ImageDraw.Draw(im)
    for cx, cy, r in ((150, 150, 100), (450, 150, 60), (150, 450, 35)):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=4)
    d.rectangle([380, 400, 520, 430], outline=0, width=4)                 # a long bar: another shape
    im.save(tmp_path / 'c.png')
    reg = pt.find_regions(tmp_path / 'c.png', size=600, log=lambda m: None)
    big = int(reg.group[reg.lab[150, 150]])
    assert len({int(reg.group[reg.lab[y, x]]) for x, y in ((150, 150), (450, 150), (150, 450))}) == 3
    ids = set(pt.alike(reg, big).tolist())
    assert {int(reg.lab[150, 150]), int(reg.lab[150, 450]), int(reg.lab[450, 150])} <= ids
    assert int(reg.lab[415, 450]) not in ids and int(reg.lab[5, 5]) not in ids


def test_a_csv_of_numbers_and_hex_is_read_as_colours(tmp_path, reg):
    p = tmp_path / 'map.csv'
    p.write_text('Number,R,G,B,RGB,HEX,Color_Name\n1,248,241,218,"RGB(248, 241, 218)",#F8F1DA,Warm Ivory\n'
                 '2,226,72,91,"RGB(226, 72, 91)",,Rose Red\n', encoding='utf-8')
    text = pt.csv_colours(str(p))
    assert text.splitlines() == ['1=F8F1DA', '2=E2485B']          # no hex: R,G,B
    area_col, *_ = pt.plan(reg, pt.parse_colors(text))
    assert area_col[1] == 'F8F1DA' and area_col[2] == 'E2485B'


def test_an_unnamed_area_takes_its_nearest_named_colour_and_a_named_white_stays(tmp_path):
    im = Image.new('L', (400, 400), 255)
    d = ImageDraw.Draw(im)
    d.ellipse([100, 100, 300, 300], outline=0, width=4)
    d.ellipse([170, 170, 230, 230], outline=0, width=4)              # a disc inside the ring
    im.save(tmp_path / 'r.png')
    reg = pt.find_regions(tmp_path / 'r.png', size=400, log=lambda m: None)
    ring, disc = int(reg.lab[130, 200]), int(reg.lab[200, 200])
    area_col, line, tiny, notes = pt.plan(reg, pt.parse_colors(f'1=FFFFFF, {ring}=laal'))
    index, pal, _ = pt.paint(reg, area_col, line, tiny)
    rgb = pal[index]
    assert tuple(rgb[5, 5]) == (255, 255, 255)                      # the white the user named
    assert tuple(rgb[200, 200]) == (0xC8, 0x10, 0x2E)               # the unnamed disc: the ring around it
    assert disc != ring and any('sabse paas' in n for n in notes)


def test_a_csv_with_a_row_per_colour_and_a_numbers_list(tmp_path, reg):
    p = tmp_path / 'g.csv'
    p.write_text('﻿Color,RGB,HEX,Numbers\nIvory,"RGB(244, 235, 211)",#F4EBD3,"1, 3"\n'
                 'Crimson,"RGB(178, 32, 46)",#B2202E,2\n', encoding='utf-8')
    assert pt.csv_colours(str(p)).splitlines() == ['1 3=F4EBD3', '2=B2202E']


def test_the_seal_can_ride_in_the_file_name(tmp_path, capsys):
    s = sketch(tmp_path / 'buta_seal3.png')
    assert main(['paint', str(s), '--out', str(tmp_path / 'o'), '--size', '600']) == 0
    assert 'seal 3 px' in capsys.readouterr().out


def test_a_csv_colour_with_its_name_and_hex_in_one_cell(tmp_path):
    p = tmp_path / 'n.csv'
    p.write_text('Number,Color\n1,Ivory / Warm Cream #FFF4D6\n2,laal\n', encoding='utf-8')
    assert pt.csv_colours(str(p)).splitlines() == ['1=FFF4D6', '2=laal']


def test_a_csv_hex_column_wins_over_a_colour_name_column(tmp_path):
    p = tmp_path / 'h.csv'
    p.write_text('Color,RGB/HEX,Numbers,Number_Count\nBlack,#150C07,"1, 5",2\n', encoding='utf-8')
    assert pt.csv_colours(str(p)).splitlines() == ['1 5=150C07']


def test_an_unnamed_area_is_one_colour_never_split(tmp_path):
    im = Image.new('L', (400, 200), 255)
    d = ImageDraw.Draw(im)
    for x in (130, 270):
        d.line([(x, 0), (x, 199)], fill=0, width=4)                 # three bands: left | middle | right
    im.save(tmp_path / 'b.png')
    reg = pt.find_regions(tmp_path / 'b.png', size=400, log=lambda m: None)
    left, mid, right = (int(reg.lab[100, x]) for x in (40, 200, 360))
    area_col, line, tiny, _ = pt.plan(reg, pt.parse_colors(f'{left}=laal, {right}=hara'))
    index, pal, _ = pt.paint(reg, area_col, line, tiny)
    assert len(np.unique(index[reg.lab == mid])) == 1             # the middle band: one colour, not half and half


def test_colours_given_twice_the_later_wins_and_a_check_sheet_is_made(tmp_path):
    s = sketch(tmp_path / 'b.png')
    out = tmp_path / 'o'
    csv_ = tmp_path / 'c.csv'
    csv_.write_text('Number,HEX\n1,#C8102E\n', encoding='utf-8')               # the ground in red, by mistake
    assert main(['paint', str(s), '--out', str(out), '--size', '600', '--colors', str(csv_),
                 '--colors', '1=F2E8CF']) == 0                                # ...corrected after
    with Image.open(out / 'b_final_600px_300dpi.png') as im:
        assert tuple(np.asarray(im.convert('RGB'))[5, 5]) == (0xF2, 0xE8, 0xCF)
    assert (out / 'b_check.png').exists()


def test_colours_read_off_a_coloured_version_of_the_design(tmp_path):
    s = sketch(tmp_path / 'r.png')
    ref = Image.open(s).convert('RGB')
    d = ImageDraw.Draw(ref)
    d.rectangle([0, 0, 599, 599], fill=(20, 15, 12))                    # black ground
    for cx, cy in ((150, 150), (450, 150), (150, 450), (450, 450)):
        d.ellipse([cx - 90, cy - 90, cx + 90, cy + 90], fill=(130, 100, 70))   # tan circles
    d.polygon([(300, 230), (360, 350), (240, 350)], fill=(230, 222, 210))      # a small white triangle
    ref = ref.resize((900, 900))                                              # another size: laid on the grid
    ref.save(tmp_path / 'ref.png')
    out = tmp_path / 'o'
    assert main(['paint', str(s), '--ref', str(tmp_path / 'ref.png'), '--out', str(out), '--size', '600']) == 0
    with Image.open(out / 'r_final_600px_300dpi.png') as im:
        rgb = np.asarray(im.convert('RGB')).astype(int)
    near = lambda p, c: np.abs(rgb[p] - np.array(c)).max() < 25
    assert near((5, 5), (20, 15, 12)) and near((150, 150), (130, 100, 70)) and near((320, 300), (230, 222, 210))
    assert (out / 'r_ref_numbers.png').exists() and (out / 'r_ref_colors.csv').exists()
