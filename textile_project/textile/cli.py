"""textile: one command for the mill's design work (ROADMAP Phase 1).

  python -m textile export  design.png --out out/ --name design01
  python -m textile verify  out/ --name design01
  python -m textile palette design.png
  python -m textile paint   sketch.png --out out/ [--colors "A=cream, B=laal, lines=coffee"]

Defaults: --size 3535 (px, the width; a square design is 3535 x 3535),
--dpi 300 (the mill's format). Every command ends with a short report.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

from . import export as ex
from . import edges as ed
from . import fill_auto as fa
from . import fill_method1 as m1
from . import fill_method2 as m2
from . import fill_method3 as m3
from . import fill_method4 as m4
from . import paint as pt
from . import palette as pl
from . import make as mk
from . import repeat as rp
from . import tile as tl
from .io_utils import hex_of, inches, load_rgb, read_cv2, rgb_of, safe_name, unique_rgb
from .verify import summary_hinglish, verify_package

SIZE, DPI = 3535, 300


def _target(w, h, size):
    """(width, height) for --size: the width is `size`, the height keeps the
    design's proportions (a square stays square)."""
    return size, max(1, round(size * h / w))


def _resize_index(index, size):
    h, w = index.shape
    tw, th = _target(w, h, size)
    if (tw, th) == (w, h):
        return index
    # an index map is only ever resized NEAREST: no new, mixed colours
    return cv2.resize(index, (tw, th), interpolation=cv2.INTER_NEAREST)


def cmd_export(a):
    rgb = load_rgb(a.image)
    min_share = 0.0 if a.keep_strays else a.min_share
    pal, is_flat = pl.extract_palette(rgb, a.max_colors, min_share, log=_log)
    index = pl.map_to_palette(rgb, pal)
    # a flat image's colours under min_share are left out of its palette and
    # their pixels go to the nearest kept colour: say which, like merge_stray does
    merged = pl.dropped_colours(rgb, pal) if is_flat else []
    if not a.keep_strays:
        index, pal, more = pl.merge_stray(index, pal, a.min_share)
        merged += more
    index = _resize_index(index, a.size)
    name = safe_name(a.name or os.path.splitext(os.path.basename(a.image))[0])
    line_index = None
    if a.line_color:
        line_index = int(((pal.astype(int) - rgb_of(a.line_color).astype(int)) ** 2).sum(1).argmin())
    done = ex.export_package(index, pal, a.out, name, a.dpi, line_index=line_index)
    v = verify_package(done['paths'], done['size_px'], a.dpi)
    report = {'design': name, 'size_px': done['size_px'], 'dpi': a.dpi,
              'print_size_inch': done['print_size_inch'], 'source': os.path.basename(a.image),
              'source_was_flat': bool(is_flat), 'colors_in_final': v['colors_in_final'],
              'channels': done['channels'], 'stray_merged': merged, 'verify': v}
    ex.write_report(done['paths']['report'], report)
    print('\n' + summary_hinglish(report))
    print(f"Files: {a.out}  (mill ko: {os.path.basename(done['paths']['tif'])})")
    return 0 if v['passed'] else 1


def cmd_fill(a):
    if a.method == 'auto':
        return _fill_auto(a)
    if a.method == '2':
        return _fill_method2(a)
    try:
        if a.method == '3':
            f = m3.fill(a.line, a.ref, a.size, a.max_colors, a.min_share, a.line_threshold, a.line_color,
                        a.reach_align, log=_log)
        elif a.method == '4':
            f = m4.fill(a.line, a.ref, a.size, a.max_colors, a.min_share, a.line_threshold, a.line_color,
                        a.seal, a.min_align, a.force, log=_log)
        else:
            f = m1.fill(a.line, a.ref, a.size, a.max_colors, a.min_share, a.line_threshold, a.line_color,
                        a.min_align, a.force, log=_log)
    except m1.FillError as e:
        print(f'STOP: {e}')
        return 1
    return _export_fill(a, f, int(a.method), *fa.merged(f, a.min_share, a.keep_strays))


def _export_fill(a, f, method, index, pal, merged, line_index, auto=None, more=()):
    """Write and verify a Method 1 or 3 fill (`auto`: what auto mode measured)."""
    name = safe_name(a.name or os.path.splitext(os.path.basename(a.line))[0])
    os.makedirs(a.out, exist_ok=True)
    if f.debug is not None:
        from PIL import Image
        Image.fromarray(f.debug).save(os.path.join(a.out, f'{name}_DEBUG_doubtful_regions.png'))
    done = ex.export_package(index, pal, a.out, name, a.dpi, line_index=line_index)
    label = {1: 'textile fill (method 1)', 3: 'textile fill (method 3: reference khiska kar)',
             4: 'textile fill (method 4: line art ke gap band)'}[method]
    ex.compare_sheet(f.reference, pal[index], os.path.join(a.out, f'{name}_compare.png'),
                     ('reference', label), more)
    v = verify_package(done['paths'], done['size_px'], a.dpi)
    report = {'design': name, 'method': method, 'size_px': done['size_px'], 'dpi': a.dpi,
              'print_size_inch': done['print_size_inch'], 'reference_was_flat': f.reference_was_flat,
              'colors_in_final': v['colors_in_final'], 'channels': done['channels'],
              'line_color_hex': hex_of(pal[line_index]), 'alignment_score': round(f.alignment_score, 3),
              'regions': f.regions, 'doubtful_regions': len(f.doubtful), 'stray_merged': merged, 'verify': v}
    if method == 3:
        report['alignment_before'] = round(f.alignment_before, 3)
        report['max_shift_px'] = f.max_shift_px
    if auto:
        report['auto'] = auto
    ex.write_report(done['paths']['report'], report)
    print('\n' + summary_hinglish(report))
    print(f"Files: {a.out}  (mill ko: {os.path.basename(done['paths']['tif'])})")
    return 0 if v['passed'] else 1


def _fill_auto(a):
    """Method 1, judged; then Method 3 or 2 (see fill_auto.choose)."""
    m2_args = _m2_args(a)
    if m2_args is None:
        return 1
    try:
        c = fa.choose(a.line, a.ref, a.size, a.max_colors, a.min_share, a.line_threshold, a.line_color, a.min_align,
                      a.max_cover_diff, a.reach_align, a.keep_strays, m2_args, log=_log)
    except (m1.FillError, ValueError) as e:
        print(f'STOP: {e}')
        return 1
    if c.method == 2:
        return _export_method2(a, c.fill, c.auto, c.others)
    return _export_fill(a, c.fill, c.method, c.index, c.pal, c.merged, c.line_index, c.auto, c.others)


def _m2_args(a):
    """Method 2's settings from the command line (None: --colors is wrong)."""
    colors = [c.strip() for c in a.colors.split(',')] if a.colors else None
    if colors and len(colors) != 2:
        print('STOP: --colors me do rang do: ground,motif (jaise 10100F,E8DFD2)')
        return None
    return dict(reach=a.reach, dark_level=a.dark_level, seed_area=a.seed_area, seed_dark=a.seed_dark,
                side_band=a.side_band, side_area=a.side_area, side_dark=a.side_dark, colors=colors)


def _fill_method2(a):
    m2_args = _m2_args(a)
    if m2_args is None:
        return 1
    try:
        f = m2.fill(a.line, a.ref, a.size, a.line_threshold, log=_log, **m2_args)
    except (m1.FillError, ValueError) as e:
        print(f'STOP: {e}')
        return 1
    return _export_method2(a, f)


def _export_method2(a, f, auto=None, more=()):
    name = safe_name(a.name or os.path.splitext(os.path.basename(a.line))[0])
    os.makedirs(a.out, exist_ok=True)
    done = ex.export_package(f.index, f.pal, a.out, name, a.dpi)
    ex.compare_sheet(f.reference, f.pal[f.index], os.path.join(a.out, f'{name}_compare.png'),
                     ('reference (sirf rang)', 'textile fill (method 2: line art ki shapes)'), more)
    v = verify_package(done['paths'], done['size_px'], a.dpi)
    report = {'design': name, 'method': 2, 'size_px': done['size_px'], 'dpi': a.dpi,
              'print_size_inch': done['print_size_inch'], 'colors_in_final': v['colors_in_final'],
              'channels': done['channels'], 'ground_hex': hex_of(f.pal[0]), 'motif_hex': hex_of(f.pal[1]),
              'regions': f.regions, 'ground_seeds': f.seeds, 'unreached': f.unreached,
              'depth_hist': f.depth_hist, 'verify': v}
    if auto:
        report['auto'] = auto
    ex.write_report(done['paths']['report'], report)
    print('\n' + summary_hinglish(report))
    print(f"Files: {a.out}  (mill ko: {os.path.basename(done['paths']['tif'])})")
    return 0 if v['passed'] else 1


def cmd_repeat(a):
    r = rp.analyze(a.image)
    print(rp.summary_hinglish(r))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        name = safe_name(a.name or os.path.splitext(os.path.basename(a.image))[0])
        ex.write_report(os.path.join(a.out, f'{name}_repeat.json'), r)
        print(f"JSON: {os.path.join(a.out, f'{name}_repeat.json')}")
    return 0


def _range(text, centre):
    """'496:514' -> range(496, 514, 4); default the centre +-8 px, step 4."""
    if text:
        lo, hi = (int(v) for v in text.split(':'))
        return range(lo, hi, 4)
    c = int(round(centre))
    return range(c - 8, c + 9, 4)


def cmd_tile(a):
    """Repeat analysis first (always), then the seamless block."""
    from PIL import Image
    name = safe_name(a.name or os.path.splitext(os.path.basename(a.image))[0])
    os.makedirs(a.out, exist_ok=True)
    r = rp.analyze(a.image)
    print(rp.summary_hinglish(r) + '\n')
    ex.write_report(os.path.join(a.out, f'{name}_repeat.json'), r)
    if r['type'] not in ('straight', 'half-drop'):
        print('STOP: is design me all-over repeat (aar-paar + upar-neeche) nahi mila, isliye tile nahi kaata. '
              + ('Panel print hai: mill ko "panel print, straight vertical repeat" bolo.' if r['type'] == 'panel' else ''))
        return 1
    s = a.shear_ratio if a.shear_ratio is not None else (r['shear'] / r['H'] if abs(r['shear']) > 5 else 0.0)
    bgr = read_cv2(a.image, cv2.IMREAD_COLOR)
    ds, valid = tl.deshear(bgr, s)
    if s:
        _log(f'[tile] deshear: har {r["H"]:.0f} px par {s * r["H"]:.1f} px seedha kiya')
    try:
        t, cost, (mse, W, H, x0, y0) = tl.extract(ds, valid, _range(a.w_range, r['W']), _range(a.h_range, r['H']),
                                                 log=_log)
    except ValueError as e:
        print(f'STOP: {e}')
        return 1
    block = cv2.cvtColor(np.clip(t, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    Image.fromarray(block).save(os.path.join(a.out, f'{name}_tile_native_{W}x{H}px.png'))
    from .verify import seam_check
    native_seam = seam_check(block)
    ground = None
    if a.clean_ground:
        block, g, share = tl.clean_ground(block, a.clean_ground)
        ground = {'hex': hex_of(g), 'share_percent': round(share * 100, 1)}
        _log(f'[tile] ground {hex_of(g)} saaf kiya ({share * 100:.1f}% pixels)')
    factor = a.size / W
    up = tl.upscale(block, a.size)
    report = {'design': name, 'source': os.path.basename(a.image), 'repeat': r,
              'block_source_px': [W, H], 'block_at': [x0, y0], 'deshear_ratio': round(s, 4),
              'seam_cost': round(float(cost), 1), 'upscale': round(factor, 2), 'clean_ground': ground,
              'native_seam': native_seam, 'dpi': a.dpi}
    if a.colors:
        pal, _ = pl.extract_palette(up, a.colors, a.min_share, log=_log)
        index = pl.map_to_palette(up, pal)
        index, pal, merged = pl.merge_stray(index, pal, a.min_share)
        done = ex.export_package(index, pal, a.out, name, a.dpi)
        v = verify_package(done['paths'], done['size_px'], a.dpi)
        final = pal[index]
        report.update({'size_px': done['size_px'], 'channels': done['channels'], 'stray_merged': merged,
                       'colors_in_final': v['colors_in_final'], 'verify': v})
        rpath, tif = done['paths']['report'], done['paths']['tif']
    else:
        final = up
        h, w = final.shape[:2]
        p = ex.paths(a.out, name, w, h, a.dpi)
        from .io_utils import save_png, save_tif, to_image
        save_png(to_image(final), p['png'], a.dpi)
        save_tif(to_image(final), p['tif'], a.dpi)
        report.update({'size_px': [w, h]})
        rpath, tif = p['report'], p['tif']
    report['seam'] = seam_check(final)
    report['print_size_inch'] = [inches(report['size_px'][0], a.dpi), inches(report['size_px'][1], a.dpi)]
    prev = Image.fromarray(tl.tiled(final))
    prev.thumbnail((1800, 1800), Image.LANCZOS)
    prev.save(os.path.join(a.out, f'{name}_3x3_preview.png'))
    ex.write_report(rpath, report)
    print('\n' + _tile_summary(report))
    print(f"Files: {a.out}  (mill ko: {os.path.basename(tif)})")
    return 0 if report['seam']['seamless'] and report.get('verify', {'passed': True})['passed'] else 1


def _tile_summary(r):
    w, h = r['size_px']
    rep = r['repeat']
    lines = [f"Tile: source me {r['block_source_px'][0]} x {r['block_source_px'][1]} px block "
             f"(jagah {r['block_at'][0]},{r['block_at'][1]}, seam cost {r['seam_cost']})",
             f"Size: {w} x {h} px = {r['print_size_inch'][0]} x {r['print_size_inch'][1]} inch @ {r['dpi']} DPI "
             f"({r['upscale']}x bada kiya)"]
    if r['upscale'] > 3:
        lines.append(f"  Source chhoti hai: {r['upscale']}x upscale soft lagega. Badi source image ho to wahi do.")
    if r['clean_ground']:
        lines.append(f"Ground {r['clean_ground']['hex']} pakka flat kiya ({r['clean_ground']['share_percent']}% pixels)")
    if 'channels' in r:
        lines.append(f"Rang (channels): {len(r['channels'])}")
        lines += [f"  {c['channel']:02d}. {c['hex']}  {c['coverage_percent']}%" for c in r['channels']]
        v = r['verify']
        lines.append('Verify: ' + ('PASS (channels mila kar final same, TIFF LZW)' if v['passed']
                                   else 'FAIL - ' + '; '.join(v['problems'])))
    else:
        lines.append('Rang: flat nahi kiye (photo jaisa tile). Channels chahiye to --colors N do.')
    lines.append('Repeat seam: ' + ('saaf (seamless)' if r['seam']['seamless'] else 'jod dikhega - 3x3 preview dekho'))
    if rep['type'] == 'half-drop':
        lines.append('Mill ko: is block ko STRAIGHT repeat me lagao (half-drop block ke andar hi bana hai).')
    else:
        lines.append('Mill ko: is block ko straight repeat me lagao.')
    return '\n'.join(lines)


def cmd_make(a):
    """An original design from a JSON config (see examples/)."""
    from PIL import Image
    from .verify import seam_check
    try:
        cfg = mk.load(a.config)
        index, pal, names = mk.make(cfg, log=_log)
    except (OSError, ValueError) as e:
        print(f'STOP: {e}')
        return 1
    name = safe_name(a.name or cfg.get('name') or os.path.splitext(os.path.basename(a.config))[0])
    dpi = int(cfg.get('dpi', 300))
    done = ex.export_package(index, pal, a.out, name, dpi)
    v = verify_package(done['paths'], done['size_px'], dpi)
    final = pal[index]
    seam = seam_check(final)
    panel = cfg.get('repeat', 'all-over') == 'panel'
    seamless = seam['top_bottom']['seamless'] if panel else seam['seamless']
    prev = Image.fromarray(np.tile(final, (3, 1, 1) if panel else (3, 3, 1)))
    prev.thumbnail((1800, 1800), Image.NEAREST)
    prev.save(os.path.join(a.out, f'{name}_repeat_preview.png'))
    report = {'design': name, 'size_px': done['size_px'], 'dpi': dpi, 'print_size_inch': done['print_size_inch'],
              'repeat': 'panel' if panel else 'all-over', 'config': cfg, 'colors_in_final': v['colors_in_final'],
              'channels': done['channels'], 'seam': seam, 'verify': v}
    ex.write_report(done['paths']['report'], report)
    print('\n' + summary_hinglish(report))
    print('Repeat: ' + ('panel (sirf upar-neeche): ' if panel else 'all-over: ')
          + ('jod saaf (seamless)' if seamless else 'JOD DIKHEGA - preview dekho'))
    print(f"Files: {a.out}  (mill ko: {os.path.basename(done['paths']['tif'])})")
    return 0 if v['passed'] and seamless else 1


def cmd_final(a):
    """A design made to the mill's exact repeat size (inches x DPI): flat colours, smooth hard edges, channels, TIF.

    The flat design is drawn straight at that size by `number`'s own colour work (`number.flat_index`: inks from
    the picture at its own size, every outline smoothed along itself onto the fine grid by `edges`). It used to be
    `number` at half size, then `crisp` curves at 2x: on the 23.5 x 20.7 inch sheet (7050 x 6210) that redraw lost
    dots inside petals and the thin cream gaps between them (and bent shapes: its smoothing scale counted the 2x
    zoom twice), took 180 s and 2.2 GB. Straight: 13 s, 0.5 GB for the flat, every detail of the flat design kept."""
    from . import number as nb
    wi, hi = (float(v) for v in a.inches.lower().replace('"', '').split('x'))
    Wt, Ht = round(wi * a.dpi), round(hi * a.dpi)                  # the file's pixels = inch x DPI
    name = safe_name(a.name or os.path.splitext(os.path.basename(a.design))[0])
    try:
        names = _ink_names(a)
    except (ValueError, OSError) as e:
        print(f'STOP: {e}')
        return 1
    rgb, note = fit_to(load_rgb(a.design), Wt, Ht)
    print(f'[final] {a.inches} inch @ {a.dpi} DPI = {Wt} x {Ht} px. {note}')
    g = nb.grain(rgb)
    woven = g >= nb.WOVEN_GRAIN
    px_mm = a.dpi / 25.4
    keep_mm2 = a.min_mm2 if getattr(a, 'min_mm2', None) else nb.DETAIL_MM2[a.detail] * (nb.WOVEN_DETAIL if woven else 1)
    pal, index = nb.flat_index(rgb, Wt, Ht, a.colors, keep_mm2 * px_mm ** 2, woven, True, g, a.small_px, bool(getattr(a, 'min_mm2', None)),
                               rescue=not getattr(a, 'no_detail_rescue', False))
    if a.merge_shades:
        index, shaded = nb.merge_shade_inks(index, pal, px_mm)
        for src, dst in shaded:
            print(f'[final] shade: {src} ko {dst} me mila diya (usi rang ka gehra/halka shade, ek screen)')
    if a.merge_similar:
        index, _, moved = nb.merge_similar_inks(index, pal)
        for src, dst in moved:
            print(f'[final] chhota rang {src} ko {dst} me mila diya (ek hi rang jaisa dikhta hai)')
    used = np.flatnonzero(np.bincount(index.ravel(), minlength=len(pal)))     # no empty channel
    remap = np.zeros(len(pal), np.uint8)
    remap[used] = np.arange(len(used))
    index, pal = remap[index], pal[used]
    return final_package(index, pal, a.out, name, a.dpi, rgb, (wi, hi), note, names, not a.no_psd)


def _ink_names(a):
    """--ink-names FILE: {'RRGGBB': the mill's name} (textile.names.read_ink_names), or None."""
    from .names import read_ink_names
    return read_ink_names(a.ink_names) if getattr(a, 'ink_names', None) else None


def fit_to(rgb, Wt, Ht):
    """The picture cut (never stretched) to a Wt x Ht repeat's proportions, centred, and what was done: `final`'s
    crop, kept in one place so a design made again at that size (LoomLab's rebuild) is matched to the same part."""
    sh, sw = rgb.shape[:2]
    want = Wt / Ht
    note = 'aspect already matches'
    if abs(sw / sh - want) > 0.01 * want:                         # crop (never stretch) to the repeat's proportions
        if sw / sh > want:
            nw = round(sh * want)
            x0 = (sw - nw) // 2
            rgb = rgb[:, x0:x0 + nw]
        else:
            nh = round(sw / want)
            y0 = (sh - nh) // 2
            rgb = rgb[y0:y0 + nh]
        note = f'design {sw}x{sh} ko {rgb.shape[1]}x{rgb.shape[0]} crop kiya (kheencha nahi); repeat seamless tha to jod ab nahi milega'
    return np.ascontiguousarray(rgb), note


def final_package(index, pal, out, name, dpi, rgb, inches, note, ink_names=None, psd=True):
    """`final`'s files from its flat design (index + inks at the mill's size): channels, TIF, SVG, the layered PSD
    (one layer per plate; no guide: number's sketch is another size and crop), verify and the report with the
    match to the (cut) picture `rgb`. Also what LoomLab runs when the operator's palette edits are put into a
    mill-size package again. `ink_names`: {'RRGGBB': the mill's name} for those plates."""
    from . import number as nb
    from . import vector as vc
    from .verify import verify_package
    wi, hi = inches
    done = ex.export_package(index, pal, out, name, dpi, ink_names=ink_names)
    _, svg, _ = vc.trace(index, pal)                              # the same shapes as vectors, for the mill's software
    with open(os.path.join(out, f'{name}_final.svg'), 'w', encoding='utf-8') as fh:
        fh.write(svg)
    psd_path = None
    if psd:                                                       # after the export: its peak memory is gone by now
        from . import psdout
        w, h = done['size_px']
        psd_path = psdout.write_psd(done['paths']['channels_zip'],
                                    os.path.join(out, f'{name}_layers_{ex.size_tag(w, h)}_{dpi}dpi.psd'), dpi)
    v = verify_package(done['paths'], done['size_px'], dpi)
    real = nb.design_match(rgb, pal[index])
    report = {'design': name, 'inches': [wi, hi], 'dpi': dpi, 'size_px': done['size_px'], 'crop': note,
              'colours': len(pal), 'channels': done['channels'], 'design_match': real, 'verify': v}
    ex.write_report(done['paths']['report'], report)
    print(f"\n{name}: {done['size_px'][0]} x {done['size_px'][1]} px @ {dpi} DPI = {wi} x {hi} inch, {len(pal)} rang, "
          f"asli design se mel {real}%")
    print(f"Mill ko: {os.path.basename(done['paths']['tif'])} | verify: {'PASS' if v['passed'] else 'FAIL'}")
    if psd_path:
        print(f"Photoshop file: {os.path.basename(psd_path)} (har rang ek layer, {w}x{h} px, {dpi} DPI)")
    return 0 if v['passed'] else 1


def cmd_crisp(a):
    """A flat design redrawn as smooth curves: SVG (sharp at any zoom) + a zoomed flat PNG."""
    import json
    from PIL import Image
    from . import crisp as cr
    with Image.open(a.image) as im:
        rgb = np.asarray(im.convert('RGB'))
        dpi = round(float(im.info.get('dpi', (DPI, DPI))[0])) or DPI
    try:
        svg, big, rep = cr.crisp(rgb, a.scale, a.zoom, a.smoothing)
    except ValueError as e:
        print(f'STOP: {e}')
        return 1
    os.makedirs(a.out, exist_ok=True)
    name = safe_name(a.name or os.path.splitext(os.path.basename(a.image))[0].split('_final_')[0])
    with open(os.path.join(a.out, f'{name}_crisp.svg'), 'w', encoding='utf-8') as fh:
        fh.write(svg)
    png = os.path.join(a.out, f'{name}_crisp_{a.zoom}x.png')
    from .io_utils import save_png, to_image
    save_png(to_image(big), png, dpi * a.zoom)
    with open(os.path.join(a.out, f'{name}_crisp_report.json'), 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, indent=1)
    print(f"{name}: {rep['colours']} rang, {rep['outlines']} curve-shapes. SVG (kitna bhi zoom saaf): {name}_crisp.svg; "
          f"flat PNG {big.shape[1]}x{big.shape[0]} px (har pixel ek rang, bina blur): {os.path.basename(png)}")
    print(f"Pixel design se farak: {rep['pixels_different_percent']}% pixel (sirf kinaron par). Sirf dekhne/edit/share ke liye; "
          f"mill ko pixel TIF hi do.")
    return 0


def cmd_vector(a):
    """Trace a flat design's channels and draw them back without anti-aliasing."""
    from PIL import Image
    from . import vector as vc
    with Image.open(a.image) as im:
        rgb = np.asarray(im.convert('RGB'))
        dpi = round(float(im.info.get('dpi', (DPI, DPI))[0])) or DPI
    pal = unique_rgb(rgb)
    if len(pal) > 64:
        print(f'STOP: is image me {len(pal)} rang hain - ye flat design nahi. Pehle fill/export/make se flat karo.')
        return 1
    index = pl.map_to_palette(rgb, pal).astype(np.uint8)
    pad = 32 if a.repeat else 0
    work = np.pad(index, pad, mode='wrap') if pad else index
    traced, svg, _ = vc.trace(work, pal, a.eps)
    if pad:
        traced = traced[pad:-pad, pad:-pad]
        _, svg, _ = vc.trace(index, pal, a.eps)          # the SVG of the tile itself
    changed = float((traced != index).mean())
    name = safe_name(a.name or os.path.splitext(os.path.basename(a.image))[0].split('_final_')[0] + '_vector')
    done = ex.export_package(traced, pal, a.out, name, dpi)
    with open(os.path.join(a.out, f'{name}.svg'), 'w') as f:
        f.write(svg)
    v = verify_package(done['paths'], done['size_px'], dpi)
    report = {'design': name, 'source': os.path.basename(a.image), 'size_px': done['size_px'], 'dpi': dpi,
              'print_size_inch': done['print_size_inch'], 'eps_px': a.eps, 'changed_percent': round(changed * 100, 2),
              'colors_in_final': v['colors_in_final'], 'channels': done['channels'], 'verify': v}
    if a.repeat:
        from .verify import seam_check
        report['seam'] = seam_check(pal[traced])
    ex.write_report(done['paths']['report'], report)
    print(summary_hinglish(report))
    print(f"Vector: kinare seedhe kiye (eps {a.eps} px = {a.eps * 25.4 / dpi:.2f} mm tak), "
          f"{changed * 100:.2f}% pixel badle. SVG: {name}.svg")
    print(f"Files: {a.out}  (mill ko: {os.path.basename(done['paths']['tif'])})")
    return 0 if v['passed'] and report.get('seam', {'seamless': True})['seamless'] else 1


def cmd_edges(a):
    """Clean edges: every ink's outline redrawn smooth and hard-edged, drawn at --size (default 3535 px)."""
    from PIL import Image
    with Image.open(a.image) as im:
        rgb = np.asarray(im.convert('RGB'))
    pal = unique_rgb(rgb)
    if len(pal) > 64:
        print(f'STOP: is image me {len(pal)} rang hain - ye flat design nahi. Pehle fill/export/make se flat karo.')
        return 1
    index = pl.map_to_palette(rgb, pal).astype(np.uint8)
    wrap = (a.repeat in ('x', 'both'), a.repeat in ('y', 'both'))
    factor = a.size / index.shape[1]
    cleaned, rep = ed.clean(index, factor, a.strength, wrap, a.specks)
    used = np.unique(cleaned)                                  # an ink made only of crumbs is gone
    remap = np.zeros(len(pal), np.uint8)
    remap[used] = np.arange(len(used))
    cleaned, pal = remap[cleaned], pal[used]
    h, w = cleaned.shape
    # how much of the design moved: against the source drawn plain (nearest) at the same size
    plain = np.asarray(Image.fromarray(rgb).resize((w, h), Image.NEAREST))
    changed = float((pal[cleaned] != plain).any(-1).mean())
    name = safe_name(a.name or os.path.splitext(os.path.basename(a.image))[0].split('_final_')[0] + '_clean')
    done = ex.export_package(cleaned, pal, a.out, name, a.dpi)
    v = verify_package(done['paths'], done['size_px'], a.dpi)
    report = {'design': name, 'source': os.path.basename(a.image), 'size_px': done['size_px'], 'dpi': a.dpi,
              'print_size_inch': done['print_size_inch'], 'edges': rep, 'changed_percent': round(changed * 100, 2),
              'colors_in_final': v['colors_in_final'], 'channels': done['channels'], 'verify': v}
    if a.repeat != 'no':
        from .verify import seam_check
        report['seam'] = seam_check(pal[cleaned])
    ex.write_report(done['paths']['report'], report)
    print(summary_hinglish(report))
    print(f"Kinare saaf (strength {a.strength}): {rep['outlines_smoothed']} outline smooth, {rep['specks_moved']} chhote tukde "
          f"padosi rang me, {changed * 100:.2f}% pixel badle. Rang wahi {len(pal)}, naya koi nahi.")
    print(f"Files: {a.out}  (mill ko: {os.path.basename(done['paths']['tif'])})")
    return 0 if v['passed'] and report.get('seam', {'seamless': True})['seamless'] else 1


def cmd_paint(a):
    """A sketch coloured as the user says: step 1 (no --colors) the lettered map,
    step 2 the design, one channel per named colour."""
    from PIL import Image
    name = safe_name(a.name or os.path.splitext(os.path.basename(a.sketch))[0])
    if a.seal is None:                              # 'peony_seal10.png': the seal rides in the file name (Windows bat)
        import re
        m = re.search(r'seal[_-]?(\d+)', os.path.basename(a.sketch), re.I)
        if m:
            a.seal = int(m.group(1))
            print(f'[seal] file ke naam se: {a.seal} px')
    try:
        reg = pt.find_regions(a.sketch, a.size, a.line_threshold, a.seal, a.group_tolerance, log=_log)
    except m1.FillError as e:
        print(f'STOP: {e}')
        return 1
    os.makedirs(a.out, exist_ok=True)
    mp = pt.maps(reg, a.out, name)
    if not a.colors and pt.filled(mp['colors']):
        a.colors = [mp['colors']]                   # the map's own file, filled in: paint with it
        print(f"[colors] {os.path.basename(mp['colors'])} me rang likhe hain: unse rang bhar raha hoon")
    ref_text, ref_rgb = None, None
    if a.ref:
        try:
            ref_text, ref_rgb = pt.from_reference(reg, a.ref, a.ref_colors, a.out, name, log=_log)
        except m1.FillError as e:
            print(f'STOP: {e}')
            return 1
        print(f'[ref] {name}_ref_numbers.png = rangeen design par har hisse ka number; '
              f'{name}_ref_colors.csv = har number ka rang (badalna ho to --colors se sudhaar do)')
    if not a.colors and not a.ref:
        print(f"\nMap: {os.path.basename(mp['map'])} (group letters), {os.path.basename(mp['numbers'])} "
              f"(har hisse ka number), list: {os.path.basename(mp['groups'])}")
        print(f'{reg.n} band hisse, {len(reg.letters)} group. Ground (sabse bada) = {reg.letters[reg.ground]}.')
        print('Ab rang batao:  --colors "A=cream, B=laal, C D=hara, 12=gold, lines=coffee"')
        print(f"  ya {os.path.basename(mp['colors'])} me likh kar:  --colors {mp['colors']}")
        return 0
    parts = [ref_text] if ref_text else []
    for c in (a.colors or []) if isinstance(a.colors, list) or a.colors is None else [a.colors]:   # later wins
        if os.path.isfile(c) and c.lower().endswith('.csv'):
            try:
                c = pt.csv_colours(c)
            except (m1.FillError, ValueError) as e:
                print(f'STOP: {e}')
                return 1
        elif os.path.isfile(c):
            with open(c, encoding='utf-8') as fh:
                c = fh.read()
        parts.append(c)
    text = '\n'.join(parts)
    old = pt.stamp_mismatch(reg, text)
    if old:
        print(f'STOP: ye rang-file doosre map ki hai ({old[2:]}), abhi {pt.stamp(reg)[2:]}. Letters/numbers '
              'size, seal aur sketch ke saath badalte hain: wahi --size/--seal do, ya naya map dekh kar rang likho.')
        return 1
    try:
        area_col, line, tiny, notes = pt.plan(reg, pt.parse_colors(text))
        index, pal, li = pt.paint(reg, area_col, line, tiny)
        names = _ink_names(a)
    except (m1.FillError, ValueError, OSError) as e:
        print(f'STOP: {e}')
        return 1
    done = ex.export_package(index, pal, a.out, name, a.dpi, line_index=li, ink_names=names)
    with Image.open(a.sketch) as im:
        sk = im.convert('RGB')
        sk = np.asarray(sk.resize((1200, max(1, round(1200 * sk.height / sk.width))), Image.LANCZOS))
    if ref_rgb is not None:
        sk = cv2.resize(ref_rgb, (sk.shape[1], sk.shape[0]), interpolation=cv2.INTER_AREA)
    ex.compare_sheet(sk, pal[index], os.path.join(a.out, f'{name}_compare.png'),
                     ('rangeen design' if ref_rgb is not None else 'sketch', 'textile paint'))
    _, biggest = pt.check(reg, index, pal, a.out, name)
    with open(os.path.join(a.out, f'{name}_colors.txt'), 'w', encoding='utf-8') as fh:
        body = '\n'.join(l for l in text.strip().splitlines() if not l.startswith('# textile paint:'))
        fh.write(pt.stamp(reg) + '\n' + body + '\n')
    v = verify_package(done['paths'], done['size_px'], a.dpi)
    small = [c['hex'] for c in done['channels'] if c['coverage_percent'] < pl.STRAY_SHARE * 100]
    if small:
        notes.append(f"{', '.join(small)}: 0.05% se kam hissa - itne chhote rang ki alag screen chahiye? "
                     'Nahi to us hisse ko kisi aur rang ka bolo.')
    report = {'design': name, 'source': os.path.basename(a.sketch), 'size_px': done['size_px'], 'dpi': a.dpi,
              'print_size_inch': done['print_size_inch'], 'colors_in_final': v['colors_in_final'],
              'channels': done['channels'],
              'paint': {'biggest_areas': biggest, 'colors': text.strip(), 'areas': reg.n, 'groups': len(reg.letters),
                        'ground_group': reg.letters[reg.ground] if reg.ground >= 0 else None,
                        'tiny_areas': int((reg.group[1:] < 0).sum()), 'seal_px': reg.seal,
                        'line_threshold': a.line_threshold, 'sketch': reg.sketch_hash,
                        'lines': line, 'notes': notes},
              'verify': v}
    ex.write_report(done['paths']['report'], report)
    print('\n' + summary_hinglish(report))
    for n in notes:
        print(f'Note: {n}')
    print('Sabse bade hisse aur unka rang (yahi galat ho to rang "failta" dikhta hai):')
    print('  ' + ', '.join(f'{i} ({share}%) {hx} {cname}' for i, share, hx, cname in biggest))
    print(f'Check: {name}_check.png = bana design + har hisse ka number (kis number ko kya rang mila)')
    print(f"Files: {a.out}  (mill ko: {os.path.basename(done['paths']['tif'])}; dekhne ko: {name}_compare.png)")
    return 0 if v['passed'] else 1


def _learn_from(r, a):
    """After a number run: print what looks wrong with a fix for each, what similar past designs needed, and note
    the run in the log. Advice only (nothing here changes the files made); never stops a run."""
    from . import learn
    try:
        problems = learn.advise(r)
        if problems:
            print('\nSalah (kya dikkat dikhi aur kya karna hai):')
            for _, msg in problems:
                print(f'  - {msg}')
        else:
            print('\nSalah: koi dikkat nahi dikhi.')
        for dist, row, verdict in learn.similar(r, 2):
            if dist < 3:
                tag = {'good': ' (aapne good kaha)', 'bad': ' (aapne bad kaha)'}.get(verdict, '')
                print(f"  Pehle ka milta-julta design: {row['name']}{tag}: line {row['result']['line_mm']} mm, "
                      f"{row['result']['inks']} rang, match {row['result']['match']}%")
        settings = {'colours': r.get('colours_limit'), 'detail': r.get('detail'), 'line_mm': r['line_mm'],
                    'size': a.size, 'circle': a.circle, 'polygons': a.polygons, 'motifs': a.motifs}
        if learn.record(r, a.design, settings, problems, getattr(a, 'run_id', None)):
            print(f"  (ye run yaad kar liya. Result achha/kharab laga to: python -m textile feedback {r['name']} good|bad \"note\")")
    except Exception as e:                      # a note must never stop the job
        print(f'  (yaad rakhne me dikkat: {e})')


def cmd_feedback(a):
    from . import learn
    run = learn.feedback(a.name, a.verdict, ' '.join(a.note))
    if run is None:
        print(f"'{a.name}' ka koi run record nahi mila. Naam wahi likho jo `number` ne chhapa.")
        return 1
    print(f"Yaad kar liya: {a.name} ({run['time']}) = {a.verdict}.")
    return 0


def cmd_stitch(a):
    """Parts of one design (a folder, a zip, or files) put back together; with --number the whole number run follows."""
    import glob
    import json
    import tempfile
    import zipfile
    from . import stitch as st
    paths = []
    tmp = None
    for item in a.parts:
        if os.path.isdir(item):
            paths += sorted(glob.glob(os.path.join(item, '**', '*.*'), recursive=True))
        elif item.lower().endswith('.zip'):
            tmp = tmp or tempfile.mkdtemp(prefix='stitch_')
            with zipfile.ZipFile(item) as z:
                for n in z.namelist():
                    base = os.path.basename(n)
                    if base and not base.startswith('.') and os.path.splitext(base)[1].lower() in st.IMAGES:
                        dst = os.path.join(tmp, base)
                        with open(dst, 'wb') as fh:
                            fh.write(z.read(n))
                        paths.append(dst)
        else:
            paths.append(item)
    grid = None
    if a.grid:
        r, c = a.grid.lower().split('x')
        grid = (int(r), int(c))
    try:
        img, rep = st.stitch(paths, grid, log=_log)
    except ValueError as e:
        print(f'STOP: {e}')
        return 1
    os.makedirs(a.out, exist_ok=True)
    name = a.name or 'stitched'
    path = os.path.join(a.out, f'{name}_stitched.png')
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    with open(os.path.join(a.out, f'{name}_stitch_report.json'), 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, indent=1)
    W, H = rep['size']
    print(f"\nJoda hua design: {os.path.basename(path)} ({W}x{H} px, {rep['grid'][0]}x{rep['grid'][1]} parts)")
    vis = [j for j in rep['joints'] if j['visible']]
    print('Jod: ' + ('sab theek, koi jod nahi dikhta' if not vis else
                     'DIKHTE HAIN: ' + ', '.join(f"{j['between']} (farak {j['jump']})" for j in vis)
                     + ' -> ye parts aapas me nahi milte (alag-alag bane?)'))
    if a.number:
        print('\n--- number (poora design: sketch, numbers, rang plates) ---')
        return main(['number', path, '--out', a.out, '--name', name] + list(a.number_args or []))
    return 0


def cmd_split(a):
    from . import stitch as st
    r, c = (int(v) for v in a.grid.lower().split('x'))
    out, plan = st.split(a.image, a.out, a.name, (r, c), a.overlap / 100.0, a.target)
    print(f"{len(out)} parts ({r}x{c}, {a.overlap:g}% overlap), har part {plan['part_size'][0]}x{plan['part_size'][1]} px -> {a.out}")
    if a.target:
        print(f"{a.target} px chaudai ka design chahiye to AI se har part {plan['ask_ai_part_px']} px ka maango "
              f"(2048 ya 1536 mile to bhi chalega: `stitch` jodta hai, `final` exact size banata hai)")
    print('Har part ko bada/saaf karke `textile stitch` ko do. Jaanchne ke liye: textile partscheck REFERENCE PARTS_FOLDER --plan *_parts_plan.json')
    return 0


def cmd_partscheck(a):
    """Every part an AI redrew, compared with the reference: moved motifs, new colours, lost / invented detail."""
    import glob
    from . import stitch as st
    files = []
    for item in a.parts:
        files += sorted(glob.glob(os.path.join(item, '*.*'))) if os.path.isdir(item) else [item]
    files = [f for f in files if os.path.splitext(f)[1].lower() in st.IMAGES]
    r, c = (int(v) for v in a.grid.lower().split('x'))
    try:
        res = st.parts_check(a.reference, files, a.plan, (r, c), a.overlap / 100.0)
    except ValueError as e:
        print(f'STOP: {e}')
        return 1
    for k, v in sorted(res['tiles'].items()):
        print(f"  {k}: " + ('theek' if not v['redo'] else 'DOBARA BANWAO: ' + '; '.join(v['redo']))
              + f"  (khisak {v['shift_percent']}%, naye rang {v['new_colour_percent']}%, kinare {v['edge_ratio']}x)")
    if res['redo']:
        print('\nDobara banwane wale parts: ' + ', '.join(sorted(res['redo'])) + '. Baaki theek hain.')
    else:
        print('\nSab parts reference se mel khaate hain.')
    return 0 if not res['redo'] else 2


def cmd_number(a):
    """A coloured design numbered: every patch of one colour gets a number (a map to plan a sketch on)."""
    from . import number as nb
    import re
    if a.hd or re.search(r'(^|[_-])hd($|[_-])', os.path.splitext(os.path.basename(a.design))[0], re.I):
        a.size *= 2                                  # same 300 DPI, twice the pixels (a bigger sheet: 7070 px = 23.6 inch)
        print(f'[hd] {a.size} px @ {a.dpi} DPI: sketch aur numbers double resolution me')
    try:
        r = nb.number(a.design, a.out, a.name, a.size, a.colors, a.detail, not a.no_smooth, a.line_mm,
                      not a.no_separators, a.min_area, dpi=a.dpi, bold_mm=a.bold_mm, bold_scale=a.bold_scale, circle=a.circle, polygons=a.polygons, motifs=a.motifs, merge_similar=a.merge_similar,
                      fair=(a.fair if a.fair is not None else a.smoothing / 20.0), merge_shades=a.merge_shades, small_px=a.small_px, log=_log,
                      detail_rescue=not a.no_detail_rescue)
    except m1.FillError as e:
        print(f'STOP: {e}')
        return 1
    w, h = r['size_px']
    with open(os.path.join(a.out, f"{r['name']}_number.json"), 'w', encoding='utf-8') as fh:   # what was made, for LoomLab
        json.dump({k: (os.path.basename(v) if isinstance(v, str) and ('/' in v or os.sep in v) else v) for k, v in r.items()},
                  fh, indent=1, default=str)
    print(f"\n{r['name']}: {w}x{h} px, {r['areas']} hisse numbered, {len(r['inks'])} rang:")
    for hx, cname, share in r['inks']:
        print(f'  {hx}  {cname}  {share}%')
    if len(r['tried']) > 1:
        print('Line motai ki tulna (sketch + CSV se wapas design):')
        for mm, m, parts in r['tried']:
            print(f"  {mm} mm: {m}% match, {parts} hisse" + ('   <- chuna' if mm == r['line_mm'] else ''))
    print(f"Sketch se wapas design: {r['match']}% match | {r['tiny']} bahut chhote hisse | {r['groups']} group (letters: "
          f"{os.path.basename(r['letters'])})")
    print(f"Asli design (aapki image) se mel: {r['design_match']}% - flat rang image ki shading/grain nahi rakhte, "
          f"isliye AI image par 85-93% achha hai (95+ mumkin nahi); upar wala 99.9% sirf sketch ka hai")
    print(f"Sketch: {os.path.basename(r['sketch'])} (rang hata kar sirf lines), "
          f"{os.path.basename(r['sketch_numbers'])} (sketch + wahi numbers)")
    print(f"Bold sketch: {os.path.basename(r['bold'])} (smooth curves, gehri moti line) + .svg (kitna bhi zoom, saaf)")
    sn = r.get('snapped') or {}
    print(f"Bold sketch: {sn.get('circles', 0)} gol pakke circle, {sn.get('polygons', 0)} saaf triangle/rectangle jaisi shape, "
          f"{sn.get('leaves', 0)} patti/petal, {sn.get('ovals', 0)} oval, baaki {sn.get('smooth', 0)} smooth curve")
    print(f"Bold sketch + numbers: {os.path.basename(r['bold_numbers'])}")
    print(f"Bold sketch rangeen: {os.path.basename(r['bold_colour'])} (rang bhara, smooth curves ke saath)")
    print(f"Rangeen: {r['name']}_rangeen.png (sketch + CSV se bana design, jaisa paint banayega)")
    print(f"Files: {r['name']}_numbers.png (rangeen design par numbers), {r['name']}_flat.png, "
          f"{r['name']}_colors.csv (har number ka rang)")
    pkg_code = 0
    if not a.no_package:                              # the colour plates: one channel per colour + the mill's TIF, from the sketch + CSV
        pkg_code = number_package(r['name'], r['sketch'], r['csv'], r['bold_numbers'], a.out, (w, h), a.dpi, not a.no_psd,
                                   use_layers=bool(getattr(a, 'psd_layers', False)))
    # Optional prepress extras (underbase/trap/film/estimate/seam); never touch number outputs when off
    pre_rep = None
    if any((getattr(a, 'underbase', False), getattr(a, 'trap', None), getattr(a, 'film', False),
            getattr(a, 'estimate', False), getattr(a, 'seam_check', False))):
        from . import prepress as pp
        try:
            pre_rep = pp.run_prepress(
                a.out, r['name'],
                underbase=bool(getattr(a, 'underbase', False)),
                underbase_choke_px=int(getattr(a, 'underbase_choke', 1) or 1),
                trap_px=getattr(a, 'trap', None),
                film=bool(getattr(a, 'film', False)),
                estimate=bool(getattr(a, 'estimate', False)),
                seam_check=bool(getattr(a, 'seam_check', False)),
                dpi=a.dpi, log=_log,
            )
        except Exception as e:
            print(f'[prepress] STOP: {e}')
            return 1
    grad_rep = None
    if getattr(a, 'gradients', False) or getattr(a, 'hybrid', False):
        from . import gradients as gr
        try:
            grad_rep = gr.run_gradients(
                a.out, r['name'], a.design,
                size_px=(w, h), dpi=a.dpi,
                lpi=int(getattr(a, 'halftone_lpi', 45) or 45),
                angle_deg=float(getattr(a, 'halftone_angle', 45.0)),
                colors_limit=int(a.colors),
                hybrid=bool(getattr(a, 'hybrid', True)),  # --gradients defaults to hybrid
                log=_log,
            )
        except Exception as e:
            print(f'[gradients] STOP: {e}')
            return 1
    # Put underbase / trapped spots / tone plates into package channels (TIF/PSD/channels/)
    want_pkg_merge = (not a.no_package) and any((
        getattr(a, 'underbase', False),
        getattr(a, 'trap', None),
        getattr(a, 'gradients', False),
        getattr(a, 'hybrid', False),
    ))
    if want_pkg_merge:
        from . import package_channels as pc
        ub = os.path.join(a.out, 'prepress', 'UNDERBASE_WHITE.png')
        trap_dir = os.path.join(a.out, 'prepress', 'trapped_plates')
        tones = (grad_rep or {}).get('tone_channels') or (grad_rep or {}).get('tone_plates') if grad_rep else None
        try:
            pc.merge_into_package(
                a.out, r['name'], dpi=a.dpi,
                underbase_path=ub if getattr(a, 'underbase', False) and os.path.isfile(ub) else None,
                trapped_dir=trap_dir if getattr(a, 'trap', None) and os.path.isdir(trap_dir) else None,
                tone_plates=tones or None,
                bold_numbers=r.get('bold_numbers'),
                write_psd=not a.no_psd,
                composite_path=(grad_rep or {}).get('composite') if grad_rep else None,
                include_composite=bool(getattr(a, 'psd_composite', False)),
                log=_log,
            )
        except Exception as e:
            print(f'[package] STOP: {e}')
            return 1
    _learn_from(r, a)
    if r['missed']:
        print(f"Note: {r['missed']} hisson ke number ke liye jagah nahi mili (sirf neela dot).")
    return pkg_code



def number_package(name, sketch, colours_csv, bold_numbers, out, size_px, dpi=DPI, psd=True, ink_names=None, use_layers=False):
    """A number run's plates, made from its sketch (`NAME_sketch_seal0.png`) + colours CSV by `paint`, into
    OUT/package/: one channel per colour, the mill's TIF, the layered PSD (guide: the bold sketch with numbers,
    hidden) and the loose channel / stacked images. `number` calls it after numbering; LoomLab calls it again
    when the operator's palette edits (the CSV's HEX column rewritten) are put into the zip. `ink_names`: a
    --ink-names file for paint (the mill's names for those plates). paint's exit code."""
    w, h = size_px
    pkg = os.path.join(out, 'package')
    print(f'\n--- Rang channels (har rang ki alag plate) -> {pkg} ---')
    pkg_code = main(['paint', sketch, '--colors', colours_csv, '--out', pkg, '--name', name,
                     '--size', str(w)] + (['--dpi', str(dpi)] if dpi != DPI else [])
                    + (['--ink-names', ink_names] if ink_names else []))
    if pkg_code == 0:
        import glob
        from . import stack
        zips = glob.glob(os.path.join(pkg, f"{name}_colored_channels_*.zip"))
        if zips:
            if psd:
                from . import psdout
                psd_path = psdout.write_psd(
                    zips[0], os.path.join(pkg, f"{name}_layers_{w}px_{dpi}dpi.psd"), dpi,
                    guides=[('GUIDE bold sketch + numbers (hidden)', bold_numbers)] if use_layers else (),
                    use_layers=use_layers,
                )
                if use_layers:
                    print(f"Photoshop file: {os.path.basename(psd_path)} (har rang ek layer, {w}x{h} px, {dpi} DPI; "
                          f"guide layer chhupi hui)")
                else:
                    print(f"Photoshop file: {os.path.basename(psd_path)} (spot CHANNELS, darkest-first, "
                          f"{w}x{h} px, {dpi} DPI; no ink layers)")
            chs, steps = stack.loose_images(zips[0], pkg, name, dpi)
            print(f"Alag-alag images ({w}x{h} px, {dpi} DPI): {len(chs)} channels -> {pkg}{os.sep}channels{os.sep}, "
                  f"{len(steps)} overlap steps (1 channel, 1+2, ... sab) -> {pkg}{os.sep}stacked{os.sep}")
    print(f"Mill ko: {pkg}{os.sep}{name}_final_{w}px_{dpi}dpi.tif ; plates: {name}_colored_channels_*.zip, "
          f"{name}_bw_separations_*.zip")
    return pkg_code

PAIR = ('_lineart', '_ref', '_colored', '_reference')
IMAGES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp')


def pairs(folder):
    """(name, line art, reference) for every NAME_lineart.* with a
    NAME_ref.* / NAME_colored.* / NAME_reference.* beside it, sorted; and the
    line arts that have none."""
    files = {f.lower(): f for f in os.listdir(folder) if f.lower().endswith(IMAGES)}
    out, alone = [], []
    for low, f in sorted(files.items()):
        stem, ext = os.path.splitext(low)
        if not stem.endswith('_lineart'):
            continue
        base = stem[:-len('_lineart')]
        ref = next((files[c] for s in PAIR[1:] for c in files if os.path.splitext(c)[0] == base + s), None)
        if ref:
            out.append((os.path.splitext(f)[0][:-len('_lineart')], os.path.join(folder, f), os.path.join(folder, ref)))
        else:
            alone.append(f)
    return out, alone


def cmd_batch(a):
    """Every pair in a folder through `fill`, each into out/NAME/; one bad
    pair never stops the rest. A summary table and batch_summary.csv."""
    import csv
    import io
    import json
    from contextlib import redirect_stdout
    found, alone = pairs(a.folder)
    if not found:
        print(f'STOP: {a.folder} me koi NAME_lineart + NAME_ref (ya _colored) jodi nahi mili.')
        return 1
    for f in alone:
        print(f'[skip] {f}: iska reference nahi mila (NAME_ref.png ya NAME_colored.png rakho)')
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for name, line, ref in found:
        out = os.path.join(a.out, safe_name(name))
        args = ['fill', '--method', a.method, '--line', line, '--ref', ref, '--out', out, '--name', name,
                '--size', str(a.size), '--dpi', str(a.dpi), '--max-colors', str(a.max_colors)]
        print(f'[batch] {name} ...', flush=True)
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                code = main(args)
        except Exception as e:                      # one broken file never stops the batch
            code, buf = 2, io.StringIO(f'STOP: {type(e).__name__}: {e}')
        text = buf.getvalue()
        with open(os.path.join(out if os.path.isdir(out) else a.out, f'{safe_name(name)}_log.txt'), 'w') as f:
            f.write(text)
        rep = os.path.join(out, f'{safe_name(name)}_report.json')
        r = json.load(open(rep)) if code != 2 and os.path.exists(rep) else {}
        stop = next((l for l in text.splitlines() if l.startswith('STOP')), '')
        rows.append({'design': name, 'result': 'OK' if code == 0 else ('STOP' if stop else 'CHECK'),
                     'method': r.get('method', ''), 'alignment': r.get('alignment_score', ''),
                     'colours': r.get('colors_in_final', ''),
                     'verify': ('PASS' if r['verify']['passed'] else 'FAIL') if r.get('verify') else '',
                     'note': stop[6:] if stop else ''})
        print(f"[batch] {name}: {rows[-1]['result']}" + (f' - {rows[-1]["note"]}' if stop else ''))
    with open(os.path.join(a.out, 'batch_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('\nDesign | Result | Method | Align | Rang | Verify')
    for r in rows:
        print(f"{r['design']} | {r['result']} | {r['method']} | {r['alignment']} | {r['colours']} | {r['verify']}")
    ok = sum(r['result'] == 'OK' for r in rows)
    print(f'\n{ok}/{len(rows)} theek. Summary: {os.path.join(a.out, "batch_summary.csv")}')
    return 0 if ok == len(rows) else 1


def cmd_verify(a):
    name = safe_name(a.name) if a.name else None
    tifs = glob.glob(os.path.join(a.folder, f"{name or '*'}_final_*dpi.tif"))
    if len(tifs) != 1:
        print(f"[ERROR] {a.folder} me ek hi *_final_*.tif chahiye, mile {len(tifs)}. --name do.")
        return 2
    tif = os.path.basename(tifs[0])
    name = tif.split('_final_')[0]
    tag = tif.split('_final_')[1][:-4]                 # e.g. 3535px_300dpi
    dpi = int(tag.rsplit('_', 1)[1][:-3])
    p = ex.paths(a.folder, name, 1, 1, dpi)            # name/report paths; files by tag below
    for k, suffix in (('png', f'_final_{tag}.png'), ('tif', f'_final_{tag}.tif'),
                      ('channels_zip', f'_colored_channels_{tag}.zip'), ('bw_zip', f'_bw_separations_{tag}.zip')):
        p[k] = os.path.join(a.folder, name + suffix)
    missing = [os.path.basename(p[k]) for k in ('png', 'tif', 'channels_zip', 'bw_zip') if not os.path.exists(p[k])]
    if missing:
        print('[ERROR] ye files nahi mili: ' + ', '.join(missing))
        return 2
    from PIL import Image
    with Image.open(p['png']) as im:
        size = list(im.size)
    v = verify_package(p, size, dpi)
    cnt = []
    import zipfile
    with zipfile.ZipFile(p['channels_zip']) as z:
        for i, n in enumerate(sorted(z.namelist()), 1):
            with Image.open(z.open(n)) as im:
                share = (np.asarray(im.convert('RGBA'))[..., 3] > 0).mean() * 100
            parts = n[:-4].split('_')
            cnt.append({'channel': i, 'hex': parts[-1], 'coverage_percent': round(share, 2),
                        **({'name': parts[2], 'role': parts[3]} if len(parts) == 5 else {})})
    report = {'size_px': size, 'dpi': dpi, 'channels': cnt, 'verify': v}
    print(summary_hinglish(report))
    return 0 if v['passed'] else 1


def cmd_palette(a):
    rgb = load_rgb(a.image)
    pal, is_flat = pl.extract_palette(rgb, a.max_colors, a.min_share, log=_log)
    index = pl.map_to_palette(rgb, pal)
    cnt = pl.coverage(index, len(pal))
    h, w = index.shape
    print(f"Image: {w} x {h} px ({'flat' if is_flat else 'flat nahi, k-means'})")
    for k in np.argsort(-cnt):
        share = cnt[k] / index.size * 100
        tag = '  <- chhota (stray), merge hoga' if cnt[k] / index.size < a.min_share else ''
        print(f"  {pl._hex(pal[k])}  {share:.2f}%{tag}")
    _, pal2, merged = pl.merge_stray(index, pal, a.min_share)
    print(f"Merge ke baad: {len(pal2)} channels" + (f" ({len(merged)} chhote rang mile)" if merged else ''))
    print(f"{a.size} px par: {inches(a.size, a.dpi)} inch @ {a.dpi} DPI")
    return 0


def _log(msg):
    print(msg, flush=True)


def cmd_auto(a):
    """AUTO: FLAT solids + crisp LINE/dots + SPRAY only where source has shading."""
    from . import auto as am
    name = a.name
    if not name:
        name = os.path.splitext(os.path.basename(a.image))[0].replace('#', '')
    r = am.run_auto(
        a.image, a.out, name=name, colors=a.colors, dpi=a.dpi,
        size=getattr(a, 'size', None),
        allow_downscale=bool(getattr(a, 'allow_downscale', False)),
        edge_scale=float(getattr(a, 'edge_scale', 2.0)),
        write_psd=not getattr(a, 'no_psd', False),
        include_composite=bool(getattr(a, 'psd_composite', False)),
        use_layers=bool(getattr(a, 'psd_layers', False)),
        halftone=getattr(a, 'halftone', 'fm'),
        log=_log,
    )
    de = (r.get('delta_e_cie76') or {})
    frac = r.get('region_fractions') or {}
    print(f"[auto] mode={r.get('auto_mode')} inks={r.get('n_inks')} "
          f"flat={frac.get('flat')}% line={frac.get('line')}% spray={frac.get('spray')}% "
          f"ΔE={de.get('overall_mean')} ground_std={r.get('ground_std')}")
    hi = r.get('hires') or {}
    print(f"[auto] output {r.get('size_px')} @ {r.get('dpi')} DPI "
          f"(base_dpi={hi.get('base_dpi')} ×{hi.get('hires_scale')} supersample)")
    print(f"[auto] composite={r.get('composite')}")
    if r.get('package'):
        print(f"[auto] package psd={r['package'].get('psd')}")
    return 0 if r.get('applied') else 1


def cmd_spray(a):
    """Pure tonal spray: continuous ink tones, FM films, package TIF/PSD."""
    from . import spray as sp
    name = a.name
    if not name:
        name = os.path.splitext(os.path.basename(a.image))[0].replace('#', '')
    r = sp.run_spray(
        a.image, a.out, name=name, colors=a.colors, dpi=a.dpi,
        lpi=getattr(a, 'halftone_lpi', 45),
        halftone=a.halftone,
        write_psd=not getattr(a, 'no_psd', False),
        include_composite=bool(getattr(a, 'psd_composite', False)),
        use_layers=bool(getattr(a, 'psd_layers', False)),
        log=_log,
    )
    de = (r.get('delta_e_cie76') or {}).get('overall_mean')
    print(f"[spray] inks={r.get('n_inks')} ΔE={de} composite={r.get('composite')}")
    if r.get('package'):
        print(f"[spray] package={r['package'].get('package_dir')} psd={r['package'].get('psd')}")
    return 0 if r.get('applied') else 1



def cmd_reconstruct(a):
    """Patch-work redrawn with a defined boundary; spray uses the spray solver."""
    from . import reconstruct as rc
    name = a.name
    if not name:
        name = os.path.splitext(os.path.basename(a.image))[0].replace('#', '')
    r = rc.run_reconstruct(
        a.image, a.out, name=name, colors=a.colors, scale=a.scale, dpi=a.dpi,
        min_feature_mm=a.min_feature_mm, ai_upscale=bool(a.ai_upscale),
        views=True, compare_auto=bool(getattr(a, 'compare_auto', False)), log=_log,
    )
    de = (r.get('delta_e_cie76') or {})
    counts = r.get('counts') or {}
    print(f"[reconstruct] {r.get('size_px')} @ {r.get('dpi')} DPI  "
          f"boost={((r.get('quality_boost') or {}).get('method'))}  "
          f"inks={r.get('n_inks')} spray={r.get('spray_percent')}%  "
          f"dE={de.get('overall_mean')} flat_std={r.get('flat_std')}")
    print(f"[reconstruct] dots {counts.get('dots_source')}->{counts.get('dots_output')}  "
          f"lines {counts.get('lines_source')}->{counts.get('lines_output')}")
    if r.get('package'):
        print(f"[reconstruct] psd={r['package'].get('psd')}")
    print(f"[reconstruct] report={r.get('report')}")
    return 0 if r.get('applied') else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog='textile', description='Mill ke designs: export, verify, palette')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, out=True):
        p.add_argument('--size', type=int, default=None,
                       help=f'Output width px (default {SIZE} = 11.78 inch at {DPI} DPI; same inches at another --dpi)')
        p.add_argument('--dpi', type=int, default=DPI, help=f'Default {DPI} (mill format). 600 sirf maangne par.')
        if out:
            p.add_argument('--out', required=True, help='Output folder')
            p.add_argument('--name', default=None, help="File prefix (default: image ka naam; '#' hata diya jaata hai)")
        p.add_argument('--max-colors', type=int, default=16, help='Non-flat image ke liye k-means colors')
        p.add_argument('--min-share', type=float, default=pl.STRAY_SHARE,
                       help='Isse kam share wala rang stray (default 0.0005 = 0.05%%)')

    e = sub.add_parser('export', help='Flat design -> TIF/PNG, channels, B/W seps, preview, report')
    e.add_argument('image')
    common(e)
    e.add_argument('--keep-strays', action='store_true', help='Chhote rang merge mat karo')
    e.add_argument('--line-color', default=None, help="Outline ka rang (hex, jaise 120F06): uska channel 'outline' kehlata hai")
    e.set_defaults(fn=cmd_export)

    f = sub.add_parser('fill', help='Line art + reference -> flat colour design (Method 1 ya 2)')
    f.add_argument('--method', choices=['1', '2', '3', '4', 'auto'], default='1',
                   help='1: images aligned (default). 2: alag AI generations, sirf 2 rang (ground + motif). '
                        '3: wahi design par reference thoda khiska/khincha, kitne bhi rang. '
                        '4: Method 1 + line art ke tootne wale gap band (AI line art ke liye). '
                        'auto: Method 1 chala kar jaancho, fail ho to 2 ya 3')
    f.add_argument('--line', required=True, help='Black & white line art')
    f.add_argument('--ref', required=True, help='Colored reference (same design, same alignment)')
    common(f)
    f.add_argument('--line-threshold', type=int, default=150, help='Gray < ye = line (0-255)')
    f.add_argument('--line-color', default='auto', help="'auto' ya hex jaise EFCE6A")
    f.add_argument('--min-align', type=float, default=0.55, help='Isse kam alignment score par STOP')
    f.add_argument('--force', action='store_true', help='Alignment warning ke bawajood chalao')
    f.add_argument('--keep-strays', action='store_true', help='Chhote rang merge mat karo (method1 jaisa)')
    f.add_argument('--max-cover-diff', type=float, default=6.0,
                   help='auto: rangon ka hissa reference se itne points se zyada alag = Method 1 fail (default 6)')
    f.add_argument('--seal', type=int, default=None,
                   help='Method 4: gap band karne ka radius (output px). Default: line art ke 1.5 px, output ke hisaab se')
    f.add_argument('--reach-align', type=int, default=96,
                   help='Method 3: reference ko kitne px tak khiska sakte hain (3535 px par, default 96)')
    m = f.add_argument_group('Method 2 (prototype ke default)')
    m.add_argument('--colors', default=None, help='ground,motif hex (default: reference se, sabse common = ground)')
    m.add_argument('--reach', type=int, default=14, help='Line ke aar-paar kitne px tak dekhna (default 14)')
    m.add_argument('--dark-level', type=int, default=110, help='Reference me isse gehra = dark (default 110)')
    m.add_argument('--seed-area', type=int, default=6000, help='Bada band hissa: area > ye (default 6000)')
    m.add_argument('--seed-dark', type=float, default=0.8, help='...aur reference me dark > ye (default 0.8)')
    m.add_argument('--side-band', type=float, default=0.27, help='Side panel: chaudai ka ye hissa (default 0.27)')
    m.add_argument('--side-area', type=int, default=9000, help='Side panel me area > ye (default 9000)')
    m.add_argument('--side-dark', type=float, default=0.55, help='Side panel me dark > ye (default 0.55)')
    f.set_defaults(fn=cmd_fill)

    v = sub.add_parser('verify', help='Ek output folder ki files dobara check karo')
    v.add_argument('folder')
    v.add_argument('--name', default=None)
    v.set_defaults(fn=cmd_verify)

    r = sub.add_parser('repeat', help='All-over design ka repeat type, size, drop, jhukav (JSON)')
    r.add_argument('image')
    r.add_argument('--out', default=None, help='JSON yahan save karo (optional)')
    r.add_argument('--name', default=None)
    r.set_defaults(fn=cmd_repeat)

    t = sub.add_parser('tile', help='All-over design -> ek seamless repeat block (pehle repeat analysis)')
    t.add_argument('image')
    common(t)
    t.add_argument('--colors', type=int, default=None, help='Tile ko itne flat rangon me karo + channels (optional)')
    t.add_argument('--clean-ground', type=float, default=0, help='Ground ke itne paas wale pixels pakka ground (RGB doori, jaise 30)')
    t.add_argument('--shear-ratio', type=float, default=None, help='Jhukav khud do (dx/dy), default analysis se')
    t.add_argument('--w-range', default=None, help="Block chaudai search, jaise 496:514 (default analysis +-8)")
    t.add_argument('--h-range', default=None, help="Block lambai search, jaise 308:326 (default analysis +-8)")
    t.set_defaults(fn=cmd_tile)

    k = sub.add_parser('make', help='Original design (repeat ya panel) ek JSON config se, flat rang')
    k.add_argument('config', help='JSON config (examples/ dekho)')
    k.add_argument('--out', required=True)
    k.add_argument('--name', default=None)
    k.set_defaults(fn=cmd_make)

    pcx = sub.add_parser('partscheck', help='AI ke dobara banaye parts ko reference se milao: khisak, naye rang, detail kam/zyada')
    pcx.add_argument('reference')
    pcx.add_argument('parts', nargs='+', help='folder ya part files (naam r1c1 ... r3c3)')
    pcx.add_argument('--plan', default=None, help='split ka *_parts_plan.json (sabse sahi)')
    pcx.add_argument('--grid', default='3x3')
    pcx.add_argument('--overlap', type=float, default=15.0)
    pcx.set_defaults(fn=cmd_partscheck)

    fin = sub.add_parser('final', help='Design ko mill ke exact repeat size (inch x DPI) par: flat rang, smooth kinare, channels, TIF')
    fin.add_argument('design')
    fin.add_argument('--out', required=True)
    fin.add_argument('--inches', required=True, help='repeat ka size WxH inch, jaise 23.5x20.7')
    fin.add_argument('--dpi', type=int, default=DPI)
    fin.add_argument('--name', default=None)
    fin.add_argument('--colors', type=int, default=8)
    fin.add_argument('--detail', default='normal', choices=['kam', 'normal', 'zyada'])
    fin.add_argument('--merge-similar', action='store_true')
    fin.add_argument('--small-px', type=float, default=20.0, dest='small_px', help='Chhote objects (tasveer me is se kam px chaude, ya patli tehni/line) tasveer ke kinaron ke mix rang se banao: daana gol aur sahi size, star ki nok, tehni ek line; bade objects pehle jaise. 0 = band (default 20)')
    fin.add_argument('--no-detail-rescue', action='store_true', dest='no_detail_rescue', help='Chhote details (daane, phool ke beech, rare rang, gehre hisse me grey daane) bachane wala step band karo: purana vyavhar (default: chalu)')
    fin.add_argument('--merge-shades', action=argparse.BooleanOptionalAction, default=True,
                     help='Ek hi rang ke gehre/halke shade (AI image ki shading: patti ki jad me gehra laal) ek screen me (default ON): dE2000 < 10 aur chhota rang apne kinare ka 30%%+ bade rang ko chhoota ho; patli line wala rang nahi milta. Band: --no-merge-shades')
    fin.add_argument('--min-mm2', type=float, default=None, help='--detail ki jagah: isse chhote hisse (sq mm) paas ke bade hisse me mila do (jaise 3)')
    fin.add_argument('--no-psd', action='store_true', help='Photoshop (.psd, har rang ek layer) mat banao')
    fin.add_argument('--ink-names', default=None, help='JSON file: {"D96A8E": "Rani Pink 12"} (ya LoomLab ka inks.json): us HEX ki plate ka naam mill ka apna (channel file, PSD layer, report); sirf bilkul wahi HEX')
    fin.set_defaults(fn=cmd_final)

    cr_ = sub.add_parser('crisp', help='Flat design ko smooth curves me dobara draw: SVG (kitna bhi zoom saaf) + bada flat PNG')
    cr_.add_argument('image')
    cr_.add_argument('--out', required=True)
    cr_.add_argument('--name', default=None)
    cr_.add_argument('--scale', type=float, default=2.0, help='source ke ek pixel me kitne design pixel (1254 px -> 3535 = 2.8)')
    cr_.add_argument('--zoom', type=int, default=2, help='PNG kitne guna bada (default 2)')
    cr_.add_argument('--smoothing', type=float, default=70.0, help='0-100, default 70')
    cr_.set_defaults(fn=cmd_crisp)

    vt = sub.add_parser('vector', help='Flat design ke kinare seedhe (trace -> bina AA dobara) + SVG')
    vt.add_argument('image', help='Flat design (jaise *_final_*.png)')
    vt.add_argument('--out', required=True)
    vt.add_argument('--name', default=None)
    vt.add_argument('--eps', type=float, default=0.8, help='Kinara pixels se kitna hil sakta hai (px, default 0.8)')
    vt.add_argument('--repeat', action='store_true', help='Repeat tile hai: kinare wrap karke trace (jod saaf rahe)')
    vt.set_defaults(fn=cmd_vector)

    ed_ = sub.add_parser('edges', help='Kinare saaf: har rang ki outline smooth, kone seedhe, patli line/dot salamat')
    ed_.add_argument('image', help='Flat design (jaise *_final_*.png, ya Reduce/fill ka result)')
    ed_.add_argument('--strength', type=int, choices=[1, 2, 3], default=2, help='1 halka, 2 normal (default), 3 zyada')
    ed_.add_argument('--specks', type=int, default=0, help='Itne pixel tak ke chhote tukde padosi rang me (default 0 = nahi; asli chhote dot bhi ja sakte hain)')
    ed_.add_argument('--repeat', choices=['no', 'x', 'y', 'both'], default='no',
                     help='Repeat tile hai: kinare jod ke aar-paar bhi saaf (x = left-right, y = upar-neeche)')
    common(ed_)
    ed_.set_defaults(fn=cmd_edges)

    pa = sub.add_parser('paint', help='Sketch (line art) + aapke bataye rang -> design, har rang alag channel')
    pa.add_argument('sketch', help='Line art / sketch (safed par kaali lines)')
    common(pa)
    pa.add_argument('--colors', default=None, action='append',
                    help='Rang: "A=cream, B=laal, C D=hara, 12 40-45=gold, lines=coffee, rest=navy" (ya us text ki '
                         'file, ya CSV: Number + HEX columns). Kai baar de sakte ho: baad wala jeetta hai (jaise CSV, phir '
                         '"171 337=cream" sudhaar). Na do to sirf map banta hai (letters + numbers)')
    pa.add_argument('--ref', default=None,
                    help='Usi design ka rangeen version: har hisse ka rang usse padha jaata hai (CSV apne aap), '
                         'aur rangeen design par numbers. --colors se upar se sudhaar')
    pa.add_argument('--ref-colors', type=int, default=8,
                    help='--ref se zyada se zyada itne rang (default 8; milte-julte shade apne aap ek)')
    pa.add_argument('--line-threshold', type=int, default=150, help='Gray < ye = line (0-255)')
    pa.add_argument('--seal', type=int, default=None,
                    help='Line ke chhote gap band karne ka radius (output px). Default: sketch ke 1.5 px; 0 = band nahi')
    pa.add_argument('--group-tolerance', type=float, default=1.0,
                    help='Ek jaise hisse pehchanne ki dheel (1 default; 1.5 = zyada hisse ek group me, 0.5 = kam)')
    pa.add_argument('--ink-names', default=None, help='JSON file: {"D96A8E": "Rani Pink 12"} (ya LoomLab ka inks.json): us HEX ki plate ka naam mill ka apna (channel file, PSD layer, report); sirf bilkul wahi HEX')
    pa.set_defaults(fn=cmd_paint)

    nu = sub.add_parser('number', help='Rangeen design ke har rang ke hisse ko number do (sketch banwane ke liye map)')
    nu.add_argument('design', help='Rangeen design (png/jpg)')
    nu.add_argument('--out', required=True)
    nu.add_argument('--name', default=None)
    nu.add_argument('--size', type=int, default=None, help='Kis chaudai par ginna (default 3535)')
    nu.add_argument('--dpi', type=int, default=DPI, help=argparse.SUPPRESS)
    nu.add_argument('--colors', type=int, default=8, help='Zyada se zyada itne rang (default 8; milte-julte shade ek)')
    nu.add_argument('--detail', choices=['kam', 'normal', 'zyada'], default='normal',
                    help='kam: 1.5 sq mm se chhote hisse paas me mila do (saaf, kam numbers); normal 0.4; zyada 0.1')
    nu.add_argument('--min-area', type=int, default=None, help='--detail ki jagah: isse chhote hisse (px) mila do')
    nu.add_argument('--line-mm', default='auto',
                    help='Sketch ki line kitni moti (mm, jaise 0.35). Default auto: 0.17, 0.35, 0.5 teeno banakar '
                         'jo design sabse sahi wapas banaye wahi rakhta hai')
    nu.add_argument('--bold-mm', type=float, default=0.5,
                    help='NAME_sketch_bold.png / .svg ki line kitni moti (mm, default 0.5): smooth curve + gehri, saaf')
    nu.add_argument('--bold-scale', type=int, default=1,
                    help='Bold sketch kitne guna bade canvas par (default 1 = 3535 px @ 300 DPI, mill jaisa; '
                         '2 = 7070 px @ 600 DPI, wahi 11.78 inch, sirf dekhne ke liye): curves seedhe bade canvas par '
                         'draw hoti hain, image bada nahi hoti. 4 = 14140 px')
    nu.add_argument('--circle', type=float, default=0.9,
                    help='Bold sketch me jo outline itne (0.9 = 90%%) gol ho wo pakka circle bane (0 = band)')
    nu.add_argument('--polygons', type=float, default=0.9,
                    help='Bold sketch me jo outline itni (0.9 = 90%%) triangle / rectangle / diamond jaisi seedhi bhujaon wali '
                         'ho wo saaf shape bane; curve wali shape kabhi seedhi nahi hoti (0 = band)')
    nu.add_argument('--motifs', type=float, default=0.9,
                    help='Bold sketch me patti / petal (do nok wali, gol bhujaen) aur oval jab itne (0.9 = 90%%) saaf ho to '
                         'pakki saaf shape (patti ki dono bhujaen barabar); tedhi / kati shape wahi rehti hai (0 = band)')
    nu.add_argument('--no-package', action='store_true',
                    help='Rang channels / mill TIF mat banao (default: sketch + CSV se `paint` package bhi banta hai, package/ me)')
    nu.add_argument('--no-psd', action='store_true', help='Photoshop (.psd, har rang ek layer) mat banao')
    nu.add_argument('--merge-similar', action='store_true',
                    help='Chhota rang (< 1.5%%) jo kisi bade rang ke bahut paas ho (dE < 12) us me mila do: ek screen kam')
    nu.add_argument('--small-px', type=float, default=20.0, dest='small_px', help='Chhote objects (tasveer me is se kam px chaude, ya patli tehni/line) tasveer ke kinaron ke mix rang se banao: daana gol aur sahi size, star ki nok, tehni ek line; bade objects pehle jaise. 0 = band (default 20)')
    nu.add_argument('--no-detail-rescue', action='store_true', dest='no_detail_rescue', help='Chhote details (daane, phool ke beech, rare rang, gehre hisse me grey daane) bachane wala step band karo: purana vyavhar (default: chalu)')
    nu.add_argument('--merge-shades', action=argparse.BooleanOptionalAction, default=True,
                    help='Ek hi rang ke gehre/halke shade (AI image ki shading: patti ki jad me gehra laal) ek screen me (default ON): dE2000 < 10 aur chhota rang apne kinare ka 30%%+ bade rang ko chhoota ho; patli line wala rang nahi milta. Band: --no-merge-shades')
    nu.add_argument('--smoothing', type=float, default=70.0,
                    help='Bold sketch ki curves ko Photoshop ke brush "Smoothing" jaisa smooth karo, 0-100 (default 70): '
                         'jitna zyada, utni aade-tedhe jhatke ghulte hain aur curve ek lagatar slope me behti hai; 0 = band. '
                         'Seedhe kone aur gol shape nahi bigadte')
    nu.add_argument('--fair', type=float, default=None, help=argparse.SUPPRESS)
    nu.add_argument('--run-id', default=None, help=argparse.SUPPRESS)       # LoomLab's job id, for its feedback
    nu.add_argument('--no-smooth', action='store_true', help='Lines smooth mat karo (pixel jaisi)')
    nu.add_argument('--hd', action='store_true',
                    help='Double resolution (7070 px @ 300 DPI). File ke naam me _hd likhne se bhi (jaise rose_hd.png)')
    nu.add_argument('--no-separators', action='store_true',
                    help='Rang-rang ke beech ki line bhi kaali (default grey: print me paas ka rang)')
    # Optional prepress (Hybrid Lab features); all OFF by default — see textile.prepress
    nu.add_argument('--underbase', action='store_true',
                    help='Selective white underbase plate (sirf un inks ke neeche jo dark fabric par chahiye; default OFF)')
    nu.add_argument('--underbase-choke', type=int, default=1, dest='underbase_choke',
                    help='Underbase choke in px (default 1; --underbase ke saath)')
    nu.add_argument('--trap', type=int, default=None, metavar='PX',
                    help='Trapping/choke between neighbouring plates, px (jaise 1; default OFF)')
    nu.add_argument('--film', action='store_true',
                    help='Film positives (black-on-white) with registration marks (default OFF)')
    nu.add_argument('--estimate', action='store_true',
                    help='Mesh count + ink-weight estimate CSV/TXT (default OFF)')
    nu.add_argument('--seam-check', action='store_true', dest='seam_check',
                    help='Repeat seam mismatch score + 2x2 tiled preview (default OFF)')
    nu.add_argument('--gradients', action='store_true',
                    help='Hybrid tonal: FLAT solids + SHADED overprint films (default OFF)')
    nu.add_argument('--hybrid', action='store_true',
                    help='Same as --gradients (FLAT solid ink + SHADED tonal overprint)')
    nu.add_argument('--psd-layers', action='store_true', dest='psd_layers',
                    help='Legacy per-ink LAYERED PSD (default: spot CHANNELS only)')
    nu.add_argument('--psd-composite', action='store_true', dest='psd_composite',
                    help='Add COMPOSITE flattened layer to tone PSD (default OFF)')
    nu.add_argument('--halftone-lpi', type=int, default=45, dest='halftone_lpi',
                    help='Halftone lines per inch for --gradients (default 45)')
    nu.add_argument('--halftone-angle', type=float, default=45.0, dest='halftone_angle',
                    help='Halftone screen angle in degrees for --gradients (default 45)')
    nu.set_defaults(fn=cmd_number)

    fb = sub.add_parser('feedback', help='Kisi run par apni raay: good / bad (+ note), taaki tool seekhta rahe')
    fb.add_argument('name', help='design ka naam (jo number ne chhapa)')
    fb.add_argument('verdict', choices=['good', 'bad'])
    fb.add_argument('note', nargs='*', help='kya achha/kharab tha (optional)')
    fb.set_defaults(fn=cmd_feedback)

    ln = sub.add_parser('learn', help='Ab tak ke runs, aam dikkatein, achhe/kharab runs ka saar')
    ln.set_defaults(fn=lambda a: print(__import__('textile.learn', fromlist=['x']).summary()) or 0)

    sti = sub.add_parser('stitch', help='Ek design ke parts (folder / zip / files) jod kar ek design; --number = poora number run bhi')
    sti.add_argument('parts', nargs='+', help='folder, zip ya part images (naam r1c1 / piece_1_2 / 1..9)')
    sti.add_argument('--out', required=True)
    sti.add_argument('--name', default=None)
    sti.add_argument('--grid', default=None, help='ROWSxCOLS, jaise 3x3 (naam se ya ginti se khud)')
    sti.add_argument('--number', action='store_true', help='jodne ke baad number chalao (sketch, numbers, plates, PSD)')
    sti.add_argument('--number-args', nargs=argparse.REMAINDER, default=[],
                     help='number ke liye aur options, sabse aakhir me: --number-args --bold-mm 0.175 --merge-similar')
    sti.set_defaults(fn=cmd_stitch)

    spl = sub.add_parser('split', help='Design ko parts me kaato (thoda overlap ke saath), har part alag se bada karne ke liye')
    spl.add_argument('image')
    spl.add_argument('--out', required=True)
    spl.add_argument('--name', default=None)
    spl.add_argument('--grid', default='3x3')
    spl.add_argument('--overlap', type=float, default=15.0, help='padosi part se kitna % share (default 15)')
    spl.add_argument('--target', type=int, default=None, help='final design kitne px chaudai ka chahiye: batata hai AI se har part kitna maangna hai')
    spl.set_defaults(fn=cmd_split)

    b = sub.add_parser('batch', help='Folder ke saare NAME_lineart + NAME_ref jode ek saath (fill)')
    b.add_argument('folder')
    b.add_argument('--out', required=True)
    b.add_argument('--method', choices=['1', '2', '3', '4', 'auto'], default='auto', help='Default auto')
    b.add_argument('--size', type=int, default=None)
    b.add_argument('--dpi', type=int, default=DPI)
    b.add_argument('--max-colors', type=int, default=16)
    b.set_defaults(fn=cmd_batch)

    p = sub.add_parser('palette', help='Image ke rang aur coverage dikhao')
    p.add_argument('image')
    common(p, out=False)
    p.set_defaults(fn=cmd_palette)


    au = sub.add_parser('auto', help='AUTO: FLAT solids + crisp LINE/dots + SPRAY tone only where shaded')
    au.add_argument('image', help='Source design image')
    au.add_argument('--out', required=True, help='Output folder')
    au.add_argument('--name', default=None, help='File prefix')
    au.add_argument('--colors', type=int, default=8, help='Max spot inks / flat colours (default 8)')
    au.add_argument('--dpi', type=int, default=DPI, help=f'DPI (default {DPI})')
    au.add_argument('--size', type=int, default=None,
                    help='Working width px (default: FULL native width; never downscales)')
    au.add_argument('--allow-downscale', action='store_true',
                    help='Opt-in: allow --size smaller than the source (destroys clarity; off by default)')
    au.add_argument('--edge-scale', type=float, default=2.0,
                    help='Supersample factor for pen-tool edge/film plates (1–4, default 2; kept hi-res)')
    au.add_argument('--halftone', choices=['fm', 'am'], default='fm',
                    help='Shade film type when spray regions exist (default fm)')
    au.add_argument('--no-psd', action='store_true', help='Skip PSD layers file')
    au.add_argument('--psd-layers', action='store_true', dest='psd_layers',
                    help='Legacy per-ink LAYERED PSD (default: spot CHANNELS only)')
    au.add_argument('--psd-composite', action='store_true', dest='psd_composite',
                    help='Add COMPOSITE flattened layer (default OFF)')
    au.set_defaults(fn=cmd_auto)

    sp = sub.add_parser('spray', help='Pure tonal spray: continuous ink tones + FM films + package (no lines/numbering)')
    sp.add_argument('image', help='Source image (soft airbrush / blotchy tones)')
    sp.add_argument('--out', required=True, help='Output folder')
    sp.add_argument('--name', default=None, help='File prefix')
    sp.add_argument('--colors', type=int, default=8, help='Max spot inks (default 8)')
    sp.add_argument('--dpi', type=int, default=DPI, help=f'DPI (default {DPI})')
    sp.add_argument('--halftone', choices=['fm', 'am'], default='fm',
                    help='Film type: fm/stochastic (default) or am')
    sp.add_argument('--halftone-lpi', type=int, default=45, dest='halftone_lpi',
                    help='LPI when --halftone am (default 45)')
    sp.add_argument('--no-psd', action='store_true', help='Skip PSD layers file')
    sp.add_argument('--psd-layers', action='store_true', dest='psd_layers',
                    help='Legacy per-ink LAYERED PSD (default: spot CHANNELS only)')
    sp.add_argument('--psd-composite', action='store_true', dest='psd_composite',
                    help='Add COMPOSITE flattened layer (default OFF)')
    sp.set_defaults(fn=cmd_spray)

    rc_ = sub.add_parser(
        'reconstruct',
        help='Patch-work ko defined boundary se dobara draw; spray/gradient wahi spray solver',
    )
    rc_.add_argument('image', help='Source design')
    rc_.add_argument('--out', required=True, help='Output folder')
    rc_.add_argument('--name', default=None, help='File prefix')
    rc_.add_argument('--colors', type=int, default=8, help='Max inks (default 8)')
    rc_.add_argument('--scale', type=float, default=2.0, help='Output scale (default 2, never below 1)')
    rc_.add_argument('--dpi', type=int, default=DPI, help=f'DPI (default {DPI})')
    rc_.add_argument('--min-feature-mm', type=float, default=0.15, dest='min_feature_mm',
                     help='Thinner than this (mm at output DPI) is thickened, never deleted (default 0.15)')
    rc_.add_argument('--ai-upscale', action='store_true', dest='ai_upscale',
                     help='Real-ESRGAN before the trace, only if installed (default off)')
    rc_.add_argument('--compare-auto', action='store_true', dest='compare_auto',
                     help='Zoom sheets also run the existing auto pipeline on a small crop')
    rc_.set_defaults(fn=cmd_reconstruct)

    a = ap.parse_args(argv)
    # AUTO keeps size=None → native image width; other commands default to mill sheet width
    if hasattr(a, 'size') and a.size is None and getattr(a, 'cmd', None) not in ('auto', 'reconstruct'):
        a.size = round(SIZE * a.dpi / DPI)          # 3535 at 300 DPI; the same 11.78 inch at any other
    if getattr(a, 'dpi', DPI) != DPI:
        print(DPI_WARNING.format(dpi=a.dpi))
    code = a.fn(a)
    if getattr(a, 'dpi', DPI) != DPI:
        print(DPI_WARNING.format(dpi=a.dpi))
    return code


DPI_WARNING = ('[CHETAVNI] {dpi} DPI file. Mill ka format 300 DPI hai: unka software ise 300 maan le to design '
               'galat size (600 par double) chhapega. Ye file mill ko tabhi bhejo jab mill ne {dpi} DPI khud maanga ho.')


if __name__ == '__main__':
    sys.exit(main())
