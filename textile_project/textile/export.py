"""Everything a finished design is sent out as, from one index map + palette:

  NAME_final_<S>px_<DPI>dpi.tif   the mill's file (LZW, DPI written in)
  NAME_final_<S>px_<DPI>dpi.png   to look at and share
  NAME_colored_channels_*.zip     one RGBA layer per colour, transparent around
  NAME_bw_separations_*.zip       1-bit black & white separations
  NAME_channels_preview.png       every channel on one sheet
  NAME_report.json                sizes, colours, coverage, checks

This is method1_colorfill.py's output step, moved here unchanged, so every
command writes the same files the same way. Channels are numbered by
coverage, biggest first, and named channel_01_olive_ground_4A5B24.png:
colour name, role (ground / outline / motif, see names.py), hex; no '#'.
"""
from __future__ import annotations

import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from .io_utils import hex_of, inches, ordered_map, png_bytes, safe_name, save_png, save_tif, to_image
from .names import colour_name, roles as colour_roles, ink_token


def size_tag(w, h):
    """'3535px' for a square design (as method1 names it), else '3535x1768px'."""
    return f'{w}px' if w == h else f'{w}x{h}px'


def paths(out_dir, name, w, h, dpi):
    tag = f'{size_tag(w, h)}_{dpi}dpi'
    j = lambda f: os.path.join(out_dir, f)
    return {'png': j(f'{name}_final_{tag}.png'), 'tif': j(f'{name}_final_{tag}.tif'),
            'channels_zip': j(f'{name}_colored_channels_{tag}.zip'),
            'bw_zip': j(f'{name}_bw_separations_{tag}.zip'),
            'preview': j(f'{name}_channels_preview.png'), 'report': j(f'{name}_report.json')}


def channel_file(i, rgb, role, ink=None):
    """channel_01_olive_ground_4A5B24.png; with the mill's own name for the ink, that name in the colour word's
    place (channel_01_Rani-Pink-12_motif_D96A8E.png): still 5 parts, the hex last, no '#'."""
    return f'channel_{i:02d}_{(ink_token(ink) if ink else "") or colour_name(rgb)}_{role}_{hex_of(rgb)}.png'


def export_package(index, pal, out_dir, name, dpi=300, line_index=None, ink_names=None):
    """Write every output for the design `pal[index]`. Returns
    {'paths', 'channels', 'size_px', 'dpi'}. `line_index` is the palette entry
    of the line art's colour (its channel is named 'outline'). `ink_names` ({'RRGGBB': name}, exact hex): the
    mill's own name for those plates, in the file names (so the PSD layers and preview too) and the report.
    Nothing is checked here (that is verify.py's job, on the files as written)."""
    ink_names = ink_names or {}
    name = safe_name(name)
    os.makedirs(out_dir, exist_ok=True)
    H, W = index.shape
    K = len(pal)
    p = paths(out_dir, name, W, H, dpi)

    flat = pal[index]
    with ThreadPoolExecutor(2) as files:                 # the two big files code while the channels do
        # one PIL image each: save() parks its settings on the image, two saves of one image at once would mix them
        png = files.submit(save_png, to_image(flat), p['png'], dpi)
        tif = files.submit(save_tif, to_image(flat), p['tif'], dpi)
        del flat

        cnt = np.bincount(index.ravel(), minlength=K)
        order = [k for k in np.argsort(-cnt) if cnt[k] > 0]
        role = colour_roles(cnt, line_index)
        tw = 500 if len(order) <= 30 else 200                 # many channels: smaller thumbs, or the sheet is huge
        thumb = (tw, tw) if W == H else (tw, max(1, round(tw * H / W)))
        # the thumbs' pixels: NEAREST picks the same source pixels whatever the image holds, so shrinking the ink
        # map once gives each channel's thumb exactly (not a full-size white sheet per channel)
        small = np.asarray(Image.fromarray(np.ascontiguousarray(index, np.int32)).resize(thumb, Image.NEAREST))

        def one(job):
            i, k = job
            r, g, b = (int(v) for v in pal[k])
            m = index == k
            rgba = m[..., None] * np.array([r, g, b, 255], np.uint8)
            fn = channel_file(i, pal[k], role[k], ink_names.get(hex_of(pal[k])))
            colour = png_bytes(to_image(rgba), dpi)
            del rgba
            bw = png_bytes(Image.fromarray(~m), dpi)       # 1-bit: ink black, the rest white
            th = np.full(small.shape + (3,), 255, np.uint8)
            th[small == k] = (r, g, b)
            return k, fn, colour, bw, Image.fromarray(th)

        thumbs, channels = [], []
        with zipfile.ZipFile(p['channels_zip'], 'w', zipfile.ZIP_DEFLATED) as zc, \
             zipfile.ZipFile(p['bw_zip'], 'w', zipfile.ZIP_DEFLATED) as zb:
            for i, (k, fn, colour, bw, th) in enumerate(ordered_map(one, enumerate(order, 1)), 1):
                hx = hex_of(pal[k])                     # file name me '#' nahi
                zc.writestr(fn, colour)
                zb.writestr(fn, bw)
                share = cnt[k] / index.size * 100
                channels.append({'channel': i, 'hex': hx, 'name': colour_name(pal[k]), 'role': role[k],
                                 'coverage_percent': round(share, 2)})
                if hx in ink_names:
                    channels[-1]['ink'] = ink_names[hx]
                label = f'{fn}  {share:.2f}%' if tw == 500 else f'{i:02d}  {hx}  {share:.2f}%'
                thumbs.append((th, label))
        png.result()
        tif.result()

    _preview_sheet(thumbs, thumb, p['preview'])
    return {'paths': p, 'channels': channels, 'size_px': [W, H], 'dpi': dpi,
            'print_size_inch': inches(W, dpi) if W == H else [inches(W, dpi), inches(H, dpi)]}


def _preview_sheet(thumbs, thumb, path):
    cols = 3 if len(thumbs) <= 30 else 8
    rows = (len(thumbs) + cols - 1) // cols
    tw, th = thumb
    sheet = Image.new('RGB', (cols * (tw + 40), rows * (th + 70)), 'white')
    d = ImageDraw.Draw(sheet)
    for j, (t, label) in enumerate(thumbs):
        x, y = (j % cols) * (tw + 40) + 20, (j // cols) * (th + 70) + 20
        sheet.paste(t, (x, y))
        d.rectangle([x, y, x + tw - 1, y + th - 1], outline='gray')
        d.text((x, y + th + 8), label, fill='black')
    sheet.save(path)


def compare_sheet(reference, final_rgb, path, label=('reference', 'output'), more=()):
    """Reference and result side by side (NEAREST, 1200 px wide each), to look
    at, never to print. `more`: further (rgb, label) panels, such as the
    method auto mode did not choose."""
    h, w = reference.shape[:2]
    panels = [(reference, label[0]), (final_rgb, label[1]), *more]
    sheet = Image.new('RGB', (w * len(panels) + 10 * (len(panels) + 1), h + 40), 'white')
    d = ImageDraw.Draw(sheet)
    for i, (rgb, text) in enumerate(panels):
        img = to_image(rgb)
        if i:
            img = img.resize((w, h), Image.NEAREST)
        sheet.paste(img, (10 + i * (w + 10), 30))
        d.text((10 + i * (w + 10), 8), text, fill='black')
    sheet.save(path)


def write_report(path, report):
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
