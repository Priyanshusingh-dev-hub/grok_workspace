"""Inject optional prepress/gradient plates into the mill package channels.

When --underbase / --trap / --gradients produced extras, merge them into
package/'s colored_channels zip (PSD + channels/ folder follow that zip).
Print order: underbase first, then spot plates (trapped if present), then tone.
With no extras this module is not called — package stays byte-identical.
"""
from __future__ import annotations

import glob
import io
import os
import re
import zipfile
from typing import Any

import numpy as np
from PIL import Image

from .io_utils import hex_of, png_bytes, save_png, save_tif, to_image


_CH_RE = re.compile(
    r'^channel_(\d+)_(.+)_([0-9A-Fa-f]{6})\.png$', re.I
)


def _read_zip_plates(channels_zip: str) -> list[dict[str, Any]]:
    plates = []
    with zipfile.ZipFile(channels_zip) as z:
        for m in sorted(n for n in z.namelist() if n.lower().endswith('.png')):
            data = z.read(m)
            im = Image.open(io.BytesIO(data)).convert('RGBA')
            base = os.path.basename(m)
            mo = _CH_RE.match(base)
            hx = mo.group(3).upper() if mo else '000000'
            role = 'motif'
            if mo:
                mid = mo.group(2)
                # mid is like "olive_ground" or "Rani-Pink-12_motif"
                if mid.endswith('_ground'):
                    role = 'ground'
                elif mid.endswith('_outline'):
                    role = 'outline'
                elif mid.endswith('_motif'):
                    role = 'motif'
                else:
                    parts = mid.rsplit('_', 1)
                    role = parts[-1] if parts else 'motif'
            plates.append({
                'name': base,
                'hex': hx,
                'role': role,
                'rgba': np.asarray(im, dtype=np.uint8),
            })
    return plates


def _mask_to_rgba(mask_l: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    """L mask (ink > 0) -> RGBA plate, transparent outside."""
    on = mask_l > 0
    rgba = np.zeros(mask_l.shape + (4,), np.uint8)
    rgba[on, 0] = rgb[0]
    rgba[on, 1] = rgb[1]
    rgba[on, 2] = rgb[2]
    rgba[on, 3] = 255
    return rgba


def _tone_to_rgba(tone_l: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    """Greyscale tone 0..255 -> RGBA (ink colour, alpha = tone %)."""
    rgba = np.zeros(tone_l.shape + (4,), np.uint8)
    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = tone_l
    return rgba


def _load_l(path: str, size: tuple[int, int]) -> np.ndarray:
    im = Image.open(path).convert('L')
    if im.size != size:
        im = im.resize(size, Image.Resampling.NEAREST)
    return np.asarray(im, dtype=np.uint8)


def _channel_filename(i: int, label: str, role: str, hx: str) -> str:
    token = re.sub(r'[^A-Za-z0-9]+', '-', label).strip('-') or 'plate'
    return f'channel_{i:02d}_{token}_{role}_{hx.upper()}.png'


def merge_into_package(
    out_dir: str,
    name: str,
    *,
    dpi: int = 300,
    underbase_path: str | None = None,
    trapped_dir: str | None = None,
    tone_plates: list[dict[str, Any]] | None = None,
    bold_numbers: str | None = None,
    write_psd: bool = True,
    composite_path: str | None = None,
    include_composite: bool = False,
    use_layers: bool = False,
    log=print,
) -> dict[str, Any]:
    """Rewrite package channels zip (+ PSD, loose channels, multipage spot TIF)."""
    pkg = os.path.join(out_dir, 'package')
    zips = sorted(glob.glob(os.path.join(pkg, f'{name}_colored_channels_*.zip')))
    if not zips:
        log('[package] no colored_channels zip — skip channel merge')
        return {'merged': False, 'reason': 'no_package_zip'}
    zpath = zips[0]
    plates = _read_zip_plates(zpath)
    if not plates:
        return {'merged': False, 'reason': 'empty_zip'}
    h, w = plates[0]['rgba'].shape[:2]
    size = (w, h)

    # --trap: replace spot plate masks with trapped versions (match by hex)
    if trapped_dir and os.path.isdir(trapped_dir):
        by_hex = {}
        for fn in os.listdir(trapped_dir):
            if not fn.lower().endswith('.png'):
                continue
            m = re.search(r'([0-9A-Fa-f]{6})\.png$', fn)
            if m:
                by_hex[m.group(1).upper()] = os.path.join(trapped_dir, fn)
        for p in plates:
            path = by_hex.get(p['hex'].upper())
            if not path:
                continue
            mask = _load_l(path, size)
            rgb = tuple(int(p['rgba'][p['rgba'][..., 3] > 0][0][i]) for i in range(3)) if (p['rgba'][..., 3] > 0).any() else (
                int(p['hex'][0:2], 16), int(p['hex'][2:4], 16), int(p['hex'][4:6], 16))
            p['rgba'] = _mask_to_rgba(mask, rgb)
            p['trapped'] = True
        log(f'[package] trap: replaced {sum(1 for p in plates if p.get("trapped"))} spot plates')

    new_plates: list[dict[str, Any]] = []

    # --underbase first
    if underbase_path and os.path.isfile(underbase_path):
        ub = _load_l(underbase_path, size)
        new_plates.append({
            'label': 'UNDERBASE-WHITE',
            'role': 'underbase',
            'hex': 'FFFFFF',
            'rgba': _mask_to_rgba(ub, (255, 255, 255)),
        })

    # Build tonal map from --gradients (greyscale tone_path preferred)
    tone_by_hex: dict[str, dict] = {}
    for tp in tone_plates or []:
        path = tp.get('tone_path') or tp.get('path')
        if not path or not os.path.isfile(path):
            continue
        ink = tp.get('ink_rgb') or [0, 0, 0]
        hx = hex_of(ink)
        tone_by_hex[hx.upper()] = {
            'label': tp.get('name') or f'INK-{hx}',
            'ink': tuple(int(x) for x in ink),
            'tone': _load_l(path, size),
        }

    used_tones: set[str] = set()
    for p in plates:
        mo = _CH_RE.match(p['name'])
        label = mo.group(2).rsplit('_', 1)[0] if mo else p['hex']
        hx = p['hex'].upper()
        if hx in tone_by_hex:
            td = tone_by_hex[hx]
            new_plates.append({
                'label': td['label'],
                'role': 'tone',
                'hex': hx,
                'rgba': _tone_to_rgba(td['tone'], td['ink']),
                'tone_l': td['tone'],
            })
            used_tones.add(hx)
        else:
            new_plates.append({
                'label': label,
                'role': p.get('role') or 'motif',
                'hex': hx,
                'rgba': p['rgba'],
            })

    # any tone inks not already in the package (append)
    for hx, td in tone_by_hex.items():
        if hx in used_tones:
            continue
        new_plates.append({
            'label': td['label'],
            'role': 'tone',
            'hex': hx,
            'rgba': _tone_to_rgba(td['tone'], td['ink']),
            'tone_l': td['tone'],
        })

    # Tone channels darkest-first (press / PSD bottom order); keep underbase first
    def _plate_lum(p):
        hx = p.get('hex') or '000000'
        try:
            r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        except Exception:
            r = g = b = 0
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    head = [p for p in new_plates if p.get('role') == 'underbase']
    rest = [p for p in new_plates if p.get('role') != 'underbase']
    tones = sorted([p for p in rest if p.get('role') == 'tone'], key=_plate_lum)
    solids = sorted([p for p in rest if p.get('role') != 'tone'], key=_plate_lum)
    new_plates = head + tones + solids

    # rewrite zip with press-order names
    tmp = zpath + '.tmp'
    bw_path = zpath.replace('_colored_channels_', '_bw_separations_')
    bw_tmp = bw_path + '.tmp' if os.path.isfile(bw_path) else None
    names_out = []
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zc:
        zb = zipfile.ZipFile(bw_tmp, 'w', zipfile.ZIP_DEFLATED) if bw_tmp else None
        try:
            for i, p in enumerate(new_plates, 1):
                fn = _channel_filename(i, p['label'], p['role'], p['hex'])
                names_out.append(fn)
                zc.writestr(fn, png_bytes(to_image(p['rgba']), dpi))
                if zb is not None:
                    if 'tone_l' in p:
                        bw_arr = (255 - p['tone_l']).astype(np.uint8)
                    else:
                        on = p['rgba'][..., 3] > 0
                        bw_arr = np.where(on, 0, 255).astype(np.uint8)
                    zb.writestr(fn, png_bytes(Image.fromarray(bw_arr, mode='L'), dpi))
        finally:
            if zb is not None:
                zb.close()
    os.replace(tmp, zpath)
    if bw_tmp and os.path.isfile(bw_tmp):
        os.replace(bw_tmp, bw_path)

    # loose channels + stacked
    from . import stack
    chs, steps = stack.loose_images(zpath, pkg, name, dpi)

    # PSD rewrite
    psd_path = None
    if write_psd:
        from . import psdout
        guides = []
        if bold_numbers and os.path.isfile(bold_numbers):
            guides = [('GUIDE bold sketch + numbers (hidden)', bold_numbers)]
        merged = None
        if composite_path and os.path.isfile(composite_path):
            merged = Image.open(composite_path).convert('RGB')
        psd_path = psdout.write_psd(
            zpath, os.path.join(pkg, f'{name}_layers_{w}px_{dpi}dpi.psd'), dpi,
            guides=guides, merged_rgb=merged, include_composite=include_composite,
            use_layers=use_layers,
        )

    # multipage spot TIF (one page per channel, L mask) — mill channel pack
    spot_tif = os.path.join(pkg, f'{name}_spot_channels_{w}px_{dpi}dpi.tif')
    pages = []
    for p in new_plates:
        if 'tone_l' in p:
            # greyscale tone: 255 = full ink (mill tone channel)
            pages.append(Image.fromarray(p['tone_l'], mode='L'))
        else:
            on = p['rgba'][..., 3] > 0
            pages.append(Image.fromarray(np.where(on, 0, 255).astype(np.uint8), mode='L'))
    if pages:
        pages[0].save(
            spot_tif, format='TIFF', save_all=True, append_images=pages[1:],
            compression='tiff_lzw', dpi=(dpi, dpi),
        )

    log(f'[package] channels updated: {len(names_out)} plates '
        f'(underbase={"yes" if underbase_path else "no"}, '
        f'tone={len(tone_plates or [])}) -> {os.path.basename(zpath)}')
    return {
        'merged': True,
        'channel_names': names_out,
        'channel_count': len(names_out),
        'channels_zip': zpath,
        'channels_dir': os.path.join(pkg, 'channels'),
        'psd': psd_path,
        'spot_tif': spot_tif if pages else None,
        'loose': chs,
    }


def channel_names_from_package(pkg_or_zip: str) -> list[str]:
    """Read channel file names (press order) from a zip or package/channels dir."""
    if os.path.isdir(pkg_or_zip):
        ch = os.path.join(pkg_or_zip, 'channels')
        if os.path.isdir(ch):
            return sorted(f for f in os.listdir(ch) if f.lower().endswith('.png'))
        zips = sorted(glob.glob(os.path.join(pkg_or_zip, '*_colored_channels_*.zip')))
        if not zips:
            return []
        pkg_or_zip = zips[0]
    with zipfile.ZipFile(pkg_or_zip) as z:
        return sorted(n for n in z.namelist() if n.lower().endswith('.png'))
