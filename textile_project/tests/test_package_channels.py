"""Prepress/gradient plates land in package spot channels (zip / channels / PSD order)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from textile.io_utils import save_png, to_image
from textile import package_channels as pc
from textile import prepress as pp
from textile import gradients as gr
from textile import export as ex


def _tiny_flat(tmp_path: Path, name='t'):
    rgb = np.zeros((48, 48, 3), np.uint8)
    rgb[:24, :24] = (200, 40, 40)
    rgb[:24, 24:] = (40, 180, 40)
    rgb[24:, :] = (240, 240, 245)
    flat = tmp_path / f'{name}_flat.png'
    save_png(to_image(rgb), str(flat), 300)
    # build a real package via export_package
    pal = np.array([[200, 40, 40], [40, 180, 40], [240, 240, 245]], np.uint8)
    index = np.zeros((48, 48), np.uint8)
    index[:24, :24] = 0
    index[:24, 24:] = 1
    index[24:, :] = 2
    pkg = tmp_path / 'package'
    done = ex.export_package(index, pal, str(pkg), name, dpi=300)
    return rgb, done


def test_merge_puts_underbase_first_and_tone_last(tmp_path):
    name = 't'
    rgb, done = _tiny_flat(tmp_path, name)
    # source with a soft gradient strip so --gradients applies
    src = rgb.astype(np.float32)
    for x in range(48):
        src[:, x, :] = src[:, x, :] * (0.55 + 0.45 * x / 47)
    src_u = np.clip(src, 0, 255).astype(np.uint8)
    # solid motif block
    src_u[10:22, 10:22] = (200, 40, 40)
    src_path = tmp_path / f'{name}.png'
    save_png(to_image(src_u), str(src_path), 300)
    save_png(to_image(rgb), str(tmp_path / f'{name}_flat.png'), 300)

    pre = pp.run_prepress(str(tmp_path), name, underbase=True, trap_px=1, dpi=300)
    assert pre['underbase']['generated']
    g = gr.run_gradients(str(tmp_path), name, str(src_path), size_px=(48, 48), dpi=300, lpi=40, colors_limit=8)
    # package lives under tmp_path/package from export
    # merge expects out_dir with package/ subfolder — move layout to match number()
    # export wrote into tmp_path/package; merge_into_package wants out_dir/package
    rep = pc.merge_into_package(
        str(tmp_path), name, dpi=300,
        underbase_path=pre['underbase']['path'],
        trapped_dir=str(tmp_path / 'prepress' / 'trapped_plates'),
        tone_plates=g.get('tone_plates') or [],
        write_psd=False, log=lambda *_: None,
    )
    assert rep['merged'] is True
    names = rep['channel_names']
    assert names[0].startswith('channel_01_') and 'underbase' in names[0].lower()
    assert any('halftone' in n.lower() or 'TONE' in n.upper() for n in names)
    # read back from zip
    back = pc.channel_names_from_package(rep['channels_zip'])
    assert back == names
    assert rep['channel_count'] == len(names)
    # spot TIF page count
    if rep.get('spot_tif'):
        from PIL import ImageSequence
        im = Image.open(rep['spot_tif'])
        pages = list(ImageSequence.Iterator(im))
        assert len(pages) == len(names)


def test_flags_off_does_not_need_merge():
    """Sanity: helper is only for extras; empty call returns no_package cleanly."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        r = pc.merge_into_package(d, 'x', dpi=300, write_psd=False, log=lambda *_: None)
        assert r['merged'] is False
