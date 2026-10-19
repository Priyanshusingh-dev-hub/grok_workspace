"""The colour plates as loose images, and the plates laid over each other one at a time.

  loose_images(channels_zip, out_dir, name, dpi) -> (channel files, stacked step files)

`textile paint` already writes every colour as its own 3535 px / 300 DPI RGBA layer, but inside a zip, and its
preview is one small sheet. This writes the same layers as ordinary files, and for each step k the first k
plates stacked on white (plate 1 alone, plates 1+2, ... all of them = the final design), every one the full
3535 px at 300 DPI, never a sheet of thumbnails. Plates are mutually exclusive, so stacking them in order only
shows how the design builds up; it changes nothing the mill gets.
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
from PIL import Image

from .io_utils import save_png, to_image


def loose_images(channels_zip, out_dir, name, dpi=300):
    ch_dir = os.path.join(out_dir, 'channels')
    st_dir = os.path.join(out_dir, 'stacked')
    os.makedirs(ch_dir, exist_ok=True)
    os.makedirs(st_dir, exist_ok=True)
    channels, steps = [], []
    canvas = None
    with zipfile.ZipFile(channels_zip) as z:
        for i, member in enumerate(sorted(n for n in z.namelist() if n.lower().endswith('.png')), 1):
            data = z.read(member)
            path = os.path.join(ch_dir, os.path.basename(member))
            with open(path, 'wb') as fh:                       # the plate itself, byte for byte
                fh.write(data)
            channels.append(path)
            rgba = np.asarray(Image.open(io.BytesIO(data)).convert('RGBA'))
            if canvas is None:
                canvas = np.full(rgba.shape[:2] + (3,), 255, np.uint8)
            on = rgba[..., 3] > 0
            canvas[on] = rgba[..., :3][on]
            step = os.path.join(st_dir, f'{name}_stack_{i:02d}_of_{{n}}.png')
            steps.append((step, canvas.copy()))
    n = len(steps)
    files = []
    for step, img in steps:
        path = step.format(n=f'{n:02d}')
        save_png(to_image(img), path, dpi)
        files.append(path)
    return channels, files
