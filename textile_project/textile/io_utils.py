"""Files in and out, the mill's way: DPI written into every image, the final
as an LZW TIFF, no '#' in any name (some phone and zip apps skip such files).

Colour images are read with PIL, which opens any path Windows can (cv2.imread
cannot read a path with non-English letters on Windows); for an 8-bit PNG the
pixels are cv2's. Line art (grey) is read with `read_cv2`: PIL's grey differs.
"""
from __future__ import annotations

import io
import os
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

def _pixel_limit() -> int:
    """Pillow's decompression-bomb guard, raised for mill-size files but never
    switched off (this module is also imported inside the LoomLab engine, so
    the setting is process-wide). 500 MP by default (a 2 m x 1.5 m repeat at
    300 DPI is ~419 MP); LOOMLAB_MAX_IMAGE_PIXELS (or MAX_IMAGE_PIXELS) changes
    it, read exactly as backend/app/core/pixels.py reads it."""
    for key in ('LOOMLAB_MAX_IMAGE_PIXELS', 'MAX_IMAGE_PIXELS'):
        raw = os.environ.get(key, '').strip().replace('_', '')
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return 500_000_000


Image.MAX_IMAGE_PIXELS = _pixel_limit()

_UNSAFE = re.compile(r'[^A-Za-z0-9._-]+')


def safe_name(text: str) -> str:
    """A name safe in a file name: letters, digits, '.', '_' and '-' only
    (so never '#', spaces or slashes)."""
    return _UNSAFE.sub('_', str(text)).strip('_') or 'design'


def load_rgb(path) -> np.ndarray:
    """An image as H x W x 3 uint8 RGB (alpha dropped)."""
    with Image.open(path) as im:
        limit = Image.MAX_IMAGE_PIXELS or _pixel_limit()
        if im.width * im.height > limit:         # Pillow itself only warns up to twice the limit
            raise Image.DecompressionBombError(
                f'{path}: {im.width} x {im.height} px is more than {limit:,} pixels '
                '(LOOMLAB_MAX_IMAGE_PIXELS badha sakte ho).')
        return np.asarray(im.convert('RGB'))


def read_cv2(path, flag):
    """cv2.imread's pixels for any path: the file's bytes decoded by OpenCV.
    (cv2.imread itself cannot open a path with non-English letters on
    Windows; PIL's own grey conversion differs from OpenCV's by a level here
    and there, which would move pixels across a threshold.) None if unreadable."""
    import cv2
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flag) if data.size else None


def to_image(arr: np.ndarray) -> Image.Image:
    """uint8 array -> PIL image (mode from the shape: L, RGB or RGBA)."""
    return Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8))


def save_png(img: Image.Image, path, dpi: int) -> Path:
    img.save(path, dpi=(dpi, dpi))
    return Path(path)


def save_tif(img: Image.Image, path, dpi: int) -> Path:
    """Lossless LZW TIFF with the DPI written in: the file the mill gets."""
    img.save(path, dpi=(dpi, dpi), compression='tiff_lzw')
    return Path(path)


def png_bytes(img: Image.Image, dpi: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, 'PNG', dpi=(dpi, dpi))
    return buf.getvalue()


def ordered_map(fn, items, workers=None):
    """`map(fn, items)` on a few threads: results in the items' order, never more than `workers` running or
    waiting to be taken (each may hold a full-size layer: 175 MB at 7050 x 6210). PIL's PNG/TIFF zlib/LZW coding
    and numpy's big array work let go of the GIL, so a package's channels code ~3x faster on 4 cores; the bytes
    are the same (each file is still made by one encoder, written in order)."""
    workers = workers or min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(workers) as pool:
        waiting = deque()
        for item in items:
            waiting.append(pool.submit(fn, item))
            if len(waiting) >= workers:
                yield waiting.popleft().result()
        while waiting:
            yield waiting.popleft().result()


def hex_of(rgb) -> str:
    """'4A5B24' (no '#', upper case)."""
    r, g, b = (int(v) for v in rgb)
    return f'{r:02X}{g:02X}{b:02X}'


def rgb_of(hex_text: str) -> np.ndarray:
    h = hex_text.strip().lstrip('#')
    if not re.fullmatch(r'[0-9A-Fa-f]{6}', h):
        raise ValueError(f'"{hex_text}" rang ka hex code nahi hai (jaise 4A5B24).')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


def inches(px: int, dpi: int) -> float:
    return round(px / dpi, 2)


def unique_rgb(rgb, return_inverse=False):
    """np.unique(rgb.reshape(-1, 3), axis=0[, return_inverse=True]): the same colours in the same order (and the
    same inverse), each pixel packed into one integer first. Unique rows sort 3 columns lexically: 15 s on a
    3535 px design, ~2 s this way (a 7050 px `final` sheet: about a minute saved)."""
    flat = np.asarray(rgb).reshape(-1, 3).astype(np.int32)
    key = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    if return_inverse:
        key, inv = np.unique(key, return_inverse=True)
    else:
        key = np.unique(key)
    cols = np.stack([key >> 16, (key >> 8) & 255, key & 255], 1).astype(np.uint8)
    return (cols, inv) if return_inverse else cols
