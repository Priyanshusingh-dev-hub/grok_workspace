"""Pillow's decompression-bomb guard stays on: textile is also imported inside
the LoomLab engine, so `Image.MAX_IMAGE_PIXELS = None` here switched it off for
the whole engine. Now a bounded limit (500 MP, LOOMLAB_MAX_IMAGE_PIXELS)."""
import struct
import zlib

import pytest
from PIL import Image

from textile import io_utils


def _chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF)


def test_importing_textile_keeps_the_guard_on():
    assert Image.MAX_IMAGE_PIXELS is not None
    assert Image.MAX_IMAGE_PIXELS >= 420_000_000          # a 2 m x 1.5 m repeat at 300 DPI still opens


def test_an_image_over_the_limit_is_refused_before_it_is_decoded(tmp_path):
    p = tmp_path / 'bomb.png'                              # the header of a 25000 x 25000 PNG, no pixels
    p.write_bytes(b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', struct.pack('>IIBBBBB', 25_000, 25_000, 1, 0, 0, 0, 0))
                  + _chunk(b'IEND', b''))
    with pytest.raises(Image.DecompressionBombError, match='25000 x 25000'):
        io_utils.load_rgb(p)


def test_a_normal_image_still_loads(tmp_path):
    p = tmp_path / 'ok.png'
    Image.new('RGB', (300, 200), '#8A1C1C').save(p)
    assert io_utils.load_rgb(p).shape == (200, 300, 3)


@pytest.mark.parametrize('env, want', [({}, 500_000_000), ({'LOOMLAB_MAX_IMAGE_PIXELS': '800000000'}, 800_000_000),
                                       ({'MAX_IMAGE_PIXELS': '300000000'}, 300_000_000),
                                       ({'LOOMLAB_MAX_IMAGE_PIXELS': '0'}, 500_000_000)])
def test_the_limit_reads_the_same_setting_as_the_engine(monkeypatch, env, want):
    monkeypatch.delenv('LOOMLAB_MAX_IMAGE_PIXELS', raising=False)
    monkeypatch.delenv('MAX_IMAGE_PIXELS', raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert io_utils._pixel_limit() == want
