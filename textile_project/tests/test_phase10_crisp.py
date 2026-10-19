"""textile crisp: a flat design redrawn as smooth curves (SVG + a bigger flat PNG)."""
import numpy as np
from PIL import Image, ImageDraw

from textile import crisp as cr
from textile.cli import main


def _design(tmp_path):
    im = Image.new('RGB', (240, 240), (242, 232, 204))
    d = ImageDraw.Draw(im)
    d.ellipse([40, 40, 100, 100], fill=(140, 55, 70))
    d.rectangle([140, 140, 200, 200], fill=(27, 45, 72))
    d.ellipse([150, 40, 160, 50], fill=(27, 45, 72))                       # a small dot
    im.save(tmp_path / 'f.png')
    return np.asarray(im)


def test_crisp_keeps_the_flat_colours_and_the_shapes_and_draws_at_the_zoom(tmp_path):
    rgb = _design(tmp_path)
    svg, big, rep = cr.crisp(rgb, scale=1.0, zoom=3)
    assert big.shape == (720, 720, 3)
    assert {tuple(c) for c in big.reshape(-1, 3)} == {tuple(c) for c in rgb.reshape(-1, 3)}   # flat colours, none new, no blur
    assert svg.startswith('<svg') and svg.count('<path') == 2             # two colours painted over the ground
    assert rep['pixels_different_percent'] < 3.0                           # only edge pixels differ from the pixel design
    # the circle (radius 30 px, so 90 px at the 3x zoom) is a true disc: its farthest pixel and its area agree
    ys, xs = np.nonzero((big == np.array([140, 55, 70])).all(2))
    r = np.hypot(xs - xs.mean(), ys - ys.mean())
    assert abs(r.max() - 90) < 4 and abs(len(xs) / (np.pi * 90 ** 2) - 1) < 0.05


def test_crisp_stops_on_a_picture_that_is_not_flat(tmp_path):
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)).save(tmp_path / 'n.png')
    assert main(['crisp', str(tmp_path / 'n.png'), '--out', str(tmp_path / 'o')]) == 1


def test_cli_crisp_writes_svg_png_and_report(tmp_path):
    _design(tmp_path)
    assert main(['crisp', str(tmp_path / 'f.png'), '--out', str(tmp_path / 'o'), '--name', 'f', '--zoom', '2', '--scale', '1']) == 0
    for n in ('f_crisp.svg', 'f_crisp_2x.png', 'f_crisp_report.json'):
        assert (tmp_path / 'o' / n).exists()
    with Image.open(tmp_path / 'o' / 'f_crisp_2x.png') as im:
        assert im.size == (480, 480)


def test_final_makes_the_exact_pixels_of_inches_times_dpi_and_never_stretches(tmp_path):
    im = Image.new('RGB', (400, 300), (242, 232, 204))                       # 4:3, the repeat is 4 x 3.4 inch (1.176:1)
    d = ImageDraw.Draw(im)
    d.ellipse([50, 50, 150, 150], fill=(140, 55, 70))
    d.rectangle([220, 120, 330, 230], fill=(27, 45, 72))
    im.save(tmp_path / 'd.png')
    assert main(['final', str(tmp_path / 'd.png'), '--out', str(tmp_path / 'o'), '--inches', '4x3.4', '--dpi', '300',
                 '--name', 't']) == 0
    tif = next((tmp_path / 'o').glob('t_final_*.tif'))
    with Image.open(tif) as t:
        assert t.size == (1200, 1020) and round(t.info['dpi'][0]) == 300      # inch x DPI, exactly
    assert next((tmp_path / 'o').glob('t_colored_channels_*.zip'))
