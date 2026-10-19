"""`final --min-mm2`: the operator's own smallest part (sq mm), instead of --detail's three steps."""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from textile.cli import main


def _parts(png):
    a = np.asarray(Image.open(png).convert('RGB')).reshape(-1, 3)
    flat = np.asarray(Image.open(png).convert('RGB'))
    n = 0
    for c in np.unique(a, axis=0):
        n += ndimage.label((flat == c).all(-1))[1]
    return n


def test_a_bigger_smallest_part_joins_flecks_into_their_neighbour(tmp_path):
    im = Image.new('RGB', (400, 300), (242, 232, 204))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 50, 220, 200], fill=(140, 55, 70))
    d.rectangle([260, 120, 380, 260], fill=(27, 45, 72))
    for x, y in ((20, 20), (240, 20), (330, 40), (30, 250), (230, 270), (300, 100)):      # round flecks about 9 px wide
        d.ellipse([x, y, x + 9, y + 9], fill=(140, 55, 70))
    im.save(tmp_path / 'p.png')
    made = {}
    for key, extra in (('fine', ['--detail', 'zyada']), ('coarse', ['--min-mm2', '20'])):
        out = tmp_path / key
        assert main(['final', str(tmp_path / 'p.png'), '--out', str(out), '--inches', '2x1.5', '--name', 'p',
                     '--no-psd', *extra]) == 0
        made[key] = _parts(next(out.glob('p_final_*.png')))
    assert made['coarse'] < made['fine']
