"""Phase 5: `textile batch` (every pair in a folder), `textile vector`
(trace + redraw without anti-aliasing, + SVG) and the 600 DPI export."""
import csv
import json
import shutil
import xml.dom.minidom

import numpy as np
from PIL import Image, ImageDraw

from conftest import SAMPLES
from textile import cli, vector
from textile.verify import seam_check


def test_batch_runs_every_pair_and_survives_a_bad_one(tmp_path, capsys):
    inp = tmp_path / 'in'; inp.mkdir()
    shutil.copy(SAMPLES / 'tree_lineart.png', inp / 'tree_lineart.png')
    shutil.copy(SAMPLES / 'tree_ref.png', inp / 'tree_colored.png')          # colorfill's naming works too
    shutil.copy(SAMPLES / 'star_lineart.png', inp / 'star_lineart.png')
    shutil.copy(SAMPLES / 'star_ref.png', inp / 'star_ref.png')
    (inp / 'broken_lineart.png').write_bytes(b'not an image')
    (inp / 'broken_ref.png').write_bytes(b'not an image')
    shutil.copy(SAMPLES / 'star_lineart.png', inp / 'alone_lineart.png')    # no reference
    assert cli.main(['batch', str(inp), '--out', str(tmp_path / 'out'), '--size', '800']) == 1
    rows = {r['design']: r for r in csv.DictReader(open(tmp_path / 'out' / 'batch_summary.csv'))}
    assert set(rows) == {'broken', 'star', 'tree'}
    assert rows['tree']['result'] == 'OK' and rows['tree']['method'] == '2' and rows['tree']['verify'] == 'PASS'
    assert rows['star']['result'] == 'OK' and rows['broken']['result'] == 'STOP'
    assert (tmp_path / 'out' / 'tree' / 'tree_final_800px_300dpi.tif').exists()
    assert 'alone_lineart.png' in capsys.readouterr().out                    # said, not silently skipped


def _flat():
    im = Image.new('P', (400, 300), 0)
    im.putpalette([240, 230, 210, 120, 30, 40, 40, 90, 60] + [0] * 759)
    d = ImageDraw.Draw(im)
    d.polygon([(30, 250), (200, 20), (370, 260)], fill=1)                  # slanted edges: staircases
    d.ellipse((150, 140, 250, 230), fill=2)
    d.rectangle((10, 10, 13, 13), fill=2)                                  # a 4 px dot stays
    return np.asarray(im.convert('RGB'))


def test_vector_straightens_edges_and_keeps_one_ink_per_pixel(tmp_path):
    src = tmp_path / 'd_final_400x300px_300dpi.png'
    Image.fromarray(_flat()).save(src, dpi=(300, 300))
    assert cli.main(['vector', str(src), '--out', str(tmp_path / 'v')]) == 0
    r = json.loads((tmp_path / 'v' / 'd_vector_report.json').read_text())
    assert r['verify']['passed'] and r['colors_in_final'] == 3 and r['changed_percent'] < 3
    out = np.asarray(Image.open(tmp_path / 'v' / 'd_vector_final_400x300px_300dpi.png').convert('RGB'))
    assert (out[10:14, 10:14] == (40, 90, 60)).all()                         # the dot is not traced away
    xml.dom.minidom.parse(str(tmp_path / 'v' / 'd_vector.svg'))             # a well-formed SVG
    assert (tmp_path / 'v' / 'd_vector.svg').read_text().count('<path') == 2


def test_vector_on_a_repeat_keeps_it_seamless():
    y, x = np.mgrid[0:240, 0:240]
    ring = ((x - 10) ** 2 + (y - 120) ** 2 < 60 ** 2) | ((x - 250) ** 2 + (y - 120) ** 2 < 60 ** 2)
    index = ring.astype(np.uint8)                                            # a disc cut by the left/right join
    pal = np.array([[20, 20, 60], [230, 200, 120]], np.uint8)
    padded = np.pad(index, 32, mode='wrap')
    traced = vector.trace(padded, pal)[0][32:-32, 32:-32]
    assert seam_check(pal[traced])['seamless']


def test_vector_refuses_a_photo(tmp_path, capsys):
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)).save(tmp_path / 'p.png')
    assert cli.main(['vector', str(tmp_path / 'p.png'), '--out', str(tmp_path / 'v')]) == 1
    assert 'flat design nahi' in capsys.readouterr().out


def test_600_dpi_keeps_the_inches_and_warns(tmp_path, capsys):
    src = tmp_path / 'd.png'
    Image.fromarray(_flat()).save(src)
    assert cli.main(['export', str(src), '--out', str(tmp_path / 'o'), '--dpi', '600']) == 0
    out = capsys.readouterr().out
    assert out.count('[CHETAVNI] 600 DPI') == 2
    r = json.loads((tmp_path / 'o' / 'd_report.json').read_text())
    assert r['size_px'] == [7070, 5302] and r['dpi'] == 600 and r['verify']['passed']
