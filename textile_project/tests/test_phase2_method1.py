"""Phase 2, Method 1: `textile fill` against reference_code/method1_colorfill.py.

With --keep-strays (method1 never merged strays) it must write what the
script wrote: the same final PNG/TIFF bytes, the same debug image, every
layer of both zips byte for byte (only the names are the new format), and
the same report numbers. By default strays merge, and the star mandala is the
4 channels the roadmap expects.
"""
import json
import os
import shutil
import zipfile

import numpy as np
import pytest
from PIL import Image

from conftest import RUNS, SAMPLES, TAG
from textile import cli


def old_name(n):
    p = n[:-4].split('_')
    return f'{p[0]}_{p[1]}_{p[-1]}.png'


def run_fill(out, name, extra=()):
    args = [a if not a.endswith('.png') else str(SAMPLES / a) for a in RUNS[name][0]]
    return cli.main(['fill', *args, '--out', str(out), '--name', name, *extra])


@pytest.mark.parametrize('name', ['floral', 'star'])
def test_fill_writes_what_method1_wrote(reference, tmp_path, name):
    assert run_fill(tmp_path, name, ['--keep-strays']) == 0
    old = reference[name]
    for f in (f'{name}_final_{TAG}.png', f'{name}_final_{TAG}.tif', f'{name}_DEBUG_doubtful_regions.png'):
        assert (old / f).read_bytes() == (tmp_path / f).read_bytes(), f
    for z in (f'{name}_colored_channels_{TAG}.zip', f'{name}_bw_separations_{TAG}.zip'):
        za, zb = zipfile.ZipFile(old / z), zipfile.ZipFile(tmp_path / z)
        assert za.namelist() == [old_name(n) for n in zb.namelist()]
        for n in zb.namelist():
            assert za.read(old_name(n)) == zb.read(n), n
    ro = json.loads((old / f'{name}_report.json').read_text())
    rn = json.loads((tmp_path / f'{name}_report.json').read_text())
    for k in ('size_px', 'dpi', 'colors_in_final', 'line_color_hex', 'alignment_score', 'regions',
              'doubtful_regions', 'reference_was_flat'):
        assert ro[k] == rn[k], k
    assert ro['channels'] == [{k: c[k] for k in ('channel', 'hex', 'coverage_percent')} for c in rn['channels']]
    assert rn['verify']['passed'] and (tmp_path / f'{name}_compare.png').exists()


def test_floral_names_its_outline(tmp_path):
    assert run_fill(tmp_path, 'floral') == 0
    r = json.loads((tmp_path / 'floral_report.json').read_text())
    assert r['line_color_hex'] == 'F0CF6A' and r['stray_merged'] == []
    roles = {c['hex']: c['role'] for c in r['channels']}
    assert roles['374C13'] == 'ground' and roles['F0CF6A'] == 'outline'
    names = zipfile.ZipFile(tmp_path / f'floral_colored_channels_{TAG}.zip').namelist()
    assert names[0] == 'channel_01_bottlegreen_ground_374C13.png'
    assert names[1] == 'channel_02_lightyellow_outline_F0CF6A.png'


def test_star_is_four_channels_once_strays_merge(tmp_path):
    assert run_fill(tmp_path, 'star') == 0
    r = json.loads((tmp_path / 'star_report.json').read_text())
    assert r['colors_in_final'] == 4 and r['verify']['passed']
    assert [(c['hex'], c['role']) for c in r['channels']] == [
        ('4A5B23', 'ground'), ('120F06', 'outline'), ('F3EDDF', 'motif'), ('CE8F09', 'motif')]
    assert {m['hex'] for m in r['stray_merged']} == {'887F64', 'C5BBA0'}


def _pair(folder, w=200, h=100):
    """A tiny aligned pair: a ring drawn in black, the same ring coloured."""
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - w / 2, yy - h / 2)
    line = np.full((h, w, 3), 255, np.uint8)
    ring = (d > 30) & (d < 34)
    line[ring] = 0
    ref = np.full((h, w, 3), (230, 220, 200), np.uint8)
    ref[d <= 32] = (180, 40, 60)
    ref[ring] = (20, 20, 20)
    os.makedirs(folder, exist_ok=True)
    Image.fromarray(line).save(os.path.join(folder, 'रेखा.png'))
    Image.fromarray(ref).save(os.path.join(folder, 'रंग.png'))
    return os.path.join(folder, 'रेखा.png'), os.path.join(folder, 'रंग.png')


def test_fill_opens_hindi_paths_and_keeps_proportions(tmp_path):
    line, ref = _pair(str(tmp_path / 'डिज़ाइन फ़ोल्डर'))
    out = tmp_path / 'out'
    assert cli.main(['fill', '--line', line, '--ref', ref, '--out', str(out), '--name', 'ring', '--size', '400']) == 0
    r = json.loads((out / 'ring_report.json').read_text())
    assert r['size_px'] == [400, 200] and r['colors_in_final'] == 3 and r['verify']['passed']
    with Image.open(out / 'ring_final_400x200px_300dpi.tif') as tif:
        a = np.asarray(tif.convert('RGB'))
    assert tuple(a[100, 200]) == (180, 40, 60) and tuple(a[5, 5]) == (230, 220, 200)


def test_fill_stops_when_the_images_do_not_line_up(tmp_path, capsys):
    line, ref = _pair(str(tmp_path / 'p'))
    moved = tmp_path / 'p' / 'moved.png'
    Image.fromarray(np.roll(np.asarray(Image.open(ref)), 40, axis=1)).save(moved)
    rc = cli.main(['fill', '--line', line, '--ref', str(moved), '--out', str(tmp_path / 'o'), '--size', '400'])
    assert rc == 1 and 'STOP: line art aur reference aligned nahi' in capsys.readouterr().out
    assert not (tmp_path / 'o').exists() or not any((tmp_path / 'o').glob('*_final_*'))
