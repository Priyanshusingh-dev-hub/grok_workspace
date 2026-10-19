"""Phase 1: the common export / verify / palette modules and the CLI.

The anchor: reference_code/method1_colorfill.py (the user's tested script) is
run on the samples, and `textile export` of its final image must write the
very same files (every PNG byte, every layer of both zips, the TIFF and the
preview) and the same report numbers.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from textile import cli
from textile import palette as pl
from textile.export import export_package
from textile.io_utils import safe_name
from textile.verify import seam_check, verify_package

from conftest import RUNS, SAMPLES, TAG, ROOT  # noqa: E402


def _files(d, name):
    return [f'{name}_final_{TAG}.png', f'{name}_final_{TAG}.tif', f'{name}_channels_preview.png']


def old_name(n):
    """channel_01_olive_ground_4A5B24.png (now) -> channel_01_4A5B24.png (method1)."""
    parts = n[:-4].split('_')
    return f'{parts[0]}_{parts[1]}_{parts[-1]}.png'


def _preview_tiles(path):
    """The preview sheet without its label lines (they carry the file names)."""
    a = np.asarray(Image.open(path).convert('RGB')).copy()
    for y in range(20 + 500, a.shape[0], 570):
        a[y:y + 50] = 0
    return a


def _assert_same_package(old, new, name):
    for f in _files(old, name)[:2]:
        assert (old / f).read_bytes() == (new / f).read_bytes(), f
    prev = f'{name}_channels_preview.png'
    assert np.array_equal(_preview_tiles(old / prev), _preview_tiles(new / prev))
    for z in (f'{name}_colored_channels_{TAG}.zip', f'{name}_bw_separations_{TAG}.zip'):
        za, zb = zipfile.ZipFile(old / z), zipfile.ZipFile(new / z)
        assert za.namelist() == [old_name(n) for n in zb.namelist()], z
        assert all('#' not in n and len(n[:-4].split('_')) == 5 for n in zb.namelist())
        for n in zb.namelist():
            assert za.read(old_name(n)) == zb.read(n), f'{z}/{n}'
    ro = json.loads((old / f'{name}_report.json').read_text())
    rn = json.loads((new / f'{name}_report.json').read_text())
    for k in ('size_px', 'dpi', 'print_size_inch', 'colors_in_final'):
        assert ro[k] == rn[k], k
    assert ro['channels'] == [{k: c[k] for k in ('channel', 'hex', 'coverage_percent')} for c in rn['channels']]
    assert ro['tif_dpi'] == rn['verify']['tif_dpi']
    assert ro['channels_overlap_equals_final'] and rn['verify']['passed']


@pytest.mark.parametrize('name', ['floral', 'star'])
def test_export_writes_what_method1_wrote(reference, tmp_path, name):
    old = reference[name]
    rc = cli.main(['export', str(old / f'{name}_final_{TAG}.png'), '--out', str(tmp_path), '--name', name,
                   '--keep-strays'])
    assert rc == 0
    _assert_same_package(old, tmp_path, name)
    assert len(json.loads((old / f'{name}_report.json').read_text())['channels']) == RUNS[name][1]


def test_star_strays_fold_into_four_channels(reference, tmp_path):
    """ROADMAP: the star mandala is 4 colours once its stray colours merge."""
    assert cli.main(['export', str(reference['star'] / f'star_final_{TAG}.png'), '--out', str(tmp_path),
                     '--name', 'star']) == 0
    r = json.loads((tmp_path / 'star_report.json').read_text())
    assert [c['hex'] for c in r['channels']] == ['4A5B23', '120F06', 'F3EDDF', 'CE8F09']
    assert [c['role'] for c in r['channels']] == ['ground', 'motif', 'motif', 'motif']
    assert {m['hex']: m['into'] for m in r['stray_merged']} == {'887F64': '4A5B23', 'C5BBA0': 'F3EDDF'}
    assert r['verify']['passed'] and r['colors_in_final'] == 4


def test_merge_stray_sends_a_stray_to_the_nearest_big_colour():
    pal = np.array([[0, 0, 0], [250, 250, 250], [240, 10, 10], [10, 10, 240]], np.uint8)
    index = np.zeros((100, 100), np.uint8)
    index[:, 50:] = 1
    index[0, 0] = 2                       # 1 px of red: a stray, nearest big is black
    index[0, 99] = 3                      # 1 px of blue: nearest big is black too
    out, pal2, merged = pl.merge_stray(index, pal, min_share=0.001)
    assert pal2.tolist() == [[0, 0, 0], [250, 250, 250]]
    assert out[0, 0] == 0 and out[0, 99] == 0 and (out[:, 50:-1] == 1).all()
    assert {m['hex'] for m in merged} == {'F00A0A', '0A0AF0'}
    same, pal3, none = pl.merge_stray(index, pal, min_share=0.0)
    assert len(pal3) == 4 and not none and (same == index).all()


def test_palette_functions_are_method1s():
    src = (ROOT / 'reference_code' / 'method1_colorfill.py').read_text()
    for fn in ('extract_palette', 'map_to_palette'):
        body = src.split(f'def {fn}(')[1].split('\ndef ')[0].split('\n\n\n')[0]
        mine = (ROOT / 'textile' / 'palette.py').read_text().split(f'def {fn}(')[1].split('\n\n\n')[0]
        # same code, apart from the log function being passed in
        assert body.strip().replace('max_colors, min_share)', 'X') == \
            mine.strip().replace('max_colors, min_share, log=print)', 'X'), fn


def _tiny_package(tmp_path, dpi=300):
    pal = np.array([[55, 76, 19], [240, 207, 106], [119, 22, 132]], np.uint8)
    index = np.zeros((60, 60), np.uint8)
    index[10:40, 10:40] = 1
    index[20:30, 20:30] = 2
    return export_package(index, pal, str(tmp_path), 'tiny#1', dpi), index, pal


def test_verify_passes_a_good_package_and_names_have_no_hash(tmp_path):
    done, _, _ = _tiny_package(tmp_path)
    assert all('#' not in Path(p).name for p in done['paths'].values())
    v = verify_package(done['paths'], done['size_px'], 300)
    assert v['passed'], v['problems']
    assert v['colors_in_final'] == v['channels'] == 3 and v['tif_lzw'] and v['tif_dpi'] == [300, 300]


def test_verify_catches_a_missing_layer_and_a_wrong_size(tmp_path):
    done, _, _ = _tiny_package(tmp_path)
    z = done['paths']['channels_zip']
    with zipfile.ZipFile(z) as src:
        keep = {n: src.read(n) for n in src.namelist()[:-1]}
    with zipfile.ZipFile(z, 'w') as dst:
        for n, b in keep.items():
            dst.writestr(n, b)
    v = verify_package(done['paths'], [61, 60], 300)
    assert not v['passed']
    text = ' | '.join(v['problems'])
    assert 'size 60x60' in text and 'channels mila kar final nahi banta' in text and 'B/W' in text


def test_verify_catches_a_wrong_dpi(tmp_path):
    done, _, _ = _tiny_package(tmp_path, dpi=600)
    v = verify_package(done['paths'], done['size_px'], 300)
    assert not v['passed'] and any('DPI' in p for p in v['problems'])


def test_export_keeps_proportions_and_never_mixes_colours(tmp_path):
    img = np.zeros((40, 80, 3), np.uint8)
    img[:, 40:] = (230, 220, 200)
    img[10:20, 10:20] = (200, 30, 40)
    src = tmp_path / 'wide design #2.png'
    Image.fromarray(img).save(src)
    assert cli.main(['export', str(src), '--out', str(tmp_path / 'o'), '--size', '800']) == 0
    r = json.loads(next((tmp_path / 'o').glob('*_report.json')).read_text())
    assert r['size_px'] == [800, 400] and r['design'] == 'wide_design_2'
    with Image.open(next((tmp_path / 'o').glob('*_final_800x400px_300dpi.tif'))) as tif:
        assert len(tif.getcolors()) == 3                     # NEAREST only: no new colours


def test_verify_command_reads_a_folder_back(tmp_path, capsys):
    _tiny_package(tmp_path)
    assert cli.main(['verify', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'Verify: PASS' in out and '60 x 60 px' in out


def test_seam_check_tells_a_tile_from_a_cut():
    tile = np.zeros((90, 90, 3), np.uint8)
    tile[:, 30:60] = 200                                    # a stripe: wraps cleanly
    assert seam_check(tile)['seamless']
    cut = np.zeros((90, 90, 3), np.uint8)
    cut[:, 60:] = 200                                       # ends at the right edge: a join shows
    s = seam_check(cut)
    assert not s['seamless'] and not s['left_right']['seamless'] and s['top_bottom']['seamless']


def test_safe_name():
    assert safe_name('design #01 (final).png') == 'design_01_final_.png'
    assert '#' not in safe_name('#####') and safe_name('#####') == 'design'
