"""The mill's own ink names on the plates (`paint --ink-names`, `final --ink-names`, `cli.number_package`):
opt-in, exact hex only, ASCII file and PSD layer names that keep the 5-part channel shape."""
import json
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from textile.cli import main, number_package
from textile.names import ink_token, read_ink_names

MAROON, NAVY = '8C3746', '1B2D48'


def test_a_name_becomes_a_safe_file_word():
    assert ink_token('Rani Pink #12') == 'Rani-Pink-12'
    assert ink_token('Navy/Blue: 3') == 'Navy-Blue-3'
    assert ink_token('Crème Brûlée') == 'Creme-Brulee'
    assert ink_token('Ink_5.2') == 'Ink-5-2'
    assert ink_token('=HYPERLINK("x")') == 'HYPERLINK-x'
    assert ink_token('रानी गुलाबी') == ''                         # nothing left: the colour word stays
    assert len(ink_token('a' * 80)) == 32


def test_names_are_read_from_a_hex_map_or_loomlabs_inks_file(tmp_path):
    f = tmp_path / 'n.json'
    f.write_text(json.dumps({'#8c3746': ' Rani Pink 12 ', NAVY: 'Navy\x07 7', 'FFFFFF': ''}), encoding='utf-8')
    assert read_ink_names(f) == {MAROON: 'Rani Pink 12', NAVY: 'Navy 7'}
    f.write_text(json.dumps({'inks': [{'name': 'रानी', 'hex': '#8C3746'}]}, ensure_ascii=False), encoding='utf-8')
    assert read_ink_names(f) == {MAROON: 'रानी'}
    for bad in ('[1, 2]', '{"red": "x"}', '{"8C3746": 5}', 'not json'):
        f.write_text(bad, encoding='utf-8')
        with pytest.raises(ValueError):
            read_ink_names(f)


@pytest.fixture(scope='module')
def numbered(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('n')
    im = Image.new('RGB', (500, 500), (242, 232, 204))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 60, 240, 240], fill=(140, 55, 70))
    d.rectangle([300, 300, 460, 440], fill=(27, 45, 72))
    im.save(tmp / 'p.png')
    out = tmp / 'o'
    assert main(['number', str(tmp / 'p.png'), '--out', str(out), '--size', '500']) == 0
    return out, json.loads((out / 'p_number.json').read_text(encoding='utf-8'))


def test_named_plates_keep_the_channel_shape_and_the_psd_layers_read(numbered, tmp_path):
    from psd_tools import PSDImage
    out, r = numbered
    names = tmp_path / 'names.json'
    names.write_text(json.dumps({MAROON: 'Rani Pink #12', NAVY: 'रानी नीला'}, ensure_ascii=False), encoding='utf-8')
    run = tmp_path / 'run'
    run.mkdir()
    assert number_package('p', str(out / r['sketch']), str(out / r['csv']), str(out / r['bold_numbers']), str(run),
                          r['size_px'], 300, True, str(names)) == 0
    pkg = run / 'package'
    zc = next(pkg.glob('p_colored_channels_*.zip'))
    members = zipfile.ZipFile(zc).namelist()
    assert any(m.endswith(f'_Rani-Pink-12_motif_{MAROON}.png') for m in members)
    assert any(m.endswith(f'_navy_motif_{NAVY}.png') for m in members)        # Devanagari: the colour word
    assert all(len(m[:-4].split('_')) == 5 and '#' not in m for m in members)
    assert sorted(zipfile.ZipFile(next(pkg.glob('p_bw_separations_*.zip'))).namelist()) == sorted(members)
    assert sorted(p.name for p in (pkg / 'channels').iterdir()) == sorted(members)
    report = json.loads((pkg / 'p_report.json').read_text(encoding='utf-8'))
    assert {c['hex']: c.get('ink') for c in report['channels']}[MAROON] == 'Rani Pink #12'
    assert report['verify']['passed']
    from psd_tools.constants import Resource
    psd = PSDImage.open(next(pkg.glob('p_layers_*.psd')))
    assert list(psd) == []  # channels-only default: no ink layers
    ch_names = list(psd.image_resources[Resource.ALPHA_NAMES_UNICODE].data)
    assert any(MAROON in n and 'Rani Pink' in n for n in ch_names), ch_names
    assert main(['verify', str(pkg)]) == 0


def test_without_names_nothing_changes_and_a_bad_file_stops_before_the_work(numbered, tmp_path):
    out, r = numbered
    bad = tmp_path / 'bad.json'
    bad.write_text('{"zz": "x"}', encoding='utf-8')
    assert main(['paint', str(out / r['sketch']), '--colors', str(out / r['csv']), '--out', str(tmp_path / 'a'),
                 '--name', 'p', '--size', str(r['size_px'][0]), '--ink-names', str(bad)]) == 1
    assert main(['final', str(out / 'p_flat.png'), '--out', str(tmp_path / 'f'), '--inches', '1x1',
                 '--ink-names', str(tmp_path / 'missing.json')]) == 1
    plain = zipfile.ZipFile(next((out / 'package').glob('p_colored_channels_*.zip'))).namelist()
    assert all(m.split('_')[2].islower() for m in plain)                     # the colour words, as always


def test_final_names_its_plates_too(numbered, tmp_path):
    """`final` picks its own inks at the mill's size: a name is put on whichever of them has that exact hex."""
    out, _ = numbered
    args = ['final', str(out / 'p_flat.png'), '--inches', '1x1', '--name', 'm']
    assert main(args + ['--out', str(tmp_path / 'a')]) == 0
    first = json.loads((tmp_path / 'a' / 'm_report.json').read_text(encoding='utf-8'))
    hx = first['channels'][-1]['hex']
    names = tmp_path / 'n.json'
    names.write_text(json.dumps({hx: 'Deep Navy'}), encoding='utf-8')
    assert main(args + ['--out', str(tmp_path / 'b'), '--ink-names', str(names)]) == 0
    members = zipfile.ZipFile(next((tmp_path / 'b').glob('m_colored_channels_*.zip'))).namelist()
    assert any(m.endswith(f'_Deep-Navy_{first["channels"][-1]["role"]}_{hx}.png') for m in members)
    again = json.loads((tmp_path / 'b' / 'm_report.json').read_text(encoding='utf-8'))
    assert again['verify']['passed'] and again['channels'][-1]['ink'] == 'Deep Navy'
