"""textile stitch / split: parts of one design back into one image."""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from textile import stitch as st
from textile.cli import main


def _design(path, size=600):
    im = Image.new('RGB', (size, size), (242, 232, 204))
    d = ImageDraw.Draw(im)
    rng = np.random.default_rng(3)
    for _ in range(40):
        x, y, r = rng.integers(0, size, 2).tolist() + [int(rng.integers(15, 60))]
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in rng.integers(20, 200, 3)),
                  outline=(10, 10, 10), width=3)
    for k in range(0, size, 37):
        d.line([(0, k), (size, size - k)], fill=(27, 45, 72), width=4)
    im.save(path)
    return np.asarray(im)


def test_parts_cut_edge_to_edge_come_back_exactly(tmp_path):
    full = _design(tmp_path / 'd.png')
    parts = tmp_path / 'p'
    parts.mkdir()
    for r in range(3):
        for c in range(3):
            Image.fromarray(full[r * 200:(r + 1) * 200, c * 200:(c + 1) * 200]).save(parts / f'piece_{r + 1}_{c + 1}_200x200.png')
    img, rep = st.stitch(sorted(str(p) for p in parts.iterdir()), log=lambda m: None)
    assert rep['grid'] == [3, 3] and img.shape == full.shape
    assert np.array_equal(img, full)                                        # edge to edge: put back pixel for pixel
    assert not any(j['visible'] for j in rep['joints'])


def test_overlapping_parts_are_lined_up_and_joined_without_a_visible_seam(tmp_path):
    full = _design(tmp_path / 'd.png')
    out, plan = st.split(str(tmp_path / 'd.png'), str(tmp_path / 's'), 'd', (3, 3), 0.2)
    assert len(out) == 9 and all('_r' in os.path.basename(p) for p in out)
    assert len({Image.open(p).size for p in out}) == 1                     # all parts one size (AI tools give squares)
    img, rep = st.stitch(out, log=lambda m: None)
    assert img.shape == full.shape and rep['gaps_px'] == 0
    assert all(j.get('overlap_px', 1) > 0 for j in rep['joints'] if 'overlap_px' in j)   # every joint found its overlap
    assert np.abs(img.astype(int) - full.astype(int)).mean() < 1.0          # the same design back


def test_parts_from_different_drawings_are_named_as_visible_joints(tmp_path):
    a = _design(tmp_path / 'a.png')
    rng = np.random.default_rng(9)
    parts = tmp_path / 'p'
    parts.mkdir()
    for r in range(2):
        for c in range(2):
            tile = a[r * 300:(r + 1) * 300, c * 300:(c + 1) * 300].copy()
            if (r, c) == (0, 1):
                tile = 255 - tile                                           # a part from some other drawing
            Image.fromarray(tile).save(parts / f'r{r + 1}c{c + 1}.png')
    img, rep = st.stitch(sorted(str(p) for p in parts.iterdir()), log=lambda m: None)
    assert any(j['visible'] for j in rep['joints'])


def test_cli_stitch_from_a_zip(tmp_path):
    import zipfile
    full = _design(tmp_path / 'd.png', 300)
    z = tmp_path / 'parts.zip'
    with zipfile.ZipFile(z, 'w') as zf:
        for r in range(3):
            for c in range(3):
                p = tmp_path / f'x_{r + 1}_{c + 1}.png'
                Image.fromarray(full[r * 100:(r + 1) * 100, c * 100:(c + 1) * 100]).save(p)
                zf.write(p, f'puzzle/{p.name}')
    assert main(['stitch', str(z), '--out', str(tmp_path / 'o'), '--name', 'd']) == 0
    got = np.asarray(Image.open(tmp_path / 'o' / 'd_stitched.png').convert('RGB'))
    assert np.array_equal(got, full)


def _tile_set(tmp_path):
    full = _design(tmp_path / 'ref.png', 600)
    out, plan = st.split(str(tmp_path / 'ref.png'), str(tmp_path / 'parts'), 'ref', (3, 3), 0.15, target=5000)
    return full, out, plan


def test_split_writes_a_plan_and_says_how_big_a_part_to_ask_an_ai_for(tmp_path):
    full, out, plan = _tile_set(tmp_path)
    import json
    saved = json.loads((tmp_path / 'parts' / 'ref_parts_plan.json').read_text())
    assert saved['grid'] == [3, 3] and saved['size'] == [600, 600] and len(saved['boxes']) == 9
    assert 1800 <= plan['ask_ai_part_px'] <= 1900                          # 5000 wide from 3 parts with 15% overlap: ~1870 px
    img, rep = st.stitch(out, log=lambda m: None)
    assert abs(img.shape[1] - 600) <= 2                                    # the plan's geometry puts the parts back to the size


def test_partscheck_passes_good_tiles_and_names_moved_recoloured_and_blurred_ones(tmp_path):
    import cv2
    full, out, plan = _tile_set(tmp_path)
    plan_file = str(tmp_path / 'parts' / 'ref_parts_plan.json')
    ai = tmp_path / 'ai'
    ai.mkdir()
    for p in out:
        a = cv2.imread(p)
        a = cv2.resize(a, (a.shape[1] * 3, a.shape[0] * 3), interpolation=cv2.INTER_LANCZOS4)       # an upscaled copy: a faithful redraw
        name = os.path.basename(p)
        if name.endswith('r1c2.png'):
            a = np.roll(a, 40, axis=1)                                                              # a motif moved
        if name.endswith('r2c2.png'):
            a = (255 - a)                                                                           # other colours
        if name.endswith('r3c3.png'):
            a = cv2.GaussianBlur(a, (0, 0), 14)                                                     # detail blurred away
        cv2.imwrite(str(ai / name), a)
    res = st.parts_check(str(tmp_path / 'ref.png'), sorted(str(f) for f in ai.iterdir()), plan_file)
    assert set(res['redo']) == {'r1c2', 'r2c2', 'r3c3'}
    assert not res['tiles']['r1c1']['redo'] and not res['tiles']['r2c3']['redo']
    assert 'khisak' in ' '.join(res['tiles']['r1c2']['redo'])
    assert 'naye rang' in ' '.join(res['tiles']['r2c2']['redo'])
    assert 'detail' in ' '.join(res['tiles']['r3c3']['redo'])
    assert main(['partscheck', str(tmp_path / 'ref.png'), str(ai), '--plan', plan_file]) == 2        # redo needed: exit code 2
