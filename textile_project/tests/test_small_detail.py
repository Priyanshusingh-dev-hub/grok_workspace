"""Small-detail rescue in `textile number` (flat_index rescue=True; --no-detail-rescue turns it off).
SYNTHETIC ground truth from tests/synth_detail.py: every object is scored for survival (IoU >= 0.5, own ink)."""
import numpy as np
import pytest
import synth_detail as sd
from textile import number as nb

KEEP = 0.4 * (300 / 25.4) ** 2          # --detail normal at 300 DPI


def _run(fn, s, K, cols, rescue):
    rgb, _ = fn(s)
    _, objs = fn(s * K)
    H, W = objs[0]['mask'].shape
    pal, idx = nb.flat_index(rgb, W, H, cols, KEEP, False, True, nb.grain(rgb), rescue=rescue)
    idx, _ = nb.merge_shade_inks(idx, pal, 300 / 25.4)
    return pal, sd.detail_metrics(objs, pal[idx])


def _kind(d, k):
    a, b = d['survival_by_kind'][k].split('/')
    return int(a), int(b)


@pytest.mark.parametrize('K', [1, 2.44])
def test_card_petals_rare_colours_and_centres_survive(K):
    _, d = _run(sd.small_object_card, 1, K, 12, True)
    for k in ('petal', 'rare', 'centre', 'ring', 'line', 'leaf'):
        a, b = _kind(d, k)
        assert a == b, (k, d['survival_by_kind'])
    assert d['merged_pairs'] == 0
    _, d0 = _run(sd.small_object_card, 1, K, 12, False)
    assert d['survival_percent'] >= d0['survival_percent'] + 25          # before: 55 / 58 %
    assert d['line_gap_retention_percent'] >= d0['line_gap_retention_percent'] - 0.5


def test_grey_dots_inside_a_dark_block_are_not_taken_for_an_edge_blend():
    _, d = _run(sd.small_object_card, 2, 1, 12, True)
    assert _kind(d, 'grey-in-dark')[0] >= 3                               # the three dots (hairline: known gap)
    assert _kind(_run(sd.small_object_card, 2, 1, 12, False)[1], "grey-in-dark")[0] == 0


def test_tiny_dots_survive_the_min_area_and_printed_size_is_kept():
    _, d = _run(sd.style_a, 1, 2.44, 8, True)
    assert d['survival_percent'] == 100 and d['merged_pairs'] == 0
    assert d['mean_abs_area_error_percent'] < 35
    _, d0 = _run(sd.style_a, 1, 2.44, 8, False)
    assert _kind(d0, 'dot')[0] < _kind(d, 'dot')[0]


def test_light_dots_on_a_black_centre_keep_their_own_tone():
    pal, d = _run(sd.style_b, 2, 1, 8, True)
    assert _kind(d, 'dot')[0] >= 17 and _kind(d, 'petal') == (75, 75) and d['merged_pairs'] == 0
    assert len(pal) <= 8


def test_rare_colour_with_no_ink_left_is_reported_not_silently_lost():
    rgb, _ = sd.small_object_card(1)
    nb.flat_index(rgb, rgb.shape[1], rgb.shape[0], 9, KEEP, False, True, nb.grain(rgb))
    assert nb.RESCUE_REPORT.get('rare_flagged')                          # budget full: listed for the log


def test_cli_flag_turns_the_rescue_off():
    from textile import cli
    import inspect
    src = inspect.getsource(cli)
    assert src.count("'--no-detail-rescue'") == 2 and 'detail_rescue=not a.no_detail_rescue' in src
    assert inspect.signature(nb.number).parameters['detail_rescue'].default is True
