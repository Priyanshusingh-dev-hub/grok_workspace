"""`fill --method auto`'s choice, in one place for the CLI and the LoomLab app.

Method 1, judged: it fails when its alignment is under `min_align`, or when
the share of each colour differs from the reference's by more than
`max_cover_diff` points in all (half the sum of the differences): on the tree
panel Method 1 gave the cream motif 10% where the reference has ~31% (13.7+
points), while the floral and star, filled right, differ by 1.7 and 3.2.
A failed Method 1 with two colours goes to Method 2; otherwise to Method 3,
judged the same way; if that fails too the shapes meet nowhere (a different
drawing) and Method 2 builds them from the line art alone, in the
reference's two main colours (`only_two`: said plainly).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2

from . import fill_method1 as m1
from . import fill_method2 as m2
from . import fill_method3 as m3
from . import fill_method4 as m4
from . import palette as pl
from .io_utils import read_cv2
from .verify import coverage_diff


@dataclass
class Choice:
    method: int
    fill: object                  # the chosen method's result (Fill, Fill2 or Fill3)
    index: object                 # strays merged (Methods 1 and 3)
    pal: object
    merged: list
    line_index: int | None        # None for Method 2 (no outline colour)
    auto: dict                    # what was measured and why, for the report
    others: list = field(default_factory=list)   # (rgb, label) of the methods not chosen


def merged(f, min_share=pl.STRAY_SHARE, keep_strays=False):
    """A Method 1/3 result with strays merged (unless kept), and the outline's
    index in the merged palette."""
    index, pal = f.index, f.pal
    line_rgb = pal[f.line_index].astype(int)
    out = []
    if not keep_strays:
        index, pal, out = pl.merge_stray(index, pal, min_share)
    line_index = int(((pal.astype(int) - line_rgb) ** 2).sum(1).argmin())
    return index, pal, out, line_index


def choose(line, ref, size=3535, max_colors=16, min_share=pl.STRAY_SHARE, line_threshold=150, line_color='auto',
           min_align=0.55, max_cover_diff=6.0, reach_align=96, keep_strays=False, m2_args=None, log=print):
    """Run Method 1, then 3, then 2 as needed. Raises FillError for a pair
    that cannot be filled at all (unreadable, different shapes)."""
    f = m1.fill(line, ref, size, max_colors, min_share, line_threshold, line_color, min_align, True, log=log)
    ref_rgb = cv2.cvtColor(read_cv2(ref, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    index, pal, mg, li = merged(f, min_share, keep_strays)
    diff = coverage_diff(ref_rgb, index, pal)
    why = []
    if f.alignment_score < min_align:
        why.append(f'alignment {f.alignment_score:.2f} < {min_align}')
    if diff > max_cover_diff:
        why.append(f'rangon ka hissa reference se {diff:.1f} points alag (> {max_cover_diff})')
    base = {'method1': {'alignment_score': round(f.alignment_score, 3), 'coverage_diff': round(diff, 1)},
            'why': why, 'limit': max_cover_diff}
    if not why:
        log(f'[auto] Method 1 theek: alignment {f.alignment_score:.2f}, coverage farak {diff:.1f} points')
        return Choice(1, f, index, pal, mg, li, base | {'chosen': 1, 'coverage_diff': round(diff, 1)})
    others = [(pal[index], 'method 1 (nahi chuna)')]

    def method2(extra):
        f2 = m2.fill(line, ref, size, line_threshold, log=log, **(m2_args or {}))
        auto = base | {'chosen': 2, 'coverage_diff': round(coverage_diff(ref_rgb, f2.index, f2.pal), 1)} | extra
        return Choice(2, f2, f2.index, f2.pal, [], None, auto, others)

    if len(pal) <= 2:
        log(f"[auto] Method 1 nahi chala ({'; '.join(why)}) -> Method 2 (2 rang, line art ki structure)")
        return method2({})
    # AI line art breaks here and there and the ground colour runs through: seal those gaps first
    log(f"[auto] Method 1 nahi chala ({'; '.join(why)}) -> Method 4 (line art ke gap band karke)")
    f4 = m4.fill(line, ref, size, max_colors, min_share, line_threshold, line_color, None, min_align, True, log=log)
    f4.reference_was_flat = f.reference_was_flat
    index4, pal4, mg4, li4 = merged(f4, min_share, keep_strays)
    diff4 = coverage_diff(ref_rgb, index4, pal4)
    base['method4'] = {'alignment_score': round(f4.alignment_score, 3), 'coverage_diff': round(diff4, 1)}
    if f4.alignment_score >= min_align and diff4 <= max_cover_diff:
        return Choice(4, f4, index4, pal4, mg4, li4, base | {'chosen': 4, 'coverage_diff': round(diff4, 1)}, others)
    others.append((pal4[index4], 'method 4 (nahi chuna)'))
    log(f"[auto] Method 4 bhi nahi chala (farak {diff4:.1f} points) -> Method 3 (reference ko line art par khiska kar)")
    f3 = m3.fill(line, ref, size, max_colors, min_share, line_threshold, line_color, reach_align, pal=f.pal, log=log)
    f3.reference_was_flat = f.reference_was_flat
    index, pal, mg, li = merged(f3, min_share, keep_strays)
    after = coverage_diff(ref_rgb, index, pal)
    base['method3'] = {'alignment_score': round(f3.alignment_score, 3), 'coverage_diff': round(after, 1)}
    if f3.alignment_score >= min_align and after <= max_cover_diff:
        return Choice(3, f3, index, pal, mg, li, base | {'chosen': 3, 'coverage_diff': round(after, 1)}, others)
    why.append(f'Method 3 bhi nahi chala (alignment {f3.alignment_score:.2f}, farak {after:.1f} points)')
    log(f"[auto] {why[-1]} -> Method 2 (2 rang, line art ki structure)")
    others.append((pal[index], 'method 3 (nahi chuna)'))
    return method2({'only_two': True})
