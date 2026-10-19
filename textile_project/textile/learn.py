"""The tool's memory: what every `textile number` run found, what went wrong, what to do about it, and
whether the operator was happy. Not a neural network (a few dozen designs is far too little to train one: it
would only memorise them): a plain record plus written rules, which is exactly what makes a decision easy and
what a real model can be trained on later.

  advise(result)            -> [(code, hinglish message)]  the problems this run shows and the fix for each
  record(result, ...)       -> one line in data/number-log.jsonl (features, settings, results, problems)
  feedback(name, verdict)   -> the operator's word ('good' / 'bad' + a note) on the latest run of NAME (or, with
                               `run`, on exactly that run: LoomLab gives each run its job id)
  similar(result)           -> past runs that look like this design, with what worked on them (advice only)
  summary()                 -> text: runs, problems by how often, what the 'good' runs had in common

Everything here is advice and notes: it never changes what a run makes. A failing log (a read-only disk) never
stops a run.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import numpy as np

from . import names as nm

LOG_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'number-log.jsonl')

SIMILAR_DE = 12.0      # two inks closer than this (Lab distance) look like one colour on cloth
SIMILAR_SHARE = 1.5    # ...and the smaller is under this % of the design: a stray shade, not a real second ink
BUSY_AREAS = 1500      # more parts than this: the sketch is very full
MATCH_LOW = 99.0       # sketch painted back with its CSV, % of the flat design


def log_path():
    return os.environ.get('TEXTILE_LEARN_LOG') or LOG_DEFAULT


def _hex_rgb(hx):
    h = hx.lstrip('#')
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def similar_inks(inks):
    """[(small, big, dE)]: an ink (hex, name, share) that is close to a bigger one and itself small."""
    out = []
    if len(inks) < 2:
        return out
    lab = nm._lab(np.array([_hex_rgb(i[0]) for i in inks], np.float64))
    for a in range(len(inks)):
        for b in range(len(inks)):
            if a != b and inks[a][2] < SIMILAR_SHARE and inks[a][2] < inks[b][2]:
                de = float(np.linalg.norm(lab[a] - lab[b]))
                if de < SIMILAR_DE:
                    out.append((inks[a], inks[b], round(de, 1)))
    seen, res = set(), []
    for s, b, de in sorted(out, key=lambda t: t[2]):
        if s[0] not in seen:                       # each small ink is named once, with its closest big one
            seen.add(s[0])
            res.append((s, b, de))
    return res


def advise(r):
    """The problems a finished run shows, each with what to do. `r` is what `number()` returned. Returns
    [(code, message)], most important first; an empty list means nothing to fix."""
    out = []
    for s, b, de in similar_inks(r['inks']):
        out.append(('similar_inks', f"{s[0]} ({s[1]}, {s[2]}%) {b[0]} ({b[1]}) ke bahut paas hai (farak {de}): shayad ek hi rang. "
                    f"CSV me dono ko ek rang do to ek channel / screen kam hogi."))
    if r.get('tiny'):
        out.append(('tiny_parts', f"{r['tiny']} bahut chhote hisse (0.1 sq mm se kam): chhape me nahi aayenge. "
                    f"`--detail kam` se paas ke hisse me mil jaate hain."))
    if r.get('missed'):
        out.append(('missed_numbers', f"{r['missed']} hisse ko number likhne ki jagah nahi mili: `--bold-scale 2` ya zoom karke dekho "
                    f"(CSV me wo number phir bhi hai)."))
    if r['match'] < MATCH_LOW:
        out.append(('match_low', f"Sketch se wapas design sirf {r['match']}% mil raha: lines bahut patli/kati hain ya hisse mile hue. "
                    f"`--line-mm 0.17` ya `--detail zyada` try karo."))
    limit = r.get('colours_limit')
    if limit and len(r['inks']) >= limit:
        out.append(('colours_at_limit', f"Rang ginti limit ({limit}) tak pahunch gayi: design me aur rang chhoot sakte hain. "
                    f"`--colors {limit + 2}` try karo aur dekho koi naya asli rang aata hai ya nahi."))
    if r['areas'] > BUSY_AREAS:
        out.append(('busy', f"{r['areas']} hisse: bahut bhara design, sketch me numbers chhote honge. "
                    f"`--detail kam` hisse ghatata hai; zoom ke liye `--bold-scale 2`."))
    if r.get('woven'):
        out.append(('woven', "Ye photo / buna kapda jaisa laga (grain zyada): daane saaf kiye gaye. Mockup photo ho to design khud crop karke do."))
    elif r.get('grain', 0) >= 2:
        out.append(('grainy', f"Thoda grain ({r['grain']}): JPEG ya AI ki mehek ho sakti hai; rang ginti dhyaan se dekho."))
    if r.get('line_mm') and r['line_mm'] > 0.17 and len(r.get('tried', [])) > 1:
        out.append(('thick_line_won', f"Patli line se match kam aaya, {r['line_mm']} mm line chuni gayi: design me bareek detail hai."))
    return out


def features(r):
    """The numbers that describe a design, for finding past designs that look like it."""
    w, h = r['size_px']
    mpx = max(w * h / 1e6, 1e-6)
    shares = sorted((i[2] for i in r['inks']), reverse=True) or [100.0]
    return {'grain': r.get('grain', 0.0), 'areas_per_mpx': round(r['areas'] / mpx, 1), 'inks': len(r['inks']),
            'top_share': shares[0], 'small_inks': int(sum(1 for s in shares if s < SIMILAR_SHARE)),
            'tiny_per_mpx': round(r.get('tiny', 0) / mpx, 2)}


def _read():
    p = log_path()
    rows = []
    if os.path.exists(p):
        with open(p, 'rb') as fh:                  # decoded line by line: a crash can cut a line inside a character
            for line in fh:
                try:
                    rows.append(json.loads(line.decode('utf-8')))
                except ValueError:                 # (UnicodeDecodeError too) a half-written line: skip it, keep the rest
                    pass
    return rows


_LOCK = threading.Lock()      # LoomLab writes runs and verdicts from several threads at once


def _append(row):
    p = log_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with _LOCK, open(p, 'a+b') as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell():                          # a crash left a half line: start on a new one, or both are lost
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b'\n':
                fh.write(b'\n')
        fh.write((json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8'))


def design_hash(path):
    try:
        with open(path, 'rb') as fh:
            return hashlib.sha1(fh.read()).hexdigest()[:12]
    except OSError:
        return ''


def record(r, design_path, settings, problems, run=None):
    """One 'run' line. Best effort: any error is swallowed (it is a note, not part of the job). `run`: an id for
    exactly this run (LoomLab's job id), so a verdict lands on it even when another run has the same name."""
    try:
        row = {'event': 'run', 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'name': r['name'],
               'design': design_hash(design_path), 'features': features(r), 'settings': settings,
               'result': {'match': r['match'], 'design_match': r.get('design_match'), 'areas': r['areas'], 'line_mm': r['line_mm'], 'missed': r['missed'],
                          'tiny': r.get('tiny', 0), 'inks': len(r['inks']), 'snapped': r.get('snapped', {})},
               'problems': [c for c, _ in problems]}
        if run:
            row['run'] = run
        _append(row)
        return True
    except OSError:
        return False


def feedback(name, verdict, note='', run=None, extra=None):
    """The operator's word on the latest run of `name` ('good' or 'bad'), or with `run` on exactly the run recorded
    with that id. `extra`: more facts for the row (who, what was changed after). Returns that run's row, or None."""
    if verdict not in ('good', 'bad'):
        raise ValueError("verdict 'good' ya 'bad' hona chahiye")
    runs = [x for x in _read() if x.get('event') == 'run'
            and (x.get('run') == run if run else x.get('name') == name)]
    if not runs:
        return None
    hit = runs[-1]
    row = {'event': 'feedback', 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'name': hit['name'],
           'design': hit.get('design', ''), 'run_time': hit['time'], 'verdict': verdict, 'note': note}
    if run:
        row['run'] = run
    _append(row | (extra or {}))
    return hit


def _verdicts(rows):
    """The latest word on each run: {(name, run_time) or ('run', id): (verdict, note)} (see `_verdict_of`)."""
    out = {}
    for x in rows:
        if x.get('event') == 'feedback':
            out[('run', x['run']) if x.get('run') else (x['name'], x['run_time'])] = (x['verdict'], x.get('note', ''))
    return out


def _verdict_of(verdicts, row):
    """A run's verdict: by its id when it has one (two runs of one name in one second stay apart), else by name
    and time (a CLI `feedback NAME`)."""
    return verdicts.get(('run', row['run'])) if row.get('run') and ('run', row['run']) in verdicts \
        else verdicts.get((row['name'], row['time']))


def similar(r, k=3):
    """Up to k past runs (not this one) whose design looks like this one: [(distance, row, verdict or None)],
    nearest first. Distance is on the features, each scaled by how much it varies in the log."""
    rows = [x for x in _read() if x.get('event') == 'run' and x.get('name') != r['name']]
    if not rows:
        return []
    keys = ['grain', 'areas_per_mpx', 'inks', 'top_share', 'small_inks']
    mine = features(r)
    mat = np.array([[x['features'][c] for c in keys] for x in rows] + [[mine[c] for c in keys]], float)
    spread = np.maximum(mat.std(0), 1e-6) if len(rows) > 1 else np.maximum(np.abs(mat).max(0), 1e-6)
    d = np.linalg.norm((mat[:-1] - mat[-1]) / spread, axis=1)
    verdicts = _verdicts(_read())
    order = np.argsort(d)[:k]
    return [(round(float(d[i]), 2), rows[i], (_verdict_of(verdicts, rows[i]) or (None,))[0]) for i in order]


def summary():
    rows = _read()
    runs = [x for x in rows if x.get('event') == 'run']
    if not runs:
        return 'Abhi tak koi run record nahi hua. `textile number` chalao, phir yahan dikhega.'
    verdicts = _verdicts(rows)
    lines = [f'{len(runs)} run, {len(verdicts)} par aapki raay (good/bad).']
    count = {}
    for x in runs:
        for c in x.get('problems', []):
            count[c] = count.get(c, 0) + 1
    if count:
        lines.append('Sabse zyada aane wali dikkatein: ' + ', '.join(f'{c} ({n})' for c, n in sorted(count.items(), key=lambda t: -t[1])))
    for v, label in (('good', 'Achhe bane (good)'), ('bad', 'Kharab bane (bad)')):
        sel = [x for x in runs if (_verdict_of(verdicts, x) or (None,))[0] == v]
        if sel:
            lines.append(f"{label}: " + '; '.join(f"{x['name']} (line {x['result']['line_mm']} mm, {x['result']['inks']} rang, match "
                                                  f"{x['result']['match']}%" + (f", dikkat: {','.join(x['problems'])}" if x['problems'] else '') + ')'
                                                  for x in sel))
    return '\n'.join(lines)
