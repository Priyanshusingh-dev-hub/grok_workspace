"""Shared: the old results. reference_code/method1_colorfill.py (the user's
tested script), unchanged, run once per test session on the samples."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# `number` notes every run in the tool's memory (textile/learn.py): tests write to a throwaway log, never to the
# user's own data/number-log.jsonl (its runs and verdicts are what `similar` and `learn` show)
os.environ.setdefault('TEXTILE_LEARN_LOG', os.path.join(tempfile.mkdtemp(prefix='textile-test-log-'), 'number-log.jsonl'))
SAMPLES = ROOT / 'tests' / 'samples'
TAG = '3535px_300dpi'
RUNS = {  # how each sample was run with method1 (and what it gave)
    'floral': (['--line', 'floral_lineart.png', '--ref', 'floral_ref.png'], 9),
    'star': (['--line', 'star_lineart.png', '--ref', 'star_ref.png', '--max-colors', '6', '--line-color', '120F06'], 6),
}

@pytest.fixture(scope='session')
def reference(tmp_path_factory):
    """method1_colorfill.py, unchanged, on each sample: the old results."""
    out = {}
    for name, (args, _) in RUNS.items():
        d = tmp_path_factory.mktemp(f'ref_{name}')
        cmd = [sys.executable, '-W', 'ignore', str(ROOT / 'reference_code' / 'method1_colorfill.py'),
               *[str(SAMPLES / a) if a.endswith('.png') else a for a in args], '--out', str(d), '--name', name]
        subprocess.run(cmd, check=True, capture_output=True)
        out[name] = d
    return out




@pytest.fixture(scope='session')
def prototype2(tmp_path_factory):
    """method2_structure_fill_PROTOTYPE.py, unchanged but for its two input
    paths, on the tree sample: writes cream.npy and depth.png where it runs."""
    d = tmp_path_factory.mktemp('proto2')
    src = (ROOT / 'reference_code' / 'method2_structure_fill_PROTOTYPE.py').read_text()
    src = (src.replace("'/mnt/user-data/uploads/23551.png'", repr(str(SAMPLES / 'tree_lineart.png')))
              .replace("'/mnt/user-data/uploads/23534.png'", repr(str(SAMPLES / 'tree_ref.png'))))
    (d / 'run.py').write_text(src)
    subprocess.run([sys.executable, 'run.py'], cwd=d, check=True, capture_output=True)
    return d
