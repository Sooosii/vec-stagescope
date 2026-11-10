import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 2_000_000


def test_no_challenge_data_tracked():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.split()
    assert not [f for f in files if f.endswith((".h5ad", ".h5", ".loom", ".npz"))]
    assert not [f for f in files if (ROOT / f).is_file() and (ROOT / f).stat().st_size > MAX_BYTES]
