"""
scripts/cleanup_temp.py
Remove leftover tmp_diadub folders under an output directory.
"""
import shutil
from pathlib import Path
import sys

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
tmp = p / "tmp_diadub"
if tmp.exists():
    shutil.rmtree(tmp)
    print("Removed", tmp)
else:
    print("Not found", tmp)
