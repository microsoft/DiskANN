"""Scan a release directory for executable binaries and write a JSON size report.

Usage: python build-stats-size.py <release_dir> <output_file>
"""
import json
import os
import sys
from pathlib import Path

release_dir = Path(sys.argv[1])
output_file = Path(sys.argv[2])

binaries = []
for p in sorted(release_dir.iterdir()):
    if not p.is_file() or p.suffix in (".d", ".rlib", ".rmeta", ".o", ".dwp"):
        continue
    if not os.access(p, os.X_OK) or p.stat().st_size < 1024:
        continue
    binaries.append({"name": p.name, "bytes": p.stat().st_size})

output_file.write_text(json.dumps(binaries, indent=2))
