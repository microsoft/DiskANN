"""Scan target/release for executable binaries and write build-stats-size.json."""
import json
import os
from pathlib import Path

binaries = []
for p in sorted(Path("target/release").iterdir()):
    if not p.is_file() or p.suffix in (".d", ".rlib", ".rmeta", ".o", ".dwp"):
        continue
    if not os.access(p, os.X_OK) or p.stat().st_size < 1024:
        continue
    binaries.append({"name": p.name, "bytes": p.stat().st_size})

Path("build-stats-size.json").write_text(json.dumps(binaries, indent=2))
