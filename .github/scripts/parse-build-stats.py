"""Parse target/release binaries into a JSON artifact."""
import json
import os
import sys
from pathlib import Path

# --- Binary sizes ---
print("| Binary | Size (bytes) | Size |")
print("|--------|-------------|------|")

binaries = []
release_dir = Path("target/release")
for p in sorted(release_dir.iterdir()):
    if not p.is_file():
        continue
    if p.suffix in (".d", ".rlib", ".rmeta", ".o", ".dwp"):
        continue
    if not os.access(p, os.X_OK):
        continue
    size = p.stat().st_size
    if size < 1024:
        continue
    if size > 1048576:
        human = f"{size / 1048576:.1f} MiB"
    else:
        human = f"{size / 1024:.1f} KiB"
    print(f"| {p.name} | {size} | {human} |")
    binaries.append({"name": p.name, "bytes": size})

Path("binary-sizes.json").write_text(json.dumps(binaries, indent=2))
