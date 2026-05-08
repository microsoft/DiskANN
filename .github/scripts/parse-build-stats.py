"""Parse cargo-timing.html and target/release binaries into JSON artifacts."""
import json
import os
import re
import sys
from pathlib import Path

html_path = Path("target/cargo-timings/cargo-timing.html")
if not html_path.exists():
    print("::warning::cargo-timing.html not found")
    sys.exit(0)

html = html_path.read_text()

# --- Build times ---
m = re.search(r"DURATION\s*=\s*(\d+(?:\.\d+)?)", html)
total_s = float(m.group(1)) if m else 0

m2 = re.search(r"Total time:</td><td>([^<]+)</td>", html)
total_display = m2.group(1).strip() if m2 else f"{total_s:.1f}s"

m = re.search(r"const UNIT_DATA\s*=\s*(\[.*?\]);", html, re.DOTALL)
if not m:
    print("::warning::Could not parse UNIT_DATA from timing report")
    sys.exit(0)

units = json.loads(m.group(1))
units_sorted = sorted(units, key=lambda u: u.get("duration", 0), reverse=True)

# Print markdown table
print(f"\n### Release Build Times (Total wall time: {total_display})\n")
print("| # | Crate | Version | Duration |")
print("|---|-------|---------|----------|")
for i, u in enumerate(units_sorted, 1):
    print(f"| {i} | {u.get('name', '?')} | {u.get('version', '?')} | {u.get('duration', 0):.1f}s |")

Path("build-times.json").write_text(json.dumps({
    "total_wall_time_s": total_s,
    "total_time_display": total_display,
    "units": [{"name": u["name"], "version": u.get("version", ""), "duration": u.get("duration", 0)} for u in units_sorted],
}, indent=2))

# --- Binary sizes ---
print("\n### Release Binary Sizes\n")
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
