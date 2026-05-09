"""Aggregate build-stats artifacts into a JS data file for the HTML report.

Reads from:
  collected/runs.tsv        — tab-separated: run_id, created_at, head_sha
  collected/<run_id>/       — contains cargo-timing.html, build-stats-size.json, cargo-bloat.txt

Usage: python build-stats-report-data.py <collected_dir> <output_dir>
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_cargo_timing(html_path: Path) -> dict:
    """Parse build times from a cargo-timing.html file."""
    if not html_path.exists():
        return {}

    html = html_path.read_text()

    m = re.search(r"DURATION\s*=\s*(\d+(?:\.\d+)?)", html)
    total_s = float(m.group(1)) if m else 0

    m2 = re.search(r"Total time:</td><td>([^<]+)</td>", html)
    total_display = m2.group(1).strip() if m2 else f"{total_s:.1f}s"

    m = re.search(r"const UNIT_DATA\s*=\s*(\[.*?\]);", html, re.DOTALL)
    if not m:
        return {"total_wall_time_s": total_s, "total_time_display": total_display, "units": []}

    units = json.loads(m.group(1))
    units_sorted = sorted(units, key=lambda u: u.get("duration", 0), reverse=True)

    return {
        "total_wall_time_s": total_s,
        "total_time_display": total_display,
        "units": [{"name": u["name"], "version": u.get("version", ""), "duration": u.get("duration", 0)} for u in units_sorted],
    }


def main():
    collected_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("collected")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("report")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_tsv = collected_dir / "runs.tsv"

    # Parse runs.tsv and load per-run artifacts
    runs = []
    for line in runs_tsv.read_text().strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        run_id, created_at, head_sha = parts[0], parts[1], parts[2]
        run_dir = collected_dir / run_id

        timing_path = run_dir / "cargo-timing.html"
        bs_path = run_dir / "build-stats-size.json"
        cb_path = run_dir / "cargo-bloat.txt"

        ll_path = run_dir / "cargo-llvm-lines.txt"

        if not timing_path.exists():
            continue  # skip runs without data

        runs.append({
            "run_id": run_id,
            "created_at": created_at,
            "head_sha": head_sha,
            "build_times": parse_cargo_timing(timing_path),
            "binary_sizes": json.loads(bs_path.read_text()) if bs_path.exists() else [],
            "cargo_bloat": cb_path.read_text() if cb_path.exists() else "",
            "cargo_llvm_lines": ll_path.read_text() if ll_path.exists() else "",
        })

    runs.sort(key=lambda r: r["created_at"])

    dates = []
    total_build_times = []
    crate_times: dict[str, list] = {}
    total_binary_sizes = []
    per_binary: dict[str, list] = {}

    for run in runs:
        dt_str = run.get("created_at", "")
        dates.append(dt_str[:10] if dt_str else "?")

        bt = run.get("build_times", {})
        total_build_times.append(bt.get("total_wall_time_s", 0))

        # Per-crate build times
        units = bt.get("units", [])
        seen = set()
        for u in units:
            name = u.get("name", "")
            if name not in crate_times:
                crate_times[name] = [None] * (len(dates) - 1)
            crate_times[name].append(u.get("duration", 0))
            seen.add(name)
        for name in crate_times:
            if name not in seen:
                crate_times[name].append(None)

        # Binary sizes
        bs = run.get("binary_sizes", [])
        total_binary_sizes.append(sum(b.get("bytes", 0) for b in bs))

        seen_bins = set()
        for b in bs:
            bname = b.get("name", "")
            if bname not in per_binary:
                per_binary[bname] = [None] * (len(dates) - 1)
            per_binary[bname].append(b.get("bytes", 0))
            seen_bins.add(bname)
        for bname in per_binary:
            if bname not in seen_bins:
                per_binary[bname].append(None)

    # Top 15 crates by average duration
    def avg(lst):
        vals = [v for v in lst if v is not None]
        return sum(vals) / len(vals) if vals else 0

    top_crates = sorted(crate_times.keys(), key=lambda c: avg(crate_times[c]), reverse=True)[:15]

    # Latest cargo bloat and llvm-lines
    latest_bloat = next((r["cargo_bloat"] for r in reversed(runs) if r.get("cargo_bloat")), "")
    latest_llvm_lines = next((r["cargo_llvm_lines"] for r in reversed(runs) if r.get("cargo_llvm_lines")), "")

    # Latest run details
    latest_run = None
    if runs:
        last = runs[-1]
        bt = last.get("build_times", {})
        latest_run = {
            "created_at": last.get("created_at", "?"),
            "head_sha": last.get("head_sha", "?")[:12],
            "total_time_display": bt.get("total_time_display", "?"),
            "units": bt.get("units", []),
            "binary_sizes": last.get("binary_sizes", []),
        }

    build_data = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dates": dates,
        "total_build_times": total_build_times,
        "total_binary_sizes_mib": [s / 1048576 for s in total_binary_sizes],
        "crate_datasets": [{"label": name, "data": crate_times[name]} for name in top_crates],
        "binary_datasets": [
            {"label": name, "data": [b / 1048576 if b is not None else None for b in per_binary[name]]}
            for name in sorted(per_binary.keys())
        ],
        "latest_cargo_bloat": latest_bloat,
        "latest_cargo_llvm_lines": latest_llvm_lines,
        "latest_run": latest_run,
    }

    js_path = output_dir / "build-stats-report.js"
    js_path.write_text(f"const BUILD_DATA = {json.dumps(build_data, indent=2)};\n")
    print(f"Generated {js_path} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
