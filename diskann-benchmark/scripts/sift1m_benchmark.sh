#!/usr/bin/env bash
# SIFT1M Pipelined Search Benchmark
#
# Downloads SIFT1M dataset, builds a disk index, and runs an ablation
# benchmark comparing BeamSearch vs PipeSearch (io_uring pipelining)
# with optional adaptive beam width (ABW) and relaxed monotonicity (RM).
#
# By default, sweeps thread counts from 1 to max_threads in strides of 4
# and produces charts (QPS, mean latency, tail latency vs threads).
#
# Prerequisites:
#   - Linux (PipeSearch requires io_uring)
#   - Rust toolchain (cargo)
#   - curl, tar, python3 with numpy and matplotlib
#   - ~2GB free disk space for data + index
#
# Usage:
#   ./diskann-benchmark/scripts/sift1m_benchmark.sh [OPTIONS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
DATA_DIR="${DATA_DIR:-$REPO_ROOT/benchmark_data/sift1m}"
MAX_THREADS="${MAX_THREADS:-48}"
THREAD_STRIDE="${THREAD_STRIDE:-4}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"
SEARCH_L="${SEARCH_L:-100}"
ABW=false
RM_L=""
SKIP_DOWNLOAD=false
SKIP_BUILD=false
SKIP_INDEX=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)      DATA_DIR="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --skip-build)    SKIP_BUILD=true; shift ;;
        --skip-index)    SKIP_INDEX=true; shift ;;
        --max-threads)   MAX_THREADS="$2"; shift 2 ;;
        --thread-stride) THREAD_STRIDE="$2"; shift 2 ;;
        --beam-width)    BEAM_WIDTH="$2"; shift 2 ;;
        --search-l)      SEARCH_L="$2"; shift 2 ;;
        --abw)           ABW=true; shift ;;
        --rm-l)          RM_L="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data-dir DIR       Data directory (default: \$REPO_ROOT/benchmark_data/sift1m)"
            echo "  --skip-download      Skip downloading SIFT1M (use existing data)"
            echo "  --skip-build         Skip building the benchmark binary"
            echo "  --skip-index         Skip building the disk index (use existing index)"
            echo "  --max-threads N      Maximum thread count for sweep (default: 48)"
            echo "  --thread-stride N    Thread count increment (default: 4)"
            echo "  --beam-width N       Beam width / pipeline width (default: 4)"
            echo "  --search-l N         Search list size L (default: 100)"
            echo "  --abw                Enable adaptive beam width for ABW variants"
            echo "  --rm-l N             Enable relaxed monotonicity with budget N for RM variants"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BIN_DIR="$DATA_DIR/bin"
INDEX_DIR="$DATA_DIR/index"
INDEX_PREFIX="$INDEX_DIR/sift1m_R64_L100"
OUTPUT_DIR="$DATA_DIR/results"

echo "=== SIFT1M Pipelined Search Benchmark ==="
echo "Data directory: $DATA_DIR"
echo "Thread sweep: 1, 4..${MAX_THREADS} (stride ${THREAD_STRIDE})"
echo "Beam width: $BEAM_WIDTH, Search L: $SEARCH_L"
echo "Adaptive beam width: $ABW"
[ -n "$RM_L" ] && echo "Relaxed monotonicity L: $RM_L"
echo ""

# -------------------------------------------------------------------
# Step 1: Download SIFT1M
# -------------------------------------------------------------------
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "--- Step 1: Downloading SIFT1M dataset ---"
    mkdir -p "$BIN_DIR"

    SIFT_URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    SIFT_TAR="$DATA_DIR/sift.tar.gz"

    if [ ! -f "$BIN_DIR/sift_base.fbin" ]; then
        if [ ! -f "$SIFT_TAR" ]; then
            echo "Downloading from $SIFT_URL ..."
            curl -L -o "$SIFT_TAR" "$SIFT_URL"
        fi

        echo "Extracting..."
        EXTRACT_DIR="$DATA_DIR/extract"
        mkdir -p "$EXTRACT_DIR"
        tar xzf "$SIFT_TAR" -C "$EXTRACT_DIR"

        echo "Converting .bvecs/.fvecs to .fbin format..."
        python3 - "$EXTRACT_DIR/sift" "$BIN_DIR" << 'PYEOF'
import sys, struct, numpy as np
from pathlib import Path

src_dir = Path(sys.argv[1])
dst_dir = Path(sys.argv[2])

def read_fvecs(path):
    data = np.fromfile(path, dtype=np.float32)
    dim = int(data[0].view(np.int32))
    return data.reshape(-1, dim + 1)[:, 1:]

def read_ivecs(path):
    data = np.fromfile(path, dtype=np.int32)
    dim = data[0]
    return data.reshape(-1, dim + 1)[:, 1:]

def write_fbin(path, data):
    npts, dim = data.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('II', npts, dim))
        data.astype(np.float32).tofile(f)

def write_ibin(path, data):
    npts, dim = data.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('II', npts, dim))
        data.astype(np.uint32).tofile(f)

base = read_fvecs(src_dir / "sift_base.fvecs")
print(f"  Base: {base.shape[0]} points, {base.shape[1]} dims")
write_fbin(dst_dir / "sift_base.fbin", base)

query = read_fvecs(src_dir / "sift_query.fvecs")
print(f"  Query: {query.shape[0]} points, {query.shape[1]} dims")
write_fbin(dst_dir / "sift_query.fbin", query)

gt = read_ivecs(src_dir / "sift_groundtruth.ivecs")
print(f"  Groundtruth: {gt.shape[0]} queries, top-{gt.shape[1]}")
write_ibin(dst_dir / "sift_groundtruth.bin", gt)
print("  Conversion complete!")
PYEOF

        rm -rf "$EXTRACT_DIR" "$SIFT_TAR"
    else
        echo "SIFT1M data already exists at $BIN_DIR, skipping download."
    fi
    echo ""
fi

# -------------------------------------------------------------------
# Step 2: Build the benchmark binary
# -------------------------------------------------------------------
if [ "$SKIP_BUILD" = false ]; then
    echo "--- Step 2: Building diskann-benchmark ---"
    cd "$REPO_ROOT"
    cargo build --release -p diskann-benchmark --features disk-index 2>&1 | tail -3
    echo ""
fi

BENCHMARK_BIN="$REPO_ROOT/target/release/diskann-benchmark"
if [ ! -x "$BENCHMARK_BIN" ]; then
    echo "ERROR: benchmark binary not found at $BENCHMARK_BIN"
    echo "Run without --skip-build or build manually:"
    echo "  cargo build --release -p diskann-benchmark --features disk-index"
    exit 1
fi

# -------------------------------------------------------------------
# Step 3: Build disk index (if needed)
# -------------------------------------------------------------------
if [ "$SKIP_INDEX" = false ] && [ ! -f "${INDEX_PREFIX}_disk.index" ]; then
    echo "--- Step 3: Building disk index (R=64, L=100, PQ_16) ---"
    mkdir -p "$INDEX_DIR"

    # Build job requires a search_phase; we include a minimal one that also
    # validates the index works after building.
    cat > "$DATA_DIR/build_config.json" << BUILDEOF
{
    "search_directories": ["$BIN_DIR"],
    "jobs": [
        {
            "type": "disk-index",
            "content": {
                "source": {
                    "disk-index-source": "Build",
                    "data_type": "float32",
                    "data": "sift_base.fbin",
                    "distance": "squared_l2",
                    "dim": 128,
                    "max_degree": 64,
                    "l_build": 100,
                    "num_threads": 4,
                    "build_ram_limit_gb": 4.0,
                    "num_pq_chunks": 16,
                    "quantization_type": "FP",
                    "save_path": "$INDEX_PREFIX"
                },
                "search_phase": {
                    "queries": "sift_query.fbin",
                    "groundtruth": "sift_groundtruth.bin",
                    "search_list": [50],
                    "beam_width": 4,
                    "recall_at": 10,
                    "num_threads": 1,
                    "is_flat_search": false,
                    "distance": "squared_l2"
                }
            }
        }
    ]
}
BUILDEOF

    "$BENCHMARK_BIN" run --input-file "$DATA_DIR/build_config.json" --output-file /dev/null
    echo ""
elif [ "$SKIP_INDEX" = true ]; then
    echo "--- Step 3: Skipping index build (--skip-index) ---"
    echo ""
else
    echo "--- Step 3: Disk index already exists, skipping build ---"
    echo ""
fi

if [ ! -f "${INDEX_PREFIX}_disk.index" ]; then
    echo "ERROR: Disk index not found at ${INDEX_PREFIX}_disk.index"
    exit 1
fi

# -------------------------------------------------------------------
# Step 4: Thread sweep benchmark
# -------------------------------------------------------------------
echo "--- Step 4: Running thread sweep benchmark ---"
mkdir -p "$OUTPUT_DIR"

# Build thread list: 1, then 4, 8, ..., MAX_THREADS
THREAD_LIST="1"
for (( t=THREAD_STRIDE; t<=MAX_THREADS; t+=THREAD_STRIDE )); do
    THREAD_LIST="$THREAD_LIST $t"
done
echo "Thread counts: $THREAD_LIST"

# Build the list of search mode configurations to benchmark.
# Always includes baseline BeamSearch and PipeSearch.
# When --abw or --rm-l are specified, adds ABW and/or ABW+RM variants for both.
MODES=()

# Helper: build a search_mode JSON fragment
beam_mode() {
    local abw="${1:-false}"
    local rm="${2:-}"
    local json="{\"mode\": \"BeamSearch\", \"adaptive_beam_width\": $abw"
    [ -n "$rm" ] && json="$json, \"relaxed_monotonicity_l\": $rm"
    echo "$json}"
}
pipe_mode() {
    local abw="${1:-true}"
    local rm="${2:-}"
    local json="{\"mode\": \"PipeSearch\", \"adaptive_beam_width\": $abw"
    [ -n "$rm" ] && json="$json, \"relaxed_monotonicity_l\": $rm"
    echo "$json}"
}

# Baseline (no ABW, no RM)
MODES+=("$(beam_mode false)")
MODES+=("$(pipe_mode false)")

# ABW variants
if [ "$ABW" = true ]; then
    MODES+=("$(beam_mode true)")
    MODES+=("$(pipe_mode true)")
fi

# ABW+RM variants (RM requires ABW for the convergence gate to work)
if [ -n "$RM_L" ]; then
    MODES+=("$(beam_mode true "$RM_L")")
    MODES+=("$(pipe_mode true "$RM_L")")
fi

NUM_MODES=${#MODES[@]}
echo "Search modes ($NUM_MODES per thread count):"
for m in "${MODES[@]}"; do echo "  $m"; done

# Generate a single config with all jobs
JOBS=""
for T in $THREAD_LIST; do
    for MODE_JSON in "${MODES[@]}"; do
        [ -n "$JOBS" ] && JOBS="$JOBS,"
        JOBS="$JOBS
        {
            \"type\": \"disk-index\",
            \"content\": {
                \"source\": {
                    \"disk-index-source\": \"Load\",
                    \"data_type\": \"float32\",
                    \"load_path\": \"$INDEX_PREFIX\"
                },
                \"search_phase\": {
                    \"queries\": \"sift_query.fbin\",
                    \"groundtruth\": \"sift_groundtruth.bin\",
                    \"search_list\": [$SEARCH_L],
                    \"beam_width\": $BEAM_WIDTH,
                    \"recall_at\": 10,
                    \"num_threads\": $T,
                    \"is_flat_search\": false,
                    \"distance\": \"squared_l2\",
                    \"search_mode\": $MODE_JSON
                }
            }
        }"
    done
done

SWEEP_CONFIG="$OUTPUT_DIR/sweep_config.json"
SWEEP_OUTPUT="$OUTPUT_DIR/sweep_results.json"

cat > "$SWEEP_CONFIG" << SWEEPEOF
{
    "search_directories": ["$BIN_DIR"],
    "jobs": [$JOBS
    ]
}
SWEEPEOF

"$BENCHMARK_BIN" run --input-file "$SWEEP_CONFIG" --output-file "$SWEEP_OUTPUT"

echo ""
echo "--- Step 5: Generating charts ---"

python3 - "$SWEEP_OUTPUT" "$OUTPUT_DIR" "$SEARCH_L" "$BEAM_WIDTH" "$ABW" "$RM_L" << 'CHARTEOF'
import json, sys, os
from collections import defaultdict

output_dir = sys.argv[2]
search_l = sys.argv[3]
beam_width = sys.argv[4]
abw_flag = sys.argv[5]
rm_l = sys.argv[6] if len(sys.argv) > 6 else ""

with open(sys.argv[1]) as f:
    data = json.load(f)

# Parse results into per-mode buckets keyed by the search_mode display string.
# Each bucket holds lists of (threads, qps, mean_lat, p95_lat, p999_lat, recall).
modes = defaultdict(lambda: {"threads": [], "qps": [], "mean_lat": [],
                              "p95_lat": [], "p999_lat": [], "recall": []})

for job in data:
    search = job.get("results", {}).get("search", {})
    if not search:
        continue
    results_per_l = search.get("search_results_per_l", [])
    if not results_per_l:
        continue
    r = results_per_l[0]
    threads = search.get("num_threads", 0)
    mode = str(search.get("search_mode", ""))

    d = modes[mode]
    d["threads"].append(threads)
    d["qps"].append(r.get("qps", 0))
    d["mean_lat"].append(r.get("mean_latency", 0))
    d["p95_lat"].append(r.get("p95_latency", 0))
    d["p999_lat"].append(r.get("p999_latency", 0))
    d["recall"].append(r.get("recall", 0))

# Sort each mode by threads
for d in modes.values():
    if d["threads"]:
        order = sorted(range(len(d["threads"])), key=lambda i: d["threads"][i])
        for k in d:
            d[k] = [d[k][i] for i in order]

# Assign short labels and colors
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4']
mode_names = sorted(modes.keys())
labels = {}
for name in mode_names:
    # Build a concise label from the search_mode string
    if "BeamSearch" in name:
        lbl = "Beam"
    elif "PipeSearch" in name:
        lbl = "Pipe"
    else:
        lbl = name[:10]
    if "abw" in name.lower() or "adaptive_beam_width: true" in name.lower():
        lbl += "+ABW"
    if "rm_l=" in name.lower() or "relaxed_monotonicity_l: Some" in name:
        lbl += "+RM"
    labels[name] = lbl

# Print table
header = f"{'Threads':>7s}"
for name in mode_names:
    lbl = labels[name]
    header += f"  {lbl+' QPS':>14s}"
header += " "
for name in mode_names:
    lbl = labels[name]
    header += f"  {lbl+' Recall':>12s}"
print(f"\n{header}")
print("=" * len(header))

max_rows = max(len(modes[n]["threads"]) for n in mode_names)
for i in range(max_rows):
    row = ""
    t = 0
    for name in mode_names:
        d = modes[name]
        if i < len(d["threads"]):
            t = d["threads"][i]
            row += f"  {d['qps'][i]:14.1f}"
        else:
            row += f"  {'':>14s}"
    row += " "
    for name in mode_names:
        d = modes[name]
        if i < len(d["threads"]):
            row += f"  {d['recall'][i]:11.2f}%"
        else:
            row += f"  {'':>12s}"
    print(f"{t:7d}{row}")

# Build plot title
title_parts = [f"L={search_l}", f"BW={beam_width}"]
if abw_flag == "true":
    title_parts.append("ABW")
if rm_l:
    title_parts.append(f"RM_L={rm_l}")
plot_title = f"SIFT1M Search Benchmark ({', '.join(title_parts)})"

# Generate charts
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(plot_title, fontsize=14)

    metrics = [
        (axes[0][0], "qps",      "QPS",              "Throughput (QPS)",    1,    False),
        (axes[0][1], "mean_lat",  "Mean Latency (ms)", "Mean Latency",      1000, True),
        (axes[1][0], "p95_lat",   "p95 Latency (ms)",  "p95 Tail Latency",  1000, True),
        (axes[1][1], "p999_lat",  "p99.9 Latency (ms)","p99.9 Tail Latency",1000, True),
    ]

    markers = ['o', 's', '^', 'D', 'v', 'P']
    for ax, key, ylabel, title, divisor, _ in metrics:
        for idx, name in enumerate(mode_names):
            d = modes[name]
            vals = [x / divisor for x in d[key]]
            ax.plot(d["threads"], vals,
                    f'{markers[idx % len(markers)]}-',
                    color=COLORS[idx % len(COLORS)],
                    label=labels[name], linewidth=2, markersize=5)
        ax.set_xlabel('Threads')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'thread_sweep.png')
    plt.savefig(chart_path, dpi=150)
    print(f"\nChart saved to: {chart_path}")
    plt.close()

except ImportError:
    print("\nmatplotlib not available — skipping chart generation.")
    print("Install with: pip install matplotlib")

# Save CSV for external plotting
csv_path = os.path.join(output_dir, 'thread_sweep.csv')
with open(csv_path, 'w') as f:
    f.write("threads,mode,qps,mean_lat_us,p95_lat_us,p999_lat_us,recall\n")
    for name in mode_names:
        d = modes[name]
        lbl = labels[name]
        for i in range(len(d["threads"])):
            f.write(f"{d['threads'][i]},{lbl},{d['qps'][i]:.1f},"
                    f"{d['mean_lat'][i]:.0f},{d['p95_lat'][i]},"
                    f"{d['p999_lat'][i]},{d['recall'][i]:.3f}\n")
print(f"CSV saved to: {csv_path}")
CHARTEOF

echo ""
echo "=== Benchmark Complete ==="
echo "Results: $SWEEP_OUTPUT"
echo "Charts:  $OUTPUT_DIR/thread_sweep.png"
echo "CSV:     $OUTPUT_DIR/thread_sweep.csv"
echo ""
echo "To re-run with different parameters:"
echo "  $0 --skip-download --skip-index --max-threads N --search-l N --abw --rm-l N"
