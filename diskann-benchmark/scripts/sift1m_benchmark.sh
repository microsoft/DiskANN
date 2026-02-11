#!/usr/bin/env bash
# SIFT1M Pipelined Search Benchmark
#
# Downloads SIFT1M dataset, builds a disk index, and runs an ablation
# benchmark comparing BeamSearch vs PipeSearch (io_uring pipelining).
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
SKIP_DOWNLOAD=false
SKIP_BUILD=false
SKIP_INDEX=false
SQPOLL_MS=""

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
        --sqpoll)        SQPOLL_MS="${2:-1000}"; shift 2 ;;
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
            echo "  --sqpoll MS          Enable SQPOLL on all PipeSearch configs (idle timeout in ms)"
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

# Generate a single config with all jobs (4 per thread count for ablation)
# Modes: BeamSearch, PipeSearch (base), PipeSearch+ABW, PipeSearch+ABW+RelaxedMono
JOBS=""
add_job() {
    local threads="$1" mode_json="$2"
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
                    \"num_threads\": $threads,
                    \"is_flat_search\": false,
                    \"distance\": \"squared_l2\",
                    \"search_mode\": $mode_json
                }
            }
        }"
}
# Optional SQPOLL suffix for PipeSearch configs
SQPOLL_JSON=""
if [ -n "$SQPOLL_MS" ]; then
    SQPOLL_JSON=", \"sqpoll_idle_ms\": $SQPOLL_MS"
fi
for T in $THREAD_LIST; do
    add_job "$T" '{"mode": "BeamSearch"}'
    add_job "$T" '{"mode": "PipeSearch", "adaptive_beam_width": false'"$SQPOLL_JSON"'}'
    add_job "$T" '{"mode": "PipeSearch", "adaptive_beam_width": true'"$SQPOLL_JSON"'}'
    add_job "$T" '{"mode": "PipeSearch", "adaptive_beam_width": true, "relaxed_monotonicity_l": '"$SEARCH_L$SQPOLL_JSON"'}'
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

python3 - "$SWEEP_OUTPUT" "$OUTPUT_DIR" "$SEARCH_L" "$BEAM_WIDTH" << 'CHARTEOF'
import json, sys, os

output_dir = sys.argv[2]
search_l = sys.argv[3]
beam_width = sys.argv[4]

with open(sys.argv[1]) as f:
    data = json.load(f)

# Classify each job into one of 4 modes based on search_mode string
# BeamSearch, PipeSearch (no abw), PipeSearch(abw), PipeSearch(abw, rm_l=N)
MODE_KEYS = [
    ("BeamSearch",      "BeamSearch"),
    ("PipeSearch(base)","PipeSearch"),      # no abw, no rm
    ("PipeSearch+ABW",  "PipeSearch(abw)"), # abw only
    ("PipeSearch+ABW+RM","PipeSearch(abw, rm_l="),  # abw + relaxed mono
]

def classify_mode(mode_str):
    s = str(mode_str)
    if "BeamSearch" in s:
        return "BeamSearch"
    if "rm_l=" in s:
        return "PipeSearch+ABW+RM"
    if "abw" in s:
        return "PipeSearch+ABW"
    # PipeSearch with no options or empty parens
    return "PipeSearch(base)"

def empty_series():
    return {"threads": [], "qps": [], "mean_lat": [], "p95_lat": [], "p999_lat": [], "recall": []}

series = {k: empty_series() for k, _ in MODE_KEYS}

for job in data:
    search = job.get("results", {}).get("search", {})
    if not search:
        continue
    results_per_l = search.get("search_results_per_l", [])
    if not results_per_l:
        continue
    r = results_per_l[0]
    threads = search.get("num_threads", 0)
    mode = classify_mode(search.get("search_mode", ""))
    d = series.get(mode)
    if d is None:
        continue
    d["threads"].append(threads)
    d["qps"].append(r.get("qps", 0))
    d["mean_lat"].append(r.get("mean_latency", 0))
    d["p95_lat"].append(r.get("p95_latency", 0))
    d["p999_lat"].append(r.get("p999_latency", 0))
    d["recall"].append(r.get("recall", 0))

# Sort each series by threads
for d in series.values():
    if d["threads"]:
        order = sorted(range(len(d["threads"])), key=lambda i: d["threads"][i])
        for k in d:
            d[k] = [d[k][i] for i in order]

# Print table
header_modes = list(series.keys())
print(f"\n{'Threads':>7s}", end="")
for m in header_modes:
    print(f"  {m+' QPS':>18s}", end="")
print()
print(f"{'':>7s}", end="")
for m in header_modes:
    print(f"  {'recall':>8s} {'mean':>8s} {'p999':>8s}", end="")
print()
print("=" * (7 + len(header_modes) * 30))

max_len = max(len(series[m]["threads"]) for m in header_modes)
for i in range(max_len):
    t = None
    for m in header_modes:
        if i < len(series[m]["threads"]):
            t = series[m]["threads"][i]
            break
    print(f"{t or 0:7d}", end="")
    for m in header_modes:
        d = series[m]
        if i < len(d["threads"]):
            print(f"  {d['qps'][i]:8.0f}qps {d['recall'][i]:6.1f}% {d['mean_lat'][i]/1000:5.1f}ms {d['p999_lat'][i]/1000:5.1f}ms", end="")
        else:
            print(f"  {'N/A':>30s}", end="")
    print()

# Chart styles per mode
STYLES = {
    "BeamSearch":         {"color": "#2196F3", "marker": "o", "ls": "-"},
    "PipeSearch(base)":   {"color": "#FF9800", "marker": "D", "ls": "--"},
    "PipeSearch+ABW":     {"color": "#FF5722", "marker": "s", "ls": "-"},
    "PipeSearch+ABW+RM":  {"color": "#4CAF50", "marker": "^", "ls": "-"},
}

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SIFT1M Ablation (L={search_l}, BW={beam_width})', fontsize=14)

    metrics = [
        (axes[0][0], "qps",      "QPS",                1,    False),
        (axes[0][1], "mean_lat", "Mean Latency (ms)",   1000, True),
        (axes[1][0], "p95_lat",  "p95 Latency (ms)",    1000, True),
        (axes[1][1], "p999_lat", "p99.9 Latency (ms)",  1000, True),
    ]

    for ax, key, title, divisor, is_latency in metrics:
        for mode_name, d in series.items():
            if not d["threads"]:
                continue
            st = STYLES.get(mode_name, {"color": "gray", "marker": ".", "ls": "-"})
            vals = [v / divisor for v in d[key]] if divisor != 1 else d[key]
            ax.plot(d["threads"], vals, marker=st["marker"], linestyle=st["ls"],
                    color=st["color"], label=mode_name, linewidth=2, markersize=5)
        ax.set_xlabel('Threads')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
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
    for mode_name, d in series.items():
        for i in range(len(d["threads"])):
            f.write(f"{d['threads'][i]},{mode_name},{d['qps'][i]:.1f},"
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
echo "  $0 --skip-download --skip-index --max-threads N --search-l N"
