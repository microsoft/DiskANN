#!/usr/bin/env bash
# SIFT1M Pipelined Search Benchmark
#
# Downloads SIFT1M dataset, builds a disk index, and runs an ablation
# benchmark comparing BeamSearch vs PipeSearch (io_uring pipelining).
#
# Prerequisites:
#   - Linux (PipeSearch requires io_uring)
#   - Rust toolchain (cargo)
#   - curl, tar, python3 with numpy
#   - ~2GB free disk space for data + index
#
# Usage:
#   ./diskann-benchmark/scripts/sift1m_benchmark.sh [--data-dir DIR] [--skip-download] [--skip-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
DATA_DIR="${DATA_DIR:-$REPO_ROOT/benchmark_data/sift1m}"
NUM_THREADS="${NUM_THREADS:-4}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"
SKIP_DOWNLOAD=false
SKIP_BUILD=false
SKIP_INDEX=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --skip-index) SKIP_INDEX=true; shift ;;
        --threads)    NUM_THREADS="$2"; shift 2 ;;
        --beam-width) BEAM_WIDTH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data-dir DIR     Data directory (default: \$REPO_ROOT/benchmark_data/sift1m)"
            echo "  --skip-download    Skip downloading SIFT1M (use existing data)"
            echo "  --skip-build       Skip building the benchmark binary"
            echo "  --skip-index       Skip building the disk index (use existing index)"
            echo "  --threads N        Number of search threads (default: 4)"
            echo "  --beam-width N     Beam width / pipeline width (default: 4)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BIN_DIR="$DATA_DIR/bin"
INDEX_DIR="$DATA_DIR/index"
INDEX_PREFIX="$INDEX_DIR/sift1m_R64_L100"
CONFIG_FILE="$DATA_DIR/benchmark_config.json"
OUTPUT_FILE="$DATA_DIR/benchmark_results.json"

echo "=== SIFT1M Pipelined Search Benchmark ==="
echo "Data directory: $DATA_DIR"
echo "Threads: $NUM_THREADS, Beam width: $BEAM_WIDTH"
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
    """Read .fvecs format: [dim(int32), vec(float32*dim)] per row."""
    data = np.fromfile(path, dtype=np.float32)
    dim = int(data[0].view(np.int32))
    return data.reshape(-1, dim + 1)[:, 1:]

def read_ivecs(path):
    """Read .ivecs format: [dim(int32), vec(int32*dim)] per row."""
    data = np.fromfile(path, dtype=np.int32)
    dim = data[0]
    return data.reshape(-1, dim + 1)[:, 1:]

def write_fbin(path, data):
    """Write DiskANN .fbin format: [npts(u32), dim(u32), data(float32*npts*dim)]."""
    npts, dim = data.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('II', npts, dim))
        data.astype(np.float32).tofile(f)

def write_ibin(path, data):
    """Write DiskANN groundtruth .bin: [npts(u32), dim(u32), data(uint32*npts*dim)]."""
    npts, dim = data.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('II', npts, dim))
        data.astype(np.uint32).tofile(f)

# Convert base vectors
base = read_fvecs(src_dir / "sift_base.fvecs")
print(f"  Base: {base.shape[0]} points, {base.shape[1]} dims")
write_fbin(dst_dir / "sift_base.fbin", base)

# Convert query vectors
query = read_fvecs(src_dir / "sift_query.fvecs")
print(f"  Query: {query.shape[0]} points, {query.shape[1]} dims")
write_fbin(dst_dir / "sift_query.fbin", query)

# Convert ground truth (take top-100)
gt = read_ivecs(src_dir / "sift_groundtruth.ivecs")
print(f"  Groundtruth: {gt.shape[0]} queries, top-{gt.shape[1]}")
write_ibin(dst_dir / "sift_groundtruth.bin", gt)

print("  Conversion complete!")
PYEOF

        # Clean up extracted files
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
                    "num_threads": $NUM_THREADS,
                    "build_ram_limit_gb": 4.0,
                    "num_pq_chunks": 16,
                    "quantization_type": "FP",
                    "save_path": "$INDEX_PREFIX"
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
# Step 4: Run ablation benchmark (BeamSearch vs PipeSearch)
# -------------------------------------------------------------------
echo "--- Step 4: Running ablation benchmark ---"
cat > "$CONFIG_FILE" << CFGEOF
{
    "search_directories": ["$BIN_DIR"],
    "jobs": [
        {
            "type": "disk-index",
            "content": {
                "source": {
                    "disk-index-source": "Load",
                    "data_type": "float32",
                    "load_path": "$INDEX_PREFIX"
                },
                "search_phase": {
                    "queries": "sift_query.fbin",
                    "groundtruth": "sift_groundtruth.bin",
                    "search_list": [10, 20, 50, 100],
                    "beam_width": $BEAM_WIDTH,
                    "recall_at": 10,
                    "num_threads": $NUM_THREADS,
                    "is_flat_search": false,
                    "distance": "squared_l2",
                    "search_mode": {"mode": "BeamSearch"}
                }
            }
        },
        {
            "type": "disk-index",
            "content": {
                "source": {
                    "disk-index-source": "Load",
                    "data_type": "float32",
                    "load_path": "$INDEX_PREFIX"
                },
                "search_phase": {
                    "queries": "sift_query.fbin",
                    "groundtruth": "sift_groundtruth.bin",
                    "search_list": [10, 20, 50, 100],
                    "beam_width": $BEAM_WIDTH,
                    "recall_at": 10,
                    "num_threads": $NUM_THREADS,
                    "is_flat_search": false,
                    "distance": "squared_l2",
                    "search_mode": {"mode": "PipeSearch"}
                }
            }
        }
    ]
}
CFGEOF

"$BENCHMARK_BIN" run --input-file "$CONFIG_FILE" --output-file "$OUTPUT_FILE"

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "To re-run with different parameters:"
echo "  $0 --skip-download --skip-index --threads N --beam-width N"
