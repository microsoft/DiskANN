#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

if [ -d "build/apps" ]; then
  BASE_PATH="build/apps"
elif [ -d "build/tests" ]; then
  BASE_PATH="build/tests"
else
  echo "Error: could not find build outputs under build/apps or build/tests" >&2
  exit 2
fi

OUT_DIR=${OUT_DIR:-"$SCRIPT_DIR/_out"}
NPTS=${NPTS:-5000}
NQ=${NQ:-200}
DIM=${DIM:-64}
K=${K:-10}
MEM_L=${MEM_L:-50}
DISK_L=${DISK_L:-50}
THREADS=${THREADS:-4}

DISK_R=${DISK_R:-16}
DISK_LBUILD=${DISK_LBUILD:-50}
DISK_SEARCH_DRAM_BUDGET_GB=${DISK_SEARCH_DRAM_BUDGET_GB:-0.25}
DISK_BUILD_DRAM_BUDGET_GB=${DISK_BUILD_DRAM_BUDGET_GB:-2}
DISK_NUM_NODES_TO_CACHE=${DISK_NUM_NODES_TO_CACHE:-0}
DISK_BEAMWIDTH=${DISK_BEAMWIDTH:-2}
DISK_PQ_BYTES=${DISK_PQ_BYTES:-8}

mkdir -p "$OUT_DIR/data" "$OUT_DIR/results"

BASE_F32="$OUT_DIR/data/base_f32.bin"
QUERY_F32="$OUT_DIR/data/query_f32.bin"
BASE_BF16="$OUT_DIR/data/base_bf16.bin"
QUERY_BF16="$OUT_DIR/data/query_bf16.bin"
GT_PREFIX="$OUT_DIR/data/gt_l2"
GT_BIN="$GT_PREFIX"

MEM_F32_PREFIX="$OUT_DIR/results/mem_f32"
MEM_BF16_PREFIX="$OUT_DIR/results/mem_bf16"

DISK_F32_FULL_PREFIX="$OUT_DIR/results/disk_f32_full"
DISK_BF16_FULL_PREFIX="$OUT_DIR/results/disk_bf16_full"
DISK_F32_PQ_PREFIX="$OUT_DIR/results/disk_f32_pq"
DISK_BF16_PQ_PREFIX="$OUT_DIR/results/disk_bf16_pq"

IDX_MEM_F32_PREFIX="$OUT_DIR/results/index_mem_f32"
IDX_MEM_BF16_PREFIX="$OUT_DIR/results/index_mem_bf16"
IDX_DISK_F32_FULL_PREFIX="$OUT_DIR/results/index_disk_f32_full"
IDX_DISK_BF16_FULL_PREFIX="$OUT_DIR/results/index_disk_bf16_full"
IDX_DISK_F32_PQ_PREFIX="$OUT_DIR/results/index_disk_f32_pq"
IDX_DISK_BF16_PQ_PREFIX="$OUT_DIR/results/index_disk_bf16_pq"

echo "[1/6] Generate float32 base/query (NPTS=$NPTS NQ=$NQ DIM=$DIM)"
"$BASE_PATH/utils/rand_data_gen" --data_type float --output_file "$BASE_F32" -D "$DIM" -N "$NPTS" --norm 1.0
"$BASE_PATH/utils/rand_data_gen" --data_type float --output_file "$QUERY_F32" -D "$DIM" -N "$NQ" --norm 1.0

echo "[2/6] Convert float32 -> bf16 (round-to-nearest-even)"
python3 "$SCRIPT_DIR/bin_convert.py" --mode float_to_bf16 --input "$BASE_F32" --output "$BASE_BF16"
python3 "$SCRIPT_DIR/bin_convert.py" --mode float_to_bf16 --input "$QUERY_F32" --output "$QUERY_BF16"

echo "[3/6] Compute float32 ground truth (L2)"
"$BASE_PATH/utils/compute_groundtruth" --data_type float --dist_fn l2 --base_file "$BASE_F32" --query_file "$QUERY_F32" --gt_file "$GT_PREFIX" --K "$K"

# compute_groundtruth historically may or may not append a ".bin" suffix depending on build.
if [[ -f "$GT_PREFIX" ]]; then
  GT_BIN="$GT_PREFIX"
elif [[ -f "$GT_PREFIX.bin" ]]; then
  GT_BIN="$GT_PREFIX.bin"
else
  echo "Error: could not find ground truth output at '$GT_PREFIX' or '$GT_PREFIX.bin'" >&2
  exit 2
fi

echo "[4/6] Memory: build + search (float vs bf16)"
"$BASE_PATH/build_memory_index" --data_type float --dist_fn l2 --data_path "$BASE_F32" --index_path_prefix "$IDX_MEM_F32_PREFIX"
"$BASE_PATH/build_memory_index" --data_type bf16 --dist_fn l2 --data_path "$BASE_BF16" --index_path_prefix "$IDX_MEM_BF16_PREFIX"

"$BASE_PATH/search_memory_index" --data_type float --dist_fn l2 --index_path_prefix "$IDX_MEM_F32_PREFIX" --query_file "$QUERY_F32" --recall_at "$K" --result_path "$MEM_F32_PREFIX" --gt_file "$GT_BIN" -L "$MEM_L" -T "$THREADS"
"$BASE_PATH/search_memory_index" --data_type bf16 --dist_fn l2 --index_path_prefix "$IDX_MEM_BF16_PREFIX" --query_file "$QUERY_BF16" --recall_at "$K" --result_path "$MEM_BF16_PREFIX" --gt_file "$GT_BIN" -L "$MEM_L" -T "$THREADS"

echo "[5/6] Disk: build + search (full-precision and PQ+reorder; float vs bf16)"
# Full-precision disk
"$BASE_PATH/build_disk_index" --data_type float --dist_fn l2 --data_path "$BASE_F32" --index_path_prefix "$IDX_DISK_F32_FULL_PREFIX" -R "$DISK_R" -L "$DISK_LBUILD" -B "$DISK_SEARCH_DRAM_BUDGET_GB" -M "$DISK_BUILD_DRAM_BUDGET_GB" --PQ_disk_bytes 0 --build_PQ_bytes 0 -T "$THREADS"
"$BASE_PATH/build_disk_index" --data_type bf16 --dist_fn l2 --data_path "$BASE_BF16" --index_path_prefix "$IDX_DISK_BF16_FULL_PREFIX" -R "$DISK_R" -L "$DISK_LBUILD" -B "$DISK_SEARCH_DRAM_BUDGET_GB" -M "$DISK_BUILD_DRAM_BUDGET_GB" --PQ_disk_bytes 0 --build_PQ_bytes 0 -T "$THREADS"

"$BASE_PATH/search_disk_index" --data_type float --dist_fn l2 --index_path_prefix "$IDX_DISK_F32_FULL_PREFIX" --query_file "$QUERY_F32" --result_path "$DISK_F32_FULL_PREFIX" --gt_file "$GT_BIN" -K "$K" -L "$DISK_L" -W "$DISK_BEAMWIDTH" -T "$THREADS" --num_nodes_to_cache "$DISK_NUM_NODES_TO_CACHE"
"$BASE_PATH/search_disk_index" --data_type bf16 --dist_fn l2 --index_path_prefix "$IDX_DISK_BF16_FULL_PREFIX" --query_file "$QUERY_BF16" --result_path "$DISK_BF16_FULL_PREFIX" --gt_file "$GT_BIN" -K "$K" -L "$DISK_L" -W "$DISK_BEAMWIDTH" -T "$THREADS" --num_nodes_to_cache "$DISK_NUM_NODES_TO_CACHE"

# Disk PQ + reorder
"$BASE_PATH/build_disk_index" --data_type float --dist_fn l2 --data_path "$BASE_F32" --index_path_prefix "$IDX_DISK_F32_PQ_PREFIX" -R "$DISK_R" -L "$DISK_LBUILD" -B "$DISK_SEARCH_DRAM_BUDGET_GB" -M "$DISK_BUILD_DRAM_BUDGET_GB" --PQ_disk_bytes "$DISK_PQ_BYTES" --build_PQ_bytes 0 --append_reorder_data -T "$THREADS"
"$BASE_PATH/build_disk_index" --data_type bf16 --dist_fn l2 --data_path "$BASE_BF16" --index_path_prefix "$IDX_DISK_BF16_PQ_PREFIX" -R "$DISK_R" -L "$DISK_LBUILD" -B "$DISK_SEARCH_DRAM_BUDGET_GB" -M "$DISK_BUILD_DRAM_BUDGET_GB" --PQ_disk_bytes "$DISK_PQ_BYTES" --build_PQ_bytes 0 --append_reorder_data -T "$THREADS"

"$BASE_PATH/search_disk_index" --data_type float --dist_fn l2 --index_path_prefix "$IDX_DISK_F32_PQ_PREFIX" --query_file "$QUERY_F32" --result_path "$DISK_F32_PQ_PREFIX" --gt_file "$GT_BIN" -K "$K" -L "$DISK_L" -W "$DISK_BEAMWIDTH" -T "$THREADS" --num_nodes_to_cache "$DISK_NUM_NODES_TO_CACHE" --use_reorder_data
"$BASE_PATH/search_disk_index" --data_type bf16 --dist_fn l2 --index_path_prefix "$IDX_DISK_BF16_PQ_PREFIX" --query_file "$QUERY_BF16" --result_path "$DISK_BF16_PQ_PREFIX" --gt_file "$GT_BIN" -K "$K" -L "$DISK_L" -W "$DISK_BEAMWIDTH" -T "$THREADS" --num_nodes_to_cache "$DISK_NUM_NODES_TO_CACHE" --use_reorder_data

echo "[6/6] Analyze float vs bf16 deltas"
python3 "$SCRIPT_DIR/analyze_results.py" \
  --gt "$GT_BIN" \
  --L "$MEM_L" \
  --K "$K" \
  --mem_float "$MEM_F32_PREFIX" \
  --mem_bf16 "$MEM_BF16_PREFIX" \
  --disk_float_full "$DISK_F32_FULL_PREFIX" \
  --disk_bf16_full "$DISK_BF16_FULL_PREFIX" \
  --disk_float_pq "$DISK_F32_PQ_PREFIX" \
  --disk_bf16_pq "$DISK_BF16_PQ_PREFIX"

echo "Done. Artifacts under: $OUT_DIR"
