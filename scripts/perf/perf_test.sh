#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

LOG_DIR=${LOG_DIR:-/app/logs}
mkdir -p "$LOG_DIR"
TIME_LOG="$LOG_DIR/time.log"

DATA_TYPE=${DATA_TYPE:-float}
if [[ "$DATA_TYPE" != "float" && "$DATA_TYPE" != "bf16" ]]; then
  echo "Unsupported DATA_TYPE='$DATA_TYPE'. Use DATA_TYPE=float or DATA_TYPE=bf16."
  exit 2
fi

# Choose which index type(s) to run: memory, disk, or both.
# Default is memory.
PERF_MODE=${PERF_MODE:-memory}
if [[ -n "$RUN_DISK" ]]; then
  # Backward compatibility: existing RUN_DISK=1 turns on disk tests.
  if [[ "$RUN_DISK" == "1" ]]; then
    PERF_MODE=both
  elif [[ "$RUN_DISK" == "0" ]]; then
    PERF_MODE=memory
  else
    echo "Unsupported RUN_DISK='$RUN_DISK'. Use RUN_DISK=0 or RUN_DISK=1."
    exit 2
  fi
fi

if [[ "$PERF_MODE" != "memory" && "$PERF_MODE" != "disk" && "$PERF_MODE" != "both" ]]; then
  echo "Unsupported PERF_MODE='$PERF_MODE'. Use PERF_MODE=memory|disk|both."
  exit 2
fi

function json_time {
  command="$@"
  echo "Executing $command"
  /usr/bin/time --quiet -o "$TIME_LOG" -a --format '{"command":"%C", "wallclock": %e, "user": %U, "sys": %S}' $command
  ret=$?
  if [ $ret -ne 0 ]; then
    echo "{\"command\": \""$command"\", \"status_code\": $ret}" >> "$TIME_LOG"
  fi
}

mkdir data
rm -f "$TIME_LOG"
touch "$TIME_LOG"
chmod 666 "$TIME_LOG"

if [ -d "build/apps" ]; then
  export BASE_PATH="build/apps"
else
  export BASE_PATH="build/tests"
fi

BASE_FILE="data/rand_${DATA_TYPE}_768D_1M_norm1.0.bin"
QUERY_FILE="data/rand_${DATA_TYPE}_768D_10K_norm1.0.bin"

GT_L2_FILE="data/l2_rand_${DATA_TYPE}_768D_1M_norm1.0_768D_10K_norm1.0_gt100"
GT_MIPS_FILE="data/mips_rand_${DATA_TYPE}_768D_1M_norm1.0_768D_10K_norm1.0_gt100"
GT_COSINE_FILE="data/cosine_rand_${DATA_TYPE}_768D_1M_norm1.0_768D_10K_norm1.0_gt100"

INDEX_L2_PREFIX="data/index_l2_rand_${DATA_TYPE}_768D_1M_norm1.0"
INDEX_MIPS_PREFIX="data/index_mips_rand_${DATA_TYPE}_768D_1M_norm1.0"
INDEX_COSINE_PREFIX="data/index_cosine_rand_${DATA_TYPE}_768D_1M_norm1.0"

json_time $BASE_PATH/utils/rand_data_gen --data_type "$DATA_TYPE" --output_file "$BASE_FILE" -D 768 -N 1000000 --norm 1.0
json_time $BASE_PATH/utils/rand_data_gen --data_type "$DATA_TYPE" --output_file "$QUERY_FILE" -D 768 -N 10000 --norm 1.0

json_time $BASE_PATH/utils/compute_groundtruth --data_type "$DATA_TYPE" --dist_fn l2 --base_file "$BASE_FILE" --query_file "$QUERY_FILE" --gt_file "$GT_L2_FILE" --K 100
json_time $BASE_PATH/utils/compute_groundtruth --data_type "$DATA_TYPE" --dist_fn mips --base_file "$BASE_FILE" --query_file "$QUERY_FILE" --gt_file "$GT_MIPS_FILE" --K 100
json_time $BASE_PATH/utils/compute_groundtruth --data_type "$DATA_TYPE" --dist_fn cosine --base_file "$BASE_FILE" --query_file "$QUERY_FILE" --gt_file "$GT_COSINE_FILE" --K 100

if [[ "$PERF_MODE" == "memory" || "$PERF_MODE" == "both" ]]; then
  json_time $BASE_PATH/build_memory_index --data_type "$DATA_TYPE" --dist_fn l2 --data_path "$BASE_FILE" --index_path_prefix "$INDEX_L2_PREFIX"
  json_time $BASE_PATH/search_memory_index --data_type "$DATA_TYPE" --dist_fn l2 --index_path_prefix "$INDEX_L2_PREFIX" --query_file "$QUERY_FILE" --recall_at 10 --result_path temp --gt_file "$GT_L2_FILE" -L 100 32
  if [[ "$DATA_TYPE" == "float" ]]; then
    json_time $BASE_PATH/search_memory_index --data_type "$DATA_TYPE" --dist_fn fast_l2 --index_path_prefix "$INDEX_L2_PREFIX" --query_file "$QUERY_FILE" --recall_at 10 --result_path temp --gt_file "$GT_L2_FILE" -L 100 32
  fi

  json_time $BASE_PATH/build_memory_index --data_type "$DATA_TYPE" --dist_fn mips --data_path "$BASE_FILE" --index_path_prefix "$INDEX_MIPS_PREFIX"
  json_time $BASE_PATH/search_memory_index --data_type "$DATA_TYPE" --dist_fn mips --index_path_prefix "$INDEX_L2_PREFIX" --query_file "$QUERY_FILE" --recall_at 10 --result_path temp --gt_file "$GT_MIPS_FILE" -L 100 32

  json_time $BASE_PATH/build_memory_index --data_type "$DATA_TYPE" --dist_fn cosine --data_path "$BASE_FILE" --index_path_prefix "$INDEX_COSINE_PREFIX"
  json_time $BASE_PATH/search_memory_index --data_type "$DATA_TYPE" --dist_fn cosine --index_path_prefix "$INDEX_L2_PREFIX" --query_file "$QUERY_FILE" --recall_at 10 --result_path temp --gt_file "$GT_COSINE_FILE" -L 100 32
fi

# Optional SSD/disk index perf (mixed RAM+SSD).
# Note: build_disk_index/search_disk_index do not support bf16 currently.
if [[ "$PERF_MODE" == "disk" || "$PERF_MODE" == "both" ]]; then
  if [[ "$DATA_TYPE" != "float" ]]; then
    echo "PERF_MODE includes disk but DATA_TYPE='$DATA_TYPE' is not supported for disk index; skipping disk tests."
  else
    DISK_R=${DISK_R:-32}
    DISK_LBUILD=${DISK_LBUILD:-50}
    DISK_SEARCH_DRAM_BUDGET_GB=${DISK_SEARCH_DRAM_BUDGET_GB:-0.5}
    DISK_BUILD_DRAM_BUDGET_GB=${DISK_BUILD_DRAM_BUDGET_GB:-8}
    DISK_PQ_DISK_BYTES=${DISK_PQ_DISK_BYTES:-0}
    DISK_BUILD_PQ_BYTES=${DISK_BUILD_PQ_BYTES:-0}
    DISK_NUM_NODES_TO_CACHE=${DISK_NUM_NODES_TO_CACHE:-10000}
    DISK_BEAMWIDTH=${DISK_BEAMWIDTH:-2}
    DISK_RECALL_AT=${DISK_RECALL_AT:-10}
    DISK_SEARCH_LISTS=${DISK_SEARCH_LISTS:-"10 20 30 40 50 100"}

    DISK_INDEX_L2_PREFIX="data/disk_index_l2_rand_${DATA_TYPE}_768D_1M_R${DISK_R}_L${DISK_LBUILD}_B${DISK_SEARCH_DRAM_BUDGET_GB}_M${DISK_BUILD_DRAM_BUDGET_GB}"
    DISK_INDEX_MIPS_PREFIX="data/disk_index_mips_rand_${DATA_TYPE}_768D_1M_R${DISK_R}_L${DISK_LBUILD}_B${DISK_SEARCH_DRAM_BUDGET_GB}_M${DISK_BUILD_DRAM_BUDGET_GB}"
    DISK_INDEX_COSINE_PREFIX="data/disk_index_cosine_rand_${DATA_TYPE}_768D_1M_R${DISK_R}_L${DISK_LBUILD}_B${DISK_SEARCH_DRAM_BUDGET_GB}_M${DISK_BUILD_DRAM_BUDGET_GB}"

    json_time $BASE_PATH/build_disk_index --data_type "$DATA_TYPE" --dist_fn l2 --data_path "$BASE_FILE" --index_path_prefix "$DISK_INDEX_L2_PREFIX" -R "$DISK_R" -L "$DISK_LBUILD" -B "$DISK_SEARCH_DRAM_BUDGET_GB" -M "$DISK_BUILD_DRAM_BUDGET_GB" --PQ_disk_bytes "$DISK_PQ_DISK_BYTES" --build_PQ_bytes "$DISK_BUILD_PQ_BYTES"
    json_time $BASE_PATH/search_disk_index --data_type "$DATA_TYPE" --dist_fn l2 --index_path_prefix "$DISK_INDEX_L2_PREFIX" --query_file "$QUERY_FILE" --gt_file "$GT_L2_FILE" -K "$DISK_RECALL_AT" -L $DISK_SEARCH_LISTS --result_path "temp/disk_l2" --num_nodes_to_cache "$DISK_NUM_NODES_TO_CACHE" -W "$DISK_BEAMWIDTH"

    json_time $BASE_PATH/build_disk_index --data_type "$DATA_TYPE" --dist_fn mips --data_path "$BASE_FILE" --index_path_prefix "$DISK_INDEX_MIPS_PREFIX" -R "$DISK_R" -L "$DISK_LBUILD" -B "$DISK_SEARCH_DRAM_BUDGET_GB" -M "$DISK_BUILD_DRAM_BUDGET_GB" --PQ_disk_bytes "$DISK_PQ_DISK_BYTES" --build_PQ_bytes "$DISK_BUILD_PQ_BYTES"
    json_time $BASE_PATH/search_disk_index --data_type "$DATA_TYPE" --dist_fn mips --index_path_prefix "$DISK_INDEX_MIPS_PREFIX" --query_file "$QUERY_FILE" --gt_file "$GT_MIPS_FILE" -K "$DISK_RECALL_AT" -L $DISK_SEARCH_LISTS --result_path "temp/disk_mips" --num_nodes_to_cache "$DISK_NUM_NODES_TO_CACHE" -W "$DISK_BEAMWIDTH"

    json_time $BASE_PATH/build_disk_index --data_type "$DATA_TYPE" --dist_fn cosine --data_path "$BASE_FILE" --index_path_prefix "$DISK_INDEX_COSINE_PREFIX" -R "$DISK_R" -L "$DISK_LBUILD" -B "$DISK_SEARCH_DRAM_BUDGET_GB" -M "$DISK_BUILD_DRAM_BUDGET_GB" --PQ_disk_bytes "$DISK_PQ_DISK_BYTES" --build_PQ_bytes "$DISK_BUILD_PQ_BYTES"
    json_time $BASE_PATH/search_disk_index --data_type "$DATA_TYPE" --dist_fn cosine --index_path_prefix "$DISK_INDEX_COSINE_PREFIX" --query_file "$QUERY_FILE" --gt_file "$GT_COSINE_FILE" -K "$DISK_RECALL_AT" -L $DISK_SEARCH_LISTS --result_path "temp/disk_cosine" --num_nodes_to_cache "$DISK_NUM_NODES_TO_CACHE" -W "$DISK_BEAMWIDTH"
  fi
fi

