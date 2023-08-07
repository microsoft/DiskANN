#!/bin/bash

function json_time {
  command="$@"
  /usr/bin/time --quiet -o /app/logs/time.log -a --format '{"command":"%C", "wallclock": %e, "user": %U, "sys": %S}' $command
  ret=$?
  if [ $ret -ne 0 ]; then
    echo "{\"command\": \""$command"\", "status_code": \"$ret\"}" >> /app/logs/time.log
  fi
}

mkdir data
rm /app/logs/time.log
touch /app/logs/time.log
chmod 666 /app/logs/time.log

if [ -d "build/apps" ]; then
	export BASE_PATH="build/apps"
else
	export BASE_PATH="build/tests"
fi

echo "Generating random vectors for index"
json_time $BASE_PATH/utils/rand_data_gen --data_type float --output_file data/rand_float_10D_10K_norm1.0.bin -D 10 -N 10000 --norm 1.0
json_time $BASE_PATH/utils/rand_data_gen --data_type int8 --output_file data/rand_int8_10D_10K_norm50.0.bin -D 10 -N 10000 --norm 50.0
json_time $BASE_PATH/utils/rand_data_gen --data_type uint8 --output_file data/rand_uint8_10D_10K_norm50.0.bin -D 10 -N 10000 --norm 50.0

echo "Generating random vectors for query"
json_time $BASE_PATH/utils/rand_data_gen --data_type float --output_file data/rand_float_10D_1K_norm1.0.bin -D 10 -N 1000 --norm 1.0
json_time $BASE_PATH/utils/rand_data_gen --data_type int8 --output_file data/rand_int8_10D_1K_norm50.0.bin -D 10 -N 1000 --norm 50.0
json_time $BASE_PATH/utils/rand_data_gen --data_type uint8 --output_file data/rand_uint8_10D_1K_norm50.0.bin -D 10 -N 1000 --norm 50.0

echo "Computing ground truth for floats across l2, mips, and cosine distance functions"
json_time $BASE_PATH/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file data/rand_float_10D_10K_norm1.0.bin --query_file data/rand_float_10D_1K_norm1.0.bin --gt_file data/l2_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 --K 100
json_time $BASE_PATH/utils/compute_groundtruth  --data_type float --dist_fn mips --base_file data/rand_float_10D_10K_norm1.0.bin --query_file data/rand_float_10D_1K_norm1.0.bin --gt_file data/mips_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 --K 100
json_time $BASE_PATH/utils/compute_groundtruth  --data_type float --dist_fn cosine --base_file data/rand_float_10D_10K_norm1.0.bin --query_file data/rand_float_10D_1K_norm1.0.bin --gt_file data/cosine_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 --K 100

echo "Computing ground truth for int8s across l2, mips, and cosine distance functions"
json_time $BASE_PATH/utils/compute_groundtruth  --data_type int8 --dist_fn l2 --base_file data/rand_int8_10D_10K_norm50.0.bin --query_file data/rand_int8_10D_1K_norm50.0.bin --gt_file data/l2_rand_int8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --K 100
json_time $BASE_PATH/utils/compute_groundtruth  --data_type int8 --dist_fn mips --base_file data/rand_int8_10D_10K_norm50.0.bin --query_file data/rand_int8_10D_1K_norm50.0.bin --gt_file data/mips_rand_int8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --K 100
json_time $BASE_PATH/utils/compute_groundtruth  --data_type int8 --dist_fn cosine --base_file data/rand_int8_10D_10K_norm50.0.bin --query_file data/rand_int8_10D_1K_norm50.0.bin --gt_file data/cosine_rand_int8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --K 100

echo "Computing ground truth for uint8s across l2, mips, and cosine distance functions"
json_time $BASE_PATH/utils/compute_groundtruth  --data_type uint8 --dist_fn l2 --base_file data/rand_uint8_10D_10K_norm50.0.bin --query_file data/rand_uint8_10D_1K_norm50.0.bin --gt_file data/l2_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --K 100
json_time $BASE_PATH/utils/compute_groundtruth  --data_type uint8 --dist_fn mips --base_file data/rand_uint8_10D_10K_norm50.0.bin --query_file data/rand_uint8_10D_1K_norm50.0.bin --gt_file data/mips_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --K 100
json_time $BASE_PATH/utils/compute_groundtruth  --data_type uint8 --dist_fn cosine --base_file data/rand_uint8_10D_10K_norm50.0.bin --query_file data/rand_uint8_10D_1K_norm50.0.bin --gt_file data/cosine_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --K 100

echo "Disk PQ Tests"
if $BASE_PATH/search_disk_index --help | grep -q "fail_if_recall_below"; then
  export DISK_RECALL="--fail_if_recall_below 70"
else
  export DISK_RECALL=""
fi
if $BASE_PATH/search_disk_index --help | grep -q "build_PQ_bytes"; then
  export BUILD_PQ_BYTES="--build_PQ_bytes 5"
else
  export BUILD_PQ_BYTES=""
fi
json_time $BASE_PATH/build_disk_index --data_type float --dist_fn l2 --data_path data/rand_float_10D_10K_norm1.0.bin --index_path_prefix data/disk_index_l2_rand_float_10D_10K_norm1.0_diskfull_oneshot -R 16 -L 32 -B 0.00003 -M 1
json_time $BASE_PATH/search_disk_index --data_type float --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_l2_rand_float_10D_10K_norm1.0_diskfull_oneshot --result_path /tmp/res --query_file data/rand_float_10D_1K_norm1.0.bin --gt_file data/l2_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

json_time $BASE_PATH/build_disk_index --data_type int8 --dist_fn l2 --data_path data/rand_int8_10D_10K_norm50.0.bin --index_path_prefix data/disk_index_l2_rand_int8_10D_10K_norm50.0_diskfull_oneshot -R 16 -L 32 -B 0.00003 -M 1
json_time $BASE_PATH/search_disk_index --data_type int8 --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_l2_rand_int8_10D_10K_norm50.0_diskfull_oneshot --result_path /tmp/res --query_file data/rand_int8_10D_1K_norm50.0.bin --gt_file data/l2_rand_int8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

json_time $BASE_PATH/build_disk_index --data_type uint8 --dist_fn l2 --data_path data/rand_uint8_10D_10K_norm50.0.bin --index_path_prefix data/disk_index_l2_rand_uint8_10D_10K_norm50.0_diskfull_oneshot -R 16 -L 32 -B 0.00003 -M 1
json_time $BASE_PATH/search_disk_index --data_type uint8 --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_l2_rand_uint8_10D_10K_norm50.0_diskfull_oneshot --result_path /tmp/res --query_file data/rand_uint8_10D_1K_norm50.0.bin --gt_file data/l2_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

json_time $BASE_PATH/build_disk_index --data_type float --dist_fn l2 --data_path data/rand_float_10D_10K_norm1.0.bin --index_path_prefix data/disk_index_l2_rand_float_10D_10K_norm1.0_diskfull_oneshot_buildpq5 -R 16 -L 32 -B 0.00003 -M 1 $BUILD_PQ_BYTES
json_time $BASE_PATH/search_disk_index --data_type float --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_l2_rand_float_10D_10K_norm1.0_diskfull_oneshot_buildpq5 --result_path /tmp/res --query_file data/rand_float_10D_1K_norm1.0.bin --gt_file data/l2_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

json_time $BASE_PATH/build_disk_index --data_type int8 --dist_fn l2 --data_path data/rand_int8_10D_10K_norm50.0.bin --index_path_prefix data/disk_index_l2_rand_int8_10D_10K_norm50.0_diskfull_oneshot_buildpq5 -R 16 -L 32 -B 0.00003 -M 1 $BUILD_PQ_BYTES
json_time $BASE_PATH/search_disk_index --data_type int8 --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_l2_rand_int8_10D_10K_norm50.0_diskfull_oneshot_buildpq5 --result_path /tmp/res --query_file data/rand_int8_10D_1K_norm50.0.bin --gt_file data/l2_rand_int8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16\

json_time $BASE_PATH/build_disk_index --data_type uint8 --dist_fn l2 --data_path data/rand_uint8_10D_10K_norm50.0.bin --index_path_prefix data/disk_index_l2_rand_uint8_10D_10K_norm50.0_diskfull_oneshot_buildpq5 -R 16 -L 32 -B 0.00003 -M 1 $BUILD_PQ_BYTES
json_time $BASE_PATH/search_disk_index --data_type uint8 --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_l2_rand_uint8_10D_10K_norm50.0_diskfull_oneshot_buildpq5 --result_path /tmp/res --query_file data/rand_uint8_10D_1K_norm50.0.bin --gt_file data/l2_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

json_time $BASE_PATH/build_disk_index --data_type float --dist_fn l2 --data_path data/rand_float_10D_10K_norm1.0.bin --index_path_prefix data/disk_index_l2_rand_float_10D_10K_norm1.0_diskfull_sharded -R 16 -L 32 -B 0.00003 -M 0.00006
json_time $BASE_PATH/search_disk_index --data_type float --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_l2_rand_float_10D_10K_norm1.0_diskfull_sharded --result_path /tmp/res --query_file data/rand_float_10D_1K_norm1.0.bin --gt_file data/l2_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

json_time $BASE_PATH/build_disk_index --data_type int8 --dist_fn l2 --data_path data/rand_int8_10D_10K_norm50.0.bin --index_path_prefix data/disk_index_l2_rand_int8_10D_10K_norm50.0_diskfull_sharded -R 16 -L 32 -B 0.00003 -M 0.00006
json_time $BASE_PATH/search_disk_index --data_type int8 --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_l2_rand_int8_10D_10K_norm50.0_diskfull_sharded --result_path /tmp/res --query_file data/rand_int8_10D_1K_norm50.0.bin --gt_file data/l2_rand_int8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

json_time $BASE_PATH/build_disk_index --data_type uint8 --dist_fn l2 --data_path data/rand_uint8_10D_10K_norm50.0.bin --index_path_prefix data/disk_index_l2_rand_uint8_10D_10K_norm50.0_diskfull_sharded -R 16 -L 32 -B 0.00003 -M 0.00006
json_time $BASE_PATH/search_disk_index --data_type uint8 --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_l2_rand_uint8_10D_10K_norm50.0_diskfull_sharded --result_path /tmp/res --query_file data/rand_uint8_10D_1K_norm50.0.bin --gt_file data/l2_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

json_time $BASE_PATH/build_disk_index --data_type float --dist_fn l2 --data_path data/rand_float_10D_10K_norm1.0.bin --index_path_prefix data/disk_index_l2_rand_float_10D_10K_norm1.0_diskpq_oneshot -R 16 -L 32 -B 0.00003 -M 1 --PQ_disk_bytes 5
json_time $BASE_PATH/search_disk_index --data_type float --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_l2_rand_float_10D_10K_norm1.0_diskpq_oneshot --result_path /tmp/res --query_file data/rand_float_10D_1K_norm1.0.bin --gt_file data/l2_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

json_time $BASE_PATH/build_disk_index --data_type int8 --dist_fn l2 --data_path data/rand_int8_10D_10K_norm50.0.bin --index_path_prefix data/disk_index_l2_rand_int8_10D_10K_norm50.0_diskpq_oneshot -R 16 -L 32 -B 0.00003 -M 1 --PQ_disk_bytes 5
json_time $BASE_PATH/search_disk_index --data_type int8 --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_l2_rand_int8_10D_10K_norm50.0_diskpq_oneshot --result_path /tmp/res --query_file data/rand_int8_10D_1K_norm50.0.bin --gt_file data/l2_rand_int8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

json_time $BASE_PATH/build_disk_index --data_type uint8 --dist_fn l2 --data_path data/rand_uint8_10D_10K_norm50.0.bin --index_path_prefix data/disk_index_l2_rand_uint8_10D_10K_norm50.0_diskpq_oneshot -R 16 -L 32 -B 0.00003 -M 1 --PQ_disk_bytes 5
json_time $BASE_PATH/search_disk_index --data_type uint8 --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_l2_rand_uint8_10D_10K_norm50.0_diskpq_oneshot --result_path /tmp/res --query_file data/rand_uint8_10D_1K_norm50.0.bin --gt_file data/l2_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

json_time $BASE_PATH/build_disk_index --data_type float --dist_fn mips --data_path data/rand_float_10D_10K_norm1.0.bin --index_path_prefix data/disk_index_mips_rand_float_10D_10K_norm1.0_diskpq_sharded -R 16 -L 32 -B 0.00003 -M 0.00006 --PQ_disk_bytes 5
json_time $BASE_PATH/search_disk_index --data_type float --dist_fn l2 $DISK_RECALL --index_path_prefix data/disk_index_mips_rand_float_10D_10K_norm1.0_diskpq_sharded --result_path /tmp/res --query_file data/rand_float_10D_1K_norm1.0.bin --gt_file data/mips_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

if $BASE_PATH/search_memory_index --help | grep -q "fail_if_recall_below"; then
  export MEMORY_RECALL="--fail_if_recall_below 70"
else
  export MEMORY_RECALL=""
fi

if $BASE_PATH/build_memory_index --help | grep -q "build_PQ_bytes"; then
  echo "In Memory PQ Tests"
  json_time $BASE_PATH/build_memory_index --data_type float --dist_fn l2 --data_path data/rand_float_10D_10K_norm1.0.bin --index_path_prefix data/index_l2_rand_float_10D_10K_norm1.0_buildpq5 --build_PQ_bytes 5
  json_time $BASE_PATH/search_memory_index --data_type float --dist_fn l2 $MEMORY_RECALL --index_path_prefix data/index_l2_rand_float_10D_10K_norm1.0_buildpq5 --query_file data/rand_float_10D_1K_norm1.0.bin --recall_at 10 --result_path temp --gt_file data/l2_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 -L  16 32

  json_time $BASE_PATH/build_memory_index --data_type int8 --dist_fn l2 --data_path data/rand_int8_10D_10K_norm50.0.bin --index_path_prefix data/index_l2_rand_int8_10D_10K_norm50.0_buildpq5 --build_PQ_bytes 5
  json_time $BASE_PATH/search_memory_index --data_type int8 --dist_fn l2 $MEMORY_RECALL --index_path_prefix data/index_l2_rand_int8_10D_10K_norm50.0_buildpq5 --query_file data/rand_int8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/l2_rand_int8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 -L  16 32

  json_time $BASE_PATH/build_memory_index --data_type uint8 --dist_fn l2 --data_path data/rand_uint8_10D_10K_norm50.0.bin --index_path_prefix data/index_l2_rand_uint8_10D_10K_norm50.0_buildpq5 --build_PQ_bytes 5
  json_time $BASE_PATH/search_memory_index --data_type uint8 --dist_fn l2 $MEMORY_RECALL --index_path_prefix data/index_l2_rand_uint8_10D_10K_norm50.0_buildpq5 --query_file data/rand_uint8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/l2_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 -L  16 32
fi

echo "In Memory No-PQ Tests"
json_time $BASE_PATH/build_memory_index --data_type float --dist_fn l2 --data_path data/rand_float_10D_10K_norm1.0.bin --index_path_prefix data/index_l2_rand_float_10D_10K_norm1.0
json_time $BASE_PATH/search_memory_index --data_type float --dist_fn  l2 $MEMORY_RECALL --index_path_prefix data/index_l2_rand_float_10D_10K_norm1.0 --query_file data/rand_float_10D_1K_norm1.0.bin --recall_at 10 --result_path temp --gt_file data/l2_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 -L  16 32

json_time $BASE_PATH/build_memory_index --data_type int8 --dist_fn l2 --data_path data/rand_int8_10D_10K_norm50.0.bin --index_path_prefix data/index_l2_rand_int8_10D_10K_norm50.0
json_time $BASE_PATH/search_memory_index --data_type int8 --dist_fn l2 $MEMORY_RECALL --index_path_prefix data/index_l2_rand_int8_10D_10K_norm50.0 --query_file data/rand_int8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/l2_rand_int8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 -L  16 32

json_time $BASE_PATH/build_memory_index --data_type uint8 --dist_fn l2 --data_path data/rand_uint8_10D_10K_norm50.0.bin --index_path_prefix data/index_l2_rand_uint8_10D_10K_norm50.0
json_time $BASE_PATH/search_memory_index --data_type uint8 --dist_fn l2 $MEMORY_RECALL --index_path_prefix data/index_l2_rand_uint8_10D_10K_norm50.0 --query_file data/rand_uint8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/l2_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 -L  16 32

json_time $BASE_PATH/search_memory_index --data_type float --dist_fn fast_l2 $MEMORY_RECALL --index_path_prefix data/index_l2_rand_float_10D_10K_norm1.0 --query_file data/rand_float_10D_1K_norm1.0.bin --recall_at 10 --result_path temp --gt_file data/l2_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 -L  16 32

json_time $BASE_PATH/build_memory_index --data_type float --dist_fn mips --data_path data/rand_float_10D_10K_norm1.0.bin --index_path_prefix data/index_mips_rand_float_10D_10K_norm1.0
json_time $BASE_PATH/search_memory_index --data_type float --dist_fn mips $MEMORY_RECALL --index_path_prefix data/index_l2_rand_float_10D_10K_norm1.0 --query_file data/rand_float_10D_1K_norm1.0.bin --recall_at 10 --result_path temp --gt_file data/mips_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 -L  16 32


json_time $BASE_PATH/build_memory_index --data_type float --dist_fn cosine --data_path data/rand_float_10D_10K_norm1.0.bin --index_path_prefix data/index_cosine_rand_float_10D_10K_norm1.0
json_time $BASE_PATH/search_memory_index --data_type float --dist_fn cosine $MEMORY_RECALL --index_path_prefix data/index_l2_rand_float_10D_10K_norm1.0 --query_file data/rand_float_10D_1K_norm1.0.bin --recall_at 10 --result_path temp --gt_file data/cosine_rand_float_10D_10K_norm1.0_10D_1K_norm1.0_gt100 -L  16 32

json_time $BASE_PATH/build_memory_index --data_type int8 --dist_fn cosine --data_path data/rand_int8_10D_10K_norm50.0.bin --index_path_prefix data/index_cosine_rand_int8_10D_10K_norm50.0
json_time $BASE_PATH/search_memory_index --data_type int8 --dist_fn cosine $MEMORY_RECALL --index_path_prefix data/index_l2_rand_int8_10D_10K_norm50.0 --query_file data/rand_int8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/cosine_rand_int8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 -L  16 32

json_time $BASE_PATH/build_memory_index --data_type uint8 --dist_fn cosine --data_path data/rand_uint8_10D_10K_norm50.0.bin --index_path_prefix data/index_cosine_rand_uint8_10D_10K_norm50.0
json_time $BASE_PATH/search_memory_index --data_type uint8 --dist_fn cosine $MEMORY_RECALL --index_path_prefix data/index_l2_rand_uint8_10D_10K_norm50.0 --query_file data/rand_uint8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/cosine_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100 -L  16 32

echo "Dynamic In-Memory tests"
json_time $BASE_PATH/test_streaming_scenario --data_type float --dist_fn l2 --data_path data/rand_float_10D_10K_norm1.0.bin --index_path_prefix data/index_stream -R 64 -L 600 --alpha 1.2 --insert_threads 4 --consolidate_threads 4 --max_points_to_insert 10000 --active_window 4000 --consolidate_interval 2000 --start_point_norm 3.2
json_time $BASE_PATH/utils/compute_groundtruth --data_type float --dist_fn l2 --base_file data/index_stream.after-streaming-act4000-cons2000-max10000.data --query_file data/rand_float_10D_1K_norm1.0.bin --K 100 --gt_file data/gt100_base-act4000-cons2000-max10000 --tags_file data/index_stream.after-streaming-act4000-cons2000-max10000.tags
json_time $BASE_PATH/search_memory_index --data_type float --dist_fn l2 $MEMORY_RECALL --index_path_prefix data/index_stream.after-streaming-act4000-cons2000-max10000 --result_path data/res_stream --query_file data/rand_float_10D_1K_norm1.0.bin --gt_file data/gt100_base-act4000-cons2000-max10000 -K 10 -L 20 40 60 80 100 -T 64 --dynamic true --tags 1

json_time $BASE_PATH/test_streaming_scenario --data_type int8 --dist_fn l2 --data_path data/rand_int8_10D_10K_norm50.0.bin --index_path_prefix data/index_stream -R 64 -L 600 --alpha 1.2 --insert_threads 4 --consolidate_threads 4 --max_points_to_insert 10000 --active_window 4000 --consolidate_interval 2000 --start_point_norm 200
json_time $BASE_PATH/utils/compute_groundtruth --data_type int8 --dist_fn l2 --base_file data/index_stream.after-streaming-act4000-cons2000-max10000.data --query_file data/rand_int8_10D_1K_norm50.0.bin --K 100 --gt_file data/gt100_base-act4000-cons2000-max10000 --tags_file data/index_stream.after-streaming-act4000-cons2000-max10000.tags
json_time $BASE_PATH/search_memory_index --data_type int8 --dist_fn l2 $MEMORY_RECALL --index_path_prefix data/index_stream.after-streaming-act4000-cons2000-max10000 --result_path res_stream --query_file data/rand_int8_10D_1K_norm50.0.bin --gt_file data/gt100_base-act4000-cons2000-max10000 -K 10 -L 20 40 60 80 100 -T 64 --dynamic true --tags 1

json_time $BASE_PATH/test_streaming_scenario --data_type uint8 --dist_fn l2 --data_path data/rand_uint8_10D_10K_norm50.0.bin --index_path_prefix data/index_stream -R 64 -L 600 --alpha 1.2 --insert_threads 4 --consolidate_threads 4 --max_points_to_insert 10000 --active_window 4000 --consolidate_interval 2000 --start_point_norm 200
json_time $BASE_PATH/utils/compute_groundtruth --data_type uint8 --dist_fn l2 --base_file data/index_stream.after-streaming-act4000-cons2000-max10000.data --query_file data/rand_uint8_10D_1K_norm50.0.bin --K 100 --gt_file data/gt100_base-act4000-cons2000-max10000 --tags_file data/index_stream.after-streaming-act4000-cons2000-max10000.tags
json_time $BASE_PATH/search_memory_index --data_type uint8 --dist_fn l2 $MEMORY_RECALL --index_path_prefix data/index_stream.after-streaming-act4000-cons2000-max10000 --result_path data/res_stream --query_file data/rand_uint8_10D_1K_norm50.0.bin --gt_file data/gt100_base-act4000-cons2000-max10000 -K 10 -L 20 40 60 80 100 -T 64 --dynamic true --tags 1


json_time $BASE_PATH/test_insert_deletes_consolidate --data_type float --dist_fn l2 --data_path data/rand_float_10D_10K_norm1.0.bin --index_path_prefix data/index_ins_del -R 64 -L 300 --alpha 1.2 -T 8 --points_to_skip 0 --max_points_to_insert 7500 --beginning_index_size 0 --points_per_checkpoint 1000 --checkpoints_per_snapshot 0 --points_to_delete_from_beginning 2500 --start_deletes_after 5000 --do_concurrent true --start_point_norm 3.2;
json_time $BASE_PATH/utils/compute_groundtruth --data_type float --dist_fn l2 --base_file data/index_ins_del.after-concurrent-delete-del2500-7500.data --query_file data/rand_float_10D_1K_norm1.0.bin --K 100 --gt_file data/gt100_random10D_1K-conc-2500-7500 --tags_file data/index_ins_del.after-concurrent-delete-del2500-7500.tags
json_time $BASE_PATH/search_memory_index --data_type float --dist_fn l2 $MEMORY_RECALL --index_path_prefix data/index_ins_del.after-concurrent-delete-del2500-7500 --result_path data/res_ins_del --query_file data/rand_float_10D_1K_norm1.0.bin --gt_file data/gt100_random10D_1K-conc-2500-7500 -K 10 -L 20 40 60 80 100 -T 8 --dynamic true --tags 1

json_time $BASE_PATH/test_insert_deletes_consolidate --data_type int8 --dist_fn l2 --data_path data/rand_int8_10D_10K_norm50.0.bin --index_path_prefix data/index_ins_del -R 64 -L 300 --alpha 1.2 -T 8 --points_to_skip 0 --max_points_to_insert 7500 --beginning_index_size 0 --points_per_checkpoint 1000 --checkpoints_per_snapshot 0 --points_to_delete_from_beginning 2500 --start_deletes_after 5000 --do_concurrent true --start_point_norm 200
json_time $BASE_PATH/utils/compute_groundtruth --data_type int8 --dist_fn l2 --base_file data/index_ins_del.after-concurrent-delete-del2500-7500.data --query_file data/rand_int8_10D_1K_norm50.0.bin --K 100 --gt_file data/gt100_random10D_1K-conc-2500-7500 --tags_file data/index_ins_del.after-concurrent-delete-del2500-7500.tags
json_time $BASE_PATH/search_memory_index --data_type int8 --dist_fn l2 $MEMORY_RECALL --index_path_prefix data/index_ins_del.after-concurrent-delete-del2500-7500 --result_path data/res_ins_del --query_file data/rand_int8_10D_1K_norm50.0.bin --gt_file data/gt100_random10D_1K-conc-2500-7500 -K 10 -L 20 40 60 80 100 -T 8 --dynamic true --tags 1

json_time $BASE_PATH/test_insert_deletes_consolidate --data_type uint8 --dist_fn l2 --data_path data/rand_uint8_10D_10K_norm50.0.bin --index_path_prefix data/index_ins_del -R 64 -L 300 --alpha 1.2 -T 8 --points_to_skip 0 --max_points_to_insert 7500 --beginning_index_size 0 --points_per_checkpoint 1000 --checkpoints_per_snapshot 0 --points_to_delete_from_beginning 2500 --start_deletes_after 5000 --do_concurrent true --start_point_norm 200
json_time $BASE_PATH/utils/compute_groundtruth --data_type uint8 --dist_fn l2 --base_file data/index_ins_del.after-concurrent-delete-del2500-7500.data --query_file data/rand_uint8_10D_1K_norm50.0.bin --K 100 --gt_file data/gt100_random10D_10K-conc-2500-7500 --tags_file data/index_ins_del.after-concurrent-delete-del2500-7500.tags
json_time $BASE_PATH/search_memory_index --data_type uint8 --dist_fn l2 $MEMORY_RECALL --index_path_prefix data/index_ins_del.after-concurrent-delete-del2500-7500 --result_path data/res_ins_del --query_file data/rand_uint8_10D_1K_norm50.0.bin --gt_file data/gt100_random10D_10K-conc-2500-7500 -K 10 -L 20 40 60 80 100 -T 8 --dynamic true --tags 1

if [ -f "$BASE_PATH/utils/generate_synthetic_labels" ]; then
  echo "Labeled Tests"
  json_time $BASE_PATH/utils/generate_synthetic_labels  --num_labels 50 --num_points 10000  --output_file data/rand_labels_50_10K.txt --distribution_type random
  json_time $BASE_PATH/utils/compute_groundtruth_for_filters  --data_type uint8 --dist_fn l2 --universal_label 0 --filter_label 10 --base_file data/rand_uint8_10D_10K_norm50.0.bin --query_file data/rand_uint8_10D_1K_norm50.0.bin --label_file data/rand_labels_50_10K.txt --gt_file data/l2_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel --K 100
  json_time $BASE_PATH/utils/compute_groundtruth_for_filters  --data_type uint8 --dist_fn mips --universal_label 0 --filter_label 10 --base_file data/rand_uint8_10D_10K_norm50.0.bin --query_file data/rand_uint8_10D_1K_norm50.0.bin --label_file data/rand_labels_50_10K.txt --gt_file data/mips_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel --K 100
  json_time $BASE_PATH/utils/compute_groundtruth_for_filters  --data_type uint8 --dist_fn cosine --universal_label 0 --filter_label 10 --base_file data/rand_uint8_10D_10K_norm50.0.bin --query_file data/rand_uint8_10D_1K_norm50.0.bin --label_file data/rand_labels_50_10K.txt --gt_file data/cosine_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel --K 100

  json_time $BASE_PATH/utils/generate_synthetic_labels  --num_labels 50 --num_points 10000  --output_file data/zipf_labels_50_10K.txt --distribution_type zipf
  json_time $BASE_PATH/utils/compute_groundtruth_for_filters  --data_type uint8 --dist_fn l2 --universal_label 0 --filter_label 5 --base_file data/rand_uint8_10D_10K_norm50.0.bin --query_file data/rand_uint8_10D_1K_norm50.0.bin --label_file data/zipf_labels_50_10K.txt --gt_file data/l2_zipf_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel --K 100
  json_time $BASE_PATH/utils/compute_groundtruth_for_filters  --data_type uint8 --dist_fn mips --universal_label 0 --filter_label 5 --base_file data/rand_uint8_10D_10K_norm50.0.bin --query_file data/rand_uint8_10D_1K_norm50.0.bin --label_file data/zipf_labels_50_10K.txt --gt_file data/mips_zipf_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel --K 100
  json_time $BASE_PATH/utils/compute_groundtruth_for_filters  --data_type uint8 --dist_fn cosine --universal_label 0 --filter_label 5 --base_file data/rand_uint8_10D_10K_norm50.0.bin --query_file data/rand_uint8_10D_1K_norm50.0.bin --label_file data/zipf_labels_50_10K.txt --gt_file data/cosine_zipf_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel --K 100

  json_time $BASE_PATH/utils/compute_groundtruth_for_filters  --data_type uint8 --dist_fn l2 --filter_label 5 --base_file data/rand_uint8_10D_10K_norm50.0.bin --query_file data/rand_uint8_10D_1K_norm50.0.bin --label_file data/zipf_labels_50_10K.txt --gt_file data/l2_zipf_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel_nouniversal --K 100
  json_time $BASE_PATH/utils/generate_synthetic_labels  --num_labels 10 --num_points 1000  --output_file data/query_labels_1K.txt --distribution_type one_per_point
  json_time $BASE_PATH/utils/compute_groundtruth_for_filters  --data_type uint8 --dist_fn l2 --universal_label 0 --filter_label_file data/query_labels_1K.txt --base_file data/rand_uint8_10D_10K_norm50.0.bin --query_file data/rand_uint8_10D_1K_norm50.0.bin --label_file data/zipf_labels_50_10K.txt --gt_file data/combined_l2_zipf_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel --K 100

  json_time $BASE_PATH/build_memory_index --data_type uint8 --dist_fn l2 --FilteredLbuild 90 --universal_label 0 --data_path data/rand_uint8_10D_10K_norm50.0.bin --label_file data/rand_labels_50_10K.txt --index_path_prefix data/index_l2_rand_uint8_10D_10K_norm50_wlabel
  json_time $BASE_PATH/build_memory_index --data_type uint8 --dist_fn cosine --FilteredLbuild 90 --universal_label 0 --data_path data/rand_uint8_10D_10K_norm50.0.bin --label_file data/rand_labels_50_10K.txt --index_path_prefix data/index_cosine_rand_uint8_10D_10K_norm50_wlabel
  json_time $BASE_PATH/search_memory_index --data_type uint8 --dist_fn l2 --filter_label 10 $MEMORY_RECALL --index_path_prefix data/index_l2_rand_uint8_10D_10K_norm50_wlabel --query_file data/rand_uint8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/l2_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel -L  16 32
  json_time $BASE_PATH/search_memory_index --data_type uint8 --dist_fn cosine --filter_label 10 $MEMORY_RECALL --index_path_prefix data/index_cosine_rand_uint8_10D_10K_norm50_wlabel --query_file data/rand_uint8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/cosine_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel -L  16 32

  json_time $BASE_PATH/build_disk_index --data_type uint8 --dist_fn l2 --universal_label 0  --FilteredLbuild 90 --data_path data/rand_uint8_10D_10K_norm50.0.bin --label_file data/rand_labels_50_10K.txt --index_path_prefix data/disk_index_l2_rand_uint8_10D_10K_norm50_wlabel -R 16 -L 32 -B 0.00003 -M 1
  json_time $BASE_PATH/search_disk_index --data_type uint8 --dist_fn l2 --filter_label 10 --fail_if_recall_below 50 --index_path_prefix data/disk_index_l2_rand_uint8_10D_10K_norm50_wlabel --result_path /tmp/res --query_file data/rand_uint8_10D_1K_norm50.0.bin --gt_file data/l2_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

  json_time $BASE_PATH/build_memory_index --data_type uint8 --dist_fn l2 --FilteredLbuild 90 --universal_label 0 --data_path data/rand_uint8_10D_10K_norm50.0.bin --label_file data/zipf_labels_50_10K.txt --index_path_prefix data/index_l2_zipf_uint8_10D_10K_norm50_wlabel
  json_time $BASE_PATH/build_memory_index --data_type uint8 --dist_fn cosine --FilteredLbuild 90 --universal_label 0 --data_path data/rand_uint8_10D_10K_norm50.0.bin --label_file data/zipf_labels_50_10K.txt --index_path_prefix data/index_cosine_zipf_uint8_10D_10K_norm50_wlabel
  json_time $BASE_PATH/search_memory_index --data_type uint8 --dist_fn l2 --filter_label 5 $MEMORY_RECALL --index_path_prefix data/index_l2_zipf_uint8_10D_10K_norm50_wlabel --query_file data/rand_uint8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/l2_zipf_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel -L  16 32
  json_time $BASE_PATH/search_memory_index --data_type uint8 --dist_fn cosine --filter_label 5 $MEMORY_RECALL --index_path_prefix data/index_cosine_zipf_uint8_10D_10K_norm50_wlabel --query_file data/rand_uint8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/cosine_zipf_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel -L  16 32

  json_time $BASE_PATH/build_disk_index --data_type uint8 --dist_fn l2 --universal_label 0  --FilteredLbuild 90 --data_path data/rand_uint8_10D_10K_norm50.0.bin --label_file data/zipf_labels_50_10K.txt --index_path_prefix data/disk_index_l2_zipf_uint8_10D_10K_norm50_wlabel -R 16 -L 32 -B 0.00003 -M 1
  json_time $BASE_PATH/search_disk_index --data_type uint8 --dist_fn l2 --filter_label 5 --fail_if_recall_below 50 --index_path_prefix data/disk_index_l2_zipf_uint8_10D_10K_norm50_wlabel --result_path /tmp/res --query_file data/rand_uint8_10D_1K_norm50.0.bin --gt_file data/l2_zipf_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

  json_time $BASE_PATH/build_memory_index --data_type uint8 --dist_fn l2 --FilteredLbuild 90 --data_path data/rand_uint8_10D_10K_norm50.0.bin --label_file data/zipf_labels_50_10K.txt --index_path_prefix data/index_l2_zipf_uint8_10D_10K_norm50_wlabel_nouniversal
  json_time $BASE_PATH/build_disk_index --data_type uint8 --dist_fn l2  --FilteredLbuild 90 --data_path data/rand_uint8_10D_10K_norm50.0.bin --label_file data/zipf_labels_50_10K.txt --index_path_prefix data/disk_index_l2_zipf_uint8_10D_10K_norm50_wlabel_nouniversal -R 16 -L 32 -B 0.00003 -M 1
  json_time $BASE_PATH/search_memory_index --data_type uint8 --dist_fn l2 --filter_label 5 $MEMORY_RECALL --index_path_prefix data/index_l2_zipf_uint8_10D_10K_norm50_wlabel_nouniversal --query_file data/rand_uint8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/l2_zipf_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel_nouniversal -L  16 32
  json_time $BASE_PATH/search_disk_index --data_type uint8 --dist_fn l2 --filter_label 5 --index_path_prefix data/disk_index_l2_zipf_uint8_10D_10K_norm50_wlabel_nouniversal --result_path /tmp/res --query_file data/rand_uint8_10D_1K_norm50.0.bin --gt_file data/l2_zipf_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel_nouniversal --recall_at 5 -L 5 12 -W 2 --num_nodes_to_cache 10 -T 16

  json_time $BASE_PATH/build_memory_index --data_type uint8 --dist_fn l2 --FilteredLbuild 90 --universal_label 0 --data_path data/rand_uint8_10D_10K_norm50.0.bin --label_file data/zipf_labels_50_10K.txt --index_path_prefix data/index_l2_zipf_uint8_10D_10K_norm50_wlabel
  json_time $BASE_PATH/search_memory_index --data_type uint8 --dist_fn l2 --query_filters_file data/query_labels_1K.txt $MEMORY_RECALL --index_path_prefix data/index_l2_zipf_uint8_10D_10K_norm50_wlabel --query_file data/rand_uint8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/combined_l2_zipf_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel -L  16 32

  json_time $BASE_PATH/build_memory_index --data_type uint8 --dist_fn l2 --FilteredLbuild 90 --universal_label 0 --data_path data/rand_uint8_10D_10K_norm50.0.bin --label_file data/rand_labels_50_10K.txt --index_path_prefix data/index_l2_rand_uint8_10D_10K_norm50_wlabel --build_PQ_bytes 5
  json_time $BASE_PATH/search_memory_index --data_type uint8 --dist_fn l2 --filter_label 10 $MEMORY_RECALL --index_path_prefix data/index_l2_rand_uint8_10D_10K_norm50_wlabel --query_file data/rand_uint8_10D_1K_norm50.0.bin --recall_at 10 --result_path temp --gt_file data/l2_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel -L 16 32

  json_time $BASE_PATH/build_stitched_index --num_threads 48 --data_type uint8 --data_path data/rand_uint8_10D_10K_norm50.0.bin --label_file data/rand_labels_50_10K.txt -R 32 -L 100 --alpha 1.2 --stitched_R 64 --index_path_prefix data/stit_rand_32_100_64_new --universal_label 0
  json_time $BASE_PATH/build_stitched_index --num_threads 48 --data_type uint8 --data_path data/rand_uint8_10D_10K_norm50.0.bin --label_file data/zipf_labels_50_10K.txt -R 32 -L 100 --alpha 1.2 --stitched_R 64 --index_path_prefix data/stit_zipf_32_100_64_new --universal_label 0
  json_time $BASE_PATH/search_memory_index --num_threads 48 --data_type uint8 --dist_fn l2 --filter_label 10 --index_path_prefix data/stit_rand_32_100_64_new --query_file data/rand_uint8_10D_1K_norm50.0.bin --result_path data/rand_stit_96_10_90_new --gt_file data/l2_rand_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel -K 10 -L 16 32 150
  json_time $BASE_PATH/search_memory_index --num_threads 48 --data_type uint8 --dist_fn l2 --filter_label 5 --index_path_prefix data/stit_zipf_32_100_64_new --query_file data/rand_uint8_10D_1K_norm50.0.bin --result_path data/zipf_stit_96_10_90_new --gt_file data/l2_zipf_uint8_10D_10K_norm50.0_10D_1K_norm50.0_gt100_wlabel -K 10 -L 16 32 150
fi
