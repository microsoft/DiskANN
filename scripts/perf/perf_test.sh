#!/bin/bash

function json_time {
  command="$@"
  echo "Executing $command"
  /usr/bin/time --quiet -o /app/logs/time.log -a --format '{"command":"%C", "wallclock": %e, "user": %U, "sys": %S}' $command
  ret=$?
  if [ $ret -ne 0 ]; then
    echo "{\"command\": \""$command"\", \"status_code\": $ret}" >> /app/logs/time.log
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

json_time $BASE_PATH/utils/rand_data_gen --data_type float --output_file data/rand_float_768D_1M_norm1.0.bin -D 768 -N 1000000 --norm 1.0
json_time $BASE_PATH/utils/rand_data_gen --data_type float --output_file data/rand_float_768D_10K_norm1.0.bin -D 768 -N 10000 --norm 1.0

json_time $BASE_PATH/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file data/rand_float_768D_1M_norm1.0.bin --query_file data/rand_float_768D_10K_norm1.0.bin --gt_file data/l2_rand_float_768D_1M_norm1.0_768D_10K_norm1.0_gt100 --K 100
json_time $BASE_PATH/utils/compute_groundtruth  --data_type float --dist_fn mips --base_file data/rand_float_768D_1M_norm1.0.bin --query_file data/rand_float_768D_10K_norm1.0.bin --gt_file data/mips_rand_float_768D_1M_norm1.0_768D_10K_norm1.0_gt100 --K 100
json_time $BASE_PATH/utils/compute_groundtruth  --data_type float --dist_fn cosine --base_file data/rand_float_768D_1M_norm1.0.bin --query_file data/rand_float_768D_10K_norm1.0.bin --gt_file data/cosine_rand_float_768D_1M_norm1.0_768D_10K_norm1.0_gt100 --K 100

json_time $BASE_PATH/build_memory_index --data_type float --dist_fn l2 --data_path data/rand_float_768D_1M_norm1.0.bin --index_path_prefix data/index_l2_rand_float_768D_1M_norm1.0
json_time $BASE_PATH/search_memory_index --data_type float --dist_fn l2 --index_path_prefix data/index_l2_rand_float_768D_1M_norm1.0 --query_file data/rand_float_768D_10K_norm1.0.bin --recall_at 10 --result_path temp --gt_file data/l2_rand_float_768D_1M_norm1.0_768D_10K_norm1.0_gt100 -L 100 32
json_time $BASE_PATH/search_memory_index --data_type float --dist_fn fast_l2 --index_path_prefix data/index_l2_rand_float_768D_1M_norm1.0 --query_file data/rand_float_768D_10K_norm1.0.bin --recall_at 10 --result_path temp --gt_file data/l2_rand_float_768D_1M_norm1.0_768D_10K_norm1.0_gt100 -L 100 32

json_time $BASE_PATH/build_memory_index --data_type float --dist_fn mips --data_path data/rand_float_768D_1M_norm1.0.bin --index_path_prefix data/index_mips_rand_float_768D_1M_norm1.0
json_time $BASE_PATH/search_memory_index --data_type float --dist_fn mips --index_path_prefix data/index_l2_rand_float_768D_1M_norm1.0 --query_file data/rand_float_768D_10K_norm1.0.bin --recall_at 10 --result_path temp --gt_file data/mips_rand_float_768D_1M_norm1.0_768D_10K_norm1.0_gt100 -L 100 32

json_time $BASE_PATH/build_memory_index --data_type float --dist_fn cosine --data_path data/rand_float_768D_1M_norm1.0.bin --index_path_prefix data/index_cosine_rand_float_768D_1M_norm1.0
json_time $BASE_PATH/search_memory_index --data_type float --dist_fn cosine --index_path_prefix data/index_l2_rand_float_768D_1M_norm1.0 --query_file data/rand_float_768D_10K_norm1.0.bin --recall_at 10 --result_path temp --gt_file data/cosine_rand_float_768D_1M_norm1.0_768D_10K_norm1.0_gt100 -L 100 32

