#!/bin/bash
TOTAL=1000000
let "TOTAL += $2"
let "REM = $TOTAL - $1"
echo $REM
echo $TOTAL
./build/tests/test_nsg_incr_index ../../SIFT1M/sift_base.bin 100 64 750 1.2 2 ../tmp-incr-${1}-${2}.nsg ${1} ${2}
 ../exp-knn/exact -o ../tmp-incr-${1}-${2}.nsg.inc.data ${TOTAL} ../sift_query.fvecs 10000 128 100 ../gs_100_tmp-incr-${1}-${2}.nsg.inc
./build/tests/utils/ivecs_to_bin ../gs_100_tmp-incr-${1}-${2}.nsg.inc.ivecs ../gs_100_tmp-incr-${1}-${2}.nsg.inc.data
./build/tests/search_memory_index float ../tmp-incr-${1}-${2}.nsg.inc.data ../tmp-incr-${1}-${2}.nsg.inc /mnt/SIFT1M/sift_query.bin 5 8 ../tmp-incr-${1}-${2}-inc_ 60 64 68 70 72
 ./build/tests/utils/calculate_recall ../gs_100_tmp-incr-${1}-${2}.nsg.inc.data ../tmp-incr-${1}-${2}-inc_60_idx_uint32.bin 5
 ./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.inc.data /mnt/aditi/tmp-incr-${1}-${2}-inc_64_idx_uint32.bin 5
 ./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.inc.data /mnt/aditi/tmp-incr-${1}-${2}-inc_68_idx_uint32.bin 5
 ./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.inc.data /mnt/aditi/tmp-incr-${1}-${2}-inc_70_idx_uint32.bin 5
 ./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.inc.data /mnt/aditi/tmp-incr-${1}-${2}-inc_72_idx_uint32.bin 5
