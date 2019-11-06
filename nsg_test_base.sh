#!/bin/bash
TOTAL=1000000
let "TOTAL += $2"
let "REM = $TOTAL - $1"
echo $REM
echo $TOTAL
./build/tests/test_nsg_incr_index /mnt/SIFT1M/sift_base.bin 100 64 750 1.2 2 /mnt/aditi/tmp-incr-${1}-${2}.nsg ${1} ${2}
 ../exp-knn/exact -o /mnt/aditi/tmp-incr-${1}-${2}.nsg.inc.data ${TOTAL} /mnt/SIFT1M/sift_query.fvecs 10000 128 100 /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.inc
./build/tests/utils/ivecs_to_bin /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.inc.ivecs /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.inc.data
./build/tests/search_memory_index float /mnt/aditi/tmp-incr-${1}-${2}.nsg.inc.data /mnt/aditi/tmp-incr-${1}-${2}.nsg.inc /mnt/SIFT1M/sift_query.bin 5 8 /mnt/aditi/tmp-incr-${1}-${2}-inc_ 60 64 68 70 72
 ./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.inc.data /mnt/aditi/tmp-incr-${1}-${2}-inc_60_idx_uint32.bin 5
 ./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.inc.data /mnt/aditi/tmp-incr-${1}-${2}-inc_64_idx_uint32.bin 5
 ./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.inc.data /mnt/aditi/tmp-incr-${1}-${2}-inc_68_idx_uint32.bin 5
 ./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.inc.data /mnt/aditi/tmp-incr-${1}-${2}-inc_70_idx_uint32.bin 5
 ./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.inc.data /mnt/aditi/tmp-incr-${1}-${2}-inc_72_idx_uint32.bin 5
