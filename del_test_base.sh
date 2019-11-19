#!/bin/bash
TOTAL=100000
let "TOTAL += $2"
let "REM = $TOTAL - $1"
echo $REM 
echo $TOTAL
./build/tests/test_nsg_incr_index ../sift_learn.bin 100 64 750 1.2 2 ../tmp-incr-${1}-${2}.nsg ${1} ${2}
../exp-knn/exact -o ../tmp-incr-${1}-${2}.nsg.del.data ${REM} ../sift_query.fvecs 10000 128 100 ../gs_100_tmp-incr-${1}-${2}.nsg.del
./build/tests/utils/ivecs_to_bin /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.del.ivecs /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.del.data
./build/tests/search_memory_index float /mnt/aditi/tmp-incr-${1}-${2}.nsg.del.data /mnt/aditi/tmp-incr-${1}-${2}.nsg.del /mnt/SIFT1M/sift_query.bin 5 8 /mnt/aditi/tmp-incr-${1}-${2}-del_ 60 64 68 70 72
./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.del.data /mnt/aditi/tmp-incr-${1}-${2}-del_60_idx_uint32.bin 5
./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.del.data /mnt/aditi/tmp-incr-${1}-${2}-del_64_idx_uint32.bin 5
./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.del.data /mnt/aditi/tmp-incr-${1}-${2}-del_68_idx_uint32.bin 5
./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.del.data /mnt/aditi/tmp-incr-${1}-${2}-del_70_idx_uint32.bin 5
./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.del.data /mnt/aditi/tmp-incr-${1}-${2}-del_72_idx_uint32.bin 5

