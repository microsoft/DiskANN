#!/bin/bash
TOTAL=100000
let "TOTAL += $2"
let "REM = $TOTAL - $1"
echo $REM
echo $TOTAL
./build/tests/test_nsg_incr_index ../sift_learn.bin 100 64 750 1.2 2 /mnt/aditi/tmp-incr-${1}-${2}.nsg ${1} ${2}
 ../exp-knn/exact -o /mnt/aditi/tmp-incr-${1}-${2}.nsg.reinc.data ${TOTAL} ../sift_query.fvecs 10000 128 100 /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.reinc
./build/tests/utils/ivecs_to_bin /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.reinc.ivecs /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.reinc.data
./build/tests/search_memory_index float /mnt/aditi/tmp-incr-${1}-${2}.nsg.reinc.data /mnt/aditi/tmp-incr-${1}-${2}.nsg.reinc /mnt/SIFT1M/sift_query.bin 5 8 /mnt/aditi/tmp-incr-${1}-${2}-reinc_ ${3} ${4} ${5} ${6}
 ./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.reinc.data /mnt/aditi/tmp-incr-${1}-${2}-reinc_${3}_idx_uint32.bin 5
./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.reinc.data /mnt/aditi/tmp-incr-${1}-${2}-reinc_${4}_idx_uint32.bin 5
./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.reinc.data /mnt/aditi/tmp-incr-${1}-${2}-reinc_${5}_idx_uint32.bin 5
./build/tests/utils/calculate_recall /mnt/aditi/gs_100_tmp-incr-${1}-${2}.nsg.reinc.data /mnt/aditi/tmp-incr-${1}-${2}-reinc_${6}_idx_uint32.bin 5
