#BASE_FILE=/home/t-adisin/sift_base.bin
#QUERY_FILE=/home/t-adisin/sift_query.bin
#GT_FILE=~/gs500_sift1m
#NUM_BASE=900000
#NUM_DELETE=50000
#NUM_INCR=50000
#NUM_CYCLES=30
#SAVE_PATH=/mnt/rakri/save/SIFT1M
#LOGFILE=~/sift1m_${NUM_BASE}_${NUM_DELETE}_${NUM_INCR}_${NUM_CYCLES}

BASE_FILE=/mnt/aditi/datasets/sift_rnd100m_data.bin
QUERY_FILE=/home/t-adisin/datasets/bigann_query_float.bin
GT_FILE=~/gs500_sift100m
NUM_BASE=90000000
NUM_DELETE=4500000
NUM_INCR=4500000
NUM_CYCLES=50
SAVE_PATH=/mnt/rakri/save/SIFT100M
LOGFILE=~/sift100m_${NUM_BASE}_${NUM_DELETE}_${NUM_INCR}_${NUM_CYCLES}

WORKING_PATH=/mnt/rakri/test3/
#WORKING_PATH=/dev/shm/test4

INDEX_FILE=merger_index_${NUM_BASE}_${NUM_DELETE}_${NUM_INCR}_${NUM_CYCLES}

#rm -rf /mnt/rakri/*pq*
./tests/utils/seed_index_merger float $NUM_BASE 1 $NUM_DELETE $NUM_INCR $NUM_CYCLES $BASE_FILE ${SAVE_PATH}/${INDEX_FILE} ${SAVE_PATH}/${INDEX_FILE}_deleted.tags  2>&1 0</dev/null 1>${LOGFILE}_build.log


# train indices
numactl --interleave=all ./tests/build_disk_index float ${SAVE_PATH}/${INDEX_FILE}_base.data ${WORKING_PATH}/${INDEX_FILE}_cycle_0 64 75 100 500 64 2>&1 0</dev/null 1>>${LOGFILE}_build.log
cp ${SAVE_PATH}/${INDEX_FILE}_base.tags ${WORKING_PATH}/${INDEX_FILE}_cycle_0_disk.index.tags

mkdir $SAVE_PATH/base_copy/
cp ${WORKING_PATH}/${INDEX_FILE}_cycle_0* $SAVE_PATH/base_copy/

#./tests/utils/compute_groundtruth float $BASE_FILE $QUERY_FILE 500 $GT_FILE

# search on base index
numactl --interleave=all ./tests/search_disk_index float $WORKING_PATH/${INDEX_FILE}_cycle_0 100000 32 4 $QUERY_FILE $GT_FILE 10 /tmp/abc 100 2>&1 0</dev/null 1>${LOGFILE}_search.log

for (( c=0; c<$NUM_CYCLES; c++ ))
do
	((nextc = c + 1))
	cp ${SAVE_PATH}/${INDEX_FILE}_cycle_${nextc}_mem_1.data $WORKING_PATH/
	cp ${SAVE_PATH}/${INDEX_FILE}_cycle_${nextc}_mem_1.tags $WORKING_PATH/
	cp ${SAVE_PATH}/${INDEX_FILE}_deleted.tags_cycle_${nextc} $WORKING_PATH/
	numactl --interleave=all ./tests/build_memory_index float $WORKING_PATH/${INDEX_FILE}_cycle_${nextc}_mem_1.data $WORKING_PATH/${INDEX_FILE}_cycle_${nextc}_mem_1.index 64 75 1.2 64
	cp $WORKING_PATH/${INDEX_FILE}_cycle_${nextc}_mem_1.tags $WORKING_PATH/${INDEX_FILE}_cycle_${nextc}_mem_1.index.tags 2>&1 0</dev/null 1>>${LOGFILE}_build.log
	/usr/bin/time numactl --interleave=all ./tests/test_index_merger float $WORKING_PATH/${INDEX_FILE}_cycle_${c} $WORKING_PATH/${INDEX_FILE}_cycle_${nextc} $WORKING_PATH/${INDEX_FILE}_deleted.tags_cycle_${nextc} 128 4 64 70 1.2 1000 $WORKING_PATH $WORKING_PATH/${INDEX_FILE}_cycle_${nextc}_mem_1.index 2>&1 0</dev/null 1>>${LOGFILE}_build.log
	numactl --interleave=all ./tests/search_disk_index float $WORKING_PATH/${INDEX_FILE}_cycle_${nextc} 100000 32 4 $QUERY_FILE $GT_FILE 10 /tmp/def 100 2>&1 0</dev/null 1>>${LOGFILE}_search.log
	sudo rm -rf  $WORKING_PATH/${INDEX_FILE}_cycle_${c}_* 
	rm -rf $SAVE_PATH/${INDEX_FILE}_cycle_${c}_*
	cp $WORKING_PATH/${INDEX_FILE}_cycle_${nextc}* $SAVE_PATH/
done

