#!/bin/bash

command_helper()
{
    echo ""
    echo "Usage: $0 -t data_type -i input_bin_file -o index_prefix -L Lvalue -R Rvalue -B PQ_vector_bytes -M number_of_pieces -S sampling_rate"
    echo -e "\t-t Base file data type: float/int8/uint8"
    echo -e "\t-i input bin file path"
    echo -e "\t-o output index prefix path (will generate path_pq_pivots.bin, path_compressed.bin, path_diskopt.bin)"
    echo -e "\t-L index construction quality (L = 30 to 100 works, 50 is good choice)"
    echo -e "\t-R index maximum degree"
    echo -e "\t-B approximate memory footprint per vector in bytes using product quantization"
    echo -e "\t-M number of shards"
    echo -e "\t-S sampling rate for k-means"
    exit 1 # Exit script after printing help
}

while getopts "t:i:o:L:R:B:S:M:" opt
do
    case "$opt" in
        t ) TYPE="$OPTARG" ;;
        i ) DATA="$OPTARG" ;;
        o ) OUTPUT_PREFIX="$OPTARG" ;;
        L ) L="$OPTARG" ;;
        R ) R="$OPTARG" ;;
        B ) B="$OPTARG" ;;
        M ) NUM_SHARDS="$OPTARG" ;;
        S ) RATE="$OPTARG" ;;
        ? ) command_helper ;; # Print command_helper in case parameter is non-existent
    esac
done

# Print command_helper in case parameters are empty
if [ -z "$TYPE" ] || [ -z "$DATA" ] || [ -z "$OUTPUT_PREFIX" ] || [ -z "$L" ] || [ -z "$R" ] || [ -z "$B" ] || [ -z "$NUM_SHARDS" ] || [ -z "$RATE" ]
then
    echo "Some or all of the parameters are empty";
    command_helper
fi

# Begin script in case all parameters are correct
echo "Building $TYPE disk-index on $DATA by partitioning with $NUM_SHARDS  shards with replication factor of 2, with L=$L, R=$R, B=$B and storing output files in prefix $OUTPUT_PREFIX"

DISK_INDEX_PATH="${OUTPUT_PREFIX}_disk.index"
UNOPT_INDEX_PATH="${OUTPUT_PREFIX}_mem.index"

let "SHARD_R = 2 * $R / 3"

echo "Shard R = $SHARD_R"


#partitions the data using k-means into $NUM_SHARDS pieces, by placing each point in 2 closest clusters to obtain overlapping clusters. k-means is run on sampled data using $RATE sampling rate
`rm ${OUTPUT_PREFIX}_subshard-*`
${BUILD_PATH}/tests/partition_data $TYPE $DATA $OUTPUT_PREFIX $RATE $NUM_SHARDS 2
#generates the compressed vectors for each vector into $B bytes by running PQ scheme on sample of data sampled at rate $RATE
${BUILD_PATH}/tests/generate_pq  $TYPE  $DATA  $OUTPUT_PREFIX  $B  $RATE
# build in-memory graphs for each piece/shard of the partitioned data

LOOP_END=`ls ${OUTPUT_PREFIX}_subshard-*_ids_uint32.bin | wc -l`
#`ls ${OUTPUT_PREFIX}_subshard-*`

let "LOOP_END = $LOOP_END - 1"
echo "LOOP END = ${LOOP_END}"

for i in `seq 0 $LOOP_END`
do 
    ${BUILD_PATH}/tests/build_memory_index  $TYPE  "${OUTPUT_PREFIX}_subshard-${i}.bin"  $L  $SHARD_R  750  2  2  "${OUTPUT_PREFIX}_subshard-${i}_mem.index" 0
done
let "LOOP_END = $LOOP_END + 1"
#merge all the shards by taking a union of the overlapping graphs
${BUILD_PATH}/tests/merge_shards "${OUTPUT_PREFIX}_subshard-" _mem.index "${OUTPUT_PREFIX}_subshard-" _ids_uint32.bin ${LOOP_END} $R $UNOPT_INDEX_PATH
#creates the disk layout by combining the full-precision data vectors and merged graph
${BUILD_PATH}/tests/create_disk_layout $TYPE $DATA $UNOPT_INDEX_PATH $DISK_INDEX_PATH
