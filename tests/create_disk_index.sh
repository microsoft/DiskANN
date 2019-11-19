#!/bin/bash

command_helper()
{
    echo ""
    echo "Usage: $0 -t data_type -i input_bin_file -o index_prefix -L Lvalue -R Rvalue -B PQ_vector_bytes"
    echo -e "\t-t Base file data type: float/int8/uint8"
    echo -e "\t-i input bin file path"
    echo -e "\t-o output index prefix path (will generate path_pq_pivots.bin, path_compressed.bin, path_diskopt.bin)"
    echo -e "\t-L index construction quality (e.g., L = 30, 50, 100 are good choices)"
    echo -e "\t-R index maximum degree (e.g., R = 64, 100 are reasonable choices)"
    echo -e "\t-B approximate memory footprint per vector in bytes using product quantization"
    exit 1 # Exit script after printing help
}

while getopts "t:i:o:L:R:B:" opt
do
    case "$opt" in
        t ) TYPE="$OPTARG" ;;
        i ) DATA="$OPTARG" ;;
        o ) OUTPUT_PREFIX="$OPTARG" ;;
        L ) L="$OPTARG" ;;
        R ) R="$OPTARG" ;;
        B ) B="$OPTARG" ;;
        ? ) command_helper ;; # Print command_helper in case parameter is non-existent
    esac
done

# Print command_helper in case parameters are empty
if [ -z "$TYPE" ] || [ -z "$DATA" ] || [ -z "$OUTPUT_PREFIX" ] || [ -z "$L" ] || [ -z "$R" ] || [ -z "$B" ]
then
    echo "Some or all of the parameters are empty";
    command_helper
fi

# Begin script in case all parameters are correct
echo "Building $TYPE disk-index on $DATA with L=$L, R=$R, B=$B and storing output files in prefix $OUTPUT_PREFIX"

DISK_INDEX_PATH="${OUTPUT_PREFIX}_disk.index"
UNOPT_INDEX_PATH="${OUTPUT_PREFIX}_mem.index"
# generating compressed vectors into $B bytes per vector, using a sampling rate of 0.01. Change if you want higher or lower sampling rate (higher is better).
${BUILD_PATH}/tests/generate_pq  $TYPE  $DATA  $OUTPUT_PREFIX  $B  0.03
#builds the in-memory graph index using 2 passes over data, and using alpha = 1.2, with size of L_construction = $L and max-degree $R. 
${BUILD_PATH}/tests/build_memory_index  $TYPE  $DATA  $L  $R  2000  2  3  $UNOPT_INDEX_PATH
#creates the disk layout by storing the graph and data together 
${BUILD_PATH}/tests/create_disk_layout $TYPE $DATA $UNOPT_INDEX_PATH $DISK_INDEX_PATH
