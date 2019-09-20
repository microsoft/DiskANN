#!/bin/bash

command_helper()
{
    echo ""
    echo "Usage: $0 -t data_type -i input_bin_file -o index_prefix -L Lvalue -r Rvalue -b PQ_vector_bytes"
    echo -e "\t-t Base file data type: float/int8/uint8"
    echo -e "\t-i input bin file path"
    echo -e "\t-o output index prefix path (will generate path_pq_pivots.bin, path_compressed.bin, path_diskopt.bin)"
    echo -e "\t-L index construction quality (L = 30 to 100 works, 50 is good choice)"
    echo -e "\t-R index maximum degree"
    echo -e "\t-b approximate memory footprint per vector in bytes using product quantization"
    exit 1 # Exit script after printing help
}

while getopts "t:i:o:L:R:b:" opt
do
    case "$opt" in
        t ) TYPE="$OPTARG" ;;
        i ) DATA="$OPTARG" ;;
        o ) OUTPUT_PREFIX="$OPTARG" ;;
        L ) L="$OPTARG" ;;
        R ) R="$OPTARG" ;;
        b ) B="$OPTARG" ;;
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

DISK_INDEX_PATH="${OUTPUT_PREFIX}_diskopt.index"
UNOPT_INDEX_PATH="${OUTPUT_PREFIX}_mem.index"

${BUILD_PATH}/tests/generate_pq  $TYPE  $DATA  $OUTPUT_PREFIX  $B  0.01
${BUILD_PATH}/tests/build_memory_index  $TYPE  $DATA  $L  $R  1250  2  1.2  $UNOPT_INDEX_PATH
${BUILD_PATH}/tests/create_disk_layout $TYPE $DATA $UNOPT_INDEX_PATH $DISK_INDEX_PATH