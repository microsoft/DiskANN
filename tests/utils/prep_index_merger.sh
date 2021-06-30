BASE_PREFIX="/dev/shm/test/sample_base"
MEM_PREFIX="/dev/shm/test/sample_mem"
DELETE_LIST="/dev/shm/sample_deleted.tags"
ONESHOT_PREFIX="/dev/shm/test/sample_oneshot"
MERGED_PREFIX="/dev/shm/test/sample_merged"
NUM_MEM_INDICES=5
# copy tags from base -> base_index
cp ${BASE_PREFIX}.tags ${BASE_PREFIX}_index_disk.index.tags
cp ${ONESHOT_PREFIX}.tags ${ONESHOT_PREFIX}_index_disk.index.tags

# copy tags file for mem indices
for i in $(seq 1 $NUM_MEM_INDICES)
do
    cp ${MEM_PREFIX}_${i}.tags ${MEM_PREFIX}_${i}_index.tags
done

# copy PQ stuff for merged from base
cp ${BASE_PREFIX}_index_pq_pivots.bin ${MERGED_PREFIX}_index_pq_pivots.bin
cp ${BASE_PREFIX}_index_pq_pivots.bin_centroid.bin ${MERGED_PREFIX}_index_pq_pivots.bin_centroid.bin
cp ${BASE_PREFIX}_index_pq_pivots.bin_chunk_offsets.bin ${MERGED_PREFIX}_index_pq_pivots.bin_chunk_offsets.bin
cp ${BASE_PREFIX}_index_pq_pivots.bin_rearrangement_perm.bin ${MERGED_PREFIX}_index_pq_pivots.bin_rearrangement_perm.bin
