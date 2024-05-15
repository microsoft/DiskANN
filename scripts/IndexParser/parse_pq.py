import parse_common as pc

def parse_compressed_vectors(file_prefix) :
    file_name = file_prefix + "_pq_compressed.bin"
    compressed_vectors = pc.DataMat('B', 1)
    compressed_vectors.load_bin(file_name)
    return compressed_vectors

def parse_pivots_file(file_prefix):
    file_name = file_prefix + "_pq_pivots.bin"
    with open(file_name, "rb") as file:
        metadata_mat = pc.DataMat('Q', 8)
        metadata_mat.load_bin_from_opened_file(file)
        num_metadata = metadata_mat.num_rows
        num_dims = metadata_mat.num_cols
        assert num_dims == 1 and (num_metadata == 4 or num_metadata == 5)


        for i in range(num_metadata):
            for j in range(num_dims):
                print (metadata_mat[i][j])
            print("\n")

        pivots = pc.DataMat('f', 4)
        pivots.load_bin_from_opened_file(file, metadata_mat[0][0])
        assert pivots.num_rows == pc.NUM_PQ_CENTROIDS

        centroids = pc.DataMat('f', 4)
        centroids.load_bin_from_opened_file(file, metadata_mat[1][0])
        assert centroids.num_rows == pivots.num_cols
        assert centroids.num_cols == 1

        #Assuming new file format =>(chunk offset is at offset 3) because we will not encounter old index formats now. 
        chunk_offsets = pc.DataMat('I', 4)
        chunk_offsets.load_bin_from_opened_file(file, metadata_mat[2][0])
        #assert chunk_offsets.num_rows == pivots.num_cols + 1 or chunk_offsets.num_rows == 0
        assert chunk_offsets.num_cols == 1
        #Ignoring rotmat for now. Also ignoring diskPQ

    return pivots, centroids, chunk_offsets
