"""Read a DiskANN index """
import argparse
import struct
import array

import parse_disk_index as pdi
import parse_pq as ppq


def main(index_path_prefix, data_type, output_file_prefix, use_pq_vectors): 
    data_type_size = 0
    data_type_code = ''
    if data_type == "float":
        data_type_size = 4
        data_type_code = 'f'
    elif data_type == "int8":
        data_type_code = 'b'
        data_type_size = 1
    elif data_type == "uint8":
        data_type_code = 'B'
        data_type_size = 1
    else:
        raise Exception("Unsupported data type. Supported data types are float, int8 and uint8")
    
    print(str.format("Parsing DiskANN index at {0} with data type {1} and writing output to {2}. Use PQ vectors: {3}", index_path_prefix, data_type, output_file_prefix, use_pq_vectors))
    
    out_disk_index_file = output_file_prefix + "_disk.index.tsv"
    out_pq_vectors_file = output_file_prefix + "_compressed_vectors.tsv"
    out_pq_pivots_file = output_file_prefix + "_pivots.tsv"
    out_pq_chunks_file = output_file_prefix + "_chunk_offsets.tsv"
    out_centroids_file = output_file_prefix + "_centroids.tsv"

    print("** Parsing PQ data **")
    compressed_vectors = ppq.parse_compressed_vectors(index_path_prefix)
    pivots, centroids, chunk_offsets = ppq.parse_pivots_file(index_path_prefix)


    
    with open(out_pq_vectors_file, "w") as out_file:
        out_file.write("Id\tvector\n")
        for i in range(len(compressed_vectors)):
            out_file.write(str(i) + "\t" + str(compressed_vectors[i].tolist()) + "\n")
    print(str.format("** Wrote PQ data to file:{} **", out_pq_vectors_file))
    
    with open(out_pq_pivots_file, "w") as out_file:
        out_file.write("Pivots\n")
        for i in range(len(pivots)):
            out_file.write(str(pivots[i].tolist()) + "\n")
    print(str.format("** Wrote PQ pivots to file:{} **", out_pq_pivots_file))

    with open(out_centroids_file, "w") as out_file:
        out_file.write("Centroids\n")
        for i in range(len(centroids)):
            out_file.write(str(centroids[i].tolist()) + "\n")
    print(str.format("** Wrote PQ centroid data to file:{} **", out_centroids_file))
    
    with open(out_pq_chunks_file, "w") as out_file:
        out_file.write("Chunk offsets\n")
        for i in range(len(chunk_offsets)):
            out_file.write(str(chunk_offsets[i].tolist()) + "\n")
    print(str.format("** Wrote chunk offsets to file:{} **", out_pq_chunks_file))
    

    if use_pq_vectors:
        pdi.parse_index_with_PQ_vectors(index_path_prefix, data_type_code, data_type_size, out_disk_index_file, compressed_vectors)
    else:
        pdi.parse_index(index_path_prefix, data_type_code, data_type_size, out_disk_index_file)
       
    print("Parsed DiskANN index and wrote output to " + out_disk_index_file + ", " + out_pq_vectors_file + ", " + out_pq_pivots_file + ", " + out_centroids_file + ", " + out_pq_chunks_file)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse a DiskANN index')
    parser.add_argument('index_path_prefix', type=str, help='Path to the DiskANN index file without the extension')
    parser.add_argument('data_type', type=str, help='Data type of the vectors in the index. Supported data types are float, int8 and uint8')
    parser.add_argument('output_file_prefix', type=str, help='Output file prefix to write index and PQ vectors. The index is written in CSV format with the following columns: Id, vector, neighbours')
    parser.add_argument('--use_pq_vectors', default=False, action='store_true', help='Whether to replace FP vectors with PQ vectors in the output file.')
    args = parser.parse_args()
    main(args.index_path_prefix, args.data_type, args.output_file_prefix, args.use_pq_vectors)
