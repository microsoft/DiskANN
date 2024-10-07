import sys
import parse_common as pc

def main(input_vector_file, data_type, ids_to_remove_file, output_file_prefix, filter_file_for_vectors):
    data_type_code, data_type_size = pc.get_data_type_code(data_type)
    vectors = pc.DataMat(data_type_code, data_type_size)
    vectors.load_bin(input_vector_file)
    vector_ids_to_remove = set()
    with open(ids_to_remove_file, "r") as f:
        for line in f:
            vector_ids_to_remove.add(int(line.strip()))

    filters = []
    if filter_file_for_vectors is not None:
        with open(filter_file_for_vectors, "r") as f:
            for line in f:
                filters.append(line.strip())
    
    vectors.remove_rows(vector_ids_to_remove)
    
    output_bin_file = output_file_prefix + "_vecs.bin"
    vectors.save_bin(output_bin_file)
    print(f"Removed {len(vector_ids_to_remove)} vectors. Output written to {output_bin_file}")

    if len(filters) > 0:
        output_filters_file = output_file_prefix + "_filters.txt"
        output_filters = [filter for idx, filter in enumerate(filters) if idx not in vector_ids_to_remove]
        with open(output_filters_file, "w") as f:
            for output_filter in output_filters:
                f.write(output_filter + "\n")
        print(f"REmoved {len(vector_ids_to_remove)} filters. Output written to {output_filters_file}")


if __name__ == "__main__":
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("Usage: <program> <input_vector_file> <data_type_format(float|uint8|int8)> <ids_to_remove_file> <output_file_prefix (program adds _vecs.bin for the vector file and _filters.txt for the filter file)> [<filter_file_for_vectors>]")
        sys.exit(1)
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)