#Important: 
# We get both the data and query files from wendi in the format:
# hash \t vector \t filter for data and 
# query id \t vector \t filter for query
# For the data file, we need to collate the vectors and filters 
# based on the hash. So we end up with one vector and several 
# filters, all associated with the same hash (i.e., point).
# For the query, we want to get the query id, vector, and filter
# as-is, because we want to do single-filter search.
import sys
vector_map = {}

class VectorAndFilter:
    def __init__(self, vector, filter):
        self.vector = vector
        self.filters = []
        self.filters.append(filter)

    def add_filter(self, filter):
        self.filters.append(filter)


def main(input_file, output_file_prefix, is_query_file_str):
    is_query_file = bool(is_query_file_str)

    with open(input_file, 'r', encoding='utf-8') as f:
        count = 0
        line = f.readline().strip()
        print(line)
        while line:
            pieces = line.split('\t')
            assert(len(pieces) == 3)
            hash = pieces[0].strip()
            vector = pieces[1].strip()
            filter = pieces[2].strip()
            if hash not in vector_map:
                vector_map[hash] = VectorAndFilter(vector, filter)
            else:
                vector_map[hash].add_filter(filter)
            line = f.readline().strip()
            if count % 100000 == 0 and count > 0:
                print("Processed " + str(count) + " lines")
            count += 1
    
    print("Found " + str(len(vector_map)) + " unique vectors")

    vec_file = open(output_file_prefix + "_vectors.tsv", 'w', encoding='utf-8')
    filter_file = open(output_file_prefix + "_filters.tsv", 'w', encoding='utf-8')
    hash_vecid_map_file = open(output_file_prefix + "_hash_vecid_map.tsv", 'w', encoding='utf-8')

    if is_query_file :
        count = 0
        for (vector_hash, vector_and_filter) in vector_map.items():
            for filter in vector_and_filter.filters:
                hash_vecid_map_file.write(vector_hash +"\t" + str(count))
                hash_vecid_map_file.write("\n")
                vec_file.write(vector_and_filter.vector.replace("|", "\t"))
                vec_file.write("\n")
                filter_file.write(filter)
                filter_file.write("\n")
                count += 1
    else:
        count = 0
        for (vector_hash, vector_and_filter) in vector_map.items():
            hash_vecid_map_file.write(vector_hash +"\t" + str(count))
            hash_vecid_map_file.write("\n")
            vec_file.write(vector_and_filter.vector.replace("|", "\t"))
            vec_file.write("\n")
            filter_file.write(",".join(vector_and_filter.filters))
            filter_file.write("\n")

            count += 1

    vec_file.close()
    filter_file.close()
    hash_vecid_map_file.close()
    print("Wrote " + str(count) + " vectors and metadata to " + ("query files" if is_query_file else "data files") + " with prefix: " + output_file_prefix)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python FileConverter.py <input_file> <output_file_prefix> <is_query_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])