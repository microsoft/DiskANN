import argparse
from utils import load_label_file, write_bloom_labels
from bloom import BloomFilter

# functions
def build_parser():
    parser = argparse.ArgumentParser(description="experiments")
    parser.add_argument('-b', metavar='base_name', type=str, default='',
                        help='path of base file')
    parser.add_argument('-q', metavar='query_name', type=str, default='',
                        help='path of query file')
    parser.add_argument('-m', metavar='filter_width', type=int, default=64,
                        help='bloom filter size')
    parser.add_argument('-k', metavar='filter_hashes', type=int, default=64,
                        help='number of bloom filter hashes')
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = build_parser()
    base_labels, unique_labels = load_label_file(args['b'])
    query_labels, _ = load_label_file(args['q'], delim='&')
    num_bits, num_hashes = args['m'], args['k']

    bloom_filter = BloomFilter(num_bits, num_hashes)
    print(f"generating bloom filters with size {len(bloom_filter)} and {bloom_filter.hash_count} hashes")

    bloomed_base_labels, bloomed_query_labels = [], []
    for curr_labels in base_labels:
        bloom_filter = BloomFilter(num_bits, num_hashes)
        for lbl in curr_labels:
            bloom_filter.add(lbl)
        bloomed_base_labels.append(bloom_filter)

    for curr_labels in query_labels:
        bloom_filter = BloomFilter(num_bits, num_hashes)
        for lbl in curr_labels:
            bloom_filter.add(lbl)
        bloomed_query_labels.append(bloom_filter)

    write_bloom_labels(bloomed_base_labels, args['b'] + "_bloom")
    write_bloom_labels(bloomed_query_labels, args['q'] + "_bloom", delim='&')




