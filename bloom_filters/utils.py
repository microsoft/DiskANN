

def load_label_file(fname, delim=','):
    filters = []
    unique_filters = set()
    with open(fname, "r") as fd:
        for line in fd:
            curr_filters = [int(x) for x in line.rstrip().split(delim)]
            filters.append(curr_filters)
            unique_filters.update(curr_filters)
    print(f"loaded {len(filters)} points with {len(unique_filters)} unique filters from {fname}")
    return filters, unique_filters

def write_bloom_labels(bloom_labels, out_fname, delim=','):
    with open(out_fname, "w+") as fd:
        for b_label in bloom_labels:
            fd.write(b_label.to_str(delim))
            fd.write('\n')
    print(f"wrote {len(bloom_labels)} bloom filters to {out_fname}")
