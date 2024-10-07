import sys

def main():
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Usage: {} <input_file> <vector_col_id (starting from zero)> <filter_col_id (starting from zero)> [filter_separater(default ,)]".format(sys.argv[0]))
        print("The program converts TSV file with vectors and multiple filters into vectors and single filters by repeating the vector for each filter. It assumes that the input file is a TSV file with no headers")
        sys.exit(1)

    vector_col_id = int(sys.argv[2])
    filter_col_id = int(sys.argv[3])
    filter_separator = ',' if len(sys.argv) == 4 else sys.argv[4]
    single_filter_file = sys.argv[1] + '.single_filter.txt'
    

    with open(single_filter_file, 'w', -1, 'UTF-8') as writer:
        with open(sys.argv[1], 'r', -1, 'UTF-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split('\t')
                vector = parts[vector_col_id]
                filters = parts[filter_col_id].split(filter_separator)
                for filter in filters:
                    filter = filter.replace(';',' ').replace('_', ' ')
                    writer.write('{}\t{}'.format(vector, filter))
                    writer.write('\n')
    
    print("Output written to {}".format(single_filter_file))

if __name__ == "__main__":
    print("In main")
    main()


    