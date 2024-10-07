import sys

def append_filter_name_to_filter_values(filter_name, filter_values_str):
    filter_values = filter_values_str.split(',')
    return [f"{filter_name}={filter_value}" for filter_value in filter_values]


def main(input_file, output_file, is_query_file):
    with open(output_file, mode="w", encoding='UTF-8') as writer:
        with open(input_file, mode='r', encoding='UTF-8') as f:
            count = 0
            for line in f:
                count += 1 
                line = line.strip()
                if line == '':
                    continue

                all_filters_str = None
                if is_query_file: 
                    parts = line.split('|')
                    if len(parts) != 3: 
                        print(f"Invalid line: {line} at line number: {count}")
                        continue
                    all_filters_str = f"c1={parts[0]}&c2={parts[1]}&c3={parts[2]}"
                else:
                    parts = line.split('|')
                    if len(parts) < 3: 
                        print(f"Invalid line: {line} at line number: {count}")
                        continue
                    all_filters = []
                    all_filters.extend(append_filter_name_to_filter_values("c1", parts[0]))
                    all_filters.extend(append_filter_name_to_filter_values("c2", parts[1]))
                    all_filters.extend(append_filter_name_to_filter_values("c3", parts[2]))
                    all_filters_str = ",".join(all_filters)

                writer.write(all_filters_str)
                writer.write('\n')

                if count % 500000 == 0:
                    print("Processed {} lines".format(count))
    
    print("Output written to {}".format(output_file))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <program> <input_filter_file> <output_filter_file> <is_query_file(True/False)>")
        sys.exit(1)
    else:
        main(sys.argv[1], sys.argv[2], bool(sys.argv[3]))