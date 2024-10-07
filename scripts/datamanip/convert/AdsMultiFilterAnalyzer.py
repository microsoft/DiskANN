import sys
import pandas as pd

base_unique_filters = set()
query_unique_filters = set()

base_joined_filters_inv_indx = {}
query_joined_filters_inv_indx = {}

joined_filters_of_point = []
joined_filters_of_query_point = []


def create_joined_filters(per_category_label_lists):
    assert(len(per_category_label_lists) == 3)
    joined_filters = []
    for l1 in per_category_label_lists[0]:
        for l2 in per_category_label_lists[1]:
            for l3 in per_category_label_lists[2]:
                joined_filters.append(f"{l1}_{l2}_{l3}")
    return joined_filters
    

def parse_filter_line(line, separator, line_number, base_filter_file):
    line = line.strip()
    if line == '':
        print(f"Empty line at line number: {line_number} in {base_filter_file}")
    parts = line.split(separator)
    if len(parts) != 3:
        print(f"line: {line} at line number: {line_number} in {base_filter_file} does not have 3 parts when split by {separator}")
    
    return parts

def append_category_name_to_labels(category_id, labels):
    category_name = f"C{category_id+1}"
    named_labels = [f"{category_name}={part}" for part in labels]
    return named_labels



def load_base_file_filters(base_filter_file):
    with open(base_filter_file, mode='r', encoding='UTF-8') as f:
        count = 0
        for line in f:
            count += 1
            cs_filters_per_category = parse_filter_line(line, '|', count, base_filter_file)
            per_category_label_lists = []
            for i in range(3):
                cat_labels = append_category_name_to_labels(i, cs_filters_per_category[i].split(','))
                assert(len(cat_labels) > 0)
                base_unique_filters.update(cat_labels)
                per_category_label_lists.append(cat_labels)

            joined_filters = create_joined_filters(per_category_label_lists)
            joined_filters_of_point.append([])
            for joined_filter in joined_filters:
                joined_filters_of_point[count-1].append(joined_filter)
                if joined_filter not in base_joined_filters_inv_indx:
                    base_joined_filters_inv_indx[joined_filter]  = []
                base_joined_filters_inv_indx[joined_filter].append(count)
            
            if count % 500000 == 0:
                print(f"Processed {count} lines in {base_filter_file}")
        
        print(f"Obtained {len(base_unique_filters)} distinct filters from {base_filter_file}, line count: {count}")
        print(f"After joining number of filters is: {len(base_joined_filters_inv_indx)}")

        
                
def load_query_file_filters(query_filter_file):
    with open(query_filter_file, mode='r', encoding='UTF-8') as f:
        count = 0
        for line in f:
            count += 1
            cs_filters_per_category = parse_filter_line(line, '|', count, query_filter_file)
            per_category_label_lists = []
            for i in range(3):
                cat_labels = append_category_name_to_labels(i, cs_filters_per_category[i].split(','))
                assert(len(cat_labels) > 0)
                query_unique_filters.update(cat_labels)
                per_category_label_lists.append(cat_labels)

            joined_filters = create_joined_filters(per_category_label_lists)
            joined_filters_of_query_point.append([])
            for joined_filter in joined_filters:
                joined_filters_of_query_point[count-1].append(joined_filter)
                if joined_filter not in query_joined_filters_inv_indx:
                    query_joined_filters_inv_indx[joined_filter]  = []
                query_joined_filters_inv_indx[joined_filter].append(count)
        
        print(f"Obtained {len(query_unique_filters)} distinct filters from {query_filter_file}, line count = {count}")
        print(f"After joining number of filters is: {len(query_joined_filters_inv_indx)}")

            

def analyze():
    missing_query_filters = query_unique_filters.difference(base_unique_filters)
    if len(missing_query_filters) > 0:
        print(f"Warning: found the following query filters not in base:{missing_query_filters}")
    

    bujf = set()
    bujf.update(base_joined_filters_inv_indx.keys())
    qujf = set()
    qujf.update(query_joined_filters_inv_indx.keys())
    missing_joined_filters = qujf.difference(bujf)
    if len(missing_joined_filters) > 0:
        print(f"Warning: found the following joined query filters not in base:{missing_joined_filters}")


    mjqf_query_ids_map = {}
    for filter in missing_joined_filters:
        mjqf_query_ids_map[filter] = []
        for index, filters_of_point in enumerate(joined_filters_of_query_point):
            if filter == filters_of_point[0]:
                mjqf_query_ids_map[filter].append(index)
    
    with open('missing_joined_query_filters.txt', mode='w', encoding='UTF-8') as f:
        for filter in mjqf_query_ids_map:
            f.write(f"{filter}\t{len(mjqf_query_ids_map[filter])}\t{mjqf_query_ids_map[filter]}\n")
    
    
    print(f"Number of unique base filters: {len(base_unique_filters)}" )
    print(f"Number of unique query filters: {len(query_unique_filters)}" )
    print(f"Number of joined base filters: {len(base_joined_filters_inv_indx)}" )
    print(f"Number of joined query filters: {len(query_joined_filters_inv_indx)}" )

def write_joined_filters(output_file_prefix):
    base_joined_filters_file = output_file_prefix + '_base_joined_filters.txt'

    with open(base_joined_filters_file, mode='w', encoding='UTF-8') as f:
        for filters_of_point in joined_filters_of_point:
            str = ','.join([x for x in filters_of_point])
            f.write(f"{str}\n")
    print(f"Base joined filters written to {base_joined_filters_file}")
    
    query_joined_filters_file = output_file_prefix + '_query_joined_filters.txt'
    with open(query_joined_filters_file, mode='w', encoding='UTF-8') as f:
        for filters_of_point in joined_filters_of_query_point:
            str = ','.join([x for x in filters_of_point])
            f.write(f"{str}\n")
    print(f"Query joined filters written to {query_joined_filters_file}")

    base_unique_filters_file = output_file_prefix + '_base_unique_filters.txt'
    with open(base_unique_filters_file , mode='w', encoding='UTF-8') as f:
        sorted_list = sorted(base_unique_filters)
        for filter in sorted_list:
            f.write(f"{filter}\n")
    print(f"Base unique filters written to {base_unique_filters_file}")

    query_unique_filters_file = output_file_prefix + '_query_unique_filters.txt'
    with open(query_unique_filters_file, mode='w', encoding='UTF-8') as f:
        sorted_list = sorted(query_unique_filters)
        for filter in sorted_list:
            f.write(f"{filter}\n")
    print(f"Query unique filters written to {query_unique_filters_file}")

    base_joined_unique_filters_file = output_file_prefix + '_base_joined_unique_filters.txt'
    with open(base_joined_unique_filters_file, mode='w', encoding='UTF-8') as f:
        sorted_list = sorted(base_joined_filters_inv_indx.keys())
        for filter in sorted_list:
            f.write(f"{filter}\t{len(base_joined_filters_inv_indx[filter])}\n")
    print(f"Base joined unique filters written to {base_joined_unique_filters_file}")

    query_unique_joined_filters_file = output_file_prefix + '_query_joined_unique_filters.txt'
    with open(query_unique_joined_filters_file, mode='w', encoding='UTF-8') as f:
        sorted_list = sorted(query_joined_filters_inv_indx.keys())
        for filter in sorted_list:
            f.write(f"{filter}\t{len(query_joined_filters_inv_indx[filter])}\n")


    


def main(base_filter_file, query_filter_file, output_path_prefix):
    load_base_file_filters(base_filter_file)
    load_query_file_filters(query_filter_file)
    analyze()
    write_joined_filters(output_path_prefix)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: AdsMultiFilterAnalyzer.py <base_filter_file> <query_filter_file> <output_file_prefix>")
        print("Both base file should have label categories separated by | and labels separated by commas")
        print("Query file should have labels separated by |")
        sys.exit(1)
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])