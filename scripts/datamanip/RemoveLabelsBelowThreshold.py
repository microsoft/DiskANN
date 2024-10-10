import sys
import re
import argparse
import parse_common as pc


#Overloading the meaning of "below_threshold" to also delete queries whose filters do not exist in the base filter set.
def filter_points_below_threshold(base_filter_points_map, base_point_filters_map, query_filter_points_map, query_point_filters_map, threshold):
    base_points_to_delete, query_points_to_delete, filters_to_remove = set(), set(), set()

    for filter, points in base_filter_points_map.items():
        if len(points) < threshold:
            print(f"Filter {filter} applies to {len(points)} points while threshold is {threshold}. Removing it.")
            filters_to_remove.add(filter)
            for point in points:
                base_point_filters_map[point].remove(filter)
                if len(base_point_filters_map[point]) == 0:
                    base_points_to_delete.add(point)

            if filter in query_filter_points_map.keys():
                for point in query_filter_points_map[filter]:
                    query_point_filters_map[point].remove(filter)
                    if len(query_point_filters_map[point]) == 0:
                        query_points_to_delete.add(point)
    
    for filter, points in query_filter_points_map.items():
        if filter not in base_filter_points_map:
            print(f"Found query filter: {filter} that is not in base. Removing {len(points)} affected queries.")
            query_points_to_delete.update(points)
    
    print(f"Found {len(filters_to_remove)} filters with frequency below {threshold} points, and {len(base_points_to_delete)} base points and {len(query_points_to_delete)} query points that will be removed as a result of removing these filters.")
    return base_points_to_delete, query_points_to_delete, filters_to_remove


def load_filters(base_filter_file):
    filter_points_map = {}
    point_filters_map = {}

    with open(base_filter_file, "r") as f:
        count = 0
        for line in f:
            filters = re.split(r'[,\|]', line.strip())
            point_filters_map[count] = filters
            for filter in filters:
                if filter not in filter_points_map:
                    filter_points_map[filter] = []
                filter_points_map[filter].append(count)
            
            count += 1
            if count % 500000 == 0:
                print(f"Processed {count} rows.")
    
    return filter_points_map, point_filters_map

def write_filters(point_filters_map, num_points, output_file):
    with open(output_file, mode="w", encoding='utf-8') as f:
        for i in range(0, num_points):
            if i in point_filters_map:
                str = ",".join(point_filters_map[i])
                f.write(str)
                f.write("\n")


def write_debug_sets(out_file, set_to_write):
    with open(out_file, "w") as f:
        t = list(set_to_write)
        t.sort()
        f.write(str(t))

def main(data_type, base_data_file, base_filter_file, query_data_file, query_filter_file, threshold, output_prefix):
    print(f"Checking {base_filter_file} for filters below threshold {threshold}. Base filter file: {base_filter_file}, query data file: {query_data_file}, query filter file: {query_filter_file}. Writing output to {output_prefix}")
    base_filter_points_map, base_point_filters_map = load_filters(base_filter_file)
    query_filter_points_map, query_point_filters_map = load_filters(query_filter_file)

    base_points_to_delete, query_points_to_delete, filters_to_remove = filter_points_below_threshold(base_filter_points_map, base_point_filters_map, query_filter_points_map, query_point_filters_map, threshold)

    print(f"Deleting {len(base_points_to_delete)} from {base_data_file}")
    data_type_code, data_type_size = pc.get_data_type_code_and_size(data_type)
    base_data  = pc.DataMat(data_type_code, data_type_size)
    base_data.load_bin(base_data_file)
    orig_base_num_rows = base_data.num_rows
    base_data.remove_rows(base_points_to_delete)
    print(f"After deleting {len(base_points_to_delete)} points, new base data has {base_data.num_rows} rows.")
    for i in base_points_to_delete: 
        del base_point_filters_map[i]
    print(f"After deleting {len(base_points_to_delete)} points, there are {base_data.num_rows} filter rows.")
    

    print(f"Deleting {len(query_points_to_delete)} points from {query_data_file}")
    query_data = pc.DataMat(data_type_code, data_type_size)
    query_data.load_bin(query_data_file)
    orig_num_queries = query_data.num_rows
    query_data.remove_rows(query_points_to_delete)
    print(f"After deleting {len(query_points_to_delete)} points, new query data has {query_data.num_rows} rows.")
    for i in query_points_to_delete:
        del query_point_filters_map[i]

    #write outputs
    out_base_file = output_prefix + "_base.bin"
    out_query_file = output_prefix + "_query.bin"
    out_base_filters_file = output_prefix + "_base_filters.txt"
    out_query_filters_file = output_prefix + "_query_filters.txt"
    base_data.save_bin(out_base_file)
    query_data.save_bin(out_query_file)
    write_filters(base_point_filters_map, orig_base_num_rows, out_base_filters_file)
    write_filters(query_point_filters_map, orig_num_queries, out_query_filters_file)

    out_base_points_deleted_file = output_prefix + "_debug_base_points_deleted.txt"
    out_query_points_deleted_file = output_prefix + "_debug_query_points_deleted.txt"
    filters_removed_file = output_prefix + "_debug_filters_removed.txt"
    write_debug_sets(out_base_points_deleted_file, base_points_to_delete)
    write_debug_sets(out_query_points_deleted_file, query_points_to_delete)
    write_debug_sets(filters_removed_file, filters_to_remove)

    print(f"New base file: {out_base_file}, new query file: {out_query_file}, new base filters file:{out_base_filters_file}, new query filters file: {out_query_filters_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a filter histogram for a base file and query file and list those filters that apply to fewer than some threshold points. If some base/query point has only those labels that were removed, list them too for removal', prog='.py')
    parser.add_argument('--base_filter_file', type=str, help='Base file file', required=True)
    parser.add_argument('--base_data_file', type=str, help='Base data file', required=True)
    parser.add_argument('--query_data_file', type=str, help='Query data file', required=True)
    parser.add_argument('--data_type', type=str, help='Data type of the vectors in the file', required=True)
    parser.add_argument('--threshold', type=int, help='Remove those filters that don\'t apply to atleast these many points.', required=True)
    parser.add_argument('--query_filter_file', type=str, help='Query filter file', required=True)
    parser.add_argument('--output_prefix', type=str, help='Output file prefix for writing the histogram and the new filter files.', required=True)

    args = parser.parse_args()
    main(args.data_type, args.base_data_file, args.base_filter_file, args.query_data_file, args.query_filter_file, args.threshold, args.output_prefix)