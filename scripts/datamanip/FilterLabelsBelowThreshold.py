import sys
import re
import argparse
import parse_common as pc


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
    
    return base_points_to_delete, query_points_to_delete, filters_to_remove


def load_filters(base_filter_file):
    filter_points_map = {}
    point_filters_map = {}

    with open(base_filter_file, "r") as f:
        count = 0
        for line in f:
            filters = re.split(",|", line.strip)
            point_filters_map[count] = filters
            for filter in filters:
                if filter not in filter_points_map:
                    filter_points_map[filter] = []
                filter_points_map[filter].append(count)
            
            count += 1
    
    return filter_points_map, point_filters_map


def main(data_type, base_data_file, base_filter_file, query_data_file, query_filter_file, threshold, output_prefix):
    base_filter_points_map, base_point_filters_map = load_filters(base_filter_file)
    query_filter_points_map, query_point_filters_map = load_filters(query_filter_file)

    base_points_to_delete, query_points_to_delete, filters_to_remove = filter_points_below_threshold(base_filter_points_map, base_point_filters_map, query_filter_points_map, query_point_filters_map, threshold)

    data_type_code, data_type_size = pc.get_data_type_code(data_type)
    base_data  = pc.DataMat(data_type_code, data_type_size)
    base_data.load_bin(base_data_file)
    base_data.remove_rows(base_points_to_delete)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a filter histogram for a base file and query file and list those filters that apply to fewer than some threshold points. If some base/query point has only those labels that were removed, list them too for removal', prog='.py')
    parser.add_argument('--base_filter_file', type=str, help='Base file file', required=True)
    parser.add_argument('--base_data_file', type=str, help='Base data file', required=True)
    parser.add_argument('--query_data_file', type=str, help='Query data file', required=True)
    parser.add_argument('--data_type', type=str, help='Data type of the vectors in the file', required=True)
    parser.add_argument('--threshold', type=int, help='Remove those filters that don\'t apply to atleast these many points.', required=True)
    parser.add_argument('--query_filter_file', type=str, help='Query filter file', required=True)
    parser.add_argument('--output_prefix', type=str, help='Output file prefix for writing the histogram and the new filter files.', required=True)

    main(parser["data_type"], parser["base_data_file"], parser["base_filter_file"], parser["query_data_file"], parser["query_filter_file"], parser["threshold"], parser["output_prefix"])