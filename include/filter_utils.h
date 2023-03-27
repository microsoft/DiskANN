// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <algorithm>
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <tuple>
#include <string>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#ifdef _WINDOWS
#include <Windows.h>
typedef HANDLE FileHandle;
#else
#include <unistd.h>
typedef int FileHandle;
#endif

#include "cached_io.h"
#include "common_includes.h"

#include "utils.h"
#include "windows_customizations.h"

// custom types (for readability)
typedef tsl::robin_set<std::string> label_set;
typedef std::string path;

// structs for returning multiple items from a function
typedef std::tuple<std::vector<label_set>, tsl::robin_map<std::string, _u32>, tsl::robin_set<std::string>>
    parse_label_file_return_values;
typedef std::tuple<std::vector<std::vector<_u32>>, _u64> load_label_index_return_values;

namespace diskann
{
template <typename T>
DISKANN_DLLEXPORT void generate_label_indices(path input_data_path, path final_index_path_prefix, label_set all_labels,
                                              unsigned R, unsigned L, float alpha, unsigned num_threads);

DISKANN_DLLEXPORT load_label_index_return_values load_label_index(path label_index_path, _u32 label_number_of_points);

DISKANN_DLLEXPORT parse_label_file_return_values parse_label_file(path label_data_path, std::string universal_label);


template <typename T>
DISKANN_DLLEXPORT tsl::robin_map<std::string, std::vector<_u32>> generate_label_specific_vector_files(
    path input_data_path, tsl::robin_map<std::string, _u32> labels_to_number_of_points,
    std::vector<label_set> point_ids_to_labels, label_set all_labels);

template <typename T>
DISKANN_DLLEXPORT tsl::robin_map<std::string, std::vector<_u32>> generate_label_specific_vector_files_compat(
    path input_data_path, tsl::robin_map<std::string, _u32> labels_to_number_of_points,
    std::vector<label_set> point_ids_to_labels, label_set all_labels);

inline std::vector<uint32_t> loadTags(const std::string &tags_file, const std::string &base_file)
{
    const bool tags_enabled = tags_file.empty() ? false : true;
    std::vector<uint32_t> location_to_tag;
    if (tags_enabled)
    {
        size_t tag_file_ndims, tag_file_npts;
        std::uint32_t *tag_data;
        diskann::load_bin<std::uint32_t>(tags_file, tag_data, tag_file_npts, tag_file_ndims);
        if (tag_file_ndims != 1)
        {
            diskann::cerr << "tags file error" << std::endl;
            throw diskann::ANNException("tag file error", -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        // check if the point count match
        size_t base_file_npts, base_file_ndims;
        diskann::get_bin_metadata(base_file, base_file_npts, base_file_ndims);
        if (base_file_npts != tag_file_npts)
        {
            diskann::cerr << "point num in tags file mismatch" << std::endl;
            throw diskann::ANNException("point num in tags file mismatch", -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        location_to_tag.assign(tag_data, tag_data + tag_file_npts);
        delete[] tag_data;
    }
    return location_to_tag;
}

} // namespace diskann
