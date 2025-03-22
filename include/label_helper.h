#pragma once
#include "label_bitmask.h"
#include "percentile_stats.h"
#include <string>

namespace diskann
{

class label_helper
{
public:
    bool parse_label_file_in_bitset(
        const std::string& label_file,
        size_t& num_points, 
        size_t num_labels,
        simple_bitmask_buf& bitmask_buf,
        TableStats& table_stats);

private:
    size_t search_string_range(const std::string& str, char ch, size_t start, size_t end);
};

}