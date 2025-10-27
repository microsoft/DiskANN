#pragma once
#include "label_bitmask.h"
#include "integer_label_vector.h"
#include "percentile_stats.h"
#include "tsl/robin_set.h"
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

    bool write_bitmask_to_file(const std::string& bitmask_label_file, simple_bitmask_buf& bitmask_buf, std::uint32_t num_points);

    bool read_bitmask_from_file(const std::string &bitmask_label_file, simple_bitmask_buf &bitmask_buf,
                                size_t& num_points);

    bool parse_label_file_in_integer(
        const std::string& label_file,
        size_t& num_points,
        integer_label_vector& integer_vector,
        tsl::robin_set<uint32_t>& labels, TableStats &table_stats);

    template <typename LabelT>
    bool load_label_map(
        const std::string& label_map_file,
        std::unordered_map<std::string, LabelT>& label_map)
    {
        std::ifstream infile(label_map_file, std::ios::binary);
        if (infile.fail())
        {
            throw diskann::ANNException(std::string("Failed to open file ") + label_map_file, -1);
        }
        infile.seekg(0, std::ios::end);
        size_t file_size = infile.tellg();

        std::string buffer(file_size, ' ');

        infile.seekg(0, std::ios::beg);
        infile.read(&buffer[0], file_size);
        infile.close();

        unsigned line_cnt = 0;

        size_t cur_pos = 0;
        size_t next_pos = 0;
        size_t lbl_pos = 0;
        std::string token;
        std::string labe_str;
        while (cur_pos < file_size && cur_pos != std::string::npos)
        {
            next_pos = buffer.find('\n', cur_pos);
            if (next_pos == std::string::npos)
            {
                break;
            }

            lbl_pos = search_string_range(buffer, '\t', cur_pos, next_pos);
            labe_str.assign(buffer.c_str() + cur_pos, lbl_pos - cur_pos);

            token.assign(buffer.c_str() + lbl_pos + 1, next_pos - lbl_pos - 1);
            LabelT label_num = (LabelT)std::stoul(token);

            label_map[labe_str] = label_num;

            cur_pos = next_pos + 1;

            line_cnt++;
        }

        return true;
    }

    template <typename LabelT>
    bool load_label_medoids(
        const std::string& label_medoids_file,
        std::unordered_map<LabelT, uint32_t>& label_to_start_id)
    {
        std::ifstream infile(label_medoids_file, std::ios::binary);
        if (infile.fail())
        {
            throw diskann::ANNException(std::string("Failed to open file ") + label_medoids_file, -1);
        }
        infile.seekg(0, std::ios::end);
        size_t file_size = infile.tellg();

        std::string buffer(file_size, ' ');

        infile.seekg(0, std::ios::beg);
        infile.read(&buffer[0], file_size);
        infile.close();

        unsigned line_cnt = 0;

        size_t cur_pos = 0;
        size_t next_pos = 0;
        size_t lbl_pos = 0;
        std::string token;
        while (cur_pos < file_size && cur_pos != std::string::npos)
        {
            next_pos = buffer.find('\n', cur_pos);
            if (next_pos == std::string::npos)
            {
                break;
            }

            lbl_pos = search_string_range(buffer, ',', cur_pos, next_pos);
            token.assign(buffer.c_str() + cur_pos, lbl_pos - cur_pos);
            LabelT label_num = (LabelT)std::stoul(token);
            
            token.assign(buffer.c_str() + lbl_pos + 1, next_pos - lbl_pos - 1);
            uint32_t medoid = (uint32_t)std::stoul(token);

            label_to_start_id[label_num] = medoid;

            cur_pos = next_pos + 1;

            line_cnt++;
        }

        return true;
    }

  private:
    size_t search_string_range(const std::string& str, char ch, size_t start, size_t end);
};

}