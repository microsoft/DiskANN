#include "label_helper.h"
#include "ann_exception.h"
#include "tsl/robin_set.h"
#include "logger.h"

#include <fstream>

namespace diskann
{

bool label_helper::parse_label_file_in_bitset(
    const std::string& label_file,
    size_t& num_points,
    size_t num_labels,
    simple_bitmask_buf& bitmask_buf,
    TableStats& table_stats)
{
    std::ifstream infile(label_file, std::ios::binary);
    if (infile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + label_file, -1);
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
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        cur_pos = next_pos + 1;

        line_cnt++;
    }

    // label is counting by 1, so additional 1 bit is needed
    bitmask_buf._bitmask_size = simple_bitmask::get_bitmask_size(num_labels + 1);
    if (num_points > line_cnt)
    {
        bitmask_buf._buf.resize(num_points * bitmask_buf._bitmask_size, 0);
    }
    else
    {
        bitmask_buf._buf.resize(line_cnt * bitmask_buf._bitmask_size, 0);
    }

    tsl::robin_set<size_t> labels;

    infile.clear();
    infile.seekg(0, std::ios::beg);
    line_cnt = 0;

    std::string label_str;
    cur_pos = 0;
    next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = search_string_range(buffer, ',', lbl_pos, next_pos);
            if (next_lbl_pos == std::string::npos) // the last label in the whole file
            {
                next_lbl_pos = next_pos;
            }

            if (next_lbl_pos > next_pos) // the last label in one line
            {
                next_lbl_pos = next_pos;
            }

            label_str.assign(buffer.c_str() + lbl_pos, next_lbl_pos - lbl_pos);
            if (label_str[label_str.length() - 1] == '\t')
            {
                label_str.erase(label_str.length() - 1);
            }

            size_t token_as_num = std::stoul(label_str);
            simple_bitmask bm(bitmask_buf.get_bitmask(line_cnt), bitmask_buf._bitmask_size);
            bm.set(token_as_num);
            labels.insert(token_as_num);
            table_stats.label_total_count++;

            lbl_pos = next_lbl_pos + 1;
        }

        cur_pos = next_pos + 1;

        line_cnt++;
    }

    num_points = (size_t)line_cnt;
    diskann::cout << "Identified " << labels.size() << " distinct label(s)" << std::endl;
    
    return true;
}

bool label_helper::write_bitmask_to_file(const std::string& bitmask_label_file, simple_bitmask_buf& bitmask_buf, std::uint32_t num_points)
{
    // format: num_points, bitmask_size, bitmask content
    std::ofstream outfile(bitmask_label_file, std::ios::binary);
    if (outfile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + bitmask_label_file, -1);
    }

    if (bitmask_buf._buf.size() != bitmask_buf._bitmask_size * num_points)
    {
        throw diskann::ANNException(std::string("Bitmask buffer is empty"), -1);
    }

    std::uint32_t bitmask_size = static_cast<std::uint32_t>(bitmask_buf._bitmask_size);
    outfile.write((char*)(&num_points), sizeof(std::uint32_t));
    outfile.write((char*)(&bitmask_size), sizeof(std::uint32_t));
    outfile.write((char *)bitmask_buf._buf.data(), bitmask_buf._buf.size() * sizeof(std::uint64_t));
    outfile.close();

    return true;
}

bool label_helper::read_bitmask_from_file(const std::string& bitmask_label_file, simple_bitmask_buf& bitmask_buf,
    size_t& num_points)
{
    std::ifstream infile(bitmask_label_file, std::ios::binary);
    if (infile.fail())
    {
        return false;
    }

    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();

    infile.seekg(0, std::ios::beg);

    std::uint32_t num_points_in_file = 0;
    std::uint32_t bitmask_size = 0;
    infile.read((char *)(&num_points_in_file), sizeof(std::uint32_t));
    infile.read((char *)(&bitmask_size), sizeof(std::uint32_t));

    size_t bitmask_data_size = num_points_in_file * bitmask_size * sizeof(std::uint64_t);
    if (file_size != (sizeof(std::uint32_t) * 2 + bitmask_data_size))
    {
        return false;
    }

    bitmask_buf._bitmask_size = bitmask_size;
    // num_points > num_points_in_file in dynamic index 
    if (num_points > static_cast<size_t>(num_points_in_file))
    {
        bitmask_buf._buf.resize(num_points * bitmask_buf._bitmask_size, 0);
    }
    else
    {
        bitmask_buf._buf.resize(num_points_in_file * bitmask_buf._bitmask_size, 0);
    }
    
    num_points = num_points_in_file;

    infile.read((char *)bitmask_buf._buf.data(), bitmask_data_size);
    infile.close();

    return true;
}

size_t label_helper::search_string_range(const std::string& str, char ch, size_t start, size_t end)
{
    for (; start != end; start++)
    {
        if (str[start] == ch)
        {
            return start;
        }
    }

    return std::string::npos;
}

bool label_helper::parse_label_file_in_integer(
    const std::string& label_file,
    size_t& num_points,
    integer_label_vector& integer_vector,
    tsl::robin_set<uint32_t>& labels, TableStats& table_stats)
{
    std::ifstream infile(label_file, std::ios::binary);
    if (infile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + label_file, -1);
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
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        cur_pos = next_pos + 1;

        line_cnt++;
    }

    const size_t rough_avg_labels_per_point = 10;

    if (num_points > line_cnt)
    {
        size_t rough_total_labels = num_points * rough_avg_labels_per_point;
        integer_vector.initialize(num_points, rough_total_labels);
    }
    else
    {
        size_t rough_total_labels = line_cnt * rough_avg_labels_per_point;
        integer_vector.initialize(line_cnt, rough_total_labels);
    }

    infile.clear();
    infile.seekg(0, std::ios::beg);
    line_cnt = 0;

    std::string label_str;
    std::vector<uint32_t> current_labels;

    cur_pos = 0;
    next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        current_labels.clear();

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = search_string_range(buffer, ',', lbl_pos, next_pos);
            if (next_lbl_pos == std::string::npos) // the last label in the whole file
            {
                next_lbl_pos = next_pos;
            }

            if (next_lbl_pos > next_pos) // the last label in one line
            {
                next_lbl_pos = next_pos;
            }

            label_str.assign(buffer.c_str() + lbl_pos, next_lbl_pos - lbl_pos);
            if (label_str[label_str.length() - 1] == '\t')
            {
                label_str.erase(label_str.length() - 1);
            }

            size_t token_as_num = std::stoul(label_str);
            current_labels.push_back(static_cast<uint32_t>(token_as_num));

            labels.insert(static_cast<uint32_t>(token_as_num));
            table_stats.label_total_count++;

            lbl_pos = next_lbl_pos + 1;
        }

        integer_vector.add_labels(line_cnt, current_labels);

        cur_pos = next_pos + 1;

        line_cnt++;
    }

    num_points = (size_t)line_cnt;
    diskann::cout << "Identified " << labels.size() << " distinct label(s)" << std::endl;

    return true;
}

}