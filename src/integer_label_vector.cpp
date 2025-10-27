#include "integer_label_vector.h"
#include "ann_exception.h"
#include <algorithm>
#include <fstream>

namespace diskann
{

bool integer_label_vector::initialize(size_t numpoints, size_t total_labels) {
    _offset.resize(numpoints + 1);
    _offset[0] = 0;

    _data.reserve(total_labels);
    return true;
}

bool integer_label_vector::initialize_from_file(const std::string& label_file, size_t& numpoints)
{
    //format:
    // format version: uint8
    //  num_points: uint32
    //  offset array: uint64[num_points + 1]
    //   label data: uint32[total_labels]
    std::ifstream infile(label_file, std::ios::binary);
    if (infile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + label_file, -1);
    }
    uint8_t format_version = 0;
    infile.read((char*)(&format_version), sizeof(uint8_t));
    if (format_version != 1)
    {
        throw diskann::ANNException(std::string("Unsupported label file format version ") +
            std::to_string(format_version),
            -1);
    }
    uint32_t num_points_in_file = 0;
    infile.read((char*)(&num_points_in_file), sizeof(uint32_t));
    _offset.resize(num_points_in_file + 1);
    infile.read((char *)_offset.data(), _offset.size() * sizeof(size_t));
    size_t total_labels = _offset[num_points_in_file];
    _data.resize(total_labels);
    infile.read((char *)_data.data(), _data.size() * sizeof(uint32_t));
    infile.close();

    numpoints = static_cast<size_t>(num_points_in_file);

    return true;
}

template <typename LabelT>
bool integer_label_vector::add_labels(uint32_t point_id, std::vector<LabelT> &labels) {
    if (point_id >= _offset.size() - 1)
    {
        return false;
    }
    
    auto start = _offset[point_id];
    for (const auto &label : labels) {
        _data.push_back(static_cast<uint32_t>(label));
    }

    _offset[point_id + 1] = _data.size();

    return true;
}

bool integer_label_vector::check_label_exists(uint32_t point_id, uint32_t label) {
    if (point_id >= _offset.size() - 1) return false;

    auto start = _offset[point_id];
    auto end = _offset[point_id + 1];
    size_t last_check = 0;
    return binary_search(start, end, label, last_check);
}

template <typename LabelT>
bool integer_label_vector::check_label_exists(uint32_t point_id, const std::vector<LabelT> &labels) {
    if (point_id >= _offset.size() - 1) return false;

    auto start = _offset[point_id];
    auto end = _offset[point_id + 1];
   
    for (const auto &label : labels) {
        size_t last_check = 0;
        if (binary_search(start, end, static_cast<uint32_t>(label), last_check)) {
            return true;
        }
        start = last_check;
    }

    return false;
}

bool integer_label_vector::check_label_full_contain(uint32_t point_id, const std::vector<uint32_t>& labels)
{
    if (point_id >= _offset.size() - 1) return false;

    auto start = _offset[point_id];
    auto end = _offset[point_id + 1];

    
    for (const auto &label : labels) {
        size_t last_check = 0;
        if (!binary_search(start, end, label, last_check)) {
            return false;
        }
        start = last_check;
    }

    return true;
}

bool integer_label_vector::check_label_full_contain(uint32_t source_point, uint32_t target_point)
{
    if (source_point >= _offset.size() - 1 || target_point >= _offset.size() - 1) return false;

    auto start = _offset[source_point];
    auto end = _offset[source_point + 1];
    auto target_start = _offset[target_point];
    auto target_end = _offset[target_point + 1];

    for (size_t i = target_start; i < target_end; i++)
    {
        size_t last_check = 0;
        if (!binary_search(start, end, _data[i], last_check)) {
            return false;
        }
        start = last_check;
    }

    return true;
}

bool integer_label_vector::binary_search(size_t start, size_t end, uint32_t label, size_t& last_check)
{
    while (start < end) {
        size_t mid = (start + end) >> 1;

        if (_data[mid] == label) {
            last_check = mid;
            return true;
        }
        if (_data[mid] < label) start = mid + 1;
        else end = mid;
    }

    last_check = start;

    return false;
}

size_t integer_label_vector::get_memory_usage() const
{
    return _offset.capacity() * sizeof(size_t) + _data.capacity() * sizeof(uint32_t);
}

bool integer_label_vector::write_to_file(const std::string& label_file) const
{
    //format:
    // format version: uint8
    //  num_points: uint32
    //  offset array: uint64[num_points + 1]
    //   label data: uint32[total_labels]
    const uint8_t format_version = 1;
    std::ofstream outfile(label_file, std::ios::binary);
    if (outfile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + label_file, -1);
    }
    outfile.write((char*)(&format_version), sizeof(uint8_t));
    
    uint32_t num_points = static_cast<uint32_t>(_offset.size() - 1);
    outfile.write((char*)(&num_points), sizeof(uint32_t));
    outfile.write((char*)_offset.data(), _offset.size() * sizeof(size_t));
    
    outfile.write((char*)_data.data(), _data.size() * sizeof(uint32_t));
    outfile.close();

    return true;
}

const std::vector<size_t> &integer_label_vector::get_offset_vector() const
{
    return _offset;
}

const std::vector<uint32_t>& integer_label_vector::get_data_vector() const
{
    return _data;
}

template bool integer_label_vector::add_labels<uint16_t>(uint32_t point_id, std::vector<uint16_t>& labels);
template bool integer_label_vector::add_labels<uint32_t>(uint32_t point_id, std::vector<uint32_t>& labels);
template bool integer_label_vector::check_label_exists<uint16_t>(uint32_t point_id, const std::vector<uint16_t>& labels);
template bool integer_label_vector::check_label_exists<uint32_t>(uint32_t point_id, const std::vector<uint32_t>& labels);

}