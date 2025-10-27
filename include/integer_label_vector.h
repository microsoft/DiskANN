#pragma once
#include <vector>
#include <string>

namespace diskann
{

class integer_label_vector
{
public:
    bool initialize(size_t numpoints, size_t total_labels);

    bool initialize_from_file(const std::string &label_file, size_t &numpoints);

    bool write_to_file(const std::string &label_file) const;

    template <typename LabelT>
    bool add_labels(uint32_t point_id, std::vector<LabelT> &labels);
    
    bool check_label_exists(uint32_t point_id, uint32_t label);

    template <typename LabelT>
    bool check_label_exists(uint32_t point_id, const std::vector<LabelT> &labels);

    bool check_label_full_contain(uint32_t point_id, const std::vector<uint32_t> &labels);

    bool check_label_full_contain(uint32_t source_point, uint32_t target_point);

    const std::vector<size_t> &get_offset_vector() const;

    const std::vector<uint32_t> &get_data_vector() const;

    size_t get_memory_usage() const;

  private:
    bool binary_search(size_t start, size_t end, uint32_t label, size_t& last_check);

private:
  std::vector<size_t> _offset;
  std::vector<uint32_t> _data;
};

}