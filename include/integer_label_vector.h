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

    bool initialize_from_buffers(const size_t *offsets, size_t num_points,
                                 const uint32_t *labels, size_t total_labels);

    // Zero-copy load path: caller pre-sizes both buffers, writes into the raw
    // pointers, and the integer_label_vector is ready to use. The two-step
    // form lets the caller skip the intermediate vector<uint8_t> + assign()
    // copies that initialize_from_buffers incurs.
    void resize_for_load(size_t num_points, size_t total_labels);
    size_t *mutable_offset_data();   // size: num_points + 1 entries (size_t each)
    uint32_t *mutable_label_data();  // size: total_labels entries (uint32_t each)

    bool write_to_file(const std::string &label_file) const;

    template <typename LabelT>
    bool add_labels(uint32_t point_id, std::vector<LabelT> &labels);
    
    bool check_label_exists(uint32_t point_id, uint32_t label);

    template <typename LabelT>
    bool check_label_exists(uint32_t point_id, const std::vector<LabelT> &labels);

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