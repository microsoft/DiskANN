#pragma once
#include <cstdint>
#include <memory>
#include <iostream>

namespace diskann
{

#define LSH_NUM_AXES 8
#define LSH_MIN_REGION_MATCH 2

class LSH
{
  public:
    DISKANN_DLLEXPORT LSH(uint32_t num_dims, uint32_t num_axes);
    DISKANN_DLLEXPORT LSH(uint32_t num_dims, uint32_t num_axes, float *axes);

    DISKANN_DLLEXPORT uint8_t get_hash(const float *vec, float *dot_product);
    DISKANN_DLLEXPORT void dump_axes_to_text_file(std::ostream &ostream);
    DISKANN_DLLEXPORT void dump_axes_to_bin_format(std::ostream &ostream);

  private:
    std::unique_ptr<float[]> _axes;
    uint32_t _num_dims, _num_axes;
};

} // namespace diskann