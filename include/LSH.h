#pragma once
#include <float.h>
#include <cstdint>
#include <memory>
#include <iostream>



//TODO: Move general utility functions like gen_random_slice from partition.* to util.*
void gen_random_slice(const std::string data_file, double p_val, float *&sampled_data, size_t &slice_size,
                      size_t &ndims);
namespace diskann
{

//typedef uint16_t HashT;
//#define LSH_NUM_AXES (sizeof(diskann::HashT) * 8)
#define LSH_MIN_REGION_MATCH 2


template<typename T>
class LSH
{
  public:
    DISKANN_DLLEXPORT LSH(uint32_t num_dims, uint32_t num_axes);
    DISKANN_DLLEXPORT LSH(uint32_t num_dims, uint32_t num_axes, float *axes);
    //takes ownership of the centroid pointer. Should not be freed by the client.
    DISKANN_DLLEXPORT LSH<T> &with_centroid(float *centroid);

    DISKANN_DLLEXPORT T get_hash(const float *vec, float *dot_product);
    DISKANN_DLLEXPORT void dump_axes_to_text_file(std::ostream &ostream);
    DISKANN_DLLEXPORT void dump_axes_to_bin_file(std::ostream &ostream);

  private:
    void check_axes_count_wrt_hash_size(uint32_t num_axes);
    void center_vector(const float *vec, float *result);
    void generate_random_axes(uint32_t dim, uint32_t num, float *vecs, float min_value = FLT_TRUE_MIN,
                              float max_value = FLT_MAX, bool normalized = false);


    std::unique_ptr<float[]> _axes; //LSH_NUM_AXES x num_dims
    std::unique_ptr<float[]> _centroid; //num_dims
    std::unique_ptr<float[]> _centered_vector; //num_dims
    uint32_t _num_dims, _num_axes;
};

} // namespace diskann