#include <memory>
#include <cstdint>
#include <random>
#include <sstream>
#include <mkl.h>

#include "utils.h"
#include "lsh.h"

namespace diskann
{

template<typename HashT>
void LSH<HashT>::generate_random_axes(uint32_t dim, uint32_t num, float *vecs, float min_value ,
                             float max_value, bool normalized)
{
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dis(min_value, max_value);

    for (uint32_t i = 0; i < num; i++)
    {
        for (uint32_t j = 0; j < dim; j++)
        {
            vecs[i * dim + j] = dis(rng);
        }
        if (normalized)
        {
            for (uint32_t i = 0; i < num; i++)
                normalize(vecs + (dim * i), dim);
        }
    }
}

template<typename HashT>
void LSH<HashT>::center_vector(const float *vec, float *result)
{
    for (auto i = 0; i < (int)_num_dims; i++)
    {
        result[i] = vec[i] - _centroid[i];
    }
}

//void generate_clustered_axes(const std::string& data_file, uint32_t dim, uint32_t num, float *vecs, float min_value = FLT_TRUE_MIN,
//                             float max_value = FLT_MAX, bool normalized = false)
//{
//    double p_val = 
//    gen_random_slice(data_file, p_val, sampled_data, slice_size, dim);
//}

template<typename HashT> 
void LSH<HashT>::check_axes_count_wrt_hash_size(uint32_t num_axes)
{
    // for now this is a simple check. The number of axes should be such that we can fit the hashes
    // into some uintX object.
    assert(num_axes <= sizeof(HashT) * 8);
    if (num_axes > sizeof(HashT) * 8)
    {
        std::stringstream ss;
        ss << "num_axes: " << num_axes << " does not fit in type specified. Size of type: " << sizeof(HashT) * 8
           << " bits." << std::endl;
        throw std::exception(ss.str().c_str());
    }
}

template<typename HashT>
LSH<HashT>::LSH(uint32_t num_dims, uint32_t num_axes) : _num_axes(num_axes), _num_dims(num_dims)
{
    check_axes_count_wrt_hash_size(num_axes);
    _axes = std::make_unique<float[]>(num_axes * num_dims);
    generate_random_axes(num_dims, num_axes, const_cast<float *>(_axes.get()), -1, 1, false);
}

template<typename HashT>
LSH<HashT>::LSH(uint32_t num_dims, uint32_t num_axes, float *axes) : _num_axes(num_axes), _num_dims(num_dims)
{
    check_axes_count_wrt_hash_size(num_axes);
    //We are assuming sufficient memory has been allocated for axes.
    _axes.reset(axes);
}

template <typename HashT> LSH<HashT> &LSH<HashT>::with_centroid(float *centroid)
{
    _centroid.reset(centroid);
    _centered_vector = std::make_unique<float[]>(_num_dims);
    return *this;
}

template<typename HashT>
HashT LSH<HashT>::get_hash(const float *vec, float *dot_product)
{
    memset(dot_product, 0, _num_axes * sizeof(float));
    if (_centroid != nullptr)
    {
        center_vector(vec, const_cast<float *>(_centered_vector.get()));
    }

    // computing Y = alpha * A (mat) * X (vec) + beta * Y
    cblas_sgemv(CblasRowMajor, CblasNoTrans, _num_axes /* M */, _num_dims /* N */, 1.0f, /* alpha */
                const_cast<float*>(_axes.get()),                                         /* A */
                _num_dims,                                                               /* lda */
                _centered_vector.get(),                                                  /* X */
                1,                                                                       /* inc x */
                0,                                                                       /* Beta */
                dot_product,                                                             /* Y */
                1                                                                        /* inc Y */
    );
    HashT lsh_value = 0;
    for (uint32_t i = 0; i < _num_axes; i++)
    {
        lsh_value = dot_product[i] < 0 ? lsh_value : lsh_value | (((HashT)1) << i);
    }
    return lsh_value;
}

template<typename HashT>
void LSH<HashT>::dump_axes_to_text_file(std::ostream &out_stream)
{
    for (uint32_t i = 0; i < _num_axes; i++)
    {
        for (uint32_t j = 0; j < _num_dims; j++)
        {
            out_stream << _axes[i * _num_dims + j] << " ";
        }
        out_stream << std::endl;
    }
}

// Assuming ostream is opened in binary mode.
template <typename HashT> 
void LSH<HashT>::dump_axes_to_bin_file(std::ostream &out_stream)
{
    out_stream.write((const char *)&_num_axes, sizeof(uint32_t));
    out_stream.write((const char *)&_num_dims, sizeof(uint32_t));
    out_stream.write((const char *)_axes.get(), sizeof(float) * _num_axes * _num_dims);
}

template DISKANN_DLLEXPORT LSH<uint8_t>::LSH(uint32_t num_dims, uint32_t num_axes);
template DISKANN_DLLEXPORT LSH<uint8_t>::LSH(uint32_t num_dims, uint32_t num_axes, float *axes);
template DISKANN_DLLEXPORT LSH<uint8_t> &LSH<uint8_t>::with_centroid(float *centroid);
template DISKANN_DLLEXPORT uint8_t LSH<uint8_t>::get_hash(const float *vec, float *dot_product);
template DISKANN_DLLEXPORT void LSH<uint8_t>::dump_axes_to_bin_file(std::ostream &out_stream);
template DISKANN_DLLEXPORT void LSH<uint8_t>::dump_axes_to_text_file(std::ostream &out_stream);

template DISKANN_DLLEXPORT LSH<uint16_t>::LSH(uint32_t num_dims, uint32_t num_axes);
template DISKANN_DLLEXPORT LSH<uint16_t>::LSH(uint32_t num_dims, uint32_t num_axes, float *axes);
template DISKANN_DLLEXPORT LSH<uint16_t> &LSH<uint16_t>::with_centroid(float *centroid);
template DISKANN_DLLEXPORT uint16_t LSH<uint16_t>::get_hash(const float *vec, float *dot_product);
template DISKANN_DLLEXPORT void LSH<uint16_t>::dump_axes_to_bin_file(std::ostream &out_stream);
template DISKANN_DLLEXPORT void LSH<uint16_t>::dump_axes_to_text_file(std::ostream &out_stream);

template DISKANN_DLLEXPORT LSH<uint32_t>::LSH(uint32_t num_dims, uint32_t num_axes);
template DISKANN_DLLEXPORT LSH<uint32_t>::LSH(uint32_t num_dims, uint32_t num_axes, float *axes);
template DISKANN_DLLEXPORT LSH<uint32_t>& LSH<uint32_t>::with_centroid(float *centroid);
template DISKANN_DLLEXPORT uint32_t LSH<uint32_t>::get_hash(const float *vec, float *dot_product);
template DISKANN_DLLEXPORT void LSH<uint32_t>::dump_axes_to_bin_file(std::ostream &out_stream);
template DISKANN_DLLEXPORT void LSH<uint32_t>::dump_axes_to_text_file(std::ostream &out_stream);

template DISKANN_DLLEXPORT LSH<uint64_t>::LSH(uint32_t num_dims, uint32_t num_axes);
template DISKANN_DLLEXPORT LSH<uint64_t>::LSH(uint32_t num_dims, uint32_t num_axes, float *axes);
template DISKANN_DLLEXPORT LSH<uint64_t> &LSH<uint64_t>::with_centroid(float *centroid);
template DISKANN_DLLEXPORT uint64_t LSH<uint64_t>::get_hash(const float *vec, float *dot_product);
template DISKANN_DLLEXPORT void LSH<uint64_t>::dump_axes_to_bin_file(std::ostream &out_stream);
template DISKANN_DLLEXPORT void LSH<uint64_t>::dump_axes_to_text_file(std::ostream &out_stream);
} //namespace diskann

