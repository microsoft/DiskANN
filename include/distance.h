#pragma once
#include "windows_customizations.h"


namespace diskann
{
enum Metric
{
    L2 = 0,
    INNER_PRODUCT = 1,
    COSINE = 2,
    FAST_L2 = 3
};

template <typename T> class Distance
{
  public:
    // distance comparison function
    virtual float compare(const T *a, const T *b, uint32_t length) const = 0;

    // For MIPS, normalization => a new dimension gets added to the vectors.
    // This function lets callers know if the normalization process
    // changes the dimension.
    virtual uint32_t new_dimension(uint32_t orig_dimension) const
    {
        return orig_dimension;
    }

    // This is for efficiency. If no normalization is required, the callers
    //can simply ignore the normalize_data_for_build() function.
    virtual bool normalization_required() const
    {
        return false;
    }
    
    //Check the normalization_required() function before calling this. 
    //Clients can call the function like this: 
    // 
    // if (metric->normalization_required()){
    //    T* normalized_data_batch; 
    //    if ( metric->new_dimension() == orig_dim ) {
    //        normalized_data_batch = nullptr;
    //        modify_data = true; 
    //     } else {
    //        normalized_data_batch = new T[batch_size * metric->new_dimension()];
    //        modify_data = false;
    //     }
    //     Split data into batches of batch_size and for each, call:
    //      metric->normalize_data_for_build(data_batch, batch_size, orig_dim
    //                                        normalized_data_batch, modify_data);
    //The default implementation is correct but very inefficient!
    virtual void normalize_data_for_build(const T *original_data, 
                                          const uint32_t num_points, const uint32_t orig_dim,
                                          T *normalized_data, bool modify_data = true)
    {
    }

    //Invokes normalization for a single vector during search. The scratch space
    //has to be created by the caller keeping track of the fact that normalization
    //might change the dimension of the query vector. 
    virtual void normalize_vector_for_search(const T *query_vec, const uint32_t query_dim, 
                                             T* scratch_query) 
    {
        memcpy(scratch_query, query_vec, query_dim * sizeof(T));
    }

    //Providing a default implementation for the virtual destructor because we don't 
    //expect most metric implementations to need it. 
    virtual ~Distance() 
    {
    }
};

class DistanceCosineInt8 : public Distance<int8_t>
{
  public:
    DISKANN_DLLEXPORT virtual float compare(const int8_t *a, const int8_t *b, uint32_t length) const;
};

class DistanceL2Int8 : public Distance<int8_t>
{
  public:
    DISKANN_DLLEXPORT virtual float compare(const int8_t *a, const int8_t *b, uint32_t size) const;
};

// AVX implementations. Borrowed from HNSW code.
class AVXDistanceL2Int8 : public Distance<int8_t>
{
  public:
    DISKANN_DLLEXPORT virtual float compare(const int8_t *a, const int8_t *b, uint32_t length) const;
};

class DistanceCosineFloat : public Distance<float>
{
  public:
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t length) const;
};

class DistanceL2Float : public Distance<float>
{
  public:
#ifdef _WINDOWS
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t size) const;
#else
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t size) const __attribute__((hot));
#endif
};

class AVXDistanceL2Float : public Distance<float>
{
  public:
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t length) const;
};

class SlowDistanceL2Float : public Distance<float>
{
  public:
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t length) const;
};

class SlowDistanceCosineUInt8 : public Distance<uint8_t>
{
  public:
    DISKANN_DLLEXPORT virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t length) const;
};

class DistanceL2UInt8 : public Distance<uint8_t>
{
  public:
    DISKANN_DLLEXPORT virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t size) const;
};

// Simple implementations for non-AVX machines. Compiler can optimize.
template <typename T> class SlowDistanceL2Int : public Distance<T>
{
  public:
    // Implementing here because this is a template function
    DISKANN_DLLEXPORT virtual float compare(const T *a, const T *b, uint32_t length) const
    {
        uint32_t result = 0;
        for (uint32_t i = 0; i < length; i++)
        {
            result += ((int32_t)((int16_t)a[i] - (int16_t)b[i])) * ((int32_t)((int16_t)a[i] - (int16_t)b[i]));
        }
        return (float)result;
    }
};

template <typename T> class DistanceInnerProduct : public Distance<T>
{
  public:
    inline float inner_product(const T *a, const T *b, unsigned size) const;

    inline float compare(const T *a, const T *b, unsigned size) const
    {
        float result = inner_product(a, b, size);
        //      if (result < 0)
        //      return std::numeric_limits<float>::max();
        //      else
        return -result;
    }
};

template <typename T> class DistanceFastL2 : public DistanceInnerProduct<T>
{
    // currently defined only for float.
    // templated for future use.
  public:
    float norm(const T *a, unsigned size) const;
    float compare(const T *a, const T *b, float norm, unsigned size) const;
};

class AVXDistanceInnerProductFloat : public Distance<float>
{
  public:
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t length) const;
};

class AVXNormalizedCosineDistanceFloat : public Distance<float>
{
  private:
    AVXDistanceInnerProductFloat _innerProduct;

  public:
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t length) const
    {
        // Inner product returns negative values to indicate distance.
        // This will ensure that cosine is between -1 and 1.
        return 1.0f + _innerProduct.compare(a, b, length);
    }
    DISKANN_DLLEXPORT virtual uint32_t new_dimension(uint32_t orig_dimension);

    DISKANN_DLLEXPORT virtual bool normalization_required() const;

    DISKANN_DLLEXPORT virtual void normalize_data_for_build(const float *original_data, const uint32_t num_points,
                                                            const uint32_t orig_dim, float *normalized_data,
                                                            bool modify_orig);

    DISKANN_DLLEXPORT virtual void normalize_vector_for_search(const float *query_vec, const uint32_t query_dim,
                                                               float* scratch_query_vector);
};

template <typename T> Distance<T> *get_distance_function(Metric m);

} // namespace diskann
