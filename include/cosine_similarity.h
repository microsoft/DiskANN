#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace diskann {
  template<typename T>
  inline float compute_l2_norm(const T* vector, uint64_t ndims) {
    float norm = 0.0f;
    for (uint64_t i = 0; i < ndims; i++) {
      norm += vector[i] * vector[i];
    }
    return std::sqrt(norm);
  }

  template<typename T>
  inline float compute_cosine_similarity(const T* left, const T* right,
                                         uint64_t ndims) {
    float left_norm = compute_l2_norm<T>(left, ndims);
    float right_norm = compute_l2_norm<T>(right, ndims);
    float dot = 0.0f;
    for (uint64_t i = 0; i < ndims; i++) {
      dot += left[i] * right[i];
    }
    float cos_sim = dot / (left_norm * right_norm);
    return cos_sim;
  }

  inline std::vector<float> compute_cosine_similarity_batch(
      const float* query, const unsigned* indices, const float* all_data,
      const unsigned ndims, const unsigned npts) {
    std::vector<float> cos_dists;
    cos_dists.reserve(npts);

    for (size_t i = 0; i < npts; i++) {
      const float* point = all_data + (size_t)(indices[i]) * (size_t)(ndims);
      cos_dists.push_back(
          compute_cosine_similarity<float>(point, query, ndims));
    }
    return cos_dists;
  }
}  // namespace diskann
