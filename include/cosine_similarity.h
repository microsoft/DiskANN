#pragma once
#include <vector>

namespace NSG {
  float              vectorMagnitude(const float* point, int ndims);
  std::vector<float> compute_cosine_similarity(
      const float* query, const unsigned* indices, const float* all_data,
      const unsigned ndims, const unsigned npts, const float queryMagnitude);
}  // namespace NSG
