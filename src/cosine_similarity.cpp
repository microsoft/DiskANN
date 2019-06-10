#include <cosine_similarity.h>
#include <vector>
#include <iostream>

namespace NSG {

  float vectorMagnitude(const float* point, int ndims) {
    float sumSquares = 0.0f;
    for (int i = 0; i < ndims; i++) {
      sumSquares += point[i] * point[i];
    }

    return sqrtf(sumSquares);
  }

  std::vector<float> compute_cosine_similarity(
      const float* query, const unsigned* indices, const float* all_data,
      const unsigned ndims, const unsigned npts, const float queryMagnitude) 
  {
    std::vector<float> cos_dists;
    cos_dists.reserve(npts);

    for (size_t i = 0; i < npts; i++) {
      float        scalarProduct = 0.0;
      float        magnitudePoint = 0.0;
      const float* point = all_data + (size_t)(indices[i] * ndims);
      for (unsigned d = 0; d < ndims; d++) {
        scalarProduct += (point[d] * query[d]);
        magnitudePoint += (point[d] * point[d]);
      }

	  auto cosDist = scalarProduct / (sqrtf(magnitudePoint) * queryMagnitude);
      cos_dists.push_back(cosDist);
    }
    return cos_dists;
  }
}  // namespace NSG