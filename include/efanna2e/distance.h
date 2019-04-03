//
// Created by 付聪 on 2017/6/21.
//

#pragma once

#include <efanna2e/util.h>
#include <intrin.h>
#include <iostream>
namespace NSG {
  enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3 };
  class Distance {
   public:
    virtual float compare(const float *a, const float *b,
                          unsigned length) const = 0;
    virtual ~Distance() {
    }
  };

  class DistanceL2 : public Distance {
   public:
    float compare(const float *a, const float *b, unsigned size) const {
      float result = 0;
// #pragma omp for simd reduction(+ : result)
      for (unsigned i = 0; i < size; i++) {
        float temp = (a[i] - b[i]);
        result += temp * temp;
	  }
      return result;
    }
  };
}  // namespace NSG