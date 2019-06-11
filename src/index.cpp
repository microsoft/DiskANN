//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//
#include <efanna2e/index.h>
namespace NSG {
  template<>
  Index<float>::Index(const size_t dimension, const size_t n, Metric metric,
                      const size_t max_points)
      : dimension_(dimension), nd_(n), has_built(false) {
    max_points_ = (max_points > 0) ? max_points : n;
    if (max_points_ < n) {
      std::cerr << "max_points must be >= n; max_points: " << max_points_
                << "  n: " << nd_ << std::endl;
      exit(-1);
    }

    switch (metric) {
      case L2:
        distance_ = new DistanceL2();
        break;
      default:
        distance_ = new DistanceL2();
        break;
    }
  }

  template<>
  Index<int8_t>::Index(const size_t dimension, const size_t n, Metric metric,
                       const size_t max_points)
      : dimension_(dimension), nd_(n), has_built(false) {
    max_points_ = (max_points > 0) ? max_points : n;
    if (max_points_ < n) {
      std::cerr << "max_points must be >= n; max_points: " << max_points_
                << "  n: " << nd_ << std::endl;
      exit(-1);
    }
    switch (metric) {
      case L2:
        distance_ = new DistanceL2Int8();
        break;
      default:
        distance_ = new DistanceL2Int8();
        break;
    }
  }

  template<>
  Index<uint8_t>::Index(const size_t dimension, const size_t n, Metric metric,
                        const size_t max_points)
      : dimension_(dimension), nd_(n), has_built(false) {
    max_points_ = (max_points > 0) ? max_points : n;
    if (max_points_ < n) {
      std::cerr << "max_points must be >= n; max_points: " << max_points_
                << "  n: " << nd_ << std::endl;
      exit(-1);
    }
    switch (metric) {
      case L2:
        distance_ = new DistanceL2UInt8();
        break;
      default:
        distance_ = new DistanceL2UInt8();
        break;
    }
  }

  template<>
  Index<float>::~Index() {
  }

  template<>
  Index<int8_t>::~Index() {
  }

  template<>
  Index<uint8_t>::~Index() {
  }
}
