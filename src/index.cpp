//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//
#include <index.h>
namespace NSG {

  template<>
  Index<float>::Index(const size_t dimension, const size_t n, Metric metric,
                      const size_t max_points)
      : _dim(dimension), _nd(n), _has_built(false) {
    _max_points = (max_points > 0) ? max_points : n;
    if (_max_points < n) {
      std::cerr << "max_points must be >= n; max_points: " << _max_points
                << "  n: " << _nd << std::endl;
      exit(-1);
    }

    switch (metric) {
      case L2:
        _distance = new DistanceL2();
        break;
      default:
        _distance = new DistanceL2();
        break;
    }
  }

  template<>
  Index<int8_t>::Index(const size_t dimension, const size_t n, Metric metric,
                       const size_t max_points)
      : _dim(dimension), _nd(n), _has_built(false) {
    _max_points = (max_points > 0) ? max_points : n;
    if (_max_points < n) {
      std::cerr << "max_points must be >= n; max_points: " << _max_points
                << "  n: " << _nd << std::endl;
      exit(-1);
    }
    switch (metric) {
      case L2:
        _distance = new DistanceL2Int8();
        break;
      default:
        _distance = new DistanceL2Int8();
        break;
    }
  }

  template<>
  Index<uint8_t>::Index(const size_t dimension, const size_t n, Metric metric,
                        const size_t max_points)
      : _dim(dimension), _nd(n), _has_built(false) {
    _max_points = (max_points > 0) ? max_points : n;
    if (_max_points < n) {
      std::cerr << "max_points must be >= n; max_points: " << _max_points
                << "  n: " << _nd << std::endl;
      exit(-1);
    }
    switch (metric) {
      case L2:
        _distance = new DistanceL2UInt8();
        break;
      default:
        _distance = new DistanceL2UInt8();
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
