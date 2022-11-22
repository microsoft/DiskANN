// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <vector>
#include "utils.h"

namespace diskann {

  struct Neighbor {
    unsigned id;
    float    distance;
    bool     flag;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f)
        : id{id}, distance{distance}, flag(f) {
    }

    inline bool operator<(const Neighbor &other) const {
      return distance < other.distance;
    }
    inline bool operator==(const Neighbor &other) const {
      return (id == other.id);
    }
  };

  struct SimpleNeighbor {
    unsigned id;
    float    distance;

    SimpleNeighbor() = default;
    SimpleNeighbor(unsigned id, float distance) : id(id), distance(distance) {
    }

    inline bool operator<(const SimpleNeighbor &other) const {
      return distance < other.distance;
    }

    inline bool operator==(const SimpleNeighbor &other) const {
      return id == other.id;
    }
  };
  struct SimpleNeighbors {
    std::vector<SimpleNeighbor> pool;
  };

  static inline unsigned InsertIntoPool(std::vector<Neighbor> &neighbors,
                                        unsigned K, unsigned cap, Neighbor nn) {
    // find the location to insert
    auto it = std::lower_bound(neighbors.begin(), neighbors.begin() + K, nn);
    unsigned ans = it - neighbors.begin();
    neighbors.insert(it, nn);
    if (neighbors.size() > cap) {
      neighbors.pop_back();
    }
    return ans;
  }
}  // namespace diskann
