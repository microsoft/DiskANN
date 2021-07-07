#pragma once

#include <cassert>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include "tsl/robin_set.h"

#include "parameters.h"

namespace diskann {

  template<typename TagT = int>
  struct Neighbor_Tag {
    TagT  tag;
    float dist;

    Neighbor_Tag() = default;

    Neighbor_Tag(TagT tag, float dist) : tag{tag}, dist{dist} {
    }
    inline bool operator<(const Neighbor_Tag &other) const {
      return (dist < other.dist);
    }
    inline bool operator==(const Neighbor_Tag &other) const {
      return (tag == other.tag);
    }
  };
}  // namespace diskann
