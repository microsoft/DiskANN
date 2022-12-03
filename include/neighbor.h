// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include "utils.h"

namespace diskann {

  struct Neighbor {
    unsigned id;
    float    distance;
    bool     checked;

    Neighbor() = default;
    Neighbor(unsigned id, float distance)
        : id{id}, distance{distance}, checked(false) {
    }

    inline bool operator<(const Neighbor &other) const {
      return distance < other.distance ||
             (distance == other.distance && id < other.id);
    }

    inline bool operator==(const Neighbor &other) const {
      return (id == other.id);
    }
  };

  // Invariant: after every `insert` and `pop`, `cur_` points to
  //            the first Neighbor which is unchecked.
  class NeighborSet {
   public:
    NeighborSet() = default;

    explicit NeighborSet(size_t capacity)
        : size_(0), capacity_(capacity), data_(capacity_ + 1) {
    }

    void insert(const Neighbor &nbr) {
      if (size_ == capacity_ && nbr.distance >= data_[size_ - 1].distance) {
        return;
      }
      int lo = 0, hi = size_;
      while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (data_[mid].distance > nbr.distance) {
          hi = mid;
        } else {
          lo = mid + 1;
        }
      }
      std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor));
      data_[lo] = {nbr.id, nbr.distance};
      if (size_ < capacity_) {
        size_++;
      }
      if (lo < cur_) {
        cur_ = lo;
      }
    }

    Neighbor pop() {
      data_[cur_].checked = true;
      size_t pre = cur_;
      while (cur_ < size_ && data_[cur_].checked) {
        cur_++;
      }
      return data_[pre];
    }

    bool has_next() const {
      return cur_ < size_;
    }

    size_t size() const {
      return size_;
    }

    size_t capacity() const {
      return capacity_;
    }

    void reserve(size_t capacity) {
      if (capacity + 1 > data_.size()) {
        data_.resize(capacity + 1);
      }
      capacity_ = capacity;
    }

    Neighbor &operator[](size_t i) {
      return data_[i];
    }

    Neighbor operator[](size_t i) const {
      return data_[i];
    }

    void clear() {
      size_ = 0;
      cur_ = 0;
    }

   private:
    size_t                size_, capacity_, cur_;
    std::vector<Neighbor> data_;
  };

}  // namespace diskann
