// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <memory>

#include "boost_dynamic_bitset_fwd.h"

namespace diskann {
  // A set of natural numbers (from 0 onwards). Made for scenario where the
  // pool of numbers is consecutive from zero to some max value and very
  // efficient methods for "add to set", "get any value from set", "is in set"
  // are needed. The memory usage of the set is determined by the largest
  // number of inserted entries (uses a vector as a backing store) as well as
  // the largest value to be placed in it (uses bitset as well).
  template<typename T>
  class natural_number_set {
   public:
    static_assert(std::is_trivial_v<T>, "Identifier must be a trivial type");

    natural_number_set();

    bool   is_empty() const;
    void   reserve(size_t count);
    void   insert(T id);
    T      pop_any();
    void   clear();
    size_t size() const;
    bool   is_in_set(T id) const;

   private:
    // Values that are currently in set.
    std::vector<T> _values_vector;

    // Values that are currently in set where each bit being set to 1
    // means that its index is in the set.
    //
    // Use a pointer here to allow for forward declaration of dynamic_bitset
    // in public headers to avoid making boost a dependency for clients
    // of DiskANN.
    std::unique_ptr<boost::dynamic_bitset<>> _values_bitset;
  };
}  // namespace diskann
