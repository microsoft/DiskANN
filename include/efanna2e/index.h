//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include "distance.h"
#include "parameters.h"

namespace efanna2e {
  struct QueryStats {
    uint64_t total_us = 0;      // total time to process query in micros
    uint64_t n_4k = 0;          // # of 4kB reads
    uint64_t n_8k = 0;          // # of 8kB reads
    uint64_t n_12k = 0;         // # of 12kB reads
    uint64_t n_ios = 0;         // total # of IOs issued
    uint64_t read_size = 0;     // total # of bytes read
    uint64_t io_us = 0;         // total time spent in IO
    uint64_t n_cmps_saved = 0;  // # cmps saved
  };

  class Index {
   public:
    explicit Index(const size_t dimension, const size_t n, Metric metric);

    virtual ~Index();

    virtual void Build(size_t n, const float *data,
                       const Parameters &parameters) = 0;

    virtual std::pair<int, int> Search(const float *query, const float *x,
                                       const size_t      K,
                                       const Parameters &parameters,
                                       unsigned *        indices) = 0;

    virtual void Save(const char *filename) = 0;

    virtual void Load(const char *filename) = 0;

    inline bool HasBuilt() const {
      return has_built;
    }

    inline size_t GetDimension() const {
      return dimension_;
    };

    inline size_t GetSizeOfDataset() const {
      return nd_;
    }

    inline const float *GetDataset() const {
      return data_;
    }

   protected:
    const size_t dimension_;
    const float *data_;
    size_t       nd_;
    bool         has_built;
    Distance *   distance_;
  };

  inline void percentile_stats(
      QueryStats *stats, uint64_t len, const char *name, const char *suffix,
      const std::function<uint64_t(const QueryStats &)> &member_fn) {
    std::vector<uint64_t> vals(len);
    for (uint64_t i = 0; i < len; i++) {
      vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(),
              [](const uint64_t &left, const uint64_t &right) {
                return left < right;
              });

    uint64_t sum_vals = std::accumulate(vals.begin(), vals.end(), (uint64_t) 0);

    std::cout << "***" << name
              << " statistics***\navg: " << sum_vals * 1.0 / len << suffix;
    std::cout << ", 50pc: " << vals[len / 2] << suffix
              << ", 90pc: " << vals[(uint64_t)(0.9 * len)];
    std::cout << suffix << ", 95pc: " << vals[(uint64_t)(0.95 * len)];
    std::cout << suffix << ", 99pc: " << vals[(uint64_t)(0.99 * len)];
    std::cout << suffix << ", 99.9pc: " << vals[(uint64_t)(0.999 * len)]
              << suffix << std::endl;
    vals.clear();
  }
}
