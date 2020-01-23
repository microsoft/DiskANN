#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>
#include "distance.h"
#include "parameters.h"

namespace diskann {
  struct QueryStats {
    uint64_t total_us = 0;      // total time to process query in micros
    uint64_t n_4k = 0;          // # of 4kB reads
    uint64_t n_8k = 0;          // # of 8kB reads
    uint64_t n_12k = 0;         // # of 12kB reads
    uint64_t n_ios = 0;         // total # of IOs issued
    uint64_t read_size = 0;     // total # of bytes read
    uint64_t io_us = 0;         // total time spent in IO
    uint64_t n_cmps_saved = 0;  // # cmps saved
    uint64_t n_cmps = 0;        // # cmps
    uint64_t n_cache_hits = 0;  // # cache_hits
    uint64_t n_hops = 0;        // # search hops
  };

  inline uint64_t get_percentile_stats(
      QueryStats *stats, uint64_t len, float percentile,
      const std::function<uint64_t(const QueryStats &)> &member_fn) {
    std::vector<uint64_t> vals(len);
    for (uint64_t i = 0; i < len; i++) {
      vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(),
              [](const uint64_t &left, const uint64_t &right) {
                return left < right;
              });

    float retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
  }

  inline float get_mean_stats(
      QueryStats *stats, uint64_t len,
      const std::function<uint64_t(const QueryStats &)> &member_fn) {
    float avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      avg += member_fn(stats[i]);
    }
    return avg / len;
  }
}
