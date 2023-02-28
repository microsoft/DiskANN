// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <typeinfo>
#include "defaults.h"

namespace diskann {
  class MutationParameters {
   public:
    MutationParameters(
        const uint32_t list_size, const uint32_t max_degree,
        const bool     saturate_graph,
        const uint32_t max_occlusion_size = defaults::MAX_OCCLUSION_SIZE,
        const float    alpha = defaults::ALPHA,
        const uint32_t num_rounds = defaults::NUM_ROUNDS,
        const uint32_t num_threads = defaults::NUM_THREADS)
        : _list_size(list_size), _max_degree(max_degree),
          _saturate_graph(saturate_graph),
          _max_occlusion_size(max_occlusion_size), _alpha(alpha),
          _num_rounds(num_rounds), _num_threads(num_threads){};

    MutationParameters(const MutationParameters &) = delete;
    MutationParameters &operator=(const MutationParameters &) = delete;

    uint32_t get_max_degree() const {
      return _max_degree;
    }
    uint32_t get_search_list_size() const {
      return _list_size;
    }
    uint32_t get_max_occlusion_size() const {
      return _max_occlusion_size;
    }
    float get_alpha() const {
      return _alpha;
    }
    uint32_t get_num_rounds() const {
      return _num_rounds;
    }
    bool is_saturate_graph() const {
      return _saturate_graph;
    }
    uint32_t get_num_threads() const {
      return _num_threads;
    }

   private:
    uint32_t _list_size;
    uint32_t _max_degree;
    bool     _saturate_graph;
    uint32_t _max_occlusion_size;
    float    _alpha;
    uint32_t _num_rounds;
    uint32_t _num_threads;
  };

  class SearchParameters {
   public:
    SearchParameters(const uint32_t list_size,
                     const uint32_t num_threads = defaults::NUM_THREADS)
        : _list_size(list_size), _num_threads(num_threads){};

    SearchParameters(const SearchParameters &) = delete;
    SearchParameters &operator=(const SearchParameters &) = delete;

    uint32_t get_search_list_size() const {
      return _list_size;
    }

    uint32_t get_num_threads() const {
      return _num_threads;
    }

   private:
    uint32_t _list_size;
    uint32_t _num_threads;
  };
}  // namespace diskann
