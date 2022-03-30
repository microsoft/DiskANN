// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <ctime>
#include <iomanip>
#include <omp.h>

#include "utils.h"
#include <restapi/search_wrapper.h>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

namespace diskann {
  const unsigned int DEFAULT_W = 1;

  SearchResult::SearchResult(unsigned int K, unsigned int elapsed_time_in_ms,
                             const unsigned* const    indices,
                             const float* const       distances,
                             const std::string* const tags)
      : _K(K), _search_time_in_ms(elapsed_time_in_ms) {
    for (unsigned i = 0; i < K; ++i) {
      this->_indices.push_back(indices[i]);
      this->_distances.push_back(distances[i]);
      if (tags != NULL)
        this->_tags.push_back(tags[i]);
    }
    if (tags != NULL)
      this->_tags_enabled = true;
  }

  BaseSearch::BaseSearch(const char* tagsFile) {
    if (tagsFile != nullptr) {
      std::ifstream in(tagsFile);

      if (!in.is_open()) {
        std::cerr << "Could not open " << tagsFile << std::endl;
      }

      std::string tag;
      while (std::getline(in, tag)) {
        _tags_str.push_back(tag);
      }

      _tags_enabled = true;

      std::cout << "Loaded " << _tags_str.size() << " tags from " << tagsFile
                << std::endl;
    } else {
      _tags_enabled = false;
    }
  }

  void BaseSearch::lookup_tags(const unsigned K, const unsigned* indices,
                               std::string* ret_tags) {
    if (_tags_enabled == false)
      throw std::runtime_error("Can not look up tags as they are not enabled.");
    else {
      for (unsigned k = 0; k < K; ++k) {
        if (indices[k] > _tags_str.size())
          throw std::runtime_error(
              "In tag lookup, index exceeded the number of tags");
        else
          ret_tags[k] = _tags_str[indices[k]];
      }
    }
  }

  template<typename T>
  InMemorySearch<T>::InMemorySearch(const char* baseFile, const char* indexFile,
                                    const char* tagsFile, Metric m)
      : BaseSearch(tagsFile) {
    _index = std::unique_ptr<diskann::Index<T>>(
        new diskann::Index<T>(m, baseFile, 0, false));

    _index->load(indexFile);
  }

  template<typename T>
  SearchResult InMemorySearch<T>::search(const T*           query,
                                         const unsigned int dimensions,
                                         const unsigned int K,
                                         const unsigned int Ls) {
    unsigned int* indices = new unsigned int[K];
    float*        distances = new float[K];

    auto startTime = std::chrono::high_resolution_clock::now();
    _index->search(query, K, Ls, indices, distances);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - startTime)
                        .count();

    std::string* tags = nullptr;
    if (_tags_enabled) {
      tags = new std::string[K];
      lookup_tags(K, indices, tags);
    }

    SearchResult result(K, (unsigned int) duration, indices, distances, tags);

    delete[] indices;
    delete[] distances;
    return result;
  }

  template<typename T>
  InMemorySearch<T>::~InMemorySearch() {
  }

  template<typename T>
  PQFlashSearch<T>::PQFlashSearch(const char*    indexPrefix,
                                  const unsigned num_nodes_to_cache,
                                  const unsigned num_threads,
                                  const char* tagsFile, Metric m)
      : BaseSearch(tagsFile) {
    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    auto ptr = new LinuxAlignedFileReader();
    reader.reset(ptr);
#endif

    _index = std::unique_ptr<diskann::PQFlashIndex<T>>(
        new diskann::PQFlashIndex<T>(reader, m));

    std::string index_prefix_path(indexPrefix);
    std::string pq_prefix = index_prefix_path + "_pq";
    std::string disk_index_file = index_prefix_path + "_disk.index";
    std::string warmup_query_file = index_prefix_path + "_sample_data.bin";

    _index->load(num_threads, pq_prefix.c_str(), disk_index_file.c_str());

    std::vector<uint32_t> node_list;
    std::cout << "Caching " << num_nodes_to_cache
              << " BFS nodes around medoid(s)" << std::endl;
    _index->cache_bfs_levels(num_nodes_to_cache, node_list);
    //_index->generate_cache_list_from_sample_queries(
    ///    warmup_query_file, 15, 6, num_nodes_to_cache, num_threads,
    ///    node_list);
    _index->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();
    omp_set_num_threads(num_threads);
  }

  template<typename T>
  SearchResult PQFlashSearch<T>::search(const T*           query,
                                        const unsigned int dimensions,
                                        const unsigned int K,
                                        const unsigned int Ls) {
    _u64*     indices_u64 = new _u64[K];
    unsigned* indices = new unsigned[K];
    float*    distances = new float[K];

    auto startTime = std::chrono::high_resolution_clock::now();
    _index->cached_beam_search(query, K, Ls, indices_u64, distances,
                               DEFAULT_W);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - startTime)
                        .count();
    for (unsigned k = 0; k < K; ++k)
      indices[k] = indices_u64[k];

    std::string* tags = nullptr;
    if (_tags_enabled) {
      tags = new std::string[K];
      lookup_tags(K, indices, tags);
    }

    SearchResult result(K, (unsigned int) duration, indices, distances, tags);

    delete[] indices_u64;
    delete[] indices;
    delete[] distances;
    return result;
  }

  template<typename T>
  PQFlashSearch<T>::~PQFlashSearch() {
  }

  template class InMemorySearch<float>;
  template class InMemorySearch<int8_t>;
  template class InMemorySearch<uint8_t>;

  template class PQFlashSearch<float>;
  template class PQFlashSearch<int8_t>;
  template class PQFlashSearch<uint8_t>;
}  // namespace diskann
