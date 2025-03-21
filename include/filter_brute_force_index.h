// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once
#include <memory>
#include "common_includes.h"
#include "windows_customizations.h"
#include "concurrent_queue.h"
#include "filter_utils.h"


namespace diskann {
  //we'll consider k * NUM_OF_PQ_RESULTS_MULTIPLIER as the shortlist of canddiates
  //for fetching FP vectors. 
  const int NUM_OF_PQ_RESULTS_MULTIPLIER = 5;

  //TODO: We MUST refactor PQ stuff out of the PQFlashIndex and use
  //PQDataStore everywhere. Why are we not doing it here? Because it 
  //will double the PQ memory consumption. :(
  template <typename T, typename LabelT> class PQFlashIndex;
  struct QueryStats;

  class BruteForceScratch
  {
    public:
      BruteForceScratch()
      {
          _p_intermediate_pq_dists = nullptr;
          _p_intermediate_pq_coords = nullptr;
          _size = 0;
      }
      BruteForceScratch(size_t max_vector_count, size_t max_pq_chunks)
      {
          _p_intermediate_pq_dists = new float[max_vector_count];
          diskann::alloc_aligned((void **)&_p_intermediate_pq_coords,
                                 max_vector_count * max_pq_chunks * sizeof(uint8_t), 256);

          _size = max_vector_count;
          if (_p_intermediate_pq_dists == nullptr || _p_intermediate_pq_coords == nullptr )
          {
              throw diskann::ANNException("Could not allocate sufficient memory for brute force scratch.", -1);
          }
          this->clear();
      }
      void clear()
      {
          memset(_p_intermediate_pq_dists, 0, _size);
          memset(_p_intermediate_pq_coords, 0, _size);
      }

      float *pq_dists_scratch()
      {
          return _p_intermediate_pq_dists;
      }
      uint8_t *pq_coords_scratch()
      {
          return _p_intermediate_pq_coords;
      }

      ~BruteForceScratch()
      {
          if (_p_intermediate_pq_dists != nullptr)
          {
              delete[] _p_intermediate_pq_dists;
              _p_intermediate_pq_dists = nullptr;
          }
          if (_p_intermediate_pq_coords != nullptr)
          {
              free(_p_intermediate_pq_coords);
              _p_intermediate_pq_coords = nullptr;
          }
          _size = 0;
      }

    private:
      //Space for storing intermediate results.
      float *_p_intermediate_pq_dists;
      uint8_t *_p_intermediate_pq_coords;
      size_t _size;
  };

  template<typename T, typename LabelT=uint32_t>
  class FilterBruteForceIndex {
  public :
    DISKANN_DLLEXPORT FilterBruteForceIndex(const std::string& disk_index_file, 
                            std::shared_ptr<PQFlashIndex<T,LabelT>> pq_flash_index);
    DISKANN_DLLEXPORT bool index_available() const; 
    DISKANN_DLLEXPORT bool brute_forceable_filter(const std::string& filter) const;
    DISKANN_DLLEXPORT int load(uint32_t num_threads);
    DISKANN_DLLEXPORT int search(const T* query, const std::string &filter,  uint32_t k, uint64_t* res_ids, float* res_dists, QueryStats* stats);
    DISKANN_DLLEXPORT ~FilterBruteForceIndex();

  private:

    void setup_thread_data(uint32_t num_threads);

    diskann::inverted_index_t _bf_filter_index;
    bool _is_loaded;
    std::string _disk_index_file;
    std::shared_ptr<PQFlashIndex<T, LabelT>> _pq_flash_index;
    std::string _filter_bf_data_file;
    ConcurrentQueue<BruteForceScratch*> _scratch;
    uint64_t _data_dim;

  };
}