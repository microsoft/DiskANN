#pragma once

#include <string>
#include <vector>

#include <distance.h>
#include <pq_flash_index.h>
#include <webservice\in_memory_nsg_search.h>

namespace diskann {
  
  class DiskIndexSearch {
   public:
    DiskIndexSearch(const char* indexFilePrefix, const char* idsFile,
                  const _u64 cache_nlevels, const _u64 nthreads);

    virtual NSGSearchResult search(const float*       query,
                                   const unsigned int dimensions,
                                   const unsigned int K);

    virtual ~DiskIndexSearch();
	
    static std::vector<std::wstring> load_ids(const char* idsFile);

   private:
    unsigned int _dimensions, _numPoints;

    std::vector<std::wstring>      _ids;
    std::unique_ptr<diskann::PQFlashIndex<float>> _pFlashIndex;
    Distance<float>*                            cosine_distance = nullptr;
  };
}  // namespace diskann
