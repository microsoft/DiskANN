#pragma once

#include <string>
#include <vector>

#include <efanna2e/distance.h>
#include <efanna2e/index_nsg.h>

namespace NSG {
  class NSGSearchResult {
   public:
    NSGSearchResult(int k, int elapsedTimeInMs)
        : K(k), searchTimeInMs(elapsedTimeInMs) {
      finalResults.reserve(k);
    }

    NSGSearchResult(const std::vector<std::wstring>& results, int k,
                    int elapsedTimeInMs)
        : finalResults(results), K(k), searchTimeInMs(elapsedTimeInMs) {
    }

    void addResult(const std::wstring& result) {
      finalResults.push_back(result);
    }

    int                      K;
    int                      searchTimeInMs;
    std::vector<std::wstring> finalResults;
  };

  class InMemoryNSGSearch {
   public:
    InMemoryNSGSearch(const char* baseFile, const char* indexFile,
                      const char* idsFile, Metric m);

    virtual NSGSearchResult search(const float* query, const unsigned int K);

    virtual ~InMemoryNSGSearch();

   private:
    float*                         _baseVectors;
    std::vector<std::wstring>      _ids;
    std::unique_ptr<NSG::IndexNSG> _nsgIndex;
  };
}  // namespace NSG
