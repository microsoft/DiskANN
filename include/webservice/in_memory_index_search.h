#pragma once

#include <string>
#include <vector>

#include <distance.h>
#include <index.h>
#include <cosine_similarity.h>
namespace diskann {
  class IndexSearchResult {
   public:
    IndexSearchResult(unsigned int k, unsigned int elapsedTimeInMs)
        : K(k), searchTimeInMs(elapsedTimeInMs) {
      finalResults.reserve(k);
    }

    void addResult(const std::wstring& result) {
      finalResults.push_back(result);
    }

    unsigned int                       K;
    unsigned int                       searchTimeInMs;
    std::vector<unsigned int> finalResultIndices;  // TEMPORARY FOR RECALL.
    std::vector<std::wstring> finalResults;
    std::vector<float>        distances;
  };

  class InMemoryIndexSearch {
   public:
    InMemoryIndexSearch(const char* baseFile, const char* indexFile,
                      const char* idsFile, Metric m);

    virtual IndexSearchResult search(const float*       query,
                                   const unsigned int dimensions,
                                   const unsigned int K);

    virtual ~InMemoryIndexSearch();

    static void load_data(const char* filename, float*& data, unsigned& num,
                          unsigned& dim);

    static std::vector<std::wstring> load_ids(const char* idsFile);

   private:
    float*       _baseVectors;
    unsigned int _dimensions, _numPoints;

    std::vector<std::wstring>      _ids;
    std::unique_ptr<diskann::Index<float>> _nsgIndex;
  };
}  // namespace diskann
