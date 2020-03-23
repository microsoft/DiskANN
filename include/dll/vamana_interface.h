#pragma once
#include "IANNIndex.h"
#include "index.h"
#include "windows_customizations.h"

namespace diskann {

  template<typename T, typename TagT>
  class Index;

  template<typename T>
  class VamanaInterface : public ANNIndex::IANNIndex {
   public:
    DISKANN_DLLEXPORT VamanaInterface(unsigned __int32       dimension,
                                      ANNIndex::DistanceType distanceType);

    DISKANN_DLLEXPORT virtual ~VamanaInterface();

    // In implementation, the file path can be a file or folder.
    DISKANN_DLLEXPORT virtual bool BuildIndex(const char* dataFilePath,
                                              const char* indexFilePath,
                                              const char* indexBuildParameters);

    // Load index form file.
    DISKANN_DLLEXPORT virtual bool LoadIndex(const char* indexFilePath,
                                             const char* queryParameters);

    // Load index from memory blob
    DISKANN_DLLEXPORT bool LoadIndex(
        const std::vector<ANNIndex::FileBlob>& files,
        const char*                            queryParameters);

    // Search several vectors, return their neighbors' distance and ids.
    // Both distances & ids are returned arraies of neighborCount elements,
    // And need to be allocated by invoker, which capicity should be greater
    // than queryCount * neighborCount.
    DISKANN_DLLEXPORT virtual void SearchIndex(const char*       vector,
                                               unsigned __int64  queryCount,
                                               unsigned __int64  neighborCount,
                                               float*            distances,
                                               unsigned __int64* ids) const;

   public:
    // Vector dimension.
    unsigned __int32       m_dimension;
    unsigned __int32       aligned_dimension;
    ANNIndex::DistanceType m_distanceType;

   private:
    std::string                        _nsgPathPrefix;
    std::unique_ptr<diskann::Index<T>> _pIndex;
    diskann::Metric                    _compareMetric;

    // flash stuff
    _u32 beam_width;
    _u32 Lsearch;
  };
}  // namespace diskann
