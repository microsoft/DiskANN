#pragma once
#include "IANNIndex.h"
#include "pq_flash_index.h"
#include "windows_customizations.h"

namespace diskann {
  const char* INDEX_PATH_PREFIX_PLACEHOLDER = "{0}";
  const size_t PATH_PREFIX_PLACEHOLDER_LEN = strlen(INDEX_PATH_PREFIX_PLACEHOLDER);

  template<typename T, typename TagT>
  class Index;

  template<typename T>
  class DiskANNInterface : public ANNIndex::IANNIndex {
   public:
    DISKANN_DLLEXPORT DiskANNInterface(
        unsigned __int32 dimension, ANNIndex::DistanceType distanceType,
        std::shared_ptr<ANNIndex::IDiskPriorityIO> diskIO = nullptr);

    DISKANN_DLLEXPORT virtual ~DiskANNInterface();

    // In implementation, the file path can be a file or folder.
    DISKANN_DLLEXPORT virtual bool BuildIndex(const char* dataFilePath,
                                              const char* indexFilePath,
                                              const char* indexBuildParameters);

    // Load index form file.
    DISKANN_DLLEXPORT virtual bool LoadIndex(const char* indexFilePath,
                                             const char* queryParameters);

    // Load index from memory blob.
    DISKANN_DLLEXPORT virtual bool LoadIndex(
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
    unsigned __int32       m_aligned_dimension;
    ANNIndex::DistanceType m_distanceType;

   private:
    // Methods
    bool writeSharedStoreIniFile(const char* indexPathPrefix);

   private:
    std::string                                _nsgPathPrefix;
    std::shared_ptr<AlignedFileReader>         _pReader;
    std::shared_ptr<ANNIndex::IDiskPriorityIO> _pDiskIO;
    std::unique_ptr<diskann::Index<T, int>> _pNsgIndex;
    diskann::Metric _compareMetric;

    // flash stuff
    std::unique_ptr<PQFlashIndex<T>> _pFlashIndex;
    _u64                             n_chunks;
    _u64                             chunk_size;
    _u64                             beam_width;
    _u64                             Lsearch;
  };
}  // namespace diskann
