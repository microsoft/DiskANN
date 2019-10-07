#pragma once

#include "windows_customizations.h"

namespace ANNIndex {

  // The Distance calculation type we can support.
  enum DistanceType { DT_L2 = 0, DT_Cosine, DT_InnerProduct, DT_Count };

  enum AlgoType { AT_IVFPQHNSW = 0, AT_KDTREERNG, AT_RandNSG, AT_Count };

  static const char* ExportCreateObjectFloatFunc = "CreateObjectFloat";
  static const char* ExportReleaseObjectFloatFunc = "ReleaseObjectFloat";

  class IANNIndex {
   public:
    DISKANN_DLLEXPORT explicit IANNIndex(unsigned __int32 dimension = 0,
                                    DistanceType     distanceType = DT_L2)
        : m_dimension(dimension), m_distanceType(distanceType) {
    }

    DISKANN_DLLEXPORT virtual ~IANNIndex() {
    }

    // In implementation, the file path can be a file or folder.
    DISKANN_DLLEXPORT virtual bool BuildIndex(const char* dataFilePath,
                                         const char* indexFilePath,
                                         const char* indexBuildParameters) = 0;

    // Load index form file.
    DISKANN_DLLEXPORT virtual bool LoadIndex(const char* indexFilePath,
                                        const char* queryParameters) = 0;

    // Search several vectors, return their neighbors' distance and ids.
    // Both distances & ids are returned arraies of neighborCount elements,
    // And need to be allocated by invoker, which capicity should be greater
    // than queryCount * neighborCount.
    DISKANN_DLLEXPORT virtual void SearchIndex(const char*       vector,
                                          unsigned __int64  queryCount,
                                          unsigned __int64  neighborCount,
                                          float*            distances,
                                          unsigned __int64* ids) const = 0;

   public:
    // Vector dimension.

    unsigned __int32 m_dimension;
    DistanceType     m_distanceType;
  };

}  // namespace ANNIndex
