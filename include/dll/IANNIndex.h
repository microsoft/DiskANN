#pragma once
#pragma once

namespace ANNIndex {

  // The Distance calculation type we can support.
  enum DistanceType { DT_L2 = 0, DT_Cosine, DT_InnerProduct, DT_Count };

  enum AlgoType { AT_IVFPQHNSW = 0, AT_KDTREERNG, AT_RandNSG, AT_Count };

  static const char* ExportCreateObjectFloatFunc = "CreateObjectFloat";
  static const char* ExportReleaseObjectFloatFunc = "ReleaseObjectFloat";

  class IANNIndex {
   public:
#ifdef __NSG_WINDOWS__
    __declspec(dllexport)
#endif
    explicit IANNIndex(unsigned __int32 dimension = 0,
                       DistanceType     distanceType = DT_L2)
        : m_dimension(dimension), m_distanceType(distanceType) {
    }

	#ifdef __NSG_WINDOWS__
    __declspec(dllexport)
#endif
    virtual ~IANNIndex() {
    }

	#ifdef __NSG_WINDOWS__
    __declspec(dllexport)
#endif
    // In implementation, the file path can be a file or folder.
    virtual bool BuildIndex(const char* dataFilePath, const char* indexFilePath,
                            const char* indexBuildParameters) = 0;

	#ifdef __NSG_WINDOWS__
    __declspec(dllexport)
#endif
    // Load index form file.
    virtual bool LoadIndex(const char* indexFilePath,
                           const char* queryParameters) = 0;

    // Search several vectors, return their neighbors' distance and ids.
    // Both distances & ids are returned arraies of neighborCount elements,
    // And need to be allocated by invoker, which capicity should be greater
    // than queryCount * neighborCount.
#ifdef __NSG_WINDOWS__
    __declspec(dllexport)
#endif
    virtual void SearchIndex(const char* vector, unsigned __int64 queryCount,
                             unsigned __int64 neighborCount, float* distances,
                             unsigned __int64* ids) const = 0;

   public:
    // Vector dimension.

    unsigned __int32 m_dimension;
    DistanceType m_distanceType;
  };

}  // namespace ANNIndex
