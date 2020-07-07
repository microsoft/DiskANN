#pragma once

#include <string>
#include <vector>
#include <memory>

#include "IDiskPriorityIO.h"

namespace ANNIndex {

  // The Distance calculation type we can support.
  enum DistanceType { DT_L2 = 0, DT_Cosine, DT_InnerProduct, DT_Count };

  enum AlgoType {
    // Memory
    AT_HNSW = 0,
    AT_KDTree,
    // SSD
    AT_IVFPQHNSW,
    AT_KDTreeRNG,
    AT_RandNSG,
    // Fresh
    AT_HNSWFresh,
    AT_SPTAG,
    AT_Count
  };

  enum VectorDataType { VDT_Byte = 0, VDT_Short, VDT_Float, VDT_Count };

  enum SSDAlgoType { SAT_IVFPQHNSW, SAT_KDTreeRNG, SAT_RandNSG, SAT_Count };

  static const char* SSDAlgoDLLNames[SAT_Count] = {
      "ANNIndexIVFPQHNSW.dll", "KDTreeRNGSSDDLL.dll", "nsg_dll.dll"};
  static const char* SSDAlgoNames[SAT_Count] = {"ivfpqhnsw", "kdtreerng",
                                                "randnsg"};

  static const char* ExportCreateObjectByteFunc = "CreateObjectByte";
  static const char* ExportReleaseObjectByteFunc = "ReleaseObjectByte";

  static const char* ExportCreateObjectShortFunc = "CreateObjectShort";
  static const char* ExportReleaseObjectShortFunc = "ReleaseObjectShort";

  static const char* ExportCreateObjectFloatFunc = "CreateObjectFloat";
  static const char* ExportReleaseObjectFloatFunc = "ReleaseObjectFloat";

  static const char* ExportCreateObjectFuncs[VDT_Count] = {
      ExportCreateObjectByteFunc, ExportCreateObjectShortFunc,
      ExportCreateObjectFloatFunc};
  static const char* ExportReleaseObjectFuncs[VDT_Count] = {
      ExportReleaseObjectByteFunc, ExportReleaseObjectShortFunc,
      ExportReleaseObjectFloatFunc};

  // NOTICE : if data is nullptr, it's means a real path has not been mampped to
  // memory blob.
  struct FileBlob {
    std::string path;
    const void* data;
    size_t      size;

    FileBlob(const std::string& filePath, const void* fileData, size_t fileSize)
        : path(filePath), data(fileData), size(fileSize) {
    }
  };

  class IANNIndex {
   public:
    explicit IANNIndex(unsigned __int32                 dimension,
                       DistanceType                     distanceType = DT_L2,
                       std::shared_ptr<IDiskPriorityIO> diskIO = nullptr)
        : m_dimension(dimension), m_distanceType(distanceType) {
    }

    virtual ~IANNIndex() {
    }

    // In implementation, the file path can be a file or folder.
    virtual bool BuildIndex(const char* dataFilePath, const char* indexFilePath,
                            const char* indexBuildParameters) = 0;

    // Load index form file.
    virtual bool LoadIndex(const char* indexFilePath,
                           const char* queryParameters) = 0;

    // Load index from memory blob.
    virtual bool LoadIndex(const std::vector<FileBlob>& files,
                           const char*                  queryParameters) = 0;

    // Search several vectors, return their neighbors' distance and ids.
    // Both distances & ids are returned arraies of neighborCount elements,
    // And need to be allocated by invoker, which capicity should be greater
    // than queryCount * neighborCount.

    // TODO :: this interface has defect as a rare case is return array size may
    // less than neighbor count.
    virtual void SearchIndex(const char*      queryVectors,
                             unsigned __int64 queryCount,
                             unsigned __int64 neighborCount, float* distances,
                             unsigned __int64* ids) const = 0;

   public:
    // Vector dimension.
    unsigned __int32 m_dimension;

    DistanceType m_distanceType;
  };

  typedef IANNIndex* (*CreateObjectFunc)(unsigned __int32, DistanceType,
                                         std::shared_ptr<IDiskPriorityIO>);
  typedef void (*ReleaseObjectFunc)(IANNIndex*);
}
