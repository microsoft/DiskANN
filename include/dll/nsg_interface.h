#pragma once
#include "IANNIndex.h"
#include "efanna2e/distance.h"

namespace NSG {

  template<typename T>
  class IndexNSG;

  
  template <typename T>
  class NSGInterface : public ANNIndex::IANNIndex {
   public:
#ifdef __NSG_WINDOWS__
    __declspec(dllexport) __cdecl
#endif
       NSGInterface(
            unsigned __int32       dimension,
            ANNIndex::DistanceType distanceType);

#ifdef __NSG_WINDOWS__
    __declspec(dllexport)
#endif
        virtual ~NSGInterface();

    // In implementation, the file path can be a file or folder.
#ifdef __NSG_WINDOWS__
    __declspec(dllexport)
#endif
        virtual bool BuildIndex(const char* dataFilePath,
                                const char* indexFilePath,
                                const char* indexBuildParameters);

    // Load index form file.
#ifdef __NSG_WINDOWS__
    __declspec(dllexport)
#endif
        virtual bool LoadIndex(const char* indexFilePath,
                               const char* queryParameters);

    // Search several vectors, return their neighbors' distance and ids.
    // Both distances & ids are returned arraies of neighborCount elements,
    // And need to be allocated by invoker, which capicity should be greater
    // than queryCount * neighborCount.
#ifdef __NSG_WINDOWS__
    __declspec(dllexport)
#endif
        virtual void SearchIndex(const char*      vector,
                                 unsigned __int64 queryCount,
                                 unsigned __int64 neighborCount,
                                 float* distances, unsigned __int64* ids) const;

   public:
    // Vector dimension.
    unsigned __int32       m_dimension;
    ANNIndex::DistanceType m_distanceType;

   private:
    std::string                           _nsgPathPrefix;
    std::unique_ptr<NSG::IndexNSG<T>>	  _pNsgIndex;
    NSG::Metric                           _compareMetric;
  };
}  // namespace NSG
