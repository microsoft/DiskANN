// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// Testing insert
#include "index.h"
#include "UTILS.H"
#include "diskann_incr_index.h"

const std::string IndexParams =
    "IndexParams=64 75 1 20 20";  // R, L, SearchMem, IndexMem, ThreadCount
const std::string SearchParams = "SearchParams=14 10 6";  // L, Threads, BW

template<typename T>
using IncrIndex = diskann::DiskANNIncrementalIndex<T>;

template<typename T>
class InMemoryIndexParallelTest {
 public:
  InMemoryIndexParallelTest(_u32 maxPoints) : _maxPoints(maxPoints) {
    std::random_device device;
    std::mt19937       generator(device());
    _pDistribution =
        std::make_shared<std::uniform_int_distribution<_u32>>(0, maxPoints);
  }

 private:
  _u32                                                _maxPoints;
  tsl::robin_set<_u64>                                _insertedIds;
  std::shared_ptr<std::uniform_int_distribution<int>> _pDistribution;

  void insertThread(T* data, _u32 ndims, IncrIndex<T>& incrIntf) {
    for (_u32 i = 0; i < 2 * maxPoints; i++)
  }
};

template<typename T>
void insertData(T* data, _u32 startIndex, _u32 endIndex, _u32 ndims,
                IncrIndex<T>& incrIntf) {
  for (_u32 i = startIndex; i < endIndex; i++) {
    incrIntf.AddData((_u64) startIndex * ndims, data + startIndex * ndims);
  }
}

template<typename T>
std::shared_ptr<IncrIndex<T>> initialCreateInMemoryIndex(
    T* data, _u32 npts, _u32 ndims, _u32 pointsPerIndex,
    const std::string& workingDir) {
  std::string parameters = IndexParams + ";" + SearchParams +
                           ";Placement=INMEMORY;WorkingDir=" + workingDir;
  std::shared_ptr<IncrIndex<T>> incrIndx = std::make_shared<IncrIndex<T>>(
      dims, ANNIndex::DT_L2, maxMemIndexPoints, parameters);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
  for (int i = 0; i <)
}

template<typename T>
void runIncrementalIntfTest(const std::string& basefile,
                            const std::string& workingDir,
                            uint32_t           maxMemIndexPoints) {
  T*     data;
  size_t npts, ndim;
  diskann::load_bin<basefile, data, npts, ndim, 0);

  if (npts / maxMemIndexPoints < 7) {
    std::cerr
        << "Very few points to test the functionality thoroughly. "
        << "Either reduce the max points per index or select a bigger base file"
        << std::endl;
    exit(1);
  }
}

int main(int argc, char** argv) {
  // Parameters: <program> <datatype> <base_file>
  // The aim of this code is to simply exercise the incremental interface. It
  // does not
  // attempt to do perf tests, e.t.c beyond a point. The goal is to mimic the
  // store
  if (argc != 4) {
    std::cout << "Usage: <program> <datatype [float|int8|uint8]> <base_file> "
                 "<working_dir> <max_num_of_mem_index_points>"
              << std::endl;
  }
  int         argi = 1;
  std::string datatype = argv[argi++];
  std::string basefile = argv[argi++];
  std::string workingDir = argv[argi++];
  uint32_t    maxMemIndexPoints = (_u32) std::atoi(argv[argi++]);

  if (datatype == "float") {
    runIncrementalIntfTest<float>(basefile, workingDir, maxMemIndexPoints);
  } else if (datatype == "uint8") {
    runIncrementalIntfTest<uint8_t>(basefile, workingDir, maxMemIndexPoints);
  } else if (datatype == "int8") {
    runIncrementalIntfTest<int8_t>(basefile, workingDir, maxMemIndexPoints);
  } else {
    std::cerr << std::string("Unknown data type: ") << datatype
              << ".Only float,uint8,and int8 are supported." << std::endl;
  }
  return 0;
}
