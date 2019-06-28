#define NOMINMAX

#include <dll/nsg_interface.h>
#include "dll/IANNIndex.h"
#include "utils.h"

void write_Tvecs_unsigned(std::string fname, _u64* input, _u64 npts,
                          _u64 ndims) {
  unsigned* out = new unsigned[npts * ndims];
  for (_u64 i = 0; i < npts * ndims; i++) {
    out[i] = (unsigned) input[i];
  }

  save_Tvecs_plain<unsigned>(fname.c_str(), out, npts, ndims);
  delete[] out;
}

template<typename T>
int aux_main(int argc, char** argv) {
  // argv[1]: data file
  // argv[2]: output_file_pattern
  if (argc != 5) {
    std::cout << "Usage: " << argv[0]
              << " <data_file> <output_file_prefix> <query_Tvecs> <query_res>"
              << std::endl;
  }

  ANNIndex::IANNIndex* intf = new NSG::NSGInterface<T>(0, ANNIndex::DT_L2);

  // for indexing
  //{
  //  // just construct index
  //  intf->BuildIndex(argv[1], argv[2], "50 64 200 32");
  //}

  // for query search
  {
    // load the index
    intf->LoadIndex(argv[2], "4 4 16");

    // load query fvecs
    T*   query = nullptr;
    unsigned nqueries, ndims;
    NSG::aligned_load_Tvecs<T>(argv[3], query, nqueries, ndims);
    ndims = ROUND_UP(ndims, 8);

    // query params/output
    _u64 k = 5, L = 30;
    _u64* query_res = new _u64[k * nqueries];
    float* query_dists = new float[k * nqueries];

    // execute queries
    intf->SearchIndex((const char*) query, nqueries, k, query_dists, query_res);

    // compute recall
    write_Tvecs_unsigned(std::string(argv[4]), query_res, nqueries, k);

    NSG::aligned_free(query);
    delete[] query_res;
    delete[] query_dists;
  }
  return 0;
}

int main(int argc, char** argv) {
  return aux_main<float>(argc, argv);
}