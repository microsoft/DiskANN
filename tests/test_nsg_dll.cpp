#define NOMINMAX

#include <dll/nsg_interface.h>
#include "dll/IANNIndex.h"
#include "utils.h"

float calc_recall(_u64 num_queries, unsigned* gold_std, _u64 dim_gs,
                  _u64* our_results, _u64 dim_or, _u64 recall_at) {
  bool*    this_point = new bool[recall_at];
  unsigned total_recall = 0;

  for (size_t i = 0; i < num_queries; i++) {
    for (unsigned j = 0; j < recall_at; j++)
      this_point[j] = false;
    for (size_t j1 = 0; j1 < recall_at; j1++)
      for (size_t j2 = 0; j2 < dim_or; j2++)
        if (gold_std[i * (size_t) dim_gs + j1] ==
            our_results[i * (size_t) dim_or + j2]) {
          if (this_point[j1] == false)
            total_recall++;
          this_point[j1] = true;
        }
  }
  return ((float) total_recall) / ((float) num_queries) *
         (100.0 / ((float) recall_at));
}

void write_Tvecs_unsigned(std::string fname, _u64* input, _u64 npts,
                          _u64 ndims) {
  unsigned* out = new unsigned[npts * ndims];
  for (_u64 i = 0; i < npts * ndims; i++) {
    out[i] = (unsigned) input[i];
  }

  NSG::save_Tvecs<unsigned>(fname.c_str(), out, npts, ndims);
  delete[] out;
}

template<typename T>
int aux_main(int argc, char** argv) {
  // argv[1]: data file
  // argv[2]: output_file_pattern
  if (argc != 5) {
    std::cout << "Usage: " << argv[0]
              << " <data_file> <output_file_prefix> <query_bin> <gt_bin>"
              << std::endl;
    return -1;
  }

  ANNIndex::IANNIndex* intf = new NSG::NSGInterface<T>(0, ANNIndex::DT_L2);

  bool res = 0;
  // for indexing
  {
    // just construct index
//    res = intf->BuildIndex(argv[1], argv[2], "50 64 200 50 0.01");
    // ERROR CHECK
    if (res == 1) {
      exit(-1);
    }
  }

  // for query search
  {
    // load the index
    bool res = intf->LoadIndex(argv[2], "12 4 4 4");
    // ERROR CHECK
    if (res == 1) {
      exit(-1);
    }

    // load query bin
    T*   query = nullptr;
    _u64 nqueries, ndims, aligned_query_dim;
    NSG::load_aligned_bin<T>(argv[3], query, nqueries, ndims,
                             aligned_query_dim);

    std::cout << "Loading ground truth..." << std::flush;
    // load ground truth
    _u32* ground_truth = nullptr;
    _u64  ngt, kgt;
    NSG::load_bin<_u32>(argv[4], ground_truth, ngt, kgt);

    if (ngt != nqueries) {
      std::cout << "mismatch in ground truth rows and number of queries"
                << std::endl;
      return -1;
    }

    // query params/output
    _u64   k = 5;
    _u64   L = 12;
    _u64*  query_res = new _u64[k * nqueries];
    float* query_dists = new float[k * nqueries];

    if (kgt < k) {
      std::cout << "number of ground truth < k" << std::endl;
      return -1;
    }
    std::cout << "done." << std::endl;
    // execute queries
    intf->SearchIndex((const char*) query, nqueries, k, query_dists, query_res);
    float avg_recall =
        calc_recall(nqueries, ground_truth, kgt, query_res, k, k);
    std::cout << "Recall@" << k << " when searching with L = " << L << " is "
              << avg_recall << std::endl;
    //  save results into ivecs
    // write_Tvecs_unsigned(argv[4], query_res, nqueries, k);

    NSG::aligned_free(query);
    delete[] ground_truth;
    delete[] query_res;
    delete[] query_dists;
  }
  return 0;
}

int main(int argc, char** argv) {
  return aux_main<float>(argc, argv);
}
