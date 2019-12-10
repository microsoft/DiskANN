#define NOMINMAX

#include <dll/vamana_interface.h>
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

  diskann::save_Tvecs<unsigned>(fname.c_str(), out, npts, ndims);
  delete[] out;
}

template<typename T>
int aux_main(int argc, char** argv) {
  ANNIndex::IANNIndex* intf =
      new diskann::VamanaInterface<T>(0, ANNIndex::DT_L2);

  bool res = 0;
  // for indexing
  {
    // just construct index: Lconstruction, Degree of Graph, Cconstruction,
    // alpha (higher is denser graphs), T (Max threads to use)
    res = intf->BuildIndex(argv[2], argv[3], "100 100 2000 1.5 32");
    // ERROR CHECK
    if (res != 1) {
      exit(-1);
    }
  }

  // for query search
  {
    // load the index (arguments are L_search, beam_width, data_file)
    std::string load_args = "110 1 " + std::string(argv[2]);
    bool        res = intf->LoadIndex(argv[3], load_args.c_str());
    // ERROR CHECK
    if (res != 1) {
      exit(-1);
    }

    // load query bin
    T*   query = nullptr;
    _u64 nqueries, ndims, aligned_query_dim;
    diskann::load_aligned_bin<T>(argv[4], query, nqueries, ndims,
                                 aligned_query_dim);

    // load ground truth
    _u32* ground_truth = nullptr;
    _u64  ngt = 0, kgt = 0;
    bool  has_ground_truth = false;
    if (std::string(argv[5]) != std::string("null")) {
      std::cout << "Loading ground truth..." << std::flush;
      diskann::load_bin<_u32>(argv[5], ground_truth, ngt, kgt);
      if (ngt != nqueries) {
        std::cout << "mismatch in ground truth rows and number of queries"
                  << std::endl;
        return -1;
      }
      has_ground_truth = true;
    }

    // query params/output
    _u64   k = 100;
    _u64   L = 110;
    _u64*  query_res = new _u64[k * nqueries];
    float* query_dists = new float[k * nqueries];

    if (kgt < k && has_ground_truth) {
      std::cout << "number of ground truth < k" << std::endl;
      return -1;
    }
    // execute queries
    intf->SearchIndex((const char*) query, nqueries, k, query_dists, query_res);
    float avg_recall = 0;
    if (has_ground_truth) {
      avg_recall = calc_recall(nqueries, ground_truth, kgt, query_res, k, k);
      std::cout << "Recall@" << k << " when searching with L = " << L << " is "
                << avg_recall << std::endl;
    }
    //  save results into ivecs
    // write_Tvecs_unsigned(argv[4], query_res, nqueries, k);
    std::cout << "Done searching." << std::endl;
    diskann::aligned_free(query);
    delete[] ground_truth;
    delete[] query_res;
    delete[] query_dists;
  }
  return 0;
}

int main(int argc, char** argv) {
  // argv[1]: datatype (int8/ uint8/ float)
  // argv[2]: data file (.bin)
  // argv[3]: output_file_prefix (will generate index with _mem.index suffix)
  // argv[4]: query_bin file
  // argv[5]: ground_truth ids file (32-byte integer entries, file size should
  // be 8 + 4*k*n_q, where gt has k entries for n_q queries)
  if (argc != 6) {
    std::cout
        << "Usage: " << argv[0]
        << " <datatype> (int8/uint8/float) <data_file> (.bin) "
           "<output_file_prefix> (index will be stored in prefix_mem.index) "
           "<query_bin> (.bin)  <gt_bin> (use \"null\" if not available) "
        << std::endl;
    return -1;
  }

  if (std::string(argv[1]) == "int8")
    return aux_main<int8_t>(argc, argv);
  else if (std::string(argv[1]) == "float")
    return aux_main<float>(argc, argv);
  else if (std::string(argv[1]) == "uint8")
    return aux_main<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported type. Use int8/uint8/float." << std::endl;
}
