#include <efanna2e/index.h>
#include <efanna2e/neighbor.h>
#include <efanna2e/pq_flash_index_nsg.h>
#include <efanna2e/timer.h>
#include <efanna2e/util.h>
#include <omp.h>
#include <atomic>
#include <cassert>
#include "utils.h"

void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error:" << filename << std::endl;
    exit(-1);
  }
  in.read((char*) &dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t             fsize = (size_t) ss;
  num = (unsigned) (fsize / (dim + 1) / 4);
  data = (float*) malloc((size_t) num * (size_t) dim * sizeof(float));

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*) (data + i * dim), dim * 4);
  }
  in.close();
}

void save_result(char* filename, std::vector<std::vector<unsigned>>& results) {
  std::cout << "Saving result to " << filename << std::endl;
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned) results[i].size();
    out.write((char*) &GK, sizeof(unsigned));
    out.write((char*) results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

template<typename T>
void aux_main(int argc, char** argv) {
  if (argc != 14) {
    std::cout << argv[0]
              << " data_bin[1] pq_tables_bin[2] n_chunks[3] chunk_size[4] "
                 "data_dim[5] nsg_disk_opt[6] query_file_fvecs[7] search_L[8] "
                 "search_K[9] result_path[10] BeamWidth[11] cache_nlevels[12] "
                 "nthreads[13]"
              << std::endl;
    exit(-1);
  }

  _u64 n_chunks = (_u64) std::atoi(argv[3]);
  _u64 chunk_size = (_u64) std::atoi(argv[4]);
  _u64 data_dim = (_u64) std::atoi(argv[5]);
  // _u64             n_threads = omp_get_max_threads();
  _u64 n_threads = (_u64) atoi(argv[13]);

  // construct FlashNSG
  NSG::PQFlashNSG<T> index;
  std::cout << "main --- tid: " << std::this_thread::get_id() << std::endl;
  std::cout << "Loading index from " << argv[1] << std::endl;
  index.load(argv[1], argv[6], argv[2], chunk_size, n_chunks, data_dim,
             n_threads);
  index.reader->register_thread();

  // load queries
  T*     query_load = NULL;
  size_t query_num, query_dim;
  std::cout << "Loading Queries from " << argv[7] << std::endl;
  load_Tvecs_plain<float, T>(argv[7], query_load, query_num, query_dim);
  std::cout << "query_dim = " << query_dim << std::endl;
  _u64 aligned_dim = ROUND_UP(query_dim, 8);
  assert(aligned_dim == index.aligned_dim);

  _u64 l_search = (_u64) atoi(argv[8]);
  _u64 k_search = (_u64) atoi(argv[9]);
  int  beam_width = atoi(argv[11]);
  _u64 cache_nlevels = (_u64) atoi(argv[12]);

  if (l_search < k_search) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  index.cache_bfs_levels(cache_nlevels);

  // align query data
  // query_load = NSG::data_align(query_load, query_num, query_dim);

  std::vector<std::vector<unsigned>> res(query_num,
                                         std::vector<unsigned>(k_search));
  std::atomic<unsigned> qcounter;
  qcounter.store(0);

  NSG::QueryStats* stats = new NSG::QueryStats[query_num];

  NSG::Timer timer;
#pragma omp  parallel for schedule(dynamic, 1) num_threads(n_threads)
  for (_s64 i = 0; i < query_num; i++) {
    unsigned val = qcounter.fetch_add(1);
    if (val % 1000 == 0) {
      std::cout << "Status: " << val << " queries done" << std::endl;
    }
    std::vector<unsigned>& query_res = res[i];

    index.cached_beam_search(query_load + i * aligned_dim, k_search, l_search,
                             query_res.data(), beam_width, stats + i);
  }

  _u64   total_query_us = timer.elapsed();
  double qps = (double) query_num / ((double) total_query_us / 1e6);
  std::cout << "QPS: " << qps << std::endl;

  NSG::percentile_stats(
      stats, query_num, "Total us / query", "us",
      [](const NSG::QueryStats& stats) { return stats.total_us; });
  NSG::percentile_stats(
      stats, query_num, "Total I/O us / query", "us",
      [](const NSG::QueryStats& stats) { return stats.io_us; });
  NSG::percentile_stats(
      stats, query_num, "Total # I/O requests / query", "",
      [](const NSG::QueryStats& stats) { return stats.n_ios; });
  NSG::percentile_stats(
      stats, query_num, "Total # 4kB requests / query", "",
      [](const NSG::QueryStats& stats) { return stats.n_4k; });
  NSG::percentile_stats(
      stats, query_num, "# cmps / query", "",
      [](const NSG::QueryStats& stats) { return stats.n_cmps; });
  NSG::percentile_stats(
      stats, query_num, "# cache hits / query", "",
      [](const NSG::QueryStats& stats) { return stats.n_cache_hits; });

  save_result(argv[10], res);
  delete[] stats;
  NSG::aligned_free(query_load);
}

int main(int argc, char** argv) {
  aux_main<float>(argc, argv);
  return 0;
}
