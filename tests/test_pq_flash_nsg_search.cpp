//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/flash_index_nsg.h>
#include <efanna2e/index.h>
#include <efanna2e/neighbor.h>
#include <efanna2e/timer.h>
#include <efanna2e/util.h>
#include <omp.h>
#include <atomic>
#include <cassert>

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

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cout << argv[0] << " data_bin pq_tables n_chunks chunk_size data_dim "
                            "nsg_disk_opt query_file_fvecs search_L search_K "
                            "result_path BeamWidth cache_nlevels"
              << std::endl;
    exit(-1);
  }

  // construct FlashNSG
  NSG::DistanceL2 dist_cmp;
  NSG::FlashNSG   index(&dist_cmp);
  std::cout << "main --- tid: " << std::this_thread::get_id() << std::endl;
  index.reader.register_thread();
  index.load(argv[1], argv[2]);

  // load queries
  float*   query_load = NULL;
  unsigned query_num, query_dim;
  NSG::aligned_load_Tvecs<float>(argv[3], query_load, query_num, query_dim);
  std::cout << "query_dim = " << query_dim << std::endl;
  _u64 aligned_dim = ROUND_UP(query_dim, 8);
  assert(aligned_dim == index.aligned_dim);

  _u64 l_search = (_u64) atoi(argv[4]);
  _u64 k_search = (_u64) atoi(argv[5]);
  int  beam_width = atoi(argv[7]);
  _u64 cache_nlevels = (_u64) atoi(argv[8]);

  if (l_search < k_search) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }
  index.cache_bfs_levels(cache_nlevels);

  // align query data
  // query_load = NSG::data_align(query_load, query_num, query_dim);

  std::vector<std::vector<unsigned>> res(query_num,
                                         std::vector<unsigned>(k_search));
  bool                  has_init = false;
  std::atomic<unsigned> qcounter;
  qcounter.store(0);

  NSG::QueryStats* stats = new NSG::QueryStats[query_num];

  NSG::Timer timer;
#pragma omp  parallel for schedule(dynamic, 128) firstprivate(has_init)
  for (_u64 i = 0; i < query_num; i++) {
    unsigned val = qcounter.fetch_add(1);
    if (val % 1000 == 0) {
      std::cout << "Status: " << val << " queries done" << std::endl;
    }
    if (!has_init) {
#pragma omp critical
      {
        index.reader.register_thread();
        std::cout << "Init complete for thread-" << omp_get_thread_num()
                  << std::endl;
        has_init = true;
      }
    }
    std::vector<unsigned>& query_res = res[i];

    auto ret = index.cached_beam_search(query_load + i * aligned_dim, k_search,
                                        l_search, query_res.data(), beam_width,
                                        stats + i);
    // auto ret = index.Search(query_load + i * dim, data_load, K, paras,
    // tmp.data());
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
      stats, query_num, "Read Size / query", "",
      [](const NSG::QueryStats& stats) { return stats.read_size; });
  NSG::percentile_stats(
      stats, query_num, "# cmps / query", "",
      [](const NSG::QueryStats& stats) { return stats.n_cmps; });
  NSG::percentile_stats(
      stats, query_num, "# cache hits / query", "",
      [](const NSG::QueryStats& stats) { return stats.n_cache_hits; });

  save_result(argv[6], res);
  delete[] stats;
  free(query_load);

  return 0;
}
