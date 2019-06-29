#include <index.h>
#include <ivfpq_flash_index_nsg.h>
#include <neighbor.h>
#include <omp.h>
#include <timer.h>
#include <util.h>
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
  if (argc != 15) {
    std::cout << argv[0]
              << " ivf_pivots[1] data_ivf_ivecs[2] pq_pivots[3] "
                 "data_pq_ivecs[4] pq_chunk_size[5] n_pts[6] data_dim[7] "
                 "nsg_disk_opt[8] query_file_fvecs[9] search_L[10] "
                 "search_K[11] result_path[12] BeamWidth[13] cache_nlevels[14]"
              << std::endl;
    exit(-1);
  }

  _u64 chunk_size = (_u64) std::atoi(argv[5]);
  _u64 n_pts = (_u64) std::atoi(argv[6]);
  _u64 data_dim = (_u64) std::atoi(argv[7]);
  // construct IVFPQ Table
  NSG::IVFPQTable ivfpq_table(argv[1], argv[2], argv[3], argv[4], chunk_size);

  // construct FlashNSG
  NSG::DistanceL2    dist_cmp;
  NSG::IVFPQFlashNSG index(&dist_cmp, &ivfpq_table);
  std::cout << "main --- tid: " << std::this_thread::get_id() << std::endl;
  index.reader->register_thread();
  index.load(argv[8], n_pts, data_dim);

  // load queries
  float*   query_load = NULL;
  unsigned query_num, query_dim;
  NSG::aligned_load_Tvecs<float>(argv[9], query_load, query_num, query_dim);
  std::cout << "query_dim = " << query_dim << std::endl;
  _u64 aligned_dim = ROUND_UP(query_dim, 8);
  assert(aligned_dim == index.aligned_dim);

  _u64 l_search = (_u64) atoi(argv[10]);
  _u64 k_search = (_u64) atoi(argv[11]);
  int  beam_width = atoi(argv[13]);
  _u64 cache_nlevels = (_u64) atoi(argv[14]);

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
  _u64             n_threads = omp_get_max_threads();
  std::cout << "Executing queries on " << n_threads << " threads\n";
  std::vector<NSG::QueryScratch> thread_scratch(n_threads);
  for (auto& scratch : thread_scratch) {
    scratch.coord_scratch = new _s8[MAX_N_CMPS * data_dim];
    NSG::alloc_aligned((void**) &scratch.sector_scratch,
                       MAX_N_SECTOR_READS * SECTOR_LEN, SECTOR_LEN);
    NSG::alloc_aligned((void**) &scratch.aligned_scratch, 256 * sizeof(float),
                       256);
    memset(scratch.aligned_scratch, 0, 256 * sizeof(float));
  }

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
        index.reader->register_thread();
        std::cout << "Init complete for thread-" << omp_get_thread_num()
                  << std::endl;
        has_init = true;
      }
    }
    std::vector<unsigned>& query_res = res[i];
    _u64                   thread_no = omp_get_thread_num();
    NSG::QueryScratch*     local_scratch = &(thread_scratch[thread_no]);

    // zero context
    local_scratch->coord_idx = 0;
    local_scratch->sector_idx = 0;

    auto ret = index.cached_beam_search(query_load + i * aligned_dim, k_search,
                                        l_search, query_res.data(), beam_width,
                                        stats + i, local_scratch);
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
      stats, query_num, "# cmps / query", "",
      [](const NSG::QueryStats& stats) { return stats.n_cmps; });
  NSG::percentile_stats(
      stats, query_num, "# cache hits / query", "",
      [](const NSG::QueryStats& stats) { return stats.n_cache_hits; });

  save_result(argv[12], res);
  delete[] stats;
  std::cerr << "Clearing scratch" << std::endl;
  for (auto& scratch : thread_scratch) {
    delete[] scratch.coord_scratch;
    free(scratch.sector_scratch);
    free(scratch.aligned_scratch);
  }
  free(query_load);

  return 0;
}
