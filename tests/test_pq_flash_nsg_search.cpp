#include <efanna2e/index.h>
#include <efanna2e/neighbor.h>
#include <efanna2e/pq_flash_index_nsg.h>
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
  size_t fsize = (size_t) ss;
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
  if (argc != 13) {
    std::cout << argv[0]
              << " data_bin[1] pq_tables_bin[2] n_chunks[3] chunk_size[4] "
                 "data_dim[5] nsg_disk_opt[6] query_file_fvecs[7] search_L[8] "
                 "search_K[9] result_path[10] BeamWidth[11] cache_nlevels[12]"
              << std::endl;
    exit(-1);
  }
  _u64 n_chunks = (_u64) std::atoi(argv[3]);
  _u64 chunk_size = (_u64) std::atoi(argv[4]);
  _u64 data_dim = (_u64) std::atoi(argv[5]);

  // construct FlashNSG
  NSG::DistanceL2 dist_cmp;
  NSG::PQFlashNSG index(&dist_cmp);
  std::cout << "main --- tid: " << std::this_thread::get_id() << std::endl;
  // index.reader.register_thread();
  std::cout << "Loading index from " << argv[1] << std::endl;
  index.load(argv[1], argv[6], argv[2], chunk_size,
                                  n_chunks, data_dim);

  // load queries
  float*   query_load = NULL;
  unsigned query_num, query_dim;
  std::cout << "Loading Queries from " << argv[7] << std::endl;
  NSG::aligned_load_Tvecs<float>(argv[7], query_load, query_num, query_dim);
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
  std::atomic<unsigned>              qcounter;
  qcounter.store(0);

  NSG::QueryStats* stats = new NSG::QueryStats[query_num];
  _u64             n_threads = 8;  //omp_get_max_threads();
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
  std::vector<bool> has_inits(n_threads, false);
#pragma omp parallel for schedule(dynamic, 128) num_threads(8)
  for (_s64 i = 0; i < query_num; i++) {
    unsigned val = qcounter.fetch_add(1);
    if (val % 1000 == 0) {
      std::cout << "Status: " << val << " queries done" << std::endl;
    }
    std::vector<unsigned>& query_res = res[i];
    _u64                   thread_no = omp_get_thread_num();
    NSG::QueryScratch*     local_scratch = &(thread_scratch[thread_no]);

    // zero context
    local_scratch->coord_idx = 0;
    local_scratch->sector_idx = 0;

	// init if not init yet
	if (!has_inits[thread_no]) {
      index.reader.register_thread();
#pragma omp critical
      std::cout << "Init complete for thread-" << omp_get_thread_num()
                << std::endl;
      has_inits[thread_no] = true;
    }

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

  save_result(argv[10], res);
  delete[] stats;
  std::cerr << "Clearing scratch" << std::endl;
  for (auto& scratch : thread_scratch) {
    delete[] scratch.coord_scratch;
    NSG::aligned_free(scratch.sector_scratch);
    NSG::aligned_free(scratch.aligned_scratch);
  }
  NSG::aligned_free(query_load);

  return 0;
}
