//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index.h>
#include <efanna2e/neighbor.h>
#include <efanna2e/pq_flash_index_nsg.h>
#include <efanna2e/timer.h>
#include <efanna2e/util.h>
#include <omp.h>
#include <atomic>
#include <cassert>

void conv_s8_to_float(_s8* in, float* out, _u64 ndims) {
  for (_u64 j = 0; j < ndims; j++) {
    out[j] = (float) in[j];
  }
}

void rerank_query(float* query, _s8* data, NSG::FixedChunkPQTable* pq_table,
                  NSG::Distance* dist_cmp, std::vector<unsigned>& results,
                  float* scratch, _u64 data_dim, _u64 aligned_dim) {
  std::vector<std::pair<unsigned, float>> id_dists;
  memset(scratch, 0, aligned_dim);
  for (const unsigned& id : results) {
    _s8* id_vec = data + (data_dim * id);
    conv_s8_to_float(id_vec, scratch, data_dim);
    float dist = dist_cmp->compare(scratch, query, aligned_dim);
    id_dists.push_back(std::make_pair(id, dist));
  }

  std::sort(id_dists.begin(), id_dists.end(),
            [](const std::pair<unsigned, float>& left,
               const std::pair<unsigned, float>& right) {
              return left.second < right.second;
            });
  for (_u64 i = 0; i < results.size(); i++) {
    results[i] = id_dists[i].first;
  }
}

void save_result(char* filename, std::vector<std::vector<unsigned>>& results,
                 _u64 k) {
  std::cout << "Saving result to " << filename << std::endl;
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  unsigned GK = (unsigned) k;
  for (unsigned i = 0; i < results.size(); i++) {
    out.write((char*) &GK, sizeof(unsigned));
    out.write((char*) results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

int main(int argc, char** argv) {
  if (argc != 15) {
    std::cout << argv[0]
              << " data_bin[1] pq_tables_bin[2] n_chunks[3] chunk_size[4] "
                 "data_dim[5] nsg_disk_opt[6] query_file_fvecs[7] search_L[8] "
                 "search_K[9] result_path[10] BeamWidth[11] cache_nlevels[12] "
                 "full_int8_coords_bin[13] k_save[14]"
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
  index.reader.register_thread();
  index.load(argv[1], argv[6], argv[2], chunk_size, n_chunks, data_dim);

  // load queries
  float*   query_load = NULL;
  unsigned query_num, query_dim;
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

  std::cout << "Loading int8_t data" << std::endl;
  _s8*     s8_data = nullptr;
  unsigned npts_u32, ndims_u32;
  NSG::load_bin<_s8>(argv[13], s8_data, npts_u32, ndims_u32);

  // rerank results
  std::cout << "Reranking results";
  float* scratch = nullptr;
  NSG::alloc_aligned((void**) &scratch, aligned_dim * sizeof(float), 32);
  memset(scratch, 0, aligned_dim);
  for (_u64 i = 0; i < query_num; i++) {
    float*                 query = query_load + i * aligned_dim;
    std::vector<unsigned>& query_results = res[i];
    rerank_query(query, s8_data, index.pq_table, &dist_cmp, query_results,
                 scratch, data_dim, aligned_dim);
  }

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

  _u64 k_save = (_u64) std::atoi(argv[14]);
  save_result(argv[10], res, k_save);
  delete[] stats;
  delete[] s8_data;
  free(query_load);
  free(scratch);

  return 0;
}
