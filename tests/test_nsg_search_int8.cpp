//
// Created by 付聪 on 2017/6/21.
//

#include <index_nsg.h>
#include <omp.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <util.h>

void load_ivecs(char* filename, unsigned*& data, unsigned& num,
                unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*) &dim, 4);
  std::cout << "data dimension: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();

  size_t fsize = (size_t) ss;
  num = (unsigned) (fsize / (dim + 1) / 4);
  std::cout << "Reading " << num << " points" << std::endl;
  data = new unsigned[(size_t) num * (size_t) dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*) (data + i * dim), dim * 4);
  }
  in.close();
}

float calc_recall(unsigned num_queries, unsigned* gold_std, unsigned dim_gs,
                  unsigned* our_results, unsigned dim_or, unsigned recall_at) {
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

void save_result(char* filename, unsigned* results, unsigned nd, unsigned nr) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < nd; i++) {
    out.write((char*) &nr, sizeof(unsigned));
    out.write((char*) (results + i * nr), nr * sizeof(unsigned));
  }
  out.close();
}

int main(int argc, char** argv) {
  if ((argc != 8)) {
    std::cout << argv[0]
              << " data_file[INT8] query_file[INT8] groundtruth nsg_path "
                 "BFS-init=1/0 beamwidth recall@"
              << std::endl;
    exit(-1);
  }

  int      bfs_init = atoi(argv[5]);
  unsigned beam_width = atoi(argv[6]);
  unsigned recall_at = atoi(argv[7]);

  _s8*      data_load = NULL;
  _s8*      query_load = NULL;
  unsigned* gt_load = NULL;
  unsigned  points_num, dim, query_num, query_dim;
  unsigned  gt_num, gt_dim;

  std::vector<unsigned> Lvec;
  unsigned              curL = 8;
  while (curL < 2048) {
    Lvec.push_back(curL);
    if (curL < 16)
      curL += 1;
    else if (curL < 32)
      curL += 2;
    else if (curL < 64)
      curL += 4;
    else if (curL < 128)
      curL += 8;
    else if (curL < 256)
      curL += 16;
    else if (curL < 512)
      curL += 32;
    else if (curL < 1024)
      curL += 64;
    else
      curL += 128;
  }

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);
  //  std::cout.precision(1);

  // load_data(argv[1], data_load, points_num, dim);
  NSG::load_Tvecs<_s8>(argv[1], data_load, points_num, dim);
  NSG::load_Tvecs<_s8>(argv[2], query_load, query_num, query_dim);

  load_ivecs(argv[3], gt_load, gt_num, gt_dim);
  if (gt_num != query_num) {
    std::cout << "Ground truth does not match number of queries. ";
    exit(-1);
  }

  if (recall_at > gt_num) {
    std::cout << "Ground truth has only " << gt_num
              << " elements. Calculating recall at " << gt_num << std::endl;
    recall_at = gt_num;
  }

  assert(dim == query_dim);
  std::cout << "Base and query data loaded" << std::endl;
  data_load = NSG::data_align_byte<_s8>(data_load, points_num, dim);
  query_load = NSG::data_align_byte<_s8>(query_load, query_num, query_dim);
  std::cout << "Data Aligned -- new dimension: " << dim << std::endl;

  NSG::IndexNSG<_s8> index(dim, points_num, NSG::L2, nullptr);
  //  if (nsg_check == 1)
  index.Load(argv[4]);  // to load NSG
                        //  else {
                        //    index.Load_nn_graph(argv[3]);  // to load EFANNA

  // ravi-comment
  // index.init_graph_outside(data_load);
  //  }
  std::cout << "Index loaded" << std::endl;

  std::vector<unsigned> start_points;
  if (bfs_init) {
    index.populate_start_points_bfs(start_points);
    std::cout << "Initialized starting points based on BFS" << std::endl;
  }

  NSG::Parameters paras;
  std::cout << "Ls\t\tLatency\t\tRecall@" << recall_at << "\tCmps\t\tHops"
            << std::endl;
  std::cout << "==============================================================="
               "======="
            << std::endl;
  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    unsigned L = Lvec[test_id];
    if (L < recall_at)
      continue;
    unsigned  K = L;
    unsigned* res = new unsigned[(size_t) query_num * K];

    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);

    long long total_hops = 0;
    long long total_cmps = 0;

    auto    s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static, 1)
    for (unsigned i = 0; i < query_num; i++) {
      auto ret =
          index.BeamSearch(query_load + i * dim, data_load, K, paras,
                           res + ((size_t) i) * K, beam_width, start_points);
// auto ret = index.Search(query_load + i * dim, data_load, K, paras,
// tmp.data());

#pragma omp atomic
      total_hops += ret.first;
#pragma omp atomic
      total_cmps += ret.second;
    }

    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    unsigned                      nthreads = omp_get_max_threads();
    //    std::cout << "search time: " << diff.count() << "\n";
    float latency = (diff.count() / query_num) * (1000000) * nthreads;
    float avg_hops = (float) total_hops / (float) query_num;
    float avg_cmps = (float) total_cmps / (float) query_num;
    float recall = calc_recall(query_num, gt_load, gt_dim, res, K, recall_at);
    std::cout << L << "\t\t" << latency << "\t\t" << recall << "\t\t"
              << avg_cmps << "\t\t" << avg_hops << std::endl;
    if (recall > 99.5) {
      break;
    }
    delete[] res;
  }
  delete[] data_load;

  return 0;
}
