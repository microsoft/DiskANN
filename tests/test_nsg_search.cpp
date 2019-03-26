//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <omp.h>

void load_data(char* filename, float*& data, unsigned& num,
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
  data = new float[(size_t) num * (size_t) dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*) (data + i * dim), dim * 4);
  }
  in.close();
}

/*void save_result(char* filename, std::vector<std::vector<unsigned> >& results)
{
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned) results[i].size();
    out.write((char*) &GK, sizeof(unsigned));
    out.write((char*) results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}*/

void save_result(char* filename, unsigned* results, unsigned nd, unsigned nr) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < nd; i++) {
    out.write((char*) &nr, sizeof(unsigned));
    out.write((char*) (results + i * nr), nr * sizeof(unsigned));
  }
  out.close();
}

int main(int argc, char** argv) {
  if ((argc < 9) || ((argc - 9) % 3 != 0)) {
    std::cout << argv[0]
              << " data_file query_file nsg_path "
                 "result_path_prefix BFS-init=1/0 L1 K1 BW1 L2 K2 BW2 ..."
              << std::endl;
    exit(-1);
  }

  int bfs_init = atoi(argv[5]);

  uint32_t  num_tests = (argc - 6) / 3;
  unsigned* Lvec = new unsigned[num_tests];
  unsigned* Kvec = new unsigned[num_tests];
  uint32_t* Bwvec = new uint32_t[num_tests];

  for (int i = 0; i < num_tests; i++) {
    Lvec[i] = std::atoi(argv[6 + 3 * i]);
    Kvec[i] = std::atoi(argv[6 + 3 * i + 1]);
    Bwvec[i] = std::atoi(argv[6 + 3 * i + 2]);
    if (Lvec[i] < Kvec[i]) {
      std::cout << "search_L cannot be smaller than search_K!" << std::endl;
      exit(-1);
    }
  }

  float*   data_load = NULL;
  unsigned points_num, dim;
  // load_data(argv[1], data_load, points_num, dim);
  NSG::load_Tvecs<float>(argv[1], data_load, points_num, dim);
  float*   query_load = NULL;
  unsigned query_num, query_dim;
  // load_data(argv[2], query_load, query_num, query_dim);
  NSG::load_Tvecs<float>(argv[2], query_load, query_num, query_dim);
  assert(dim == query_dim);
  std::cout << "Base and query data loaded" << std::endl;
  data_load = NSG::data_align(data_load, points_num, dim);
  query_load = NSG::data_align(query_load, query_num, query_dim);
  std::cout << "Data Aligned" << std::endl;

  NSG::IndexNSG index(dim, points_num, NSG::L2, nullptr);
  //  if (nsg_check == 1)
  index.Load(argv[3]);  // to load NSG
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

  for (uint32_t test_id = 0; test_id < num_tests; test_id++) {
    unsigned L = Lvec[test_id];
    unsigned K = Kvec[test_id];
    uint32_t beam_width = Bwvec[test_id];
  unsigned*       res = new unsigned[(size_t) query_num * K];

    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);

    auto s = std::chrono::high_resolution_clock::now();

    long long total_hops = 0;
    long long total_cmps = 0;

    std::cout << "NSG search using L = " << L << ", K = " << K
              << ", BeamWidth = " << beam_width << std::endl;
#pragma omp parallel for schedule(static, 1000)
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
    std::cout << "search time: " << diff.count() << "\n";

    std::cout << "Average hops: " << (float) total_hops / (float) query_num
              << std::endl
              << "Average cmps: " << (float) total_cmps / (float) query_num
              << std::endl;
    std::string output_file = std::string(argv[4]);
    output_file += "_search-" + std::to_string(L) + "-" + std::to_string(K) +
                   "-" + std::to_string(beam_width) + ".ivecs";
    char* out_file = new char[output_file.size()+1];
out_file[output_file.size()] = 0;
std::memcpy(out_file, output_file.c_str(), output_file.size());
    save_result(out_file, res, query_num, K);
  delete[] out_file;
  delete[] res;
  }
  delete[] data_load;

  return 0;
}
