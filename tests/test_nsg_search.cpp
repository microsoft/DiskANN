//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include "MemoryMapper.h"


void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream   in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*) &dim, 4);
  std::cout << "data dimension: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type  ss = in.tellg();

  size_t  fsize = (size_t) ss;
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

void load_bvecs(const char* filename, float*& data, unsigned& num,
                unsigned& dim) {
  unsigned new_dim = 0;
  char*    buf;
  int      fd;

  NSG::MemoryMapper mapper(filename);
  buf = mapper.getBuf();
  //  assert(buf);
  // size_t x=4;
  uint32_t file_dim;
  std::memcpy(&file_dim, buf, 4);
  dim = file_dim;
  if (new_dim == 0)
    new_dim = dim;

  if (new_dim < dim)
    std::cout << "load_bvecs " << filename << ". Current Dimension: " << dim
              << ". New Dimension: First " << new_dim << " columns. "
              << std::flush;
  else if (new_dim > dim)
    std::cout << "load_bvecs " << filename << ". Current Dimension: " << dim
              << ". New Dimension: " << new_dim
              << " (added columns with 0 entries). " << std::flush;
  else
    std::cout << "load_bvecs " << filename << ". Dimension: " << dim << ". "
              << std::flush;

  float* zeros = new float[new_dim];
  for (size_t i = 0; i < new_dim; i++)
    zeros[i] = 0;

  num = (unsigned) (mapper.getFileSize() / (dim + 4));
  data = new float[(size_t) num * (size_t) new_dim];

  std::cout << "# Points: " << num << ".." << std::flush;

#pragma omp parallel for schedule(static, 65536)
  for (int64_t i = 0; i < num; i++) { //Gopal. changed from size_t to int64_t
    uint32_t row_dim;
    char*    reader = buf + (i * (dim + 4));
    std::memcpy((char*) &row_dim, reader, sizeof(uint32_t));
    if (row_dim != dim)
      std::cerr << "ERROR: row dim does not match" << std::endl;
    std::memcpy(data + (i * new_dim), zeros, new_dim * sizeof(float));
    if (new_dim > dim) {
      //	std::memcpy(data + (i * new_dim), (reader + 4),
      //		    dim * sizeof(float));
      for (size_t j = 0; j < dim; j++) {
        uint8_t cur;
        std::memcpy((char*) &cur, (reader + 4 + j), sizeof(uint8_t));
        data[i * new_dim + j] = (float) cur;
      }
    } else {
      for (size_t j = 0; j < new_dim; j++) {
        uint8_t cur;
        std::memcpy((char*) &cur, (reader + 4 + j), sizeof(uint8_t));
        data[i * new_dim + j] = (float) cur;
        //	std::memcpy(data + (i * new_dim),
        //(reader + 4), 		    new_dim * sizeof(float));
      }
    }
  }

  std::cout << "done." << std::endl;
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
  float*   query_load = NULL;
  unsigned points_num, dim, query_num, query_dim;
  // load_data(argv[1], data_load, points_num, dim);
  std::string bvecs("bvecs");
  if (std::string(argv[1]).find(bvecs) != std::string::npos)
    load_bvecs(argv[1], data_load, points_num, dim);
  else
    load_data(argv[1], data_load, points_num, dim);

  if (std::string(argv[2]).find(bvecs) != std::string::npos)
    load_bvecs(argv[2], query_load, query_num, query_dim);
  else
    load_data(argv[2], query_load, query_num, query_dim);

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
    unsigned  L = Lvec[test_id];
    unsigned  K = Kvec[test_id];
    uint32_t  beam_width = Bwvec[test_id];
    unsigned* res = new unsigned[(size_t) query_num * K];

    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);

    auto s = std::chrono::high_resolution_clock::now();

    long long total_hops = 0;
    long long total_cmps = 0;

    std::cout << "NSG search using L = " << L << ", K = " << K
              << ", BeamWidth = " << beam_width << std::endl;
#pragma omp parallel for schedule(static, 1000)
    for (int i = 0; i < query_num; i++) {  // GOPAL renamed "unsigned i" to "int i"
      auto ret =
          index.BeamSearch(query_load + i * dim, data_load, K, paras,
                           res + ((size_t) i) * K, beam_width, start_points);

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

    // GOPAL Commented the below lines to get this to work on aether.
    // std::string output_file = std::string(argv[4]);
    // output_file += "_search-" + std::to_string(L) + "-" + std::to_string(K) +
    //               "-" + std::to_string(beam_width) + ".ivecs";
    // char* out_file = new char[output_file.size() + 1];
    // out_file[output_file.size()] = 0;
    // std::memcpy(out_file, output_file.c_str(), output_file.size());
    save_result(argv[4], res, query_num, K);
    // delete[] out_file;
    delete[] res;
    std::cout << "Wrote query stats and results to output file: " << argv[4]
              << std::endl;
  }
#ifdef __NSG_WINDOWS__
  _aligned_free(data_load);
#else
  delete[] data_load;
#endif

  std::cout << "Closing program." << std::endl;
  return 0;
}
