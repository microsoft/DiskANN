//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/flash_index_nsg.h>
#include <efanna2e/util.h>
#include <omp.h>
#include <cassert>

void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

void save_result(char* filename, std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cout << argv[0]
              << " embedded_index index_node_sizes n_base n_dims query_file search_L search_K result_path BeamWidth"
              << std::endl;
    exit(-1);
  }
  
  std::string index_file(argv[1]);
  std::string sizes_file(argv[2]);
  unsigned n_pts = (unsigned) std::atoi(argv[3]);
  unsigned n_dims = (unsigned) std::atoi(argv[4]);

  // construct FlashNSGIndex
  efanna2e::FlashIndexNSG index(n_dims, n_pts, efanna2e::L2, nullptr);
  index.load_embedded_index(index_file, sizes_file);

  // load queries
  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  assert(n_dims == query_dim);

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);
  int beam_width = atoi(argv[7]);

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  // align query data
  query_load = efanna2e::data_align(query_load, query_num, query_dim);
  index.Load(argv[3]);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);

  auto s = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<unsigned> > res(query_num, std::vector<unsigned>(K));
  long long total_hops=0; long long total_cmps=0;
  std::vector<long long> hops(query_num), cmps(query_num);
  bool has_init = false;
  #pragma omp parallel for firstprivate(has_init)
  for (unsigned i = 0; i < query_num; i++) {
    if(!has_init){
      #pragma omp critical
      {
        index.graph_reader.register_thread();
        std::cout << "Init complete for thread-"<<omp_get_thread_num() << std::endl;
        has_init = true;
      }
    }
    std::vector<unsigned> &query_res = res[i];
    auto ret = index.BeamSearch(query_load + i * query_dim, nullptr, K, paras, query_res.data(), beam_width);
    //auto ret = index.Search(query_load + i * dim, data_load, K, paras, tmp.data());
    hops[i] = ret.first;
    cmps[i] = ret.second;
  }
  total_hops = std::accumulate(hops.begin(), hops.end(), 0L);
  total_cmps = std::accumulate(cmps.begin(), cmps.end(), 0L);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "search time: " << diff.count() << "\n";

  std::cout << "Average hops: " << (float)total_hops/(float)query_num << std::endl
	    << "Average cmps: " << (float)total_cmps/(float)query_num << std::endl;
  
  save_result(argv[6], res);

  return 0;
}
