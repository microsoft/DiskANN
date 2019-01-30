//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/flash_index_nsg.h>
#include <efanna2e/util.h>
#include <omp.h>

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
              << " embedded_index_prefix query_file search_L search_K result_path BeamWidth"
              << std::endl;
    exit(-1);
  }
  
  unsigned points_num, dim;
  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  assert(dim == query_dim);

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);
  int beam_width = atoi(argv[7]);

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build
  query_load = efanna2e::data_align(query_load, query_num, query_dim);
  efanna2e::FlashIndexNSG index(dim, points_num, efanna2e::L2, nullptr);
  index.Load(argv[3]);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);

  auto s = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<unsigned> > res(query_num, std::vector<unsigned>(K));
  long long total_hops=0; long long total_cmps=0;
  std::vector<long long> hops(query_num), cmps(query_num);
  #pragma omp parallel for
  for (unsigned i = 0; i < query_num; i++) {
    std::vector<unsigned> &query_res = res[i];
    auto ret = index.BeamSearch(query_load + i * dim, data_load, K, paras, query_res.data(), beam_width);
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
