//
// Created by 付聪 on 2017/6/21.
//

#include <random>
#include <set>

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

// Include Efanna for NNDescent
//#include <efanna2e/index_graph.h>
//#include <efanna2e/index_random.h>
//#include <efanna2e/util.h>
//#include <efanna2e/distance.h>

void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*) &dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t             fsize = (size_t) ss;
  num = (unsigned) (fsize / (dim + 1) / 4);
  data = new float[(size_t) num * (size_t) dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*) (data + i * dim), dim * 4);
  }
  in.close();
}

template<class T>
void sample_data(size_t dim, T* src, unsigned src_npoints, T* dst,
                 unsigned dst_npoints, std::vector<unsigned>& picked_pts) {
  assert(picked_pts.size() == 0);
  picked_pts.reserve(dst_npoints);

  std::set<unsigned>              picked;
  std::random_device              rd;
  std::mt19937                    gen(rd());
  std::uniform_int_distribution<> dis(0, src_npoints - 1);

  while (picked.size() < dst_npoints) {
    unsigned r = dis(gen);
    if (picked.find(r) == picked.end())
      picked.insert(r);
  }

  for (auto p : picked) {
    memcpy(((char*) dst) + picked_pts.size() * dim * sizeof(T),
           ((char*) src) + p * dim * sizeof(T), dim * sizeof(T));
    picked_pts.push_back(p);
  }
}

int main(int argc, char** argv) {
  if (argc != 9 && argc != 10) {
    std::cout << argv[0]
              << " data_file nn_graph_path L R C NS NR output_graph_prefix"
              << std::endl;
    exit(-1);
  }
  float alpha = 1;
  if (argc == 11)
    alpha = (float) atof(argv[9]);

  float*   data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);

  std::cout << "Data loaded" << std::endl;

  std::string nn_graph_path(argv[2]);
  unsigned    L = (unsigned) atoi(argv[3]);
  unsigned    R = (unsigned) atoi(argv[4]);
  unsigned    C = (unsigned) atoi(argv[5]);
  unsigned    NS = (unsigned) atoi(argv[6]);
  unsigned    NR = (unsigned) atoi(argv[7]);

  float*                data_sampled = new float[(size_t) NS * dim];
  std::vector<unsigned> picked_pts;
  sample_data(dim, data_load, points_num, data_sampled, NS,
              picked_pts);  // Sample and copy before
  std::cout << "Data sampled" << std::endl;

  data_load = NSG::data_align(data_load, points_num, dim);
  data_sampled = NSG::data_align(data_sampled, NS, dim);
  std::cout << "File data and sample data aligned" << std::endl;

  /*{
    efanna2e::Parameters ef_paras;
    paras.Set<unsigned>("K", 50);
    paras.Set<unsigned>("L", 100);
    paras.Set<unsigned>("iter", 4);
    paras.Set<unsigned>("S", 20);
    paras.Set<unsigned>("R", 200);
    efanna2e::IndexRandom init_index(dim, NS);
    efanna2e::IndexGraph  index(dim, NS, efanna2e::L2,
                               (efanna2e::Index*) (&init_index));
    index.Build(NS, data_sampled, ef_paras);
  }*/

  NSG::IndexNSG rand_index(dim, NS, NSG::L2, nullptr);
  {
    auto            s = std::chrono::high_resolution_clock::now();
    NSG::Parameters paras;
    paras.Set<unsigned>("L", L / 2);
    paras.Set<unsigned>("R", R / 2);
    paras.Set<unsigned>("C", C / 2);
    paras.Set<float>("alpha", alpha);
    std::cout << "Params set. Rand Build start..." << std::endl;
    rand_index.BuildFromER(NS, NR, data_sampled, paras);
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "NSG(Rand) time: " << diff.count() << std::endl;
  }
  std::string rand_index_path = std::string(argv[8]) + std::string(".rand");
  const char* rand_index_path_c = rand_index_path.c_str();
  rand_index.Save(rand_index_path_c);

  NSG::IndexNSG small_index(dim, NS, NSG::L2, nullptr);
  {
    NSG::Parameters paras;
    paras.Set<unsigned>("L", L / 2);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C / 2);
    paras.Set<float>("alpha", alpha);
    paras.Set<float>("is_nsg", true);
    paras.Set<std::string>("nn_graph_path", rand_index_path_c);
    std::cout << "Params set. Small Build start..." << std::endl;
    auto s = std::chrono::high_resolution_clock::now();
    small_index.Build(NS, data_sampled, paras);
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "NSG(NSG(Rand)) time: " << diff.count() << std::endl;
  }
  small_index.SaveSmallIndex(argv[8], picked_pts);

  NSG::IndexNSG index(dim, points_num, NSG::L2, nullptr);
  {
    NSG::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<float>("alpha", alpha);
    paras.Set<std::string>("nn_graph_path", nn_graph_path);
    std::cout << "Params set. Build start..." << std::endl;
    auto s = std::chrono::high_resolution_clock::now();
    index.BuildFromSmall(points_num, data_load, paras, small_index, picked_pts);
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Big indexing time: " << diff.count() << std::endl;
  }
  index.Save(argv[8]);

  return 0;
}
