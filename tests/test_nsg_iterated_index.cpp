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

int main(int argc, char** argv) {
  if (argc != 7 && argc != 8) {
    std::cout << argv[0] << " data_file L R C iters output_graph_prefix"
              << std::endl;
    exit(-1);
  }
  float alpha = 1.0f;
  if (argc == 8)
    alpha = (float) atof(argv[7]);
  // std::string nn_graph_path(argv[2]);
  unsigned L = (unsigned) atoi(argv[2]);
  unsigned R = (unsigned) atoi(argv[3]);
  unsigned C = (unsigned) atoi(argv[4]);
  unsigned iter = (unsigned) atoi(argv[5]);
  if (iter > 3) {
    std::cout << "Please use iter = 1, 2, or 3." << std::endl;
    exit(-1);
  }

  float*   data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  std::cout << "Data loaded" << std::endl;

  data_load = NSG::data_align(data_load, points_num, dim);
  std::cout << "File data aligned" << std::endl;

  std::string first_index_path = std::string(argv[6]) + std::string(".iter.1");
  const char* first_index_path_c = first_index_path.c_str();
  {
    NSG::IndexNSG   first_index(dim, points_num, NSG::L2, nullptr);
    NSG::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", iter == 1 ? R : 2 * R / 3);
    paras.Set<unsigned>("C", iter == 1 ? C : C / 2);
    paras.Set<float>("alpha", alpha);
    auto s = std::chrono::high_resolution_clock::now();
    first_index.BuildFromER(points_num, R / 2, data_load, paras);
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "NSG(Rand) time: " << diff.count() << std::endl;
    first_index.Save(first_index_path_c);
  }
  if (iter == 1)
    return 0;

  std::string second_index_path = std::string(argv[6]) + std::string(".iter.2");
  const char* second_index_path_c = second_index_path.c_str();
  {
    NSG::IndexNSG   second_index(dim, points_num, NSG::L2, nullptr);
    NSG::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<float>("alpha", alpha);
    paras.Set<float>("is_nsg", true);
    paras.Set<std::string>("nn_graph_path", first_index_path_c);
    auto s = std::chrono::high_resolution_clock::now();
    second_index.Build(points_num, data_load, paras);
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "NSG(NSG(Rand)) time: " << diff.count() << std::endl;
    second_index.Save(second_index_path_c);
  }
  if (iter == 2)
    return 0;

  std::string third_index_path = std::string(argv[6]) + std::string(".iter.3");
  const char* third_index_path_c = third_index_path.c_str();
  {
    NSG::IndexNSG   third_index(dim, points_num, NSG::L2, nullptr);
    NSG::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<float>("alpha", alpha);
    paras.Set<float>("is_nsg", true);
    paras.Set<std::string>("nn_graph_path", second_index_path_c);
    auto s = std::chrono::high_resolution_clock::now();
    third_index.Build(points_num, data_load, paras);
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "NSG(NSG(NSG(Rand))) time: " << diff.count() << std::endl;
    third_index.Save(third_index_path_c);
  }

  return 0;
}
