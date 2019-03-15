//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

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
  if (argc != 9) {
    std::cout << argv[0] << "<index> data_file nn_graph L R C nsg_output "
              << "alpha<1 if you dont know> is_nsg<0 on efanna>" << std::endl;
    exit(-1);
  }
  std::string nn_graph_path(argv[2]);
  unsigned    L = (unsigned) atoi(argv[3]);
  unsigned    R = (unsigned) atoi(argv[4]);
  unsigned    C = (unsigned) atoi(argv[5]);
  float       alpha = (float) std::atof(argv[7]);
  bool        is_nsg = (bool) std::atoi(argv[8]);

  float*   data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  std::cout << "Data loaded" << std::endl;

  data_load = NSG::data_align(data_load, points_num, dim);
  NSG::IndexNSG index(dim, points_num, NSG::L2, nullptr);

  auto            s = std::chrono::high_resolution_clock::now();
  NSG::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("is_nsg", is_nsg);
  paras.Set<std::string>("nn_graph_path", nn_graph_path);

  std::cout << "Params set" << std::endl;

  index.Build(points_num, data_load, paras);
  auto                          e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  std::cout << "indexing time: " << diff.count() << "\n";
  index.Save(argv[6]);

  return 0;
}
