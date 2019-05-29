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
  if (argc != 16) {
    std::cout << "Correct usage\n"
              << argv[0] << " data_file efanna/nsg_graph_path L R C "
              << "save_graph_file  alpha<1>   p_val<0.1> "
              << "num_hier<1>  num_syncs<10> num_rounds<1> is_nsg (1) innerL "
                 "(L) innerR innerC (C)"
              << std::endl;
    exit(-1);
  }

  float*   data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  data_load = NSG::data_align(data_load, points_num, dim);
  std::cout << "Data loaded and aligned" << std::endl;

  std::string nn_graph_path(argv[2]);
  unsigned    L = (unsigned) atoi(argv[3]);
  unsigned    R = (unsigned) atoi(argv[4]);
  unsigned    C = (unsigned) atoi(argv[5]);
  float       alpha = (float) std::atof(argv[7]);
  float       p_val = (float) std::atof(argv[8]);
  unsigned    num_hier = (float) std::atof(argv[9]);
  unsigned    num_syncs = (float) std::atof(argv[10]);
  unsigned    num_rnds = (bool) std::atoi(argv[11]);
  bool        is_nsg = (bool) std::atoi(argv[12]);
  unsigned    innerL = (unsigned) atoi(argv[13]);
  unsigned    innerR = (unsigned) atoi(argv[14]);
  unsigned    innerC = (unsigned) atoi(argv[15]);

  NSG::Parameters paras;
  paras.Set<unsigned>("L", L);  // search list size during index construction
  paras.Set<unsigned>("R", R);  // max degree of the index
  paras.Set<unsigned>("C", C);  // candidate list size
  paras.Set<unsigned>("innerL", L);
  paras.Set<unsigned>("innerR", R);
  paras.Set<unsigned>("innerC", C);
  paras.Set<std::string>("nn_graph_path", nn_graph_path);
  paras.Set<unsigned>("num_syncs",
                      num_syncs);  // number of batches used for creating index
  paras.Set<unsigned>("num_hier", num_hier);  // ?
  paras.Set<unsigned>(
      "num_rnds",
      num_rnds);  // number of rounds for creating index, usually 2 rounds
  paras.Set<float>("alpha",
                   alpha);  // aggressiveness of adding edges in prune procedure
  paras.Set<float>("p_val", p_val);  // deprecated
  paras.Set<bool>("is_nsg", is_nsg);
  paras.Set<bool>("is_rnd_nn", 0);
  //  paras.Set<std::string>("nn_graph_path", nn_graph_path);
  std::cout << "Params set" << std::endl;

  {
    NSG::IndexNSG index(dim, points_num, NSG::L2, nullptr);
    auto          s = std::chrono::high_resolution_clock::now();
    index.BuildRandomHierarchical(points_num, data_load, paras);
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "indexing time: " << diff.count() << "\n";
    index.Save(argv[6]);
    index.Save_Inner_Vertices(argv[6]);
  }

  return 0;
}
