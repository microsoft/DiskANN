//
// Created by 付聪 on 2017/6/21.
//

#include <set>
#include <random>

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>


void load_data(
    char* filename,
    float*& data,
    unsigned& num,
    unsigned& dim) 
{  // load data with sift10K pattern
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

template <class T>
void sample_data(
    size_t dim,
    T* src,
    unsigned src_npoints,
    T* dst, 
    unsigned dst_npoints,
    std::vector<unsigned>& picked_pts)
{
    assert(picked_pts.size() == 0);
    picked_pts.reserve(dst_npoints);

    std::set picked;
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, src_npoints);

    while (picked.size() < dst_npoints) {
        unsigned r = dis(gen);
        if (!picked.contains(r))
            picked.insert(r);
    }

    for (auto p : picked) {
        memcpy((char*)dst + picked_pts.size() * dim * sizeof(T),
            src + p * dim * sizeof(T), dim * sizeof(T));
        picked_pts.push_back(p);
    }
}

int main(int argc, char** argv) 
{
  if (argc != 9 && argc !=10) {
    std::cout << argv[0] << " data_file nn_graph_path L R C L1 B1 save_graph_file"
              << std::endl;
    exit(-1);
  }
  float alpha = 1;
  if (argc == 10)  alpha = (float) atof(argv[9]);
	
  float *data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);

  std::cout << "Data loaded" << std::endl;
  
  std::string nn_graph_path(argv[2]);
  unsigned L = (unsigned)atoi(argv[3]);
  unsigned R = (unsigned)atoi(argv[4]);
  unsigned C = (unsigned)atoi(argv[5]);
  unsigned NS = (unsigned)atoi(argv[6]);
  unsigned B1 = (unsigned)atoi(argv[7]);


  float *data_sampled = new float[(size_t)L1 * dim]; 
  std::vector<unsigned> picked_pts;
  sample_data(dim, data_load, points_num, data_sampled, NS, picked_pts);  // Sample and copy before 
  
  data_load = efanna2e::data_align(data_load, points_num, dim);
  data_sampled = efanna2e::data_align(data_sampled, NS, dim);

  efanna2e::IndexNSG small_index(dim, NS, efanna2e::L2, nullptr);
  {
    auto s = std::chrono::high_resolution_clock::now();
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L", 2000);
    paras.Set<unsigned>("R", 500);
    paras.Set<unsigned>("C", 20000);
    paras.Set<float>("alpha", alpha);
    std::cout << "Params set. Build start..." << std::endl;
    small_index.BuildFromAlltoAll(NS, data_sampled, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Small indexing time: " << diff.count() << "\n";
  }

  efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
  {
    auto s = std::chrono::high_resolution_clock::now();
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<float>("alpha", alpha);
    paras.Set<std::string>("nn_graph_path", nn_graph_path);
    std::cout << "Params set. Build start..." << std::endl;
    index.Build(points_num, data_load, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Big indexing time: " << diff.count() << "\n";
  }

  index.Save(argv[8]);

  return 0;
}
