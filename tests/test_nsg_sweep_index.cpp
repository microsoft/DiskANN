//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <omp.h>

#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

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

void load_bvecs(const char* filename, float*& data, unsigned& num,
                unsigned& dim) {
  unsigned new_dim = 0;
  char*    buf;
  int      fd;
  fd = open(filename, O_RDONLY);
  if (!(fd > 0)) {
    std::cerr << "Data file " << filename
              << " not found. Program will stop now." << std::endl;
    assert(false);
  }
  struct stat sb;
  fstat(fd, &sb);
  off_t fileSize = sb.st_size;
  //  assert(sizeof(off_t) == 8);

  buf = (char*) mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
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

  num = (unsigned) (fileSize / (dim + 4));
  data = new float[(size_t) num * (size_t) new_dim];

  std::cout << "# Points: " << num << ".." << std::flush;

#pragma omp parallel for schedule(static, 65536)
  for (size_t i = 0; i < num; i++) {
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
  int val = munmap(buf, fileSize);
  close(fd);
  std::cout << "done." << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 11) {
    std::cout << "Correct usage\n"
              << argv[0] << " data_file L R C "
              << "save_graph_file  alpha<1>  p_val<0.1> "
              << "num_syncs <100> num_pass<1> innerR <R>" << std::endl;
    exit(-1);
  }

  float*   data_load = NULL;
  unsigned points_num, dim;

  std::string bvecs(".bvecs");
  std::string base_file(argv[1]);
  if (base_file.find(bvecs) == std::string::npos) {
    //		std::cout << "Loading base set as fvecs" << std::endl;
    load_data(argv[1], data_load, points_num, dim);
  } else {
    //		std::cout << "Loading training set as bvecs" << std::endl;
    load_bvecs(argv[1], data_load, points_num, dim);
  }

  //  load_data(argv[1], data_load, points_num, dim);
  data_load = NSG::data_align(data_load, points_num, dim);
  std::cout << "Data loaded and aligned" << std::endl;

  //  unsigned    nn_graph_deg = (unsigned) atoi(argv[3]);
  unsigned    L = (unsigned) atoi(argv[2]);
  unsigned    R = (unsigned) atoi(argv[3]);
  unsigned    C = (unsigned) atoi(argv[4]);
  std::string save_path(argv[5]);
  float       alpha = (float) std::atof(argv[6]);
  float       p_val = (float) std::atof(argv[7]);
  //  unsigned    num_hier = (float) std::atof(argv[8]);
  unsigned num_syncs = (float) std::atof(argv[8]);
  unsigned num_rnds = (unsigned) std::atoi(argv[9]);
  //  unsigned    innerL = (unsigned) atoi(argv[11]);
  unsigned innerR = (unsigned) atoi(argv[10]);
  //  unsigned    innerC = (unsigned) atoi(argv[13]);

  //  if (nn_graph_deg > R) {
  //    std::cerr << "Error: nn_graph_degree must be <= R" << std::endl;
  //    exit(-1);
  //  }

  NSG::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<std::string>("save_path", save_path);
  paras.Set<float>("alpha", alpha);
  paras.Set<float>("p_val", p_val);
  //  paras.Set<unsigned>("innerL", innerL);
  //  paras.Set<unsigned>("innerC", innerC);
  paras.Set<unsigned>("num_syncs", num_syncs);
  paras.Set<unsigned>("num_rnds", num_rnds);
  paras.Set<unsigned>("innerR", innerR);
  //  paras.Set<unsigned>("num_hier", num_hier);
  paras.Set<bool>("is_nsg", 0);
  paras.Set<bool>("is_rnd_nn", 1);
  //  paras.Set<std::string>("nn_graph_path", nn_graph_path);
  std::cout << "Params set" << std::endl;

  {
    NSG::IndexNSG index(dim, points_num, NSG::L2, nullptr);
    auto          s = std::chrono::high_resolution_clock::now();
    index.BuildRandomHierarchical(points_num, data_load, paras);
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "indexing time: " << diff.count() << "\n";
    index.Save(argv[5]);
    //    index.Save_Inner_Vertices(argv[5]);
  }

  return 0;
}
