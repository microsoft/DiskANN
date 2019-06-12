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

void load_fvecs(const char* filename, float*& data, unsigned& num,
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
  size_t dim_t = dim;
  if (new_dim == 0)
    new_dim = dim;

  if (new_dim < dim)
    std::cout << "load_fvecs " << filename << ". Current Dimension: " << dim
              << ". New Dimension: First " << new_dim << " columns. "
              << std::flush;
  else if (new_dim > dim)
    std::cout << "load_fvecs " << filename << ". Current Dimension: " << dim
              << ". New Dimension: " << new_dim
              << " (added columns with 0 entries). " << std::flush;
  else
    std::cout << "load_fvecs " << filename << ". Dimension: " << dim << ". "
              << std::flush;

  float* zeros = new float[new_dim];
  for (size_t i = 0; i < new_dim; i++)
    zeros[i] = 0;

  size_t num_t = (fileSize / (4 * dim_t + 4));
  num = (unsigned) num_t;
  data = new float[(size_t) num * (size_t) new_dim];

  std::cout << "# Points: " << num << ".." << std::flush;

#pragma omp parallel for schedule(static, 65536)
  for (size_t i = 0; i < num; i++) {
    uint32_t row_dim;
    char*    reader = buf + (i * (4 * dim_t + 4));
    std::memcpy((char*) &row_dim, reader, sizeof(uint32_t));
    if (row_dim != dim)
      std::cerr << "ERROR: row dim does not match" << std::endl;
    std::memcpy(data + (i * new_dim), zeros,
                ((size_t) new_dim) * sizeof(float));
    if (new_dim > dim) {
      std::memcpy(data + (i * new_dim), reader + 4, dim_t * sizeof(float));
      //	std::memcpy(data + (i * new_dim), (reader + 4),
      //		    dim * sizeof(float));
    } else {
      std::memcpy(data + (i * new_dim), reader + 4,
                  ((size_t) new_dim) * sizeof(float));
      //	std::memcpy(data + (i * new_dim),
      //(reader + 4), 		    new_dim * sizeof(float));
    }
  }
  int val = munmap(buf, fileSize);
  close(fd);
  std::cout << "done." << std::endl;
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
  if (argc != 8) {
    std::cout << "Correct usage: " << argv[0]
              << " data_file L R C alpha num_rounds "
              << "save_graph_file  " << std::endl;
    exit(-1);
  }

  float*   data_load = NULL;
  unsigned points_num, dim;

  std::string bvecs(".bvecs");
  std::string base_file(argv[1]);
  if (base_file.find(bvecs) == std::string::npos) {
    load_fvecs(argv[1], data_load, points_num, dim);
  } else {
    load_bvecs(argv[1], data_load, points_num, dim);
  }

  data_load = NSG::data_align(data_load, points_num, dim);
  std::cout << "Data loaded and aligned" << std::endl;

  unsigned    L = (unsigned) atoi(argv[2]);
  unsigned    R = (unsigned) atoi(argv[3]);
  unsigned    C = (unsigned) atoi(argv[4]);
  float       alpha = (float) std::atof(argv[5]);
  unsigned    num_rnds = (unsigned) std::atoi(argv[6]);
  std::string save_path(argv[7]);

  NSG::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<unsigned>("num_rnds", num_rnds);

  unsigned num_incr = 10000;

  NSG::IndexNSG index(dim, points_num - num_incr, NSG::L2, nullptr, points_num,
                      true);
  {
    std::vector<tag_t> tags(points_num - num_incr);
    std::iota(tags.begin(), tags.end(), 0);

    auto s = std::chrono::high_resolution_clock::now();
    index.BuildRandomHierarchical(data_load, paras, tags);
    std::chrono::duration<double> diff =
        std::chrono::high_resolution_clock::now() - s;
    std::cout << "Indexing time: " << diff.count() << "\n";
  }

  std::vector<NSG::Neighbor>       pool, tmp;
  tsl::robin_set<unsigned>         visited;
  std::vector<NSG::SimpleNeighbor> cut_graph;

  {
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = points_num - num_incr; i < points_num; ++i)
      index.insert_point(data_load + (size_t) i * (size_t) dim, paras, pool,
                         tmp, visited, cut_graph, i);
    std::chrono::duration<double> diff =
        std::chrono::high_resolution_clock::now() - s;
    std::cout << "Incremental time: " << diff.count() << "\n";
  }

  index.Save(save_path.c_str());

  tsl::robin_set<unsigned> delete_list;
  while (delete_list.size() < num_incr)
    delete_list.insert((rand() * rand() * rand()) % points_num);

  {
    auto s = std::chrono::high_resolution_clock::now();
    index.enable_delete();
    for (auto p : delete_list)
      if (index.delete_point(p) != 0)
        std::cerr << "Delete tag " << p << " not found" << std::endl;
    if (index.disable_delete(paras, true) != 0) {
      std::cerr << "Disable delete failed" << std::endl;
      return -1;
    }
    std::chrono::duration<double> diff =
        std::chrono::high_resolution_clock::now() - s;
    std::cout << "Delete time: " << diff.count() << "\n";
  }

  {
    auto s = std::chrono::high_resolution_clock::now();
    for (auto p : delete_list)
      index.insert_point(data_load + (size_t) p * (size_t) dim, paras, pool,
                         tmp, visited, cut_graph, p);
    std::chrono::duration<double> diff =
        std::chrono::high_resolution_clock::now() - s;
    std::cout << "Re-Incremental time: " << diff.count() << "\n";
  }

  auto save_path_reinc = save_path + ".reinc";
  index.Save(save_path_reinc.c_str());

  return 0;
}