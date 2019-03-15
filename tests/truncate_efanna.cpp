//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <cassert>

float*   data_ = NULL;
size_t   num;
unsigned dimension_;
//#define EFANNA_NOT_SORTED 0

NSG::Distance* distance_ = new NSG::DistanceL2;

typedef std::vector<std::vector<unsigned>> CompactGraph2;

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

void open_linux_mmapped_file_handle(char* filename, float*& data, size_t& num,
                                    unsigned& dim) {
  char* buf;
  int   fd;
  fd = open(filename, O_RDONLY);
  if (!(fd > 0)) {
    std::cerr << "Data file " << filename
              << " not found. Program will stop now." << std::endl;
    assert(false);
  }
  struct stat sb;
  // int val = fstat(fd, &sb);
  off_t fileSize = sb.st_size;
  assert(sizeof(off_t) == 8);

  buf = (char*) mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
  assert(buf);
  // size_t x=4;
  std::memcpy(&dim, buf, 4);
  num = (size_t)(fileSize / (dim + 1) / 4);
  data = new float[(size_t) num * (size_t) dim];

#pragma omp parallel for schedule(static, 65536)
  for (size_t i = 0; i < num; i++) {
    char* reader = buf + (i * (dim + 1) * 4);
    std::memcpy(data + (i * dim), (reader + 4), dim * sizeof(float));
  }
}

void Load_and_truncate_nn_graph(const char*    filename,
                                CompactGraph2& final_graph_, unsigned& k,
                                unsigned d_nn, unsigned d_rnd) {
  char* buf;
  int   fd;
  fd = open(filename, O_RDONLY);
  if (!(fd > 0)) {
    std::cerr << "Data file " << filename
              << " not found. Program will stop now." << std::endl;
    assert(false);
  }
  struct stat sb;
  int         val = fstat(fd, &sb);
  if (val != 0)
    std::cout << "FILE LOAD ERROR. CHECK!" << std::endl;

  off_t fileSize = sb.st_size;
  std::cout << fileSize << std::endl;
  buf = (char*) mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);

  std::memcpy(&k, buf, sizeof(unsigned));
  size_t num = (fileSize) / ((k + 1) * 4);

  std::cout << "k is and num is " << k << " " << num << std::endl;
  final_graph_.resize(num);
  final_graph_.reserve(num);

  unsigned  kk = (k + 3) / 4 * 4;
#pragma omp parallel for schedule(static, 65536)
  for (size_t i = 0; i < num; i++) {
    final_graph_[i].resize(d_nn + d_rnd);
    final_graph_[i].reserve(d_nn + d_rnd);
    char* reader = buf + (i * (k + 1) * sizeof(unsigned));

#ifdef EFANNA_NOT_SORTED

    std::memcpy(final_graph_[i].data(), reader + sizeof(unsigned),
                k * sizeof(unsigned));

    std::vector<NSG::Neighbor> pool;
    for (unsigned nn = 0; nn < final_graph_[i].size(); nn++) {
      unsigned id = final_graph_[i][nn];
      float    dist = distance_->compare(data_ + dimension_ * (size_t) i,
                                      data_ + dimension_ * (size_t) id,
                                      (unsigned) dimension_);
      pool.push_back(NSG::Neighbor(id, dist, true));
    }

    std::sort(pool.begin(), pool.end());

    for (unsigned nn = 0; nn < d_nn; nn++) {
      final_graph_[i][nn] = pool[nn].id;
    }
#else
    std::memcpy(final_graph_[i].data(), reader + sizeof(unsigned),
                d_nn * sizeof(unsigned));

#endif

    for (unsigned nn = d_nn; nn < d_nn + d_rnd; nn++) {
      final_graph_[i][nn] = rand() % num;
    }

    final_graph_[i].resize(d_nn + d_rnd);
  }
  val = munmap(buf, fileSize);
  close(fd);
  if (val != 0)
    std::cout << "ERROR unmapping. CHECK!" << std::endl;
  std::cout << "Loaded EFANNA graph. Set ep_ to 0" << std::endl;
}

void Save_nn_graph(const char* filename, CompactGraph2& final_graph_,
                   unsigned& k) {
  char* buf;
  int   fd;
  fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0666);
  if (!(fd > 0)) {
    std::cerr << "Data file " << filename
              << " not found. Program will stop now." << std::endl;
    assert(false);
  }
  unsigned num = final_graph_.size();
  std::cout << num;
  size_t fileSize = num * (sizeof(unsigned) * (k + 1));
  std::cout << "filesize " << fileSize << std::endl;
  ftruncate(fd, fileSize);
  buf = (char*) mmap(NULL, fileSize, PROT_WRITE, MAP_SHARED, fd, 0);

  if (buf == MAP_FAILED)
    std::cout << "error opening buffer" << std::endl;
  std::cout << "filesize " << fileSize << std::endl;
//	std::memcpy(buf, &k, sizeof(unsigned));

#pragma omp parallel for schedule(static, 65536)
  for (size_t i = 0; i < num; i++) {
    char* reader = buf + (i * (k + 1) * sizeof(unsigned));
    std::memcpy(reader, &k, sizeof(unsigned));
    std::memcpy(reader + sizeof(unsigned), final_graph_[i].data(),
                k * sizeof(unsigned));
  }
  std::cout << "here" << std::endl;
  int val = munmap(buf, fileSize);
  close(fd);
  if (val != 0)
    std::cout << "ERROR unmapping. CHECK!" << std::endl;
  std::cout << "Saved EFANNA graph. Set ep_ to 0" << std::endl;
}

int main(int argc, char** argv) {
  srand(time(NULL));
  if (argc != 5) {
    std::cout << argv[0] << "nn_graph_path d_nn d_rnd save_graph_file"
              << std::endl;
    exit(-1);
  }
  float* data_load = NULL;
  //  size_t points_num = std::atol(argv[2]);
  // unsigned  dim;
  //  open_linux_mmapped_file_handle(argv[1], data_load, points_num, dim);
  // data_ = data_load;
  // dimension_ = dim;
  // num = points_num;

  //  std::cout << "Data loaded" << std::endl;
  CompactGraph2 final_graph_;
  unsigned      k;
  std::string   nn_graph_path(argv[1]);
  unsigned      d_nn = (unsigned) atoi(argv[2]);
  unsigned      d_rnd = (unsigned) atoi(argv[3]);
  Load_and_truncate_nn_graph(nn_graph_path.c_str(), final_graph_, k, d_nn,
                             d_rnd);

  //  data_load = NSG::data_align(data_load, points_num, dim);//one must
  // align the data before build
  //  NSG::IndexNSG index(dim, points_num, NSG::L2, nullptr);

  //  NSG::Parameters paras;
  //  paras.Set<unsigned>("L", L);
  //  paras.Set<unsigned>("R", R);
  //  paras.Set<unsigned>("C", C);
  //  paras.Set<std::string>("nn_graph_path", nn_graph_path);
  //  std::cout << "Params set" << std::endl;

  //  index.Build(points_num, data_load, paras);
  //  auto e = std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double> diff = e - s;
  unsigned new_deg = d_nn + d_rnd;
  Save_nn_graph(argv[4], final_graph_, new_deg);

  return 0;
}
