
//
// Created by 付聪 on 2017/6/21.
//
#include <fcntl.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
//#include <mkl.h>
#include <omp.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <queue>
#include <random>
#include <vector>

unsigned get_num_fvecs(
    const char* filename) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  unsigned dim, num;
  in.read((char*) &dim, 4);
  std::cout << "data dimension: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();

  size_t fsize = (size_t) ss;
  num = (unsigned) (fsize / (dim + 1) / 4);
  in.close();
}

unsigned get_nsg_ep(const char* filename) {
  std::ifstream in(filename, std::ios::binary);
  unsigned      width;
  unsigned      ep_;
  in.read((char*) &width, sizeof(unsigned));
  in.read((char*) &ep_, sizeof(unsigned));
  in.close();
  std::cout << "read EP " << ep_ << std::endl;
  return ep_;
}

void load_bvecs(const char* filename, float*& data, size_t& num, size_t& dim) {
  size_t new_dim = 0;
  char*  buf;
  int    fd;
  fd = open(filename, O_RDONLY);
  if (!(fd > 0)) {
    std::cerr << "Data file " << filename
              << " not found. Program will stop now." << std::endl;
    assert(false);
  }
  struct stat sb;
  fstat(fd, &sb);
  off_t fileSize = sb.st_size;
  assert(sizeof(off_t) == 8);

  buf = (char*) mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
  assert(buf);
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

  num = (size_t)(fileSize / (dim + 4));
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

void load_fvecs(const char* filename, float*& data, unsigned& num,
                unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*) &dim, 4);
  std::cout << "data dimension: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();

  size_t fsize = (size_t) ss;
  num = (unsigned) (fsize / (dim + 1) / 4);
  std::cout << "Reading " << num << " points" << std::endl;
  data = new float[(size_t) num * (size_t) dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*) (data + i * dim), dim * 4);
  }
  in.close();
}

void save_ivecs(const char* filename, unsigned* data, unsigned num,
                unsigned dim) {  // load data with sift10K pattern
  std::ofstream in(filename, std::ios::binary);
  for (size_t i = 0; i < num; i++) {
    in.write((char*) &dim, 4);
    in.write((char*) (data + i * dim), dim * 4);
  }
  in.close();
}

void save_fvecs(const char* filename, float* data, unsigned num,
                unsigned dim) {  // load data with sift10K pattern
  std::ofstream in(filename, std::ios::binary);
  for (size_t i = 0; i < num; i++) {
    in.write((char*) &dim, 4);
    in.write((char*) (data + i * dim), dim * 4);
  }
  in.close();
}

void load_ivecs(const char* filename, unsigned*& data, unsigned& num,
                unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*) &dim, 4);
  std::cout << "data dimension: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();

  size_t fsize = (size_t) ss;
  num = (unsigned) (fsize / (dim + 1) / 4);
  std::cout << "Reading " << num << " points in dimension" << dim << std::endl;
  data = new unsigned[(size_t) num * (size_t) dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*) (data + i * dim), dim * 4);
  }
  in.close();
}

void save_nsg(const char*                        filename,
              std::vector<std::vector<unsigned>> final_graph_, unsigned width,
              unsigned ep_) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  //    assert(final_graph_.size() == nd_);
  unsigned  nd_ = final_graph_.size();
  long long total_gr_edges = 0;
  out.write((char*) &width, sizeof(unsigned));
  out.write((char*) &ep_, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    unsigned GK = (unsigned) final_graph_[i].size();
    out.write((char*) &GK, sizeof(unsigned));
    out.write((char*) final_graph_[i].data(), GK * sizeof(unsigned));
    total_gr_edges += GK;
  }
  out.close();

  std::cout << "Avg degree: " << ((float) total_gr_edges) / ((float) nd_)
            << std::endl;
}

std::vector<std::vector<unsigned>> load_nsg(const char* filename,
                                            unsigned& width, unsigned& ep_) {
  std::vector<std::vector<unsigned>> final_graph_;
  std::ifstream                      in(filename, std::ios::binary);

  in.read((char*) &width, sizeof(unsigned));
  in.read((char*) &ep_, sizeof(unsigned));
  // width=100;
  size_t   cc = 0;
  unsigned nodes = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char*) &k, sizeof(unsigned));
    if (in.eof())
      break;
    cc += k;
    ++nodes;
    std::vector<unsigned> tmp(k);
    in.read((char*) tmp.data(), k * sizeof(unsigned));
    final_graph_.push_back(tmp);

    if (nodes % 5000000 == 0)
      std::cout << "Loaded " << nodes << " nodes, and " << cc << " neighbors"
                << std::endl;
  }
  std::cout << "Loaded " << nodes << " nodes, and " << cc << " neighbors"
            << std::endl;
  return final_graph_;
  //    cc /= nd_;

  //    Load_Inner_Vertices(filename);
  // std::cout<<cc<<std::endl;
}

int main(int argc, char** argv) {
  if ((argc != 8)) {
    std::cout << argv[0]
              << ": num_shards nsg_prefix nsg_suffix "
                 "shard_base_prefix full_base_file output_file_prefix p_val"
              << std::endl;
    exit(-1);
  }

  size_t      num_shards = std::atoi(argv[1]);
  std::string nsg_prefix(argv[2]);
  std::string nsg_suffix(argv[3]);
  std::string base_prefix(argv[4]);
  std::string output_file(argv[6]);
  float*      base_data;
  float*      check_base;

  unsigned*  num_pts_in_shard = new unsigned[num_shards];
  unsigned** renaming_ids = new unsigned*[num_shards];
  unsigned*  eps = new unsigned[num_shards];
  size_t     total_num_pts = 0;
  size_t     dim;

  //#pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < num_shards; i++) {
    std::string cur_nsg_file = nsg_prefix + std::to_string(i) + nsg_suffix;
    std::cout << cur_nsg_file << std::endl;
    eps[i] = get_nsg_ep(cur_nsg_file.c_str());
    std::cout << "EP: " << eps[i] << std::endl;
  }

  std::string output_inner_ids = output_file + "_medoids.ivecs";
  save_ivecs(output_inner_ids.c_str(), eps, num_shards, 1);
  return 0;
}
