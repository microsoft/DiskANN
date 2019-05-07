
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
    unsigned    tmp_dim;
    std::string cur_renaming_file =
        base_prefix + std::to_string(i) + "_ids.ivecs";
    load_ivecs(cur_renaming_file.c_str(), renaming_ids[i], num_pts_in_shard[i],
               tmp_dim);
    std::cout << "loaded renaming ivecs for " << num_pts_in_shard[i]
              << " points with " << tmp_dim << " dim \n";
    std::cout << "Entry point " << i << ": " << renaming_ids[i][eps[i]]
              << std::endl;
#pragma omp critical
    total_num_pts += num_pts_in_shard[i];
  }

  std::cout << "Loaded total of " << total_num_pts << "points \n";

  size_t   final_base_num_pts = total_num_pts;
  unsigned tmp_final_base_num_pts, tmp_base_dim;
  //  base_data;  = new float[(total_num_pts) *dim];
  if (std::string(argv[5]).find(std::string("bvecs")) != std::string::npos)
    load_bvecs(argv[5], base_data, final_base_num_pts, dim);
  else {
    load_fvecs(argv[5], base_data, tmp_final_base_num_pts, tmp_base_dim);
    final_base_num_pts = tmp_final_base_num_pts;
    dim = tmp_base_dim;
  }

  if (final_base_num_pts != total_num_pts)
    std::cerr << "Error! mismatch in number of points " << std::endl;
  std::cout << "loaded full base \n";
  std::random_device               rd;
  std::mt19937                     gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  float                            p_val = std::atof(argv[7]);
  bool*                            is_inner = new bool[total_num_pts];

  for (size_t i = 0; i < total_num_pts; i++) {
    is_inner[i] = false;
  }

  bool inner_files_flag = true;
  for (size_t i = 0; i < num_shards; i++) {
    std::string nsg_inner_file =
        nsg_prefix + std::to_string(i) + nsg_suffix + ".inner";
    std::ifstream in(nsg_inner_file.c_str(), std::ios::binary);
    if (!in.is_open()) {
      std::cout
          << "inner file does not exist. generating all inner data at random."
          << std::endl;
      inner_files_flag = false;
      break;
      //      exit(-1);
      //      is_inner[i] = new bool[num_pts_in_shard[i]];
    } else {
      in.seekg(0, std::ios::end);
      std::ios::pos_type ss = in.tellg();

      in.seekg(0, std::ios::beg);
      size_t fsize = (size_t) ss;
      if (fsize != num_pts_in_shard[i])
        std::cout << "Error in inner vertices size mismatch \n";
      else
        std::cout << "loaded inner file for subshard " << i << " over " << fsize
                  << " points \n";

      //      is_inner[i] = new bool[num_pts_in_shard[i]];
      bool* is_inner_shard = new bool[num_pts_in_shard[i]];
      in.read((char*) is_inner_shard, num_pts_in_shard[i]);
      in.close();

      for (size_t j = 0; j < num_pts_in_shard[i]; j++)
        if (is_inner_shard[j])
          is_inner[renaming_ids[i][j]] = true;

      delete[] is_inner_shard;
    }
  }

  if (inner_files_flag == false) {
    std::cout << "inner files regenerating at random" << std::endl;
    for (size_t i = 0; i < total_num_pts; i++) {
      is_inner[i] = false;
      float candidate = dis(gen);
      if (candidate < p_val) {
        is_inner[i] = true;
      }
    }
  }

  //  for (size_t i = 0; i < num_shards; i++) {
  //    is_inner[renaming_ids[i][eps[i]]] = true;
  //  }

  size_t total_inner = 0;
  for (size_t i = 0; i < total_num_pts; i++) {
    total_inner += is_inner[i];
  }

  std::cout << "total inner count " << total_inner << std::endl;

  float*    inner_base = new float[total_inner * dim];
  unsigned* inner_ids = new unsigned[total_inner];
  size_t    inner_iter = 0;
  std::cout << "total num. points " << total_num_pts << std::endl;
  for (size_t i = 0; i < total_num_pts; i++) {
    if (is_inner[i]) {
      std::memcpy(inner_base + inner_iter * dim, base_data + i * dim, dim * 4);
      inner_ids[inner_iter] = i;
      inner_iter++;
    }
  }
  std::cout << "final inner_iter count: " << inner_iter;

  std::string output_inner_base = output_file + ".fvecs";
  save_fvecs(output_inner_base.c_str(), inner_base, total_inner, dim);

  std::string output_inner_ids = output_file + "_ids.ivecs";
  save_ivecs(output_inner_ids.c_str(), inner_ids, total_inner, 1);
  return 0;
}
