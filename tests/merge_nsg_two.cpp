
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

void save_fvecs(const char* filename, float* data, unsigned num,
                unsigned dim) {  // load data with sift10K pattern
  std::ofstream in(filename, std::ios::binary);
  //  if (!in.is_open()) {
  //    std::cout << "open file error" << std::endl;
  //    exit(-1);
  //  }
  //  in.write((char*) &dim, 4);
  //  std::cout << "data dimension: " << dim << std::endl;
  //  in.seekg(0, std::ios::end);
  //  std::ios::pos_type ss = in.tellg();

  //  size_t fsize = (size_t) ss;
  //  num = (unsigned) (fsize / (dim + 1) / 4);
  //  std::cout << "Reading " << num << " points" << std::endl;
  //  data = new float[(size_t) num * (size_t) dim];

  //  in.seekg(0, std::ios::beg);
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
  std::cout << "Reading " << num << " points" << std::endl;
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
  if ((argc != 7)) {
    std::cout << argv[0] << ": base_num_shards hier_num_shards nsg_prefix "
                            "nsg_suffix base_prefix output_file"
              << std::endl;
    exit(-1);
  }

  size_t      base_num_shards = std::atoi(argv[1]);
  size_t      hier_num_shards = std::atoi(argv[2]);
  size_t      num_shards = base_num_shards + hier_num_shards;
  std::string nsg_prefix(argv[3]);
  std::string nsg_suffix(argv[4]);
  std::string base_prefix(argv[5]);
  std::string output_file(argv[6]);
  float*      base_data;
  float*      check_base;

  std::vector<std::vector<unsigned>>* nsgs =
      new std::vector<std::vector<unsigned>>[num_shards];
  unsigned** renaming_ids = new unsigned*[num_shards];
  unsigned*  num_pts_in_shard = new unsigned[num_shards];
  float**    shard_base_data = new float*[num_shards];
  unsigned*  eps = new unsigned[num_shards];
  unsigned*  widths = new unsigned[num_shards];
  size_t     total_num_pts = 0;
  //  unsigned final_ep;
  size_t   dim;
  unsigned tmp_dim;

  size_t num_pts_in_base = 0;
  //#pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < num_shards; i++) {
    std::string cur_nsg_file = nsg_prefix + std::to_string(i) + nsg_suffix;
    nsgs[i] = load_nsg(cur_nsg_file.c_str(), widths[i], eps[i]);
    std::string cur_renaming_file =
        base_prefix + std::to_string(i) + "_ids.ivecs";
    //   std::string cur_base_file = base_prefix + std::to_string(i) + ".fvecs";
    load_ivecs(cur_renaming_file.c_str(), renaming_ids[i], num_pts_in_shard[i],
               tmp_dim);
    std::cout << "loaded renaming ivecs for " << num_pts_in_shard[i]
              << " points with " << tmp_dim << " dim \n";
//    load_fvecs(cur_base_file.c_str(), shard_base_data[i], num_pts_in_shard[i],
//               tmp_dim);
//#pragma omp critical
//    dim = tmp_dim;
//    std::cout << "loaded base for " << num_pts_in_shard[i] << " points in "
//              << dim << "dim \n";
#pragma omp critical
    total_num_pts += num_pts_in_shard[i];
    if (i < base_num_shards) {
#pragma omp critical
      num_pts_in_base += num_pts_in_shard[i];
    }
  }

  std::cout << "Loaded total of " << total_num_pts << "points and " << num_pts_in_base <<"base points \n";

  //  size_t final_base_num_pts = total_num_pts;
  //  base_data = new float[(total_num_pts) *dim];

  unsigned                           final_ep = eps[num_shards - 1];
  unsigned                           final_width = 0;
  std::vector<std::vector<unsigned>> final_graph;
  final_graph.resize(num_pts_in_base);

#pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < base_num_shards; i++) {
    for (size_t j = 0; j < num_pts_in_shard[i]; j++) {
      //      std::memcpy(base_data + (size_t) renaming_ids[i][j] * dim,
      //                 shard_base_data[i] + j * dim, dim * 4);

      final_graph[renaming_ids[i][j]].resize(nsgs[i][j].size());
      //      final_width =
      //          final_width > nsgs[i][j].size() ? final_width :
      //          nsgs[i][j].size();

      for (size_t k = 0; k < nsgs[i][j].size(); k++)
        final_graph[renaming_ids[i][j]][k] = renaming_ids[i][nsgs[i][j][k]];

      final_width = final_width > final_graph[renaming_ids[i][j]].size()
                        ? final_width
                        : final_graph[renaming_ids[i][j]].size();
    }
  }

  std::cout << "max degree after base graphs " << final_width << std::endl;

  for (size_t i = base_num_shards; i < num_shards; i++) {
    for (size_t j = 0; j < num_pts_in_shard[i]; j++) {
      //    if (final_graph[renaming_ids[num_shards - 1][j]].size() > 48)
      //      std::cerr << "Error1! "
      //                << final_graph[renaming_ids[num_shards - 1][j]].size()
      //                <<
      //                "\n";
      //    if (nsgs[num_shards - 1][j].size() > 48)
      //      std::cerr << "Error2! \n";
      for (size_t k = 0; k < nsgs[i][j].size(); k++)
        if (std::find(final_graph[renaming_ids[i][j]].begin(),
                      final_graph[renaming_ids[i][j]].end(),
                      renaming_ids[i][nsgs[i][j][k]]) ==
            final_graph[renaming_ids[i][j]].end())
          final_graph[renaming_ids[i][j]].push_back(
              renaming_ids[i][nsgs[i][j][k]]);

      final_width = final_width > final_graph[renaming_ids[i][j]].size()
                        ? final_width
                        : final_graph[renaming_ids[i][j]].size();
    }
  }

  /*  unsigned tmp_total_pts;
    load_fvecs("/mnt/SIFT1M/sift_base.fvecs", check_base, tmp_total_pts,
  tmp_dim);
    for (size_t i = 0; i < total_num_pts; i++)
      for (size_t j = 0; j < dim; j++)
        if (check_base[i * dim + j] != base_data[i * dim + j]) {
          std::cout << "value " << i << " " << j << " not matching\n";
          return -1;
        }


    bool**   is_inner = new bool*[num_shards];
    size_t total_inner_pts = 0;
    for (size_t i = 0; i < num_shards; i++) {
      std::string nsg_inner_file = nsg_prefix + std::to_string(i) + nsg_suffix +
  ".inner";
      std::ifstream in(nsg_inner_file.c_str(), std::ios::binary);
      if (!in.is_open()) {
        std::cout << "inner file does not exist. generating at random." <<
  std::endl;
  //      exit(-1);
        is_inner[i] = new bool[num_pts_in_shard[i]];

      std::random_device               rd;
      std::mt19937                     gen(rd());
      std::uniform_real_distribution<> dis(0, 1);
        float p_val = 0.01;
        for (size_t j = 0; j < num_pts_in_shard[i]; j++) {
     is_inner[i][j] = false;
          float candidate = dis(gen);
          if (candidate < p_val ) {
            is_inner[i][j] = true;
          }
        }
      is_inner[i][eps[i]] = true;
      }
      else {
      in.seekg(0, std::ios::end);
      std::ios::pos_type ss = in.tellg();

      in.seekg(0, std::ios::beg);
      size_t fsize = (size_t) ss;
      if (fsize != num_pts_in_shard[i])
        std::cout << "Error in inner vertices size mismatch \n";
      else
        std::cout << "loaded inner file for subshard " << i << " over " << fsize
                  << " points \n";

      is_inner[i] = new bool[num_pts_in_shard[i]];

      in.read((char*) is_inner[i], num_pts_in_shard[i]);
      in.close();
  }
      for (size_t j = 0; j < num_pts_in_shard[i]; j++)
        if (is_inner[i][j])
          total_inner_pts++;
    }

    std::cout << "total inner vertices number " << total_inner_pts << "\n";

    unsigned* inner_renaming = new unsigned[total_inner_pts];
    size_t  inner_iter = 0;
    float*    inner_base = new float[total_inner_pts * dim];

    for (size_t i = 0; i < num_shards; i++)
      for (size_t j = 0; j < num_pts_in_shard[i]; j++)
        if (is_inner[i][j]) {
          inner_renaming[inner_iter] = renaming_ids[i][j];
          std::memcpy(inner_base + inner_iter * dim,
                      base_data + ((size_t) (renaming_ids[i][j])) * (size_t)
  dim, 4 * dim);
          inner_iter++;
        }

    std::string   output_inner_nsg = output_file + "_inner.nsg";
    std::ifstream file(output_inner_nsg.c_str(), std::ios::binary);
    unsigned      inner_width, inner_ep;
    if (file.is_open()) {
      std::cout << "Inner NSG found. Merging it with NSGs\n";
      std::vector<std::vector<unsigned>> inner_nsg =
          load_nsg(output_inner_nsg.c_str(), inner_width, inner_ep);
      final_ep = inner_renaming[inner_ep];

      for (size_t j = 0; j < total_inner_pts; j++) {
        for (size_t k = 0; k < inner_nsg[j].size(); k++)
          if (std::find(final_graph[inner_renaming[j]].begin(),
                        final_graph[inner_renaming[j]].end(),
                        inner_renaming[inner_nsg[j][k]]) ==
              final_graph[inner_renaming[j]].end())
            final_graph[inner_renaming[j]].push_back(
                inner_renaming[inner_nsg[j][k]]);

        final_width = final_width > final_graph[inner_renaming[j]].size()
                          ? final_width
                          : final_graph[inner_renaming[j]].size();
      }

    } else
      std::cout << "Inner NSG not found. Skipping the step\n";
  */

  //  std::string output_base = output_file + ".fvecs";
  //  save_fvecs(output_base.c_str(), base_data, final_base_num_pts, dim);

  // std::string output_inner_base = output_file + "_inner.fvecs";
  // save_fvecs(output_inner_base.c_str(), inner_base, total_inner_pts, dim);

  std::string output_nsg = output_file + ".nsg";
  save_nsg(output_nsg.c_str(), final_graph, final_width, final_ep);
  std::cout << "Max degree: " << final_width << std::endl;
  return 0;
}
