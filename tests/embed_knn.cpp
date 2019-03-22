#include <efanna2e/util.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

void blk_load_nsg(std::ifstream &in, std::vector<std::vector<unsigned>> &nsg,
                  _u64 blk_size) {
  nsg.clear();
  while (!in.eof() && nsg.size() < blk_size) {
    unsigned k;
    in.read((char *) &k, sizeof(unsigned));
    if (in.eof())
      break;
    std::vector<unsigned> tmp(k);
    in.read((char *) tmp.data(), k * sizeof(unsigned));
    nsg.push_back(tmp);
  }
}

void blk_write_nsg(std::ofstream &                     out,
                   std::vector<std::vector<unsigned>> &nsg) {
  for (auto &nhood : nsg) {
    unsigned k = nhood.size();
    out.write((char *) &k, sizeof(unsigned));
    out.write((char *) nhood.data(), k * sizeof(unsigned));
  }
}

void load_knng(const char *filename, std::vector<std::vector<unsigned>> &knng) {
  std::ifstream in(filename, std::ios::binary);
  std::cout << "Reading knng" << std::endl;
  // width=100;
  while (!in.eof()) {
    unsigned k;
    in.read((char *) &k, sizeof(unsigned));
    if (in.eof())
      break;
    std::vector<unsigned> tmp(k);
    in.read((char *) tmp.data(), k * sizeof(unsigned));
    knng.push_back(tmp);
  }
  in.close();
}

void augment(std::vector<unsigned> &nsg, std::vector<unsigned> &knn) {
  nsg.insert(nsg.end(), knn.begin(), knn.end());
  auto last = std::unique(nsg.begin(), nsg.end());
  nsg.erase(last, nsg.end());
}

void block_augment_graph(std::vector<std::vector<unsigned>> &graph,
                         std::vector<std::vector<unsigned>> &knng,
                         _u64                                blk_start) {
  _u64      blk_size = graph.size();
#pragma omp parallel for
  for (_u64 i = 0; i < blk_size; i++) {
    std::vector<unsigned> &nsg = graph[i];
    std::vector<unsigned> &knn = knng[blk_start + i];
    augment(nsg, knn);
  }
}

void augment_write_nsg(const char *nsg_in, const char *knng_in,
                       const char *nsg_out) {
  unsigned width, medoid;
  // read and write header first
  std::ifstream nsg_reader(nsg_in, std::ios::binary),
      knng_reader(knng_in, std::ios::binary);
  nsg_reader.read((char *) &width, sizeof(unsigned));
  nsg_reader.read((char *) &medoid, sizeof(unsigned));
  std::cout << "NSG Header: width = " << width << ", medoid = " << medoid
            << std::endl;
  std::ofstream nsg_writer(nsg_out, std::ios::binary);
  nsg_writer.write((char *) &width, sizeof(unsigned));
  nsg_writer.write((char *) &medoid, sizeof(unsigned));

  _u64                               blk_size = 1048576;
  std::vector<std::vector<unsigned>> nsg, knng;
  std::cout << "Loading knng" << std::endl;
  load_knng(knng_in, knng);
  _u64 n_nodes = knng.size();
  _u64 n_blks = ROUND_UP(n_nodes, blk_size) / blk_size;
  std::cout << "# nodes: " << n_nodes << ", # blks: " << n_blks << std::endl;
  for (_u64 i = 0; i < n_blks; i++) {
    _u64 blk_start = i * blk_size;
    nsg.clear();
    std::cout << "Block #" << i << std::endl;
    blk_load_nsg(nsg_reader, nsg, blk_size);
    block_augment_graph(nsg, knng, blk_start);
    blk_write_nsg(nsg_writer, nsg);
  }
  nsg_reader.close();
  nsg_writer.close();
  nsg.clear();
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << argv[0] << " nsg_in knng_in[ivecs] nsg_out" << std::endl;
    exit(-1);
  }
  augment_write_nsg(argv[1], argv[2], argv[3]);
  std::cout << "Output file written\n";
  return 0;
}