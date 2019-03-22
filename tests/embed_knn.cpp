#include <efanna2e/util.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

void load_nsg(const char *filename, std::vector<std::vector<unsigned>> &graph,
              unsigned &width_u32, unsigned &medoid_u32) {
  std::ifstream in(filename, std::ios::binary);
  std::cout << "Reading nsg" << std::endl;
  in.read((char *) &width_u32, sizeof(unsigned));  // ignored
  in.read((char *) &medoid_u32, sizeof(unsigned));
  std::cout << "Medoid: " << medoid_u32 << std::endl;
  // width=100;
  while (!in.eof()) {
    unsigned k;
    in.read((char *) &k, sizeof(unsigned));
    if (in.eof())
      break;
    std::vector<unsigned> tmp(k);
    in.read((char *) tmp.data(), k * sizeof(unsigned));
    graph.push_back(tmp);
  }
  in.close();
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

void write_nsg(const char *filename, std::vector<std::vector<unsigned>> &graph,
               const unsigned width_u32, const unsigned medoid_u32) {
  std::ofstream out(filename, std::ios::binary);
  out.write((char *) &width_u32, sizeof(unsigned));   // ignored
  out.write((char *) &medoid_u32, sizeof(unsigned));  // ignored
  std::cout << "Medoid: " << medoid_u32 << std::endl;
  // width=100;
  for (auto &nhood : graph) {
    unsigned k = nhood.size();
    out.write((char *) &k, sizeof(unsigned));
    out.write((char *) nhood.data(), k * sizeof(unsigned));
  }
  out.close();
}

void augment(std::vector<unsigned> &nsg, std::vector<unsigned> &knn) {
  nsg.insert(nsg.end(), knn.begin(), knn.end());
  auto last = std::unique(nsg.begin(), nsg.end());
  nsg.erase(last, nsg.end());
}

void augment_graph(std::vector<std::vector<unsigned>> &graph,
                   std::vector<std::vector<unsigned>> &knng) {
  _u64      nnodes = graph.size();
#pragma omp parallel for schedule(dynamic, 65536)
  for (_u64 i = 0; i < nnodes; i++) {
    std::vector<unsigned> &nsg = graph[i];
    std::vector<unsigned> &knn = knng[i];
    augment(nsg, knn);
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << argv[0] << " nsg_in knng_in[ivecs] nsg_out" << std::endl;
    exit(-1);
  }
  unsigned                           width, medoid;
  std::vector<std::vector<unsigned>> nsg, knng;
  load_nsg(argv[1], nsg, width, medoid);
  load_knng(argv[2], knng);
  std::cout << "Starting augmentation" << std::endl;
  augment_graph(nsg, knng);
  write_nsg(argv[3], nsg, width, medoid);
  std::cout << "Output file written\n";
  return 0;
}