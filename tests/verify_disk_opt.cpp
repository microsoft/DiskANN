#include <efanna2e/util.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#define SECTOR_LEN 4096

void load_nsg(const char *filename, std::vector<std::vector<unsigned>> &graph) {
  std::ifstream in(filename, std::ios::binary);
  unsigned      width_u32, medoid_u32;
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

_u64 read_header(const char *filename, _u64 &npts, _u64 &ndims, _u64 &medoid,
                 float &scale_factor, _u64 *&sizes) {
  _u64          read_size = 0;
  std::ifstream reader(filename, std::ios::binary);
  reader.read((char *) &npts, sizeof(_u64));
  read_size += sizeof(_u64);
  reader.read((char *) &ndims, sizeof(_u64));
  read_size += sizeof(_u64);
  reader.read((char *) &medoid, sizeof(_u64));
  read_size += sizeof(_u64);
  reader.read((char *) &scale_factor, sizeof(float));
  read_size += sizeof(float);

  sizes = new _u64[npts];
  reader.read((char *) sizes, npts * sizeof(_u64));
  read_size += npts * sizeof(_u64);
  std::cout << "Header size: " << read_size << std::endl;
  std::cout << "npts: " << npts << std::endl;
  std::cout << "ndims: " << ndims << std::endl;
  std::cout << "medoid: " << medoid << std::endl;
  std::cout << "scale_factor: " << scale_factor << std::endl;
  reader.close();
  return read_size;
}

void verify_nbrs(std::ifstream &                           reader,
                 const std::vector<std::vector<unsigned>> &nsg, _u64 idx,
                 _u64 *offsets, _u64 *sizes, char *buf) {
  reader.seekg(offsets[idx], std::ios::beg);
  reader.read(buf, sizes[idx]);
  unsigned  nnbrs = *(unsigned *) buf;
  unsigned *nbrs = ((unsigned *) buf + 1);
  std::cout << "idx: " << idx << ", offset: " << offsets[idx]
            << ", size: " << sizes[idx] << ", nnbrs: " << nnbrs << "\n";
  assert(nnbrs == (unsigned) nsg[idx].size());
  // verify neighbors
  if (memcmp(nsg[idx].data(), nbrs, nnbrs * sizeof(unsigned))) {
    for (_u64 i = 0; i < nnbrs; i++) {
      std::cout << "i: " << i << ", nsg[idx][i]: " << nsg[idx][i]
                << ", nbrs[i]: " << nbrs[i] << "\n";
    }
    assert(false);
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << argv[0] << " nsg_file composite_nsg" << std::endl;
    exit(-1);
  }

  std::vector<std::vector<unsigned>> nsg;
  load_nsg(argv[1], nsg);

  _u64  npts, ndims, medoid;
  _u64 *sizes;
  float scale_factor;
  _u64  header_size =
      read_header(argv[2], npts, ndims, medoid, scale_factor, sizes);
  _u64  first_off = ROUND_UP(header_size, SECTOR_LEN);
  _u64 *offs = new _u64[npts];
  offs[0] = first_off;
  for (_u64 i = 0; i < npts - 1; i++) {
    offs[i + 1] = offs[i] + sizes[i];
  }
  char *        buf = new char[1048576];
  std::ifstream reader(argv[2], std::ios::binary);
  for (_u64 i = 0; i < npts - 1; i++) {
    verify_nbrs(reader, nsg, i, offs, sizes, buf);
  }
  return 0;
}
