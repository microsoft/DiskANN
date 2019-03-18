#include <efanna2e/util.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#define SECTOR_LEN 4096

void load_nsg(const char *filename, std::vector<std::vector<unsigned>> &nsg,
              _u64 &medoid) {
  std::ifstream in(filename, std::ios::binary);
  unsigned      width_u32, medoid_u32;
  in.read((char *) &width_u32, sizeof(unsigned));  // ignored
  in.read((char *) &medoid_u32, sizeof(unsigned));
  medoid = (_u64) medoid_u32;
  // width=100;
  while (!in.eof()) {
    unsigned k;
    in.read((char *) &k, sizeof(unsigned));
    if (in.eof())
      break;
    std::vector<unsigned> tmp(k);
    in.read((char *) tmp.data(), k * sizeof(unsigned));
    nsg.push_back(tmp);
  }
  in.close();
}

void write_nsg(const char* filename, const std::vector<std::vector<unsigned>> &nsg, const _u64 medoid){
  
  std::ofstream writer(filename, std::ios::binary);
  _u64 nnodes = nsg.size();
  writer.write((char*)&nnodes, sizeof(_u64));
  writer.write((char*)&medoid, sizeof(_u64));

_u64 max_degree;
  for (auto nhood : nsg) {
    max_degree = std::max((size_t) max_degree, nhood.size());
  }

  _u64 max_node_len = (max_degree + 1)*sizeof(unsigned);
  writer.write((char*)max_node_len, sizeof(_u64));
  std::cout << "max write-per-node: " << max_node_len << "B" << std::endl;

  _u64 nnodes_per_sector = SECTOR_LEN / max_node_len;
  writer.write((char*)nnodes_per_sector, sizeof(_u64));
  std::cout << "# nodes per sector: " << nnodes_per_sector << std::endl;

  _u64 nsectors_per_blk = 65536;
  _u64 nnodes_per_blk = nsectors_per_blk * nnodes_per_sector;
  _u64 nblks = ROUND_UP(nnodes, nnodes_per_blk) / nnodes_per_blk;
  writer.seekp(SECTOR_LEN, std::ios::beg);
  std::cout << "# blocks: " << nblks << std::endl;

  // buf for each block
  char* blk_buf = new char[nsectors_per_blk * SECTOR_LEN];
  for(_u64 b = 0;b<nblks;b++){
    _u64 bstart = b * nnodes_per_blk;
    _u64 bend = std::min(nnodes, nnodes_per_blk * (b+1));
    _u64 bsize = bend - bstart;

    // clear blk_buf
    memset(blk_buf, 0, nsectors_per_blk * SECTOR_LEN / sizeof(unsigned));

    // copy nsg blk to blk_buf
    char* sector_buf = blk_buf;
    for(_u64 i = bstart; i<bend;i++){
      _u64 j = i-bstart;
      unsigned nhood_size = nsg[i].size();

      // choose new sector if previous sector is exhausted
      if (j % nnodes_per_sector == 0){
        sector_buf = blk_buf + (j / nnodes_per_sector) * SECTOR_LEN ;
      }
      char* node_buf = (sector_buf + (j % nnodes_per_sector)*max_node_len);
      *(unsigned*)node_buf = nhood_size;
      const unsigned* nhood = nsg[i].data();
      memcpy(node_buf + sizeof(unsigned), nhood, nhood_size * sizeof(unsigned));
    }

    // write buf to disk
    writer.write(blk_buf, nsectors_per_blk * SECTOR_LEN);
    std::cout << "Block #" << b << " written, # nodes: " << bsize << std::endl;
  }
  writer.close();
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << argv[0]
              << " nsg_in nsg_out"
              << std::endl;
    exit(-1);
  }

  std::vector<std::vector<unsigned>> nsg;
  _u64                               medoid;
  
  load_nsg(argv[1], nsg, medoid);
  std::cout << "NSG loaded\n";

  write_nsg(argv[2], nsg, medoid);
  std::cout << "Output file written\n";
  return 0;
}