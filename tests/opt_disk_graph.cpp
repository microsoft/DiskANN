#include <cached_io.h>
#include <efanna2e/util.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#define SECTOR_LEN 4096

/*
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
    if (nsg.size() % 5000000 == 0) {
      std::cout << "Loaded " << nsg.size() << " nodes" << std::endl;
    }
  }
  in.close();
}

// layout: [UINT32 NHOOD]..[PADDING]..[UINT32 NHOOD]...
void write_plain_nsg(const char *filename, NSG::OneShotNSG &nsg) {
  std::ofstream writer(filename, std::ios::binary);
  _u64          nnodes = nsg.nnodes;
  _u64          medoid = nsg.medoid;
  writer.write((char *) &nnodes, sizeof(_u64));
  writer.write((char *) &medoid, sizeof(_u64));

  _u64 max_degree = 0;
  for (_u64 i = 0; i < nsg.size(); i++) {
    max_degree = max((size_t) max_degree, nsg.nnbrs(i));
  }
  std::cout << "max degree: " << max_degree << std::endl;

  _u64 max_node_len = (max_degree + 1) * sizeof(unsigned);
  writer.write((char *) &max_node_len, sizeof(_u64));
  std::cout << "max write-per-node: " << max_node_len << "B" << std::endl;

  _u64 nnodes_per_sector = SECTOR_LEN / max_node_len;
  writer.write((char *) &nnodes_per_sector, sizeof(_u64));
  std::cout << "# nodes per sector: " << nnodes_per_sector << std::endl;

  _u64 nsectors_per_blk = 65536;
  _u64 nnodes_per_blk = nsectors_per_blk * nnodes_per_sector;
  _u64 nblks = ROUND_UP(nnodes, nnodes_per_blk) / nnodes_per_blk;
  writer.seekp(SECTOR_LEN, std::ios::beg);
  std::cout << "# blocks: " << nblks << std::endl;

  // buf for each block
  char *blk_buf = new char[nsectors_per_blk * SECTOR_LEN];
  char *sector_buf, *node_buf;
  for (_u64 b = 0; b < nblks; b++) {
    _u64 bstart = b * nnodes_per_blk;
    _u64 bend = min(nnodes, nnodes_per_blk * (b + 1));
    _u64 bsize = bend - bstart;

    // clear blk_buf
    memset(blk_buf, 0, nsectors_per_blk * SECTOR_LEN / sizeof(unsigned));

    // copy nsg blk to blk_buf
    for (_u64 i = bstart; i < bend; i++) {
      unsigned nhood_size = nsg.nnbrs(i);
      _u64     sector_no = i / nnodes_per_sector;
      _u64     sector_off = (i % nnodes_per_sector) * max_node_len;

      // std::cout << "node : " << i << ", sector: " << sector_no ;
      // std::cout<< ", sector_off: " << sector_off << std::endl;

      // get sector buf and node buf
      sector_buf = blk_buf + (sector_no % nsectors_per_blk) * SECTOR_LEN;
      node_buf = sector_buf + sector_off;
      *((unsigned *) node_buf) = nhood_size;
      const unsigned *nhood = nsg.data(i);
      memcpy(node_buf + sizeof(unsigned), nhood, nhood_size * sizeof(unsigned));
    }

    // write buf to disk
    writer.write(blk_buf, nsectors_per_blk * SECTOR_LEN);
    std::cout << "Block #" << b << " written, # nodes: " << bsize << std::endl;
  }
  writer.close();
  delete[] blk_buf;
}

// layout: [INT8 COORDS][UINT32 NHOOD]..[PADDING]..[INT8 COORDS]...
void write_embedded_nsg(const char *filename, NSG::OneShotNSG &nsg, _s8 *data,
                        _u64 npts, _u64 ndims) {
  std::ofstream writer(filename, std::ios::binary);
  _u64          nnodes = nsg.nnodes;
  _u64          medoid = nsg.medoid;
  writer.write((char *) &nnodes, sizeof(_u64));
  writer.write((char *) &medoid, sizeof(_u64));

  _u64 max_degree = 0;
  for (_u64 i = 0; i < nsg.size(); i++) {
    max_degree = max((size_t) max_degree, nsg.nnbrs(i));
  }
  std::cout << "max degree: " << max_degree << std::endl;

  _u64 max_node_len =
      (ndims * sizeof(_s8)) + (max_degree + 1) * sizeof(unsigned);
  writer.write((char *) &max_node_len, sizeof(_u64));
  std::cout << "max write-per-node: " << max_node_len << "B" << std::endl;

  _u64 nnodes_per_sector = SECTOR_LEN / max_node_len;
  writer.write((char *) &nnodes_per_sector, sizeof(_u64));
  std::cout << "# nodes per sector: " << nnodes_per_sector << std::endl;

  _u64 nsectors_per_blk = 65536;
  _u64 nnodes_per_blk = nsectors_per_blk * nnodes_per_sector;
  _u64 nblks = ROUND_UP(nnodes, nnodes_per_blk) / nnodes_per_blk;
  writer.seekp(SECTOR_LEN, std::ios::beg);
  std::cout << "# blocks: " << nblks << std::endl;

  // buf for each block
  char *    blk_buf = new char[nsectors_per_blk * SECTOR_LEN];
  char *    sector_buf, *node_buf;
  unsigned *node_nhood_buf;
  for (_u64 b = 0; b < nblks; b++) {
    _u64 bstart = b * nnodes_per_blk;
    _u64 bend = min(nnodes, nnodes_per_blk * (b + 1));
    _u64 bsize = bend - bstart;

    // clear blk_buf
    memset(blk_buf, 0, nsectors_per_blk * SECTOR_LEN / sizeof(unsigned));

    // copy nsg blk to blk_buf
    for (_u64 i = bstart; i < bend; i++) {
      unsigned nhood_size = nsg.nnbrs(i);

      // global values
      _u64 sector_no = (i / nnodes_per_sector);
      _u64 sector_off = (i % nnodes_per_sector) * max_node_len;

      // std::cout << "node : " << i << ", sector: " << sector_no ;
      // std::cout<< ", sector_off: " << sector_off << std::endl;

      // get sector buf and node buf
      _u64 blk_sector_no = (sector_no % nsectors_per_blk);
      sector_buf = blk_buf + (blk_sector_no * SECTOR_LEN);
      node_buf = sector_buf + sector_off;
      memcpy(node_buf, data + i * ndims, ndims * sizeof(_s8));
      node_nhood_buf = (unsigned *) (node_buf + ndims * sizeof(_s8));
      *node_nhood_buf = nhood_size;
      const unsigned *nhood = nsg.data(i);
      memcpy(node_nhood_buf + 1, nhood, nhood_size * sizeof(unsigned));
    }

    // write buf to disk
    writer.write(blk_buf, nsectors_per_blk * SECTOR_LEN);
    std::cout << "Block #" << b << " written, # nodes: " << bsize << std::endl;
  }
  writer.close();
  delete[] blk_buf;
}
*/

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << argv[0] << " nsg_in nsg_out data_bin" << std::endl;
    exit(-1);
  }
  unsigned npts, ndims;
  _s8 *    data = nullptr;
  std::cout << "Embedding node coords with its nhood" << std::endl;
  NSG::load_bin<_s8>(argv[3], data, npts, ndims);

  // amount to read in one shot
  _u64 read_blk_size = 64 * 1024 * 1024;
  _u64 write_blk_size = read_blk_size;

  // create cached reader + writer
  cached_ifstream nsg_reader(argv[1], read_blk_size);
  cached_ofstream nsg_writer(argv[2], write_blk_size);

  // metadata: width, medoid
  unsigned width_u32, medoid_u32;
  nsg_reader.read((char *) &width_u32, sizeof(unsigned));
  nsg_reader.read((char *) &medoid_u32, sizeof(unsigned));

  // compute
  _u64 nnodes, medoid, max_node_len, nnodes_per_sector;
  nnodes = (_u64) npts;
  medoid = (_u64) medoid_u32;
  max_node_len =
      (((_u64) width_u32 + 1) * sizeof(unsigned)) + (ndims * sizeof(_s8));
  nnodes_per_sector = SECTOR_LEN / max_node_len;

  std::cout << "medoid: " << medoid << "B\n";
  std::cout << "max_node_len: " << max_node_len << "B\n";
  std::cout << "nnodes_per_sector: " << nnodes_per_sector << "B\n";

  // SECTOR_LEN buffer for each sector
  char *    sector_buf = new char[SECTOR_LEN];
  char *    node_buf = new char[max_node_len];
  unsigned &nnbrs = *(unsigned *) (node_buf + ndims * sizeof(_s8));
  unsigned *nhood_buf =
      (unsigned *) (node_buf + (ndims * sizeof(_s8)) + sizeof(unsigned));

  // write first sector with metadata
  *(_u64 *) sector_buf = nnodes;
  *(_u64 *) (sector_buf + sizeof(_u64)) = medoid;
  *(_u64 *) (sector_buf + 2 * sizeof(_u64)) = max_node_len;
  *(_u64 *) (sector_buf + 3 * sizeof(_u64)) = nnodes_per_sector;
  nsg_writer.write(sector_buf, SECTOR_LEN);

  _u64 n_sectors = ROUND_UP(nnodes, nnodes_per_sector) / nnodes_per_sector;
  std::cout << "# sectors: " << n_sectors << "\n";
  _u64 cur_node_id = 0;
  for (_u64 sector = 0; sector < n_sectors; sector++) {
    if (sector % 100000 == 0) {
      std::cout << "Sector #" << sector << "written\n";
    }
    memset(sector_buf, 0, SECTOR_LEN);
    for (_u64 sector_node_id = 0; sector_node_id < nnodes_per_sector;
         sector_node_id++) {
      memset(node_buf, 0, max_node_len);
      // read cur node's nnbrs
      nsg_reader.read((char *) &nnbrs, sizeof(unsigned));

      // sanity checks on nnbrs
      assert(nnbrs > 0);
      assert(nnbrs <= width_u32);

      // read node's nhood
      nsg_reader.read((char *) nhood_buf, nnbrs * sizeof(unsigned));

      // write coords of node first
      _s8 *node_coords = data + ((_u64) ndims * cur_node_id);
      memcpy(node_buf, node_coords, ndims * sizeof(_s8));

      // write nnbrs
      *(unsigned *) (node_buf + ndims * sizeof(_s8)) = nnbrs;

      // write nhood next
      memcpy(node_buf + ndims * sizeof(_s8) + sizeof(unsigned), nhood_buf,
             nnbrs * sizeof(unsigned));

      // get offset into sector_buf
      char *sector_node_buf = sector_buf + (sector_node_id * max_node_len);

      // copy node buf into sector_node_buf
      memcpy(sector_node_buf, node_buf, max_node_len);
      cur_node_id++;
    }
    // flush sector to disk
    nsg_writer.write(sector_buf, SECTOR_LEN);
  }
  delete[] sector_buf;
  delete[] node_buf;

  std::cout << "Output file written\n";
  return 0;
}
