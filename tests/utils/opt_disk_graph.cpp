#include <utils.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include "cached_io.h"

#define SECTOR_LEN 4096

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << argv[0] << " nsg_in nsg_out data_bin" << std::endl;
    exit(-1);
  }
  unsigned npts, ndims;
  _s8 *    data = nullptr;
  std::cout << "Embedding node coords with its nhood" << std::endl;
  size_t npts_64, ndims_64;
  NSG::load_bin<_s8>(argv[3], data, npts_64, ndims_64);
  npts = npts_64;
  ndims = ndims_64;

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
    for (_u64 sector_node_id = 0;
         sector_node_id < nnodes_per_sector && cur_node_id < nnodes;
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
