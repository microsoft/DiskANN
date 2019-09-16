#include <utils.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include "cached_io.h"

#define SECTOR_LEN 4096

template<typename T>
int aux_main(int argc, char **argv) {
  unsigned    npts, ndims;
  std::string base_file(argv[2]);
  std::string rand_nsg_file(argv[3]);
  std::string output_file(argv[4]);

  // amount to read or write in one shot
  _u64            read_blk_size = 64 * 1024 * 1024;
  _u64            write_blk_size = read_blk_size;
  cached_ifstream base_reader(base_file, read_blk_size);
  base_reader.read((char *) &npts, sizeof(uint32_t));
  base_reader.read((char *) &ndims, sizeof(uint32_t));

  size_t npts_64, ndims_64;
  npts_64 = npts;
  ndims_64 = ndims;

  // create cached reader + writer
  size_t          actual_file_size = get_file_size(rand_nsg_file);
  cached_ifstream nsg_reader(rand_nsg_file, read_blk_size);
  cached_ofstream nsg_writer(output_file, write_blk_size);

  // metadata: width, medoid
  unsigned width_u32, medoid_u32;
  size_t   index_file_size;

  nsg_reader.read((char *) &index_file_size, sizeof(uint64_t));
  if (index_file_size != actual_file_size) {
    std::cout << "Rand-NSG Index file size does not match expected size per "
                 "meta-data."
              << std::endl;
    exit(-1);
  }
  nsg_reader.read((char *) &width_u32, sizeof(unsigned));
  nsg_reader.read((char *) &medoid_u32, sizeof(unsigned));

  // compute
  _u64 medoid, max_node_len, nnodes_per_sector;
  npts_64 = (_u64) npts;
  medoid = (_u64) medoid_u32;
  max_node_len =
      (((_u64) width_u32 + 1) * sizeof(unsigned)) + (ndims_64 * sizeof(T));
  nnodes_per_sector = SECTOR_LEN / max_node_len;

  std::cout << "medoid: " << medoid << "B\n";
  std::cout << "max_node_len: " << max_node_len << "B\n";
  std::cout << "nnodes_per_sector: " << nnodes_per_sector << "B\n";

  // SECTOR_LEN buffer for each sector
  char *    sector_buf = new char[SECTOR_LEN];
  char *    node_buf = new char[max_node_len];
  unsigned &nnbrs = *(unsigned *) (node_buf + ndims_64 * sizeof(T));
  unsigned *nhood_buf =
      (unsigned *) (node_buf + (ndims_64 * sizeof(T)) + sizeof(unsigned));

  // number of sectors (1 for meta data)
  _u64 n_sectors = ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector;
  _u64 disk_index_file_size = (n_sectors + 1) * SECTOR_LEN;
  // write first sector with metadata
  *(_u64 *) (sector_buf + 0 * sizeof(_u64)) = disk_index_file_size;
  *(_u64 *) (sector_buf + 1 * sizeof(_u64)) = npts_64;
  *(_u64 *) (sector_buf + 2 * sizeof(_u64)) = medoid;
  *(_u64 *) (sector_buf + 3 * sizeof(_u64)) = max_node_len;
  *(_u64 *) (sector_buf + 4 * sizeof(_u64)) = nnodes_per_sector;
  nsg_writer.write(sector_buf, SECTOR_LEN);

  T *cur_node_coords = new T[ndims_64];
  std::cout << "# sectors: " << n_sectors << "\n";
  _u64 cur_node_id = 0;
  for (_u64 sector = 0; sector < n_sectors; sector++) {
    if (sector % 100000 == 0) {
      std::cout << "Sector #" << sector << "written\n";
    }
    memset(sector_buf, 0, SECTOR_LEN);
    for (_u64 sector_node_id = 0;
         sector_node_id < nnodes_per_sector && cur_node_id < npts_64;
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
      //  T *node_coords = data + ((_u64) ndims_64 * cur_node_id);
      base_reader.read((char *) cur_node_coords, sizeof(T) * ndims_64);
      memcpy(node_buf, cur_node_coords, ndims_64 * sizeof(T));

      // write nnbrs
      *(unsigned *) (node_buf + ndims_64 * sizeof(T)) = nnbrs;

      // write nhood next
      memcpy(node_buf + ndims_64 * sizeof(T) + sizeof(unsigned), nhood_buf,
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
  delete[] cur_node_coords;
  std::cout << "Output file written\n";
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout
        << argv[0]
        << " data_type <float/int8/uint8> data_bin rand-nsg_path output_file"
        << std::endl;
    exit(-1);
  }
  int ret_val = -1;
  if (std::string(argv[1]) == std::string("float"))
    ret_val = aux_main<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    ret_val = aux_main<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    ret_val = aux_main<uint8_t>(argc, argv);
  else {
    std::cout << "unsupported type. use int8/uint8/float " << std::endl;
    ret_val = -2;
  }
  return ret_val;
}
