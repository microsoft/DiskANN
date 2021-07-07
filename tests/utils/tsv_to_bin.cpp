// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

void block_convert(std::ifstream& reader, std::ofstream& writer,
                   float* read_buf, _u64 npts, _u64 ndims) {
  auto  cursor = read_buf;
  float val;

  for (_u64 i = 0; i < npts; i++) {
    for (_u64 d = 0; d < ndims; ++d) {
      reader >> val;
      *cursor = val;
      cursor++;
    }
  }
  writer.write((char*) read_buf, npts * ndims * sizeof(float));
}

int main(int argc, char** argv) {
  if (argc != 5) {
    diskann::cout << argv[0]
                  << " input_filename.tsv output_filename.bin dim num_pts>"
                  << std::endl;
    exit(-1);
  }

  _u64 ndims = atoi(argv[3]);
  _u64 npts = atoi(argv[4]);

  std::ifstream reader(argv[1], std::ios::binary | std::ios::ate);
  //  _u64          fsize = reader.tellg();
  reader.seekg(0, std::ios::beg);
  reader.seekg(0, std::ios::beg);

  _u64 blk_size = 131072;
  _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
  diskann::cout << "# blks: " << nblks << std::endl;
  std::ofstream writer(argv[2], std::ios::binary);
  int           npts_s32 = (_s32) npts;
  int           ndims_s32 = (_s32) ndims;
  writer.write((char*) &npts_s32, sizeof(_s32));
  writer.write((char*) &ndims_s32, sizeof(_s32));
  float* read_buf = new float[npts * (ndims + 1)];
  for (_u64 i = 0; i < nblks; i++) {
    _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
    block_convert(reader, writer, read_buf, cblk_size, ndims);
    diskann::cout << "Block #" << i << " written" << std::endl;
  }

  delete[] read_buf;

  reader.close();
  writer.close();
}
