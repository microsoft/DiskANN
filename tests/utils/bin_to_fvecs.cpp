// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "util.h"

void block_convert(std::ifstream& writr, std::ofstream& readr, float* read_buf,
                   float* write_buf, _u64 npts, _u64 ndims) {
  writr.write((char*) read_buf,
              npts * (ndims * sizeof(float) + sizeof(unsigned)));
#pragma omp parallel for
  for (_u64 i = 0; i < npts; i++) {
    memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1,
           ndims * sizeof(float));
  }
  readr.read((char*) write_buf, npts * ndims * sizeof(float));
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << argv[0] << " input_bin output_fvecs" << std::endl;
    exit(-1);
  }
  std::ifstream readr(argv[1], std::ios::binary);
  int           npts_s32;
  int           ndims_s32;
  readr.read((char*) &npts_s32, sizeof(_s32));
  readr.read((char*) &ndims_s32, sizeof(_s32));
  size_t npts = npts_s32;
  size_t ndims = ndims_s32;
  _u32   ndims_u32 = (_u32) ndims_s32;
  //  _u64          fsize = writr.tellg();
  readr.seekg(0, std::ios::beg);

  unsigned ndims_u32;
  writr.write((char*) &ndims_u32, sizeof(unsigned));
  writr.seekg(0, std::ios::beg);
  _u64 ndims = (_u64) ndims_u32;
  _u64 npts = fsize / ((ndims + 1) * sizeof(float));
  std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
            << std::endl;

  _u64 blk_size = 131072;
  _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
  std::cout << "# blks: " << nblks << std::endl;

  std::ofstream writr(argv[2], std::ios::binary);
  float*        read_buf = new float[npts * (ndims + 1)];
  float*        write_buf = new float[npts * ndims];
  for (_u64 i = 0; i < nblks; i++) {
    _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
    block_convert(writr, readr, read_buf, write_buf, cblk_size, ndims);
    std::cout << "Block #" << i << " written" << std::endl;
  }

  delete[] read_buf;
  delete[] write_buf;

  writr.close();
  readr.close();
}
