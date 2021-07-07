// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <limits>
#include "utils.h"

void block_convert(std::ofstream& writr, std::ifstream& readr, float* read_buf,
                   float* write_buf, _u64 npts, _u64 ndims,
                   bool normalize = false) {
  readr.read((char*) read_buf, npts * ndims * sizeof(float));
  _u32 ndims_u32 = (_u32) ndims;
#pragma omp parallel for
  for (_s64 i = 0; i < (_s64) npts; i++) {
    if (normalize) {
      float norm_pt = std::numeric_limits<float>::epsilon();
      for (_u32 dim = 0; dim < ndims_u32; dim++) {
        norm_pt +=
            *(read_buf + i * ndims + dim) * *(read_buf + i * ndims + dim);
      }
      norm_pt = std::sqrt(norm_pt);
      for (_u32 dim = 0; dim < ndims_u32; dim++) {
        *(read_buf + i * ndims + dim) = *(read_buf + i * ndims + dim) / norm_pt;
      }
    }

    memcpy(write_buf + i * (ndims + 1), &ndims_u32, sizeof(float));
    memcpy(write_buf + i * (ndims + 1) + 1, (read_buf + i * ndims),
           ndims * sizeof(float));
  }
  writr.write((char*) write_buf,
              npts * (ndims * sizeof(float) + sizeof(unsigned)));
}

int main(int argc, char** argv) {
  if (argc != 4) {
    diskann::cout << argv[0]
                  << ": [input_bin] [output_fvecs] [normalize? (1 for yes]"
                  << std::endl;
    exit(-1);
  }
  bool          normalize = (bool) std::atoi(argv[3]);
  std::ifstream readr(argv[1], std::ios::binary);
  int           npts_s32;
  int           ndims_s32;
  readr.read((char*) &npts_s32, sizeof(_s32));
  readr.read((char*) &ndims_s32, sizeof(_s32));
  //  size_t npt = npts_s32;
  //  size_t ndim = ndims_s32;
  _u32 ndims_u32 = (_u32) ndims_s32;
  _u32 npts_u32 = (_u32) npts_s32;
  // readr.seekg(0, std::ios::end);
  //_u64 fsize = readr.tellg();

  std::ofstream writr(argv[2], std::ios::binary);
  // writr.write((char*) &ndims_u32, sizeof(unsigned));
  //   writr.seekg(0, std::ios::beg);
  _u64 ndims = (_u64) ndims_u32;
  _u64 npts = (_u64) npts_u32;
  diskann::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
                << std::endl;

  _u64 blk_size = 131072;
  _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
  diskann::cout << "# blks: " << nblks << std::endl;

  float* read_buf = new float[npts * ndims];
  float* write_buf = new float[npts * (ndims + 1)];
  for (_u64 i = 0; i < nblks; i++) {
    _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
    block_convert(writr, readr, read_buf, write_buf, cblk_size, ndims,
                  normalize);
    diskann::cout << "Block #" << i << " written" << std::endl;
  }
  delete[] read_buf;
  delete[] write_buf;
  writr.close();
  readr.close();
}
