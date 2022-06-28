// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

void block_convert(std::ifstream& reader, std::ofstream& writer,
                   float* read_buf, float* write_buf, _u64 npts, _u64 ndims) {
  reader.read((char*) read_buf,
              npts * (ndims * sizeof(float) + sizeof(unsigned)));
  for (_u64 i = 0; i < npts; i++) {
    memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1,
           ndims * sizeof(float));
  }
  writer.write((char*) write_buf, npts * ndims * sizeof(float));
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << argv[0] << " input_bin   output_bin    npoints" << std::endl;
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

  std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
            << std::endl;

  _s32 rpts = atoi(argv[3]);
  std::cout << "Adding " << rpts << " random points to new file" << std::endl; 

  std::vector<_s32> random_ids(npts);
  std::iota(random_ids.begin(), random_ids.end(), 0);
  std::random_shuffle(random_ids.begin(), random_ids.end());
  random_ids.resize(rpts);

  _u64 blk_size = 131072;
  _u64 nblks = ROUND_UP(rpts, blk_size) / blk_size;
  std::cout << "# blks: " << nblks << std::endl;
  std::ofstream writer(argv[2], std::ios::binary);

  writer.write((char*) &rpts, sizeof(_s32));
  writer.write((char*) &ndims_s32, sizeof(_s32));
  float* read_buf = new float[rpts * (ndims + 1)];
  float* write_buf = new float[rpts * ndims];
  for(_s32 id : random_ids){
  	
  }

  delete[] read_buf;
  delete[] write_buf;

  readr.close();
  writer.close();
}