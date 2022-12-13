// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

// Convert float types
void block_convert_float(std::ifstream& reader, std::ofstream& writer,
                   float* read_buf, float* write_buf, _u64 npts, _u64 ndims) {
  reader.read((char*) read_buf,
              npts * (ndims * sizeof(float) + sizeof(unsigned)));
  for (_u64 i = 0; i < npts; i++) {
    memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1,
           ndims * sizeof(float));
  }
  writer.write((char*) write_buf, npts * ndims * sizeof(float));
}

// Convert byte types
void block_convert_byte(std::ifstream& reader, std::ofstream& writer, _u8* read_buf,
                   _u8* write_buf, _u64 npts, _u64 ndims) {
  reader.read((char*) read_buf,
              npts * (ndims * sizeof(_u8) + sizeof(unsigned)));
  for (_u64 i = 0; i < npts; i++) {
    memcpy(write_buf + i * ndims,
           (read_buf + i * (ndims + sizeof(unsigned))) + sizeof(unsigned),
           ndims * sizeof(_u8));
  }
  writer.write((char*) write_buf, npts * ndims * sizeof(_u8));
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << argv[0]
              << " <float/int8/uint8> input_vecs output_bin"
              << std::endl;
    exit(-1);
  }

  int datasize = sizeof(float);

  if (strcmp(argv[1], "uint8") == 0 || strcmp(argv[1], "int8") == 0) {
    datasize = sizeof(_u8);
  } else if (strcmp(argv[1], "float") != 0) {
    std::cout << "Error: type not supported. Use float/int8/uint8"
              << std::endl;
    exit(-1);
  }

  std::ifstream reader(argv[2], std::ios::binary | std::ios::ate);
  _u64          fsize = reader.tellg();
  reader.seekg(0, std::ios::beg);

  unsigned ndims_u32;
  reader.read((char*) &ndims_u32, sizeof(unsigned));
  reader.seekg(0, std::ios::beg);
  _u64 ndims = (_u64) ndims_u32;
  _u64 npts = fsize / ((ndims * datasize) + sizeof(unsigned));
  std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
            << std::endl;

  _u64 blk_size = 131072;
  _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
  std::cout << "# blks: " << nblks << std::endl;
  std::ofstream writer(argv[3], std::ios::binary);
  _s32          npts_s32 = (_s32) npts;
  _s32          ndims_s32 = (_s32) ndims;
  writer.write((char*) &npts_s32, sizeof(_s32));
  writer.write((char*) &ndims_s32, sizeof(_s32));

  _u64 chunknpts = std::min(npts, blk_size);
  _u8* read_buf = new _u8[chunknpts * ((ndims * datasize) + sizeof(unsigned))];
  _u8* write_buf = new _u8[chunknpts * ndims * datasize];

  for (_u64 i = 0; i < nblks; i++) {
    _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
    if (datasize == sizeof(float)) {
      block_convert_float(reader, writer, (float*) read_buf, (float*) write_buf,
                    cblk_size, ndims);
    } else {
      block_convert_byte(reader, writer, read_buf, write_buf, cblk_size, ndims);
    }
    std::cout << "Block #" << i << " written" << std::endl;
  }

  delete[] read_buf;
  delete[] write_buf;

  reader.close();
  writer.close();
}
