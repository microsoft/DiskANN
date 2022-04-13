// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

void block_convert_float(std::ifstream& reader, std::ofstream& writer,
                         _u64 npts, _u64 ndims) {
  auto read_buf = new float[npts * (ndims + 1)];

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
  delete[] read_buf;
}

void block_convert_int8(std::ifstream& reader, std::ofstream& writer, _u64 npts,
                        _u64 ndims) {
  auto read_buf = new int8_t[npts * (ndims + 1)];

  auto cursor = read_buf;
  int  val;

  for (_u64 i = 0; i < npts; i++) {
    for (_u64 d = 0; d < ndims; ++d) {
      reader >> val;
      *cursor = (int8_t) val;
      cursor++;
    }
  }
  writer.write((char*) read_buf, npts * ndims * sizeof(uint8_t));
  delete[] read_buf;
}

void block_convert_uint8(std::ifstream& reader, std::ofstream& writer,
                         _u64 npts, _u64 ndims) {
  auto read_buf = new uint8_t[npts * (ndims + 1)];

  auto cursor = read_buf;
  int  val;

  for (_u64 i = 0; i < npts; i++) {
    for (_u64 d = 0; d < ndims; ++d) {
      reader >> val;
      *cursor = (uint8_t) val;
      cursor++;
    }
  }
  writer.write((char*) read_buf, npts * ndims * sizeof(uint8_t));
  delete[] read_buf;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << argv[0]
              << "<float/int8/uint8> input_filename.tsv output_filename.bin "
                 "dim num_pts>"
              << std::endl;
    exit(-1);
  }

  if (std::string(argv[1]) != std::string("float") &&
      std::string(argv[1]) != std::string("int8") &&
      std::string(argv[1]) != std::string("uint8")) {
    std::cout << "Unsupported type. float, int8 and uint8 types are supported."
              << std::endl;
  }

  _u64 ndims = atoi(argv[4]);
  _u64 npts = atoi(argv[5]);

  std::ifstream reader(argv[2], std::ios::binary | std::ios::ate);
  //  _u64          fsize = reader.tellg();
  reader.seekg(0, std::ios::beg);
  reader.seekg(0, std::ios::beg);

  _u64 blk_size = 131072;
  _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
  std::cout << "# blks: " << nblks << std::endl;
  std::ofstream writer(argv[3], std::ios::binary);
  auto          npts_s32 = (_u32) npts;
  auto          ndims_s32 = (_u32) ndims;
  writer.write((char*) &npts_s32, sizeof(_u32));
  writer.write((char*) &ndims_s32, sizeof(_u32));

  for (_u64 i = 0; i < nblks; i++) {
    _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
    if (std::string(argv[1]) == std::string("float")) {
      block_convert_float(reader, writer, cblk_size, ndims);
    } else if (std::string(argv[1]) == std::string("int8")) {
      block_convert_int8(reader, writer, cblk_size, ndims);
    } else if (std::string(argv[1]) == std::string("uint8")) {
      block_convert_uint8(reader, writer, cblk_size, ndims);
    }
    std::cout << "Block #" << i << " written" << std::endl;
  }

  reader.close();
  writer.close();
}
