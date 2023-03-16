// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

template<class T>
void block_convert(std::ofstream& writer, std::ifstream& reader, T* read_buf,
                   _u64 npts, _u64 ndims) {
  reader.read((char*) read_buf, npts * ndims * sizeof(float));

  for (_u64 i = 0; i < npts; i++) {
    for (_u64 d = 0; d < ndims; d++) {
      writer << read_buf[d + i * ndims];
      if (d < ndims - 1)
        writer << "\t";
      else
        writer << "\n";
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << argv[0] << " <float/int8/uint8> input_bin output_tsv"
              << std::endl;
    exit(-1);
  }
  std::string type_string(argv[1]);
  if ((type_string != std::string("float")) &&
      (type_string != std::string("int8")) &&
      (type_string != std::string("uin8"))) {
    std::cerr << "Error: type not supported. Use float/int8/uint8" << std::endl;
  }

  std::ifstream reader(argv[2], std::ios::binary);
  _u32          npts_u32;
  _u32          ndims_u32;
  reader.read((char*) &npts_u32, sizeof(_s32));
  reader.read((char*) &ndims_u32, sizeof(_s32));
  size_t npts = npts_u32;
  size_t ndims = ndims_u32;
  std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
            << std::endl;

  _u64 blk_size = 131072;
  _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;

  std::ofstream writer(argv[3]);
  char*         read_buf = new char[blk_size * ndims * 4];
  for (_u64 i = 0; i < nblks; i++) {
    _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
    if (type_string == std::string("float"))
      block_convert<float>(writer, reader, (float*) read_buf, cblk_size, ndims);
    else if (type_string == std::string("int8"))
      block_convert<int8_t>(writer, reader, (int8_t*) read_buf, cblk_size,
                            ndims);
    else if (type_string == std::string("uint8"))
      block_convert<uint8_t>(writer, reader, (uint8_t*) read_buf, cblk_size,
                             ndims);
    std::cout << "Block #" << i << " written" << std::endl;
  }

  delete[] read_buf;

  writer.close();
  reader.close();
}
