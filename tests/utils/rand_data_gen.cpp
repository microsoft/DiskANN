// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>
#include "utils.h"

int block_write_float(std::ofstream& writer, _u64 ndims, _u64 npts,
                      float norm) {
  auto vec = new float[ndims];

  std::random_device         rd{};
  std::mt19937               gen{rd()};
  std::normal_distribution<> normal_rand{0, 1};

  for (_u64 i = 0; i < npts; i++) {
    float sum = 0;
    for (_u64 d = 0; d < ndims; ++d)
      vec[d] = normal_rand(gen);
    for (_u64 d = 0; d < ndims; ++d)
      sum += vec[d] * vec[d];
    for (_u64 d = 0; d < ndims; ++d)
      vec[d] = vec[d] * norm / std::sqrt(sum);

    writer.write((char*) vec, ndims * sizeof(float));
  }

  delete[] vec;
  return 0;
}

int block_write_int8(std::ofstream& writer, _u64 ndims, _u64 npts, float norm) {
  auto vec = new float[ndims];
  auto vec_T = new int8_t[ndims];

  std::random_device         rd{};
  std::mt19937               gen{rd()};
  std::normal_distribution<> normal_rand{0, 1};

  for (_u64 i = 0; i < npts; i++) {
    float sum = 0;
    for (_u64 d = 0; d < ndims; ++d)
      vec[d] = normal_rand(gen);
    for (_u64 d = 0; d < ndims; ++d)
      sum += vec[d] * vec[d];
    for (_u64 d = 0; d < ndims; ++d)
      vec[d] = vec[d] * norm / std::sqrt(sum);

    for (_u64 d = 0; d < ndims; ++d) {
      vec_T[d] = std::round<int>(vec[d]);
    }

    writer.write((char*) vec_T, ndims * sizeof(int8_t));
  }

  delete[] vec;
  delete[] vec_T;
  return 0;
}

int block_write_uint8(std::ofstream& writer, _u64 ndims, _u64 npts,
                      float norm) {
  auto vec = new float[ndims];
  auto vec_T = new int8_t[ndims];

  std::random_device         rd{};
  std::mt19937               gen{rd()};
  std::normal_distribution<> normal_rand{0, 1};

  for (_u64 i = 0; i < npts; i++) {
    float sum = 0;
    for (_u64 d = 0; d < ndims; ++d)
      vec[d] = normal_rand(gen);
    for (_u64 d = 0; d < ndims; ++d)
      sum += vec[d] * vec[d];
    for (_u64 d = 0; d < ndims; ++d)
      vec[d] = vec[d] * norm / std::sqrt(sum);

    for (_u64 d = 0; d < ndims; ++d) {
      vec_T[d] = 128 + std::round<int>(vec[d]);
    }

    writer.write((char*) vec_T, ndims * sizeof(uint8_t));
  }

  delete[] vec;
  delete[] vec_T;
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << argv[0] << " <float/int8/uint8> ndims npts norm output.bin"
              << std::endl;
    exit(-1);
  }

  if (std::string(argv[1]) != std::string("float") &&
      std::string(argv[1]) != std::string("int8") &&
      std::string(argv[1]) != std::string("uint8")) {
    std::cout << "Unsupported type. float, int8 and uint8 types are supported."
              << std::endl;
  }

  _u64  ndims = atoi(argv[2]);
  _u64  npts = atoi(argv[3]);
  float norm = atof(argv[4]);

  if (norm <= 0.0) {
    std::cerr << "Error: Norm must be a positive number" << std::endl;
    return -1;
  }

  if ((std::string(argv[1]) == std::string("int8")) ||
      (std::string(argv[1]) == std::string("uint8"))) {
    if (norm > 127) {
      std::cerr << "Error: for in8/uint8 datatypes, L2 norm can not be greater "
                   "than 127"
                << std::endl;
      return -1;
    }
  }

  std::ofstream writer(argv[5], std::ios::binary);
  auto          npts_s32 = (_u32) npts;
  auto          ndims_s32 = (_u32) ndims;
  writer.write((char*) &npts_s32, sizeof(_u32));
  writer.write((char*) &ndims_s32, sizeof(_u32));

  _u64 blk_size = 131072;
  _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
  std::cout << "# blks: " << nblks << std::endl;

  int ret = 0;
  for (_u64 i = 0; i < nblks; i++) {
    _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
    if (std::string(argv[1]) == std::string("float")) {
      ret = block_write_float(writer, ndims, cblk_size, norm);
    } else if (std::string(argv[1]) == std::string("int8")) {
      ret = block_write_int8(writer, ndims, cblk_size, norm);
    } else if (std::string(argv[1]) == std::string("uint8")) {
      ret = block_write_uint8(writer, ndims, cblk_size, norm);
    }
    if (ret == 0)
      std::cout << "Block #" << i << " written" << std::endl;
    else {
      writer.close();
      std::cout << "failed to write" << std::endl;
      return -1;
    }
  }

  writer.close();
  return 0;
}
