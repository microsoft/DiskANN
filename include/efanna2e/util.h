//
// Created by 付聪 on 2017/6/21.
//

#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H
#include <algorithm>
#include <cassert>
#include <unistd.h>
#include <cstdlib>
#include <fcntl.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

// taken from
// https://github.com/Microsoft/BLAS-on-flash/blob/master/include/utils.h
// round up X to the nearest multiple of Y
#define ROUND_UP(X, Y) \
  ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

// round down X to the nearest multiple of Y
#define ROUND_DOWN(X, Y) (((uint64_t)(X) / (Y)) * (Y))

// alignment tests
#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)
#define IS_4096_ALIGNED(X) IS_ALIGNED(X, 4096)

namespace efanna2e {

  static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size,
                        unsigned N) {
    for (unsigned i = 0; i < size; ++i) {
      addr[i] = rng() % (N - size);
    }

    std::sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i) {
      if (addr[i] <= addr[i - 1]) {
        addr[i] = addr[i - 1] + 1;
      }
    }
    unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i) {
      addr[i] = (addr[i] + off) % N;
    }
  }

  inline float *data_align(float *data_ori, unsigned point_num, unsigned &dim) {
#ifdef __GNUC__
#ifdef __AVX__
#define DATA_ALIGN_FACTOR 8
#else
#ifdef __SSE2__
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif
#endif
#endif

    // std::cout << "align with : "<<DATA_ALIGN_FACTOR << std::endl;
    float *  data_new = 0;
    unsigned new_dim =
        (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
// std::cout << "align to new dim: "<<new_dim << std::endl;
#ifdef __APPLE__
    data_new = new float[(size_t) new_dim * (size_t) point_num];
#else
    data_new = (float *) memalign(
        DATA_ALIGN_FACTOR * 4,
        (size_t) point_num * (size_t) new_dim * sizeof(float));
#endif

    for (size_t i = 0; i < point_num; i++) {
      memcpy(data_new + i * (size_t) new_dim, data_ori + i * (size_t) dim,
             dim * sizeof(float));
      memset(data_new + i * (size_t) new_dim + dim, 0,
             (new_dim - dim) * sizeof(float));
    }
    dim = new_dim;
#ifdef __APPLE__
    delete[] data_ori;
#else
    delete[] data_ori;
#endif
    return data_new;
  }

  inline void alloc_aligned(void **ptr, size_t size, size_t align) {
    *ptr = nullptr;
    assert(IS_ALIGNED(size, align));
    *ptr = ::aligned_alloc(align, size);
    assert(*ptr != nullptr);
    // std::cout << "ALLOC_ALIGNED:: " << ptr << "->" << *ptr << "\n";
  }

  template<typename T>
  inline void load_Tvecs(char *filename, T *&data, unsigned &num,
                         unsigned &dim) {
    // check validity of file
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
      std::cout << "open file error" << std::endl;
      exit(-1);
    }

    in.read((char *) &dim, sizeof(unsigned));
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    in.close();

    // calculate vector size
    size_t fsize = (size_t) ss;
    size_t per_row = sizeof(unsigned) + dim * sizeof(T);
    num = fsize / per_row;
    std::cout << "# points = " << num << ", original dimension = " << dim
              << std::endl;

    // create aligned buf
    unsigned actual_dim = dim;
    dim = ROUND_UP(dim, 8);
    std::cout << "Aligned dimesion = " << dim << std::endl;

    // data = new T[(size_t) num * (size_t) dim];
    alloc_aligned((void **) &data, (size_t) num * (size_t) dim * sizeof(T), 32);
    memset((void *) data, 0, (size_t) num * (size_t) dim * sizeof(T));

    // open classical fd
    int fd = open(filename, O_RDONLY);
    assert(fd != -1);

// parallel read each vector at the desired offset
#pragma omp parallel for schedule(static, 32768) num_threads(32)
    for (size_t i = 0; i < num; i++) {
      // computed using actual dimension
      uint64_t file_offset = (per_row * i) + sizeof(unsigned);
      // computed using aligned dimension
      T * buf = data + i * dim;
      int ret = pread(fd, (char *) buf, actual_dim * sizeof(T), file_offset);
      // std::cout << "ret = " << ret << "\n";
      if (ret != actual_dim * sizeof(T)) {
        std::cout << "read=" << ret << ", expected=" << actual_dim * sizeof(T);
        assert(ret == actual_dim * sizeof(T));
      }
    }
    std::cout << "Finished reading Tvecs" << std::endl;
    close(fd);
  }
}

#endif  // EFANNA2E_UTIL_H
