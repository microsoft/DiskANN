//
// Created by 付聪 on 2017/6/21.
//

#pragma once
#include <fcntl.h>
#include <Windows.h>

#include "FileAbstractions.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
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

typedef uint64_t _u64;
typedef int64_t  _s64;
typedef uint32_t _u32;
typedef int32_t  _s32;
typedef uint16_t _u16;
typedef int16_t  _s16;
typedef uint8_t  _u8;
typedef int8_t   _s8;

namespace NSG {

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

  /*
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
#else
	#ifdef __AVX__
		#define DATA_ALIGN_FACTOR 8
	#endif
#endif

    // std::cout << "align with : "<<DATA_ALIGN_FACTOR << std::endl;
    float *  data_new = 0;
    unsigned new_dim =
        (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
// std::cout << "align to new dim: "<<new_dim << std::endl;
#ifdef __APPLE__
    data_new = new float[(size_t) new_dim * (size_t) point_num];
#elif __NSG_WINDOWS__
    data_new = (float *) _aligned_malloc(
        (size_t) point_num * (size_t) new_dim * sizeof(float),
        DATA_ALIGN_FACTOR * 4);
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

  */
  inline void alloc_aligned(void **ptr, size_t size, size_t align) {
    *ptr = nullptr;
    assert(IS_ALIGNED(size, align));
#ifndef __NSG_WINDOWS__
    *ptr = ::aligned_alloc(align, size);
#else
    *ptr = ::_aligned_malloc(size, align);  // note the swapped arguments!
#endif

    assert(*ptr != nullptr);
    // std::cout << "ALLOC_ALIGNED:: " << ptr << "->" << *ptr << "\n";
  }

  inline void aligned_free(void* ptr) 
  {
	  //Gopal. Must have a check here if the pointer was actually allocated by _alloc_aligned
	  if (ptr == nullptr) {
		  return;
	  }
	  #ifndef __NSG_WINDOWS__
		  free(ptr);
	  #else
		  ::_aligned_free(ptr);
	  #endif
  }

  template<typename T>
  inline void load_Tvecs(const char *filename, T *&data, unsigned &num,
                         unsigned &dim) {
    // check validity of file
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
      std::cout << "Error opening file: " << filename << std::endl;
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
    std::cout << "# points = " << num << ", dimensions = " << dim << std::endl;

    // data = new T[(size_t) num * (size_t) dim];
    data = (T *) malloc((size_t) num * (size_t) dim * sizeof(T));
    memset((void *) data, 0, (size_t) num * (size_t) dim * sizeof(T));

    // open classical fd
    FileHandle fd;
#ifndef __NSG_WINDOWS__
    fd = open(filename, O_RDONLY);
    assert(fd != -1);
#else
    fd = CreateFileA(
        filename,      // ASCII file name-hope this is not an issue
        GENERIC_READ,  // Read only
        0,             // No shared access
        nullptr,  // No security attributes, no sharing with child processes
        OPEN_EXISTING,            // assuming here that the file exists
        FILE_FLAG_RANDOM_ACCESS,  // optimize for random seeks
        nullptr                   // we are opening an existing file
    );
    assert(fd != nullptr);
#endif

// parallel read each vector at the desired offset
    for (size_t i = 0; i < num; i++) {
      // computed using actual dimension
      uint64_t file_offset = (per_row * i) + sizeof(unsigned);
      // computed using aligned dimension
      T *buf = data + i * dim;

      // Gopal. Assuming synchronous read.
      DWORD ret = -1;
#ifndef __NSG_WINDOWS__
      ret = pread(fd, (char *) buf, dim * sizeof(T), file_offset);
#else
      OVERLAPPED overlapped;
      memset(&overlapped, 0, sizeof(overlapped));
      overlapped.OffsetHigh =
          (uint32_t)((file_offset & 0xFFFFFFFF00000000LL) >> 32);
      overlapped.Offset = (uint32_t)(file_offset & 0xFFFFFFFFLL);
      if (!ReadFile(fd, (LPVOID) buf, dim * sizeof(T), &ret, &overlapped)) {
        std::cout << "Read file returned error: " << GetLastError()
                  << std::endl;
      }

#endif

      // std::cout << "ret = " << ret << "\n";
      if ((size_t) ret != dim * sizeof(T)) {
        std::cout << "read=" << ret << ", expected=" << dim * sizeof(T);
        assert((size_t) ret == dim * sizeof(T));
      }
    }
    std::cout << "Finished reading Tvecs" << std::endl;

#ifndef __NSG_WINDOWS__
    close(fd);
#else
    CloseHandle(fd);
#endif
  }

  // each row in returned matrix is aligned to 32-byte boundary
  template<typename T>
  inline void aligned_load_Tvecs(char *filename, T *&data, unsigned &num,
                                 unsigned &dim) {
    // check validity of file
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
      std::cout << "Error opening file: " << filename << std::endl;
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
    unsigned aligned_dim = ROUND_UP(dim, 8);
    std::cout << "Aligned dimesion = " << aligned_dim << std::endl;

    // data = new T[(size_t) num * (size_t) dim];
    alloc_aligned((void **) &data,
                  (size_t) num * (size_t) aligned_dim * sizeof(T), 32);

    memset((void *) data, 0, (size_t) num * (size_t) aligned_dim * sizeof(T));

    // open classical fd
    FileHandle fd;
#ifndef __NSG_WINDOWS__
    fd = open(filename, O_RDONLY);
    assert(fd != -1);
#else
    fd = CreateFileA(filename, GENERIC_READ, 0, nullptr, OPEN_EXISTING,
                     FILE_FLAG_RANDOM_ACCESS, nullptr);
#endif

    // parallel read each vector at the desired offset
    // #pragma omp parallel for schedule(static, 32768)
    for (size_t i = 0; i < num; i++) {
      // computed using actual dimension
      uint64_t file_offset = (per_row * i) + sizeof(unsigned);
      // computed using aligned dimension
      T *   buf = data + i * aligned_dim;
      DWORD ret = -1;
#ifndef __NSG_WINDOWS__
      ret = pread(fd, (char *) buf, dim * sizeof(T), file_offset);
#else
      OVERLAPPED overlapped;
      memset(&overlapped, 0, sizeof(overlapped));
      overlapped.OffsetHigh =
          (uint32_t)((file_offset & 0xFFFFFFFF00000000LL) >> 32);
      overlapped.Offset = (uint32_t)(file_offset & 0xFFFFFFFFLL);
      if (!ReadFile(fd, (LPVOID) buf, dim * sizeof(T), &ret, &overlapped)) {
        std::cout << "Read file returned error: " << GetLastError()
                  << std::endl;
      }
#endif

      // std::cout << "ret = " << ret << "\n";
      if (ret != dim * sizeof(T)) {
        std::cout << "read=" << ret << ", expected=" << dim * sizeof(T);
        assert(ret == dim * sizeof(T));
      }
    }
    std::cout << "Finished reading Tvecs" << std::endl;

#ifndef __NSG_WINDOWS__
    close(fd);
#else
    CloseHandle(fd);
#endif
  }

  template<typename T>
  inline void load_bin(const char *filename, T *&data, unsigned &npts,
                       unsigned &ndims) {
    std::ifstream reader(filename, std::ios::binary);
    std::cout << "Reading bin: " << filename << "\n";
    int           npts_i32, ndims_i32;
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &ndims_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    ndims = (unsigned) ndims_i32;
    _u64 npts_u64 = (_u64) npts;
    _u64 ndims_u64 = (_u64) ndims;
    std::cout << "bin: #pts = " << npts << ", #dims = " << ndims
              << ", size = " << npts_u64 * ndims_u64 * sizeof(T) << "B"
              << std::endl;

    data = new T[npts_u64 * ndims_u64];
    reader.read((char *) data, npts_u64 * ndims_u64 * sizeof(T));
    reader.close();
    std::cout << "Finished reading bin" << std::endl;
  }

  struct OneShotNSG {
    _u64                  medoid, width;
    unsigned *            nsg = nullptr;
    std::vector<unsigned> nnbrs_;
    std::vector<_u64>     offsets;
    _u64                  nnodes;

    void read(char *filename) {
      std::ifstream reader(filename, std::ios::binary | std::ios::ate);
      _u64 nsg_len = reader.tellg() - (std::streamoff)(2 * sizeof(unsigned));
      reader.seekg(0, std::ios::beg);
      unsigned medoid_u32, width_u32;
      reader.read((char *) &width_u32, sizeof(unsigned));
      reader.read((char *) &medoid_u32, sizeof(unsigned));
      medoid = (_u64) medoid_u32;
      width = (_u64) width_u32;
      std::cout << "Medoid: " << medoid << ", width: " << width << std::endl;
      std::cout << "NSG Size: " << nsg_len << "B\n";
      nsg = (unsigned *) (new char[nsg_len]);
      reader.read((char *) nsg, nsg_len);

      // compute # nodes
      nnodes = 0;
      _u64 cur_off = 0;
      while (cur_off * sizeof(unsigned) < nsg_len) {
        nnodes++;
        unsigned cur_nnbrs_ = *(nsg + cur_off);
        // offset to start of node nhood
        offsets.push_back(cur_off + 1);
        // # nbrs in nhood
        nnbrs_.push_back(cur_nnbrs_);
        // offset to start of next node nhood
        cur_off += (cur_nnbrs_ + 1);
      }
      std::cout << "# nodes: " << nnodes << std::endl;
    }

    ~OneShotNSG() {
      if (nsg != nullptr) {
        delete[] nsg;
      }
    }

    unsigned *data(_u64 idx) {
      return (nsg + offsets[idx]);
    }

    _u64 nnbrs(_u64 idx) {
      return nnbrs_[idx];
    }

    _u64 size() {
      return nnodes;
    }
  };

  template<typename Dtype, typename Mtype>
  inline void block_load_convert_Tvecs(const char *filename, Mtype *&data,
                                       _u64 &num, _u64 &dim) {
    // check validity of file
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
      std::cout << "Error opening file: " << filename << std::endl;
      exit(-1);
    }
    _u32 dim_u32;
    in.read((char *) &dim_u32, sizeof(unsigned));
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    in.close();
    dim = (_u64) dim_u32;

    // calculate vector size
    _u64 fsize = (_u64) ss;
    _u64 per_row = sizeof(unsigned) + dim * sizeof(Dtype);
    num = fsize / per_row;
    std::cout << "# points = " << num << ", dimensions = " << dim << std::endl;

    // data = new T[(size_t) num * (size_t) dim];
    data = (Mtype *) malloc(num * dim * sizeof(Mtype));
    memset((void *) data, 0, num * dim * sizeof(Mtype));

    // block read buf
    _u64  blk_size = 5 * 1048576;
    char *block_read_buf = new char[per_row * blk_size];
    _u64  n_blks = ROUND_UP(num, blk_size) / blk_size;
    std::cout << "# blks: " << n_blks << ", blk_size: " << blk_size << "\n";

    // open classical fd
    FileHandle fd;
#ifndef __NSG_WINDOWS__
    fd = open(filename, O_RDONLY);
    assert(fd != -1);
#else
    fd = CreateFileA(filename, GENERIC_READ,
                     0,        // no sharing
                     nullptr,  // default security
                     OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
    assert(fd != nullptr);
#endif

    for (_u64 blk = 0; blk < n_blks; blk++) {
      // block stats
      _u64 cur_blk_npts = (std::min)(num - blk * blk_size, blk_size);
      _u64 cur_blk_offset = blk * blk_size * per_row;
      _u64 cur_blk_size = cur_blk_npts * per_row;

      // read blk into block_read_buf
      DWORD ret = -1;
#ifndef __NSG_WINDOWS__
      ret = pread(fd, block_read_buf, cur_blk_size, cur_blk_offset);
#else
      OVERLAPPED overlapped;
      memset(&overlapped, 0, sizeof(overlapped));
      overlapped.OffsetHigh =
          (uint32_t)((cur_blk_offset & 0xFFFFFFFF00000000LL) >> 32);
      overlapped.Offset = (uint32_t)(cur_blk_offset & 0xFFFFFFFFLL);
      if (!ReadFile(fd, (LPVOID) block_read_buf, cur_blk_size, &ret,
                    &overlapped)) {
        std::cout << "Read file returned error: " << GetLastError()
                  << std::endl;
      }
#endif
      if ((_u64) ret != cur_blk_size) {
        std::cout << "read=" << ret << ", expected=" << cur_blk_size;
        exit(-1);
      }
#pragma omp parallel for schedule(static, 32768)
      for (_s64 j = 0; j < cur_blk_npts; j++) {
        Mtype *mem_vec = data + dim * (blk_size * blk + j);
        Dtype *disk_vec =
            (Dtype *) (block_read_buf + (per_row * j) + sizeof(unsigned));
        for (_u64 d = 0; d < dim; d++) {
          assert(disk_vec[d] < 32768);
          mem_vec[d] = (Mtype) disk_vec[d];
        }
      }
      std::cout << "Block #" << blk << " read\n";
    }
#ifndef __NSG_WINDOWS__
    close(fd);
#else
    CloseHandle(fd);
#endif
    delete[] block_read_buf;
  }
}  // namespace NSG
