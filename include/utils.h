
#pragma once
#include <fcntl.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#ifdef _WINDOWS
#include <Windows.h>
typedef HANDLE FileHandle;
#else
#include <unistd.h>
typedef int FileHandle;
#endif

#include "cached_io.h"
#include "common_includes.h"
#include "windows_customizations.h"

// taken from
// https://github.com/Microsoft/BLAS-on-flash/blob/master/include/utils.h
// round up X to the nearest multiple of Y
#define ROUND_UP(X, Y) \
  ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define DIV_ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0))

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

namespace diskann {

  enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3 };

  DISKANN_DLLEXPORT float calc_recall_set(unsigned  num_queries,
                                          unsigned *gold_std, float *gs_dist,
                                          unsigned  dim_gs,
                                          unsigned *our_results,
                                          unsigned dim_or, unsigned recall_at);

  DISKANN_DLLEXPORT void read_idmap(const std::string &    fname,
                                    std::vector<unsigned> &ivecs);

  DISKANN_DLLEXPORT int merge_shards(
      const std::string &nsg_prefix, const std::string &nsg_suffix,
      const std::string &idmaps_prefix, const std::string &idmaps_suffix,
      const _u64 nshards, const std::string &output_nsg, unsigned max_degree);

  template<typename T>
  DISKANN_DLLEXPORT int build_merged_vamana_index(
      std::string base_file, diskann::Metric _compareMetric, size_t L, size_t R,
      float sampling_rate, double ram_budget, std::string mem_index_path);

  inline void alloc_aligned(void **ptr, size_t size, size_t align) {
    *ptr = nullptr;
    assert(IS_ALIGNED(size, align));
#ifndef _WINDOWS
    *ptr = ::aligned_alloc(align, size);
#else
    *ptr = ::_aligned_malloc(size, align);  // note the swapped arguments!
#endif
    assert(*ptr != nullptr);
  }

  inline void aligned_free(void *ptr) {
    // Gopal. Must have a check here if the pointer was actually allocated by
    // _alloc_aligned
    if (ptr == nullptr) {
      return;
    }
#ifndef _WINDOWS
    free(ptr);
#else
    ::_aligned_free(ptr);
#endif
  }

  inline void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size,
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

  inline void get_bin_metadata(const std::string &bin_file, size_t &nrows,
                               size_t &ncols) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    uint32_t      nrows_32, ncols_32;
    reader.read((char *) &nrows_32, sizeof(uint32_t));
    reader.read((char *) &ncols_32, sizeof(uint32_t));
    nrows = nrows_32;
    ncols = ncols_32;
    reader.close();
  }

  template<typename T>
  inline void load_bin(const std::string &bin_file, T *&data, size_t &npts,
                       size_t &dim) {
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    std::cout << "Reading bin file " << bin_file.c_str() << " ..."
              << std::flush;
    size_t actual_file_size = reader.get_file_size();

    int npts_i32, dim_i32;
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..."
              << std::flush;

    size_t expected_actual_file_size =
        npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size) {
      std::cout << "Error. File size mismatch. Actual size is "
                << actual_file_size << " while expected size is  "
                << expected_actual_file_size << std::endl;
      exit(-1);
    }

    data = new T[npts * dim];
    reader.read((char *) data, npts * dim * sizeof(T));
    std::cout << "Finished reading bin file." << std::endl;
  }

  template<typename T>
  inline void save_bin(const std::string &filename, T *data, size_t npts,
                       size_t ndims) {
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    std::cout << "Writing bin: " << filename.c_str() << "\n";
    int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    writer.write((char *) &npts_i32, sizeof(int));
    writer.write((char *) &ndims_i32, sizeof(int));
    std::cout << "bin: #pts = " << npts << ", #dims = " << ndims
              << ", size = " << npts * ndims * sizeof(T) + 2 * sizeof(int)
              << "B" << std::endl;

    //    data = new T[npts_u64 * ndims_u64];
    writer.write((char *) data, npts * ndims * sizeof(T));
    writer.close();
    std::cout << "Finished writing bin." << std::endl;
  }

  template<typename T>
  inline void load_aligned_bin(const std::string bin_file, T *&data,
                               size_t &npts, size_t &dim, size_t &rounded_dim) {
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    std::cout << "Reading bin file " << bin_file << " ..." << std::flush;
    size_t actual_file_size = reader.get_file_size();

    int npts_i32, dim_i32;
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    size_t expected_actual_file_size =
        npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size) {
      std::cout << "Error. File size mismatch. Actual size is "
                << actual_file_size << " while expected size is  "
                << expected_actual_file_size << std::endl;
      exit(-1);
    }

    rounded_dim = ROUND_UP(dim, 8);

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim
              << ", aligned_dim = " << rounded_dim << "..." << std::flush;
    size_t allocSize = npts * rounded_dim * sizeof(T);
    std::cout << "allocating aligned memory, " << allocSize << " bytes..."
              << std::flush;
    alloc_aligned(((void **) &data), allocSize, 8 * sizeof(T));
    std::cout << "done. Copying data..." << std::flush;

    for (size_t i = 0; i < npts; i++) {
      reader.read((char *) (data + i * rounded_dim), dim * sizeof(T));
      memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
    std::cout << " done." << std::endl;
  }

  template<typename InType, typename OutType>
  void convert_types(const InType *srcmat, OutType *destmat, size_t npts,
                     size_t dim) {
#pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (_s64) npts; i++) {
      for (uint64_t j = 0; j < dim; j++) {
        destmat[i * dim + j] = srcmat[i * dim + j];
      }
    }
  }

  /********* templated load functions *********/
  template<typename T>
  void load_Tvecs(const char *filename, T *&data, size_t &num, size_t &dim) {
    // check validity of file
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
      std::cout << "Error opening file: " << filename << std::endl;
      exit(-1);
    }
    _u64 fsize = in.tellg();
    in.seekg(0, std::ios::beg);
    _u32 dim_u32;
    in.read((char *) &dim_u32, sizeof(unsigned));
    in.close();
    dim = dim_u32;

    _u64 ndims = (_u64) dim;
    _u64 disk_vec_size = ndims * sizeof(T) + sizeof(unsigned);
    _u64 mem_vec_size = ndims * sizeof(T);
    _u64 npts = fsize / disk_vec_size;
    num = npts;
    std::cout << "Tvecs: " << filename << ", npts: " << npts
              << ", ndims: " << ndims << "\n";
    // allocate memory
    data = new T[npts * ndims];

    cached_ifstream reader(std::string(filename), 256 * 1024 * 1024);
    unsigned        dummy_ndims;
    for (_u64 i = 0; i < npts; i++) {
      T *cur_vec = data + (i * ndims);
      // read and ignore dummy ndims
      reader.read((char *) &dummy_ndims, sizeof(unsigned));

      // read vec
      reader.read((char *) cur_vec, mem_vec_size);
    }
    return;
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
#ifndef _WINDOWS
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
      T *buf = data + i * aligned_dim;

#ifndef _WINDOWS
      int ret = -1;
      ret = pread(fd, (char *) buf, dim * sizeof(T), file_offset);
#else
      DWORD      ret = -1;
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

#ifndef _WINDOWS
    close(fd);
#else
    CloseHandle(fd);
#endif
  }

  // plain saves data as npts X ndims array into filename
  template<typename T>
  void save_Tvecs(const char *filename, T *data, size_t npts, size_t ndims) {
    std::string fname(filename);

    // create cached ofstream with 64MB cache
    cached_ofstream writer(fname, 64 * 1048576);

    unsigned dims_u32 = (unsigned) ndims;

    // start writing
    for (uint64_t i = 0; i < npts; i++) {
      // write dims in u32
      writer.write((char *) &dims_u32, sizeof(unsigned));

      // get cur point in data
      T *cur_pt = data + i * ndims;
      writer.write((char *) cur_pt, ndims * sizeof(T));
    }
  }

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector(const char *vec, size_t vecsize) {
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64)
      _mm_prefetch((const char *) vec + d, _MM_HINT_T0);
  }

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector_l2(const char *vec, size_t vecsize) {
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64)
      _mm_prefetch((const char *) vec + d, _MM_HINT_T1);
  }
}  // namespace diskann

struct PivotContainer {
  PivotContainer() = default;

  PivotContainer(size_t pivo_id, float pivo_dist)
      : piv_id{pivo_id}, piv_dist{pivo_dist} {
  }

  bool operator<(const PivotContainer &p) const {
    return p.piv_dist < piv_dist;
  }

  bool operator>(const PivotContainer &p) const {
    return p.piv_dist > piv_dist;
  }

  size_t piv_id;
  float  piv_dist;
};

inline bool file_exists(const std::string &name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

inline _u64 get_file_size(const std::string &fname) {
  if (!file_exists(fname))
    return 0;
  else {
    std::ifstream reader(fname, std::ios::binary | std::ios::ate);
    _u64          end_pos = reader.tellg();
    reader.close();
    return end_pos;
  }
}

inline bool validate_file_size(const std::string &name) {
  std::ifstream in(std::string(name), std::ios::binary);
  in.seekg(0, in.end);
  size_t actual_file_size = in.tellg();
  in.seekg(0, in.beg);
  size_t expected_file_size;
  in.read((char *) &expected_file_size, sizeof(uint64_t));
  if (actual_file_size != expected_file_size) {
    std::cout << "Error loading" << name << ". Expected "
                                            "size (metadata): "
              << expected_file_size
              << ", actual file size : " << actual_file_size << ". Exitting."
              << std::endl;
    in.close();
    exit(-1);
  }
  in.close();
  return 1;
}
