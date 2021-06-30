// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <fcntl.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <random>
#include <set>
#include <sstream>
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

#include "distance.h"
#include "logger.h"
#include "ann_exception.h"
#include "common_includes.h"
#include "windows_customizations.h"

#ifdef EXEC_ENV_OLS
#include "content_buf.h"
#include "memory_mapped_files.h"
#endif

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
#define METADATA_SIZE \
  4096  // all metadata of individual sub-component files is written in first
        // 4KB for unified files
typedef uint64_t _u64;
typedef int64_t  _s64;
typedef uint32_t _u32;
typedef int32_t  _s32;
typedef uint16_t _u16;
typedef int16_t  _s16;
typedef uint8_t  _u8;
typedef int8_t   _s8;

inline bool file_exists(const std::string& name, bool dirCheck = false) {
  int val;
#ifndef _WINDOWS
  struct stat buffer;
  val = stat(name.c_str(), &buffer);
#else
  struct _stat64 buffer;
  val = _stat64(name.c_str(), &buffer);
#endif

  diskann::cout << " Stat(" << name.c_str() << ") returned: " << val
                << std::endl;
  if (val != 0) {
    switch (errno) {
      case EINVAL:
        diskann::cout << "Invalid argument passed to stat()" << std::endl;
        break;
      case ENOENT:
        diskann::cout << "File " << name.c_str() << " does not exist"
                      << std::endl;
        break;
      default:
        diskann::cout << "Unexpected error in stat():" << errno << std::endl;
        break;
    }
    return false;
  } else {
    // the file entry exists. If reqd, check if this is a directory.
    return dirCheck ? buffer.st_mode & S_IFDIR : true;
  }
}

inline std::string getTempFilePath(const std::string& workingDir,
                                   const std::string& suffix) {
#ifdef _WINDOWS
  char        temp[MAX_PATH];
  std::string retFile;

  do {
    if (!tmpnam_s(temp, MAX_PATH) == 0) {
      throw diskann::ANNException("Could not create temporary name.", -1);
    }

    std::string tempFile(temp);
    memset(temp, 0, MAX_PATH);
    // GetTempPath returns number of chars in path incl a trailing '\'
    int numCharsInTempPath = 0;
    if ((numCharsInTempPath = GetTempPathA(MAX_PATH, temp)) == 0) {
      throw diskann::ANNException("GetTempPathA failed with error code: ",
                                  GetLastError());
    }

    tempFile.erase(0, numCharsInTempPath);
    retFile = workingDir + "\\" + tempFile + "_" + suffix;
  } while (file_exists(
      retFile));  // To handle the rare case that the file may exist already.
  return retFile;
#endif
  return "";
}

inline void open_file_to_write(std::ofstream&     writer,
                               const std::string& filename) {
  writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  if (!file_exists(filename))
    writer.open(filename, std::ios::binary | std::ios::out);
  else
    writer.open(filename, std::ios::binary | std::ios::in | std::ios::out);

  if (writer.fail()) {
    diskann::cerr << std::string("Failed to open file") + filename +
                         " for write because "
                  << std::strerror(errno) << std::endl;
    throw diskann::ANNException(
        std::string("Failed to open file ") + filename +
            " for write because: " + std::strerror(errno),
        -1);
  }
}

inline _u64 get_file_size(const std::string& fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    _u64 end_pos = reader.tellg();
    reader.close();
    return end_pos;
  } else {
    diskann::cerr << "Could not open file: " << fname << std::endl;
    return 0;
  }
}

inline int delete_file(const std::string& fileName) {
  if (file_exists(fileName)) {
    auto rc = ::remove(fileName.c_str());
    if (rc != 0) {
      diskann::cerr
          << "Could not delete file: " << fileName
          << " even though it exists. This might indicate a permissions issue. "
             "If you see this message, please contact the diskann team."
          << std::endl;
    }
    return rc;
  } else {
    return 0;
  }
}

namespace diskann {
  static const size_t MAX_SIZE_OF_STREAMBUF = 2LL * 1024 * 1024 * 1024;

  enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3, COSINE = 4 };

  DISKANN_DLLEXPORT float calc_recall_set_tags(
      unsigned num_queries, unsigned* gold_std, unsigned dim_gs,
      unsigned* our_results_tags, unsigned dim_or, unsigned recall_at,
      unsigned subset_size, std::string gt_tag_filename,
      std::string current_tag_filename);

  inline void alloc_aligned(void** ptr, size_t size, size_t align) {
    *ptr = nullptr;
    assert(IS_ALIGNED(size, align));
#ifndef _WINDOWS
    *ptr = ::aligned_alloc(align, size);
#else
    *ptr = ::_aligned_malloc(size, align);  // note the swapped arguments!
#endif
    assert(*ptr != nullptr);
  }

  inline void realloc_aligned(void** ptr, size_t size, size_t align) {
    assert(IS_ALIGNED(size, align));
#ifdef _WINDOWS
    *ptr = ::_aligned_realloc(*ptr, size, align);
#endif
    assert(*ptr != nullptr);
  }

  inline void realloc_aligned(void** ptr, void** ptr_new, size_t old_size,
                              size_t new_size, size_t align) {
    assert(IS_ALIGNED(new_size, align));
#ifndef _WINDOWS
    alloc_aligned((void**) &ptr_new, new_size, align);
    memcpy(*ptr_new, *ptr, old_size);
    ::free(*ptr);
    *ptr = *ptr_new;
#endif
    assert(*ptr != nullptr);
  }

  inline void check_stop(std::string arnd) {
    int brnd;
    diskann::cout << arnd << std::endl;
    std::cin >> brnd;
  }

  inline void aligned_free(void* ptr) {
    if (ptr == nullptr) {
      return;
    }
#ifndef _WINDOWS
    free(ptr);
#else
    ::_aligned_free(ptr);
#endif
  }

  inline void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size,
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

  // get_bin_metadata functions START
  inline void get_bin_metadata_impl(std::basic_istream<char>& reader,
                                    size_t& nrows, size_t& ncols,
                                    size_t offset = 0) {
    int nrows_32, ncols_32;
    reader.seekg(offset, reader.beg);
    reader.read((char*) &nrows_32, sizeof(int));
    reader.read((char*) &ncols_32, sizeof(int));
    nrows = nrows_32;
    ncols = ncols_32;
  }

#ifdef EXEC_ENV_OLS
  inline void get_bin_metadata(MemoryMappedFiles& files,
                               const std::string& bin_file, size_t& nrows,
                               size_t& ncols, size_t offset = 0) {
    diskann::cout << "Getting metadata for file: " << bin_file << std::endl;
    auto     fc = files.getContent(bin_file);
    int      nrows_32, ncols_32;
    int32_t* metadata_ptr = (int32_t*) ((char*) fc._content + offset);
    nrows_32 = *metadata_ptr;
    ncols_32 = *(metadata_ptr + 1);
    nrows = nrows_32;
    ncols = ncols_32;
  }
#endif

  inline void get_bin_metadata(const std::string& bin_file, size_t& nrows,
                               size_t& ncols, size_t offset = 0) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    get_bin_metadata_impl(reader, nrows, ncols, offset);
  }
  // get_bin_metadata functions END

  template<typename T>
  inline std::string getValues(T* data, size_t num) {
    std::stringstream stream;
    stream << "[";
    for (size_t i = 0; i < num; i++) {
      stream << std::to_string(data[i]) << ",";
    }
    stream << "]" << std::endl;

    return stream.str();
  }

  // load_bin functions START
  template<typename T>
  inline void load_bin_impl(std::basic_istream<char>& reader, T*& data,
                            size_t& npts, size_t& dim, size_t file_offset = 0) {
    int npts_i32, dim_i32;

    reader.seekg(file_offset, reader.beg);
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    diskann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..."
                  << std::endl;

    data = new T[npts * dim];
    reader.read((char*) data, npts * dim * sizeof(T));
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  inline void load_bin(MemoryMappedFiles& files, const std::string& bin_file,
                       T*& data, size_t& npts, size_t& dim, size_t offset = 0) {
    diskann::cout << "Reading bin file " << bin_file.c_str()
                  << " at offset: " << offset << "..." << std::endl;
    auto fc = files.getContent(bin_file);

    uint32_t  t_npts, t_dim;
    uint32_t* contentAsIntPtr = (uint32_t*) ((char*) fc._content + offset);
    t_npts = *(contentAsIntPtr);
    t_dim = *(contentAsIntPtr + 1);

    npts = t_npts;
    dim = t_dim;

    data = (T*) ((char*) fc._content + offset +
                 2 * sizeof(uint32_t));  // No need to copy!
  }
#endif

  template<typename T>
  inline void load_bin(const std::string& bin_file, T*& data, size_t& npts,
                       size_t& dim, size_t offset = 0) {
    // OLS
    //_u64            read_blk_size = 64 * 1024 * 1024;
    // cached_ifstream reader(bin_file, read_blk_size);
    // size_t actual_file_size = reader.get_file_size();
    // END OLS
    diskann::cout << "Reading bin file " << bin_file.c_str() << " ..."
                  << std::endl;
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    //    uint64_t      fsize = reader.tellg();
    reader.seekg(0);

    load_bin_impl<T>(reader, data, npts, dim, offset);
  }
  // load_bin functions END

  inline void load_truthset(const std::string& bin_file, uint32_t*& ids,
                            float*& dists, size_t& npts, size_t& dim,
                            uint32_t** tags = nullptr) {
    std::ifstream reader(bin_file, std::ios::binary);
    diskann::cout << "Reading truthset file " << bin_file.c_str() << "..."
                  << std::endl;
    size_t actual_file_size = get_file_size(bin_file);

    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    diskann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..."
                  << std::endl;

    int truthset_type = -1;  // 1 means truthset has ids and distances, 2 means
                             // only ids, -1 is error
    size_t expected_file_size_with_dists =
        2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists)
      truthset_type = 1;

    size_t expected_file_size_just_ids =
        npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    size_t with_tags_actual_file_size =
        3 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_just_ids)
      truthset_type = 2;

    if (actual_file_size == with_tags_actual_file_size)
      truthset_type = 3;

    if (truthset_type == -1) {
      std::stringstream stream;
      stream << "Error. File size mismatch. File should have bin format, with "
                "npts followed by ngt followed by npts*ngt ids and optionally "
                "followed by npts*ngt distance values; actual size: "
             << actual_file_size
             << ", expected: " << expected_file_size_with_dists << " or "
             << expected_file_size_just_ids;
      diskann::cout << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    ids = new uint32_t[npts * dim];
    reader.read((char*) ids, npts * dim * sizeof(uint32_t));

    if ((truthset_type == 1) || (truthset_type == 3)) {
      dists = new float[npts * dim];
      reader.read((char*) dists, npts * dim * sizeof(float));
    }
    if (truthset_type == 3) {
      *tags = new uint32_t[npts * dim];
      reader.read((char*) *tags, npts * dim * sizeof(uint32_t));
    }
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  inline void load_bin(MemoryMappedFiles& files, const std::string& bin_file,
                       std::unique_ptr<T[]>& data, size_t& npts, size_t& dim,
                       size_t offset = 0) {
    T* ptr;
    load_bin<T>(files, bin_file, ptr, npts, dim, offset);
    data.reset(ptr);
  }

#endif

  template<typename T>
  inline void load_bin(const std::string& bin_file, std::unique_ptr<T[]>& data,
                       size_t& npts, size_t& dim, size_t offset = 0) {
    T* ptr;
    load_bin<T>(bin_file, ptr, npts, dim, offset);
    data.reset(ptr);
  }

  template<typename T>
  inline uint64_t save_bin(const std::string& filename, T* data, size_t npts,
                           size_t ndims, size_t offset = 0) {
    std::ofstream writer;
    open_file_to_write(writer, filename);

    diskann::cout << "Writing bin: " << filename.c_str() << std::endl;
    writer.seekp(offset, writer.beg);
    int    npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
    writer.write((char*) &npts_i32, sizeof(int));
    writer.write((char*) &ndims_i32, sizeof(int));
    diskann::cout << "bin: #pts = " << npts << ", #dims = " << ndims
                  << ", size = " << bytes_written << "B" << std::endl;

    writer.write((char*) data, npts * ndims * sizeof(T));
    writer.close();
    diskann::cout << "Finished writing bin." << std::endl;
    return bytes_written;
  }

  // load_aligned_bin functions START

  template<typename T>
  inline void load_aligned_bin_impl(std::basic_istream<char>& reader, T*& data,
                                    size_t& npts, size_t& dim,
                                    size_t& rounded_dim, size_t offset = 0) {
    int npts_i32, dim_i32;
    reader.seekg(offset, reader.beg);
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));

    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;
    rounded_dim = ROUND_UP(dim, 8);
    diskann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim
                  << ", aligned_dim = " << rounded_dim << "..." << std::flush;
    size_t allocSize = npts * rounded_dim * sizeof(T);
    alloc_aligned(((void**) &data), allocSize, 8 * sizeof(T));

    for (size_t i = 0; i < npts; i++) {
      reader.read((char*) (data + i * rounded_dim), dim * sizeof(T));
      memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
    diskann::cout << " Allocated " << allocSize << "bytes and copied data "
                  << std::endl;
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  inline void load_aligned_bin(MemoryMappedFiles& files,
                               const std::string& bin_file, T*& data,
                               size_t& npts, size_t& dim, size_t& rounded_dim,
                               size_t offset = 0) {
    diskann::cout << "Reading bin file " << bin_file << " at offset: " << offset
                  << "..." << std::flush;
    FileContent fc = files.getContent(bin_file);
    // ContentBuf               buf((char*) fc._content, fc._size);

    char* read_addr = (((char*) fc._content) + offset);

    int npts_32 = *((int*) read_addr);
    int ndim_32 = *((int*) (read_addr + sizeof(int)));

    npts = (uint32_t) npts_32;
    dim = (uint32_t) ndim_32;

    char* data_start = ((char*) fc._content) + offset + 2 * sizeof(int);
    rounded_dim = ROUND_UP(dim, 8);
    diskann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim
                  << ", aligned_dim = " << rounded_dim << "..." << std::flush;

    size_t allocSize = npts * rounded_dim * sizeof(T);

    diskann::cout << "allocating aligned memory, " << allocSize << " bytes..."
                  << std::flush;

    alloc_aligned(((void**) &data), allocSize, 8 * sizeof(T));
    diskann::cout << "done. Copying data..." << std::flush;

    for (size_t i = 0; i < npts; i++) {
      memcpy((data + i * rounded_dim), data_start, dim * sizeof(T));
      memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
      data_start += dim * sizeof(T);
    }

    if (data_start - read_addr != dim * sizeof(T) * npts + 2 * sizeof(int)) {
      diskann::cerr << "Read " << data_start - read_addr
                    << " bytes of data instead of: " << dim * sizeof(T) * npts
                    << std::endl;
    }
    diskann::cout << " done." << std::endl;
  }
#endif

  template<typename T>
  inline void load_aligned_bin(const std::string& bin_file, T*& data,
                               size_t& npts, size_t& dim, size_t& rounded_dim,
                               size_t offset = 0) {
    diskann::cout << "Reading bin file " << bin_file << " at offset " << offset
                  << "..." << std::flush;
    // START OLS
    //_u64            read_blk_size = 64 * 1024 * 1024;
    // cached_ifstream reader(bin_file, read_blk_size);
    // size_t actual_file_size = reader.get_file_size();
    // END OLS

    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    //    uint64_t      fsize = reader.tellg();
    reader.seekg(0);

    load_aligned_bin_impl(reader, data, npts, dim, rounded_dim, offset);
  }

  template<typename InType, typename OutType>
  void convert_types(const InType* srcmat, OutType* destmat, size_t npts,
                     size_t dim) {
#pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (_s64) npts; i++) {
      for (uint64_t j = 0; j < dim; j++) {
        destmat[i * dim + j] = (OutType) srcmat[i * dim + j];
      }
    }
  }

  template<typename T>
  inline void load_aligned_bin(const std::string&    bin_file,
                               std::unique_ptr<T[]>& data, size_t& npts,
                               size_t& dim, size_t& rounded_dim,
                               size_t offset = 0) {
    T* ptr;
    load_aligned_bin<T>(bin_file, ptr, npts, dim, rounded_dim, offset);
    data.reset(ptr);
  }

  template<typename T>
  inline uint64_t save_data_in_base_dimensions(const std::string& filename,
                                               T* data, size_t npts,
                                               size_t ndims, size_t aligned_dim,
                                               size_t offset = 0) {
    std::ofstream writer;  //(filename, std::ios::binary | std::ios::out);
    open_file_to_write(writer, filename);
    int  npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    _u64 bytes_written = 2 * sizeof(uint32_t) + npts * ndims * sizeof(T);
    writer.seekp(offset, writer.beg);
    writer.write((char*) &npts_i32, sizeof(int));
    writer.write((char*) &ndims_i32, sizeof(int));
    for (size_t i = 0; i < npts; i++) {
      writer.write((char*) (data + i * aligned_dim), ndims * sizeof(T));
    }
    writer.close();
    return bytes_written;
  }

  template<typename T>
  inline void copy_aligned_data_from_file(const std::string bin_file, T*& data,
                                          size_t& npts, size_t& dim,
                                          const size_t& rounded_dim,
                                          size_t        offset = 0) {
    if (data == nullptr) {
      diskann::cout << "Memory was not allocated for " << data
                    << " before calling the load function. Exiting..."
                    << std::endl;
      exit(-1);
    }
    std::ifstream reader(bin_file, std::ios::binary);
    reader.seekg(offset, reader.beg);

    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    for (size_t i = 0; i < npts; i++) {
      reader.read((char*) (data + i * rounded_dim), dim * sizeof(T));
      memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
  }

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector(const char* vec, size_t vecsize) {
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64)
      _mm_prefetch((const char*) vec + d, _MM_HINT_T0);
  }

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector_l2(const char* vec, size_t vecsize) {
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64)
      _mm_prefetch((const char*) vec + d, _MM_HINT_T1);
  }

  // NOTE: Implementation in utils.cpp.
  void block_convert(std::ofstream& writr, std::ifstream& readr,
                     float* read_buf, _u64 npts, _u64 ndims);

  DISKANN_DLLEXPORT void normalize_data_file(const std::string& inFileName,
                                             const std::string& outFileName);

  template<typename T>
  Distance<T>* get_distance_function(Metric m);
}  // namespace diskann

struct PivotContainer {
  PivotContainer() = default;

  PivotContainer(size_t pivo_id, float pivo_dist)
      : piv_id{pivo_id}, piv_dist{pivo_dist} {
  }

  bool operator<(const PivotContainer& p) const {
    return p.piv_dist < piv_dist;
  }

  bool operator>(const PivotContainer& p) const {
    return p.piv_dist > piv_dist;
  }

  size_t piv_id;
  float  piv_dist;
};

inline bool validate_file_size(const std::string& name) {
  std::ifstream in(std::string(name), std::ios::binary);
  in.seekg(0, in.end);
  size_t actual_file_size = in.tellg();
  in.seekg(0, in.beg);
  size_t expected_file_size;
  in.read((char*) &expected_file_size, sizeof(uint64_t));
  if (actual_file_size != expected_file_size) {
    diskann::cerr << "Error loading" << name
                  << ". Expected "
                     "size (metadata): "
                  << expected_file_size
                  << ", actual file size : " << actual_file_size << ". Exiting."
                  << std::endl;
    in.close();
    return false;
  }
  in.close();
  return true;
}

template<typename T>
diskann::Distance<T>* get_distance_function(diskann::Metric m);

extern bool AvxSupportedCPU;
extern bool Avx2SupportedCPU;

#ifdef _WINDOWS
#include <intrin.h>
#include <Psapi.h>

inline size_t getMemoryUsage() {
  PROCESS_MEMORY_COUNTERS_EX pmc;
  GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*) &pmc,
                       sizeof(pmc));
  return pmc.PrivateUsage;
}

inline std::string getWindowsErrorMessage(DWORD lastError) {
  char* errorText;
  FormatMessageA(
      // use system message tables to retrieve error text
      FORMAT_MESSAGE_FROM_SYSTEM
          // allocate buffer on local heap for error text
          | FORMAT_MESSAGE_ALLOCATE_BUFFER
          // Important! will fail otherwise, since we're not
          // (and CANNOT) pass insertion parameters
          | FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL,  // unused with FORMAT_MESSAGE_FROM_SYSTEM
      lastError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR) &errorText,  // output
      0,                   // minimum size for output buffer
      NULL);               // arguments - see note

  return errorText != nullptr ? std::string(errorText) : std::string();
}

inline void printProcessMemory(const char* message) {
  PROCESS_MEMORY_COUNTERS counters;
  HANDLE                  h = GetCurrentProcess();
  GetProcessMemoryInfo(h, &counters, sizeof(counters));
  diskann::cout << message << " [Peaking Working Set size: "
                << counters.PeakWorkingSetSize * 1.0 / (1024.0 * 1024 * 1024)
                << "GB Working set size: "
                << counters.WorkingSetSize * 1.0 / (1024.0 * 1024 * 1024)
                << "GB Private bytes "
                << counters.PagefileUsage * 1.0 / (1024 * 1024 * 1024) << "GB]"
                << std::endl;
}
#else

// need to check and change this
inline bool avx2Supported() {
  return true;
}
inline void printProcessMemory(const char*) {
}
#endif
