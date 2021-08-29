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

#include "logger.h"
#include "cached_io.h"
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

typedef uint64_t _u64;
typedef int64_t  _s64;
typedef uint32_t _u32;
typedef int32_t  _s32;
typedef uint16_t _u16;
typedef int16_t  _s16;
typedef uint8_t  _u8;
typedef int8_t   _s8;

namespace diskann {
  static const size_t MAX_SIZE_OF_STREAMBUF = 2LL * 1024 * 1024 * 1024;

  enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3 };

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

  inline void aligned_free(void* ptr) {
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
                                    size_t& nrows, size_t& ncols) {
    int nrows_32, ncols_32;
    reader.read((char*) &nrows_32, sizeof(int));
    reader.read((char*) &ncols_32, sizeof(int));
    nrows = nrows_32;
    ncols = ncols_32;
  }

#ifdef EXEC_ENV_OLS
  inline void get_bin_metadata(MemoryMappedFiles& files,
                               const std::string& bin_file, size_t& nrows,
                               size_t& ncols) {
    diskann::cout << "Getting metadata for file: " << bin_file << std::endl;
    auto                     fc = files.getContent(bin_file);
    auto                     cb = ContentBuf((char*) fc._content, fc._size);
    std::basic_istream<char> reader(&cb);
    get_bin_metadata_impl(reader, nrows, ncols);
  }
#endif

  inline void get_bin_metadata(const std::string& bin_file, size_t& nrows,
                               size_t& ncols) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    get_bin_metadata_impl(reader, nrows, ncols);
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
  inline void load_bin_impl(std::basic_istream<char>& reader,
                            size_t actual_file_size, T*& data, size_t& npts,
                            size_t& dim) {
    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    diskann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..."
                  << std::endl;

    size_t expected_actual_file_size =
        npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size) {
      std::stringstream stream;
      stream << "Error. File size mismatch. Actual size is " << actual_file_size
             << " while expected size is  " << expected_actual_file_size
             << " npts = " << npts << " dim = " << dim
             << " size of <T>= " << sizeof(T) << std::endl;
      diskann::cout << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    data = new T[npts * dim];
    reader.read((char*) data, npts * dim * sizeof(T));
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  inline void load_bin(MemoryMappedFiles& files, const std::string& bin_file,
                       T*& data, size_t& npts, size_t& dim) {
    diskann::cout << "Reading bin file " << bin_file.c_str() << " ..."
                  << std::endl;

    auto fc = files.getContent(bin_file);

    uint32_t  t_npts, t_dim;
    uint32_t* contentAsIntPtr = (uint32_t*) (fc._content);
    t_npts = *(contentAsIntPtr);
    t_dim = *(contentAsIntPtr + 1);

    npts = t_npts;
    dim = t_dim;

    auto actual_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != fc._size) {
      std::stringstream stream;
      stream << "Error. File size mismatch. Actual size is " << fc._size
             << " while expected size is  " << actual_file_size
             << " npts = " << npts << " dim = " << dim
             << " size of <T>= " << sizeof(T) << std::endl;
      diskann::cout << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    data =
        (T*) ((char*) fc._content + 2 * sizeof(uint32_t));  // No need to copy!
  }
#endif

  inline void wait_for_keystroke() {
    int a;
    std::cout << "Press any number to continue.." << std::endl;
    std::cin >> a;
  }

  template<typename T>
  inline void load_bin(const std::string& bin_file, T*& data, size_t& npts,
                       size_t& dim) {
    diskann::cout << "Reading bin file " << bin_file.c_str() << " ..."
                  << std::endl;
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    uint64_t      fsize = reader.tellg();
    reader.seekg(0);

    load_bin_impl<T>(reader, fsize, data, npts, dim);
  }
  // load_bin functions END

  inline void load_truthset(const std::string& bin_file, uint32_t*& ids,
                            float*& dists, size_t& npts, size_t& dim) {
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    diskann::cout << "Reading truthset file " << bin_file.c_str() << " ..."
                  << std::endl;
    size_t actual_file_size = reader.get_file_size();

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

    if (actual_file_size == expected_file_size_just_ids)
      truthset_type = 2;

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

    if (truthset_type == 1) {
      dists = new float[npts * dim];
      reader.read((char*) dists, npts * dim * sizeof(float));
    }
  }

  inline void prune_truthset_for_range(const std::string& bin_file, float range, std::vector<std::vector<_u32>> &groundtruth, 
                            size_t& npts) {
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    diskann::cout << "Reading truthset file " << bin_file.c_str() << " ..."
                  << std::endl;
    size_t actual_file_size = reader.get_file_size();

    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    _u64 dim = (unsigned) dim_i32;
    _u32* ids;
    float* dists;

    diskann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..."
                  << std::endl;

    int truthset_type = -1;  // 1 means truthset has ids and distances, 2 means
                             // only ids, -1 is error
    size_t expected_file_size_with_dists =
        2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists)
      truthset_type = 1;

    if (truthset_type == -1) {
      std::stringstream stream;
      stream << "Error. File size mismatch. File should have bin format, with "
                "npts followed by ngt followed by npts*ngt ids and optionally "
                "followed by npts*ngt distance values; actual size: "
             << actual_file_size
             << ", expected: " << expected_file_size_with_dists;
      diskann::cout << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    ids = new uint32_t[npts * dim];
    reader.read((char*) ids, npts * dim * sizeof(uint32_t));
 
    if (truthset_type == 1) {
      dists = new float[npts * dim];
      reader.read((char*) dists, npts * dim * sizeof(float));
    }
    float min_dist = std::numeric_limits<float>::max();
    float max_dist = 0;
    groundtruth.resize(npts);
    for (_u32 i = 0; i < npts; i++) {
      groundtruth[i].clear();
      for (_u32 j = 0; j < dim; j++) {
        if (dists[i*dim + j] <= range) {
          groundtruth[i].emplace_back(ids[i*dim+j]);
        }
        min_dist = min_dist > dists[i*dim+j] ? dists[i*dim + j] : min_dist;
        max_dist = max_dist < dists[i*dim+j] ? dists[i*dim + j] : max_dist;
      }
    }
    std::cout<<"Min dist: " << min_dist <<", Max dist: "<< max_dist << std::endl;
    delete[] ids;
    delete[] dists;
  }


  inline void load_range_truthset(const std::string& bin_file, std::vector<std::vector<_u32>> &groundtruth, _u64 & gt_num) {
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    diskann::cout << "Reading truthset file " << bin_file.c_str() << " ..."
                  << std::endl;
    size_t actual_file_size = reader.get_file_size();

    int npts_u32, total_u32;
    reader.read((char*) &npts_u32, sizeof(int));
    reader.read((char*) &total_u32, sizeof(int));

    gt_num = (_u64) npts_u32;
    _u64 total_res = (_u64) total_u32;
    
    diskann::cout << "Metadata: #pts = " << gt_num << ", #total_results = " << total_res << "..."
                  << std::endl;

    size_t expected_file_size =
        2*sizeof(_u32) + gt_num*sizeof(_u32) + total_res*sizeof(_u32);

    if (actual_file_size != expected_file_size) {
      std::stringstream stream;
      stream << "Error. File size mismatch in range truthset. actual size: "
             << actual_file_size
             << ", expected: " << expected_file_size;
      diskann::cout << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    groundtruth.clear();
    groundtruth.resize(gt_num);
    std::vector<_u32> gt_count(gt_num);
    


    reader.read((char*) gt_count.data(), sizeof(_u32)*gt_num);

    std::vector<_u32> gt_stats(gt_count);
    std::sort(gt_stats.begin(), gt_stats.end());
 
    std::cout<<"GT count percentiles:" << std::endl;
      for (_u32 p = 0; p < 100; p += 5)
    std::cout << "percentile " << p << ": "
              << gt_stats[std::floor((p / 100.0) * gt_num)] << std::endl;
  std::cout << "percentile 100"
            << ": " << gt_stats[gt_num - 1] << std::endl;


   for (_u32 i = 0; i < gt_num; i++) {
      groundtruth[i].clear();
      groundtruth[i].resize(gt_count[i]);
      if (gt_count[i]!=0)
      reader.read((char*) groundtruth[i].data(), sizeof(_u32)*gt_count[i]);
   }
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  inline void load_bin(MemoryMappedFiles& files, const std::string& bin_file,
                       std::unique_ptr<T[]>& data, size_t& npts, size_t& dim) {
    T* ptr;
    load_bin<T>(files, bin_file, ptr, npts, dim);
    data.reset(ptr);
  }
#endif

  template<typename T>
  inline void load_bin(const std::string& bin_file, std::unique_ptr<T[]>& data,
                       size_t& npts, size_t& dim) {
    T* ptr;
    load_bin<T>(bin_file, ptr, npts, dim);
    data.reset(ptr);
  }

  template<typename T>
  inline void save_bin(const std::string& filename, T* data, size_t npts,
                       size_t ndims) {
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    diskann::cout << "Writing bin: " << filename.c_str() << std::endl;
    int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    writer.write((char*) &npts_i32, sizeof(int));
    writer.write((char*) &ndims_i32, sizeof(int));
    diskann::cout << "bin: #pts = " << npts << ", #dims = " << ndims
                  << ", size = " << npts * ndims * sizeof(T) + 2 * sizeof(int)
                  << "B" << std::endl;

    writer.write((char*) data, npts * ndims * sizeof(T));
    writer.close();
    diskann::cout << "Finished writing bin." << std::endl;
  }


  template<typename T>
  inline void load_aligned_bin_impl(std::basic_istream<char>& reader,
                                    size_t actual_file_size, T*& data,
                                    size_t& npts, size_t& dim,
                                    size_t& rounded_dim) {
    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    size_t expected_actual_file_size =
        npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size) {
      std::stringstream stream;
      stream << "Error. File size mismatch. Actual size is " << actual_file_size
             << " while expected size is  " << expected_actual_file_size
             << " npts = " << npts << " dim = " << dim
             << " size of <T>= " << sizeof(T) << std::endl;
      diskann::cout << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    rounded_dim = ROUND_UP(dim, 8);
    diskann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim
                  << ", aligned_dim = " << rounded_dim << "..." << std::flush;
    size_t allocSize = npts * rounded_dim * sizeof(T);
    diskann::cout << "allocating aligned memory, " << allocSize << " bytes..."
                  << std::flush;
    alloc_aligned(((void**) &data), allocSize, 8 * sizeof(T));
    diskann::cout << "done. Copying data..." << std::flush;

    for (size_t i = 0; i < npts; i++) {
      reader.read((char*) (data + i * rounded_dim), dim * sizeof(T));
      memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
    diskann::cout << " done." << std::endl;
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  inline void load_aligned_bin(MemoryMappedFiles& files,
                               const std::string& bin_file, T*& data,
                               size_t& npts, size_t& dim, size_t& rounded_dim) {
    diskann::cout << "Reading bin file " << bin_file << " ..." << std::flush;
    FileContent              fc = files.getContent(bin_file);
    ContentBuf               buf((char*) fc._content, fc._size);
    std::basic_istream<char> reader(&buf);

    size_t actual_file_size = fc._size;
    load_aligned_bin_impl(reader, actual_file_size, data, npts, dim,
                          rounded_dim);
  }
#endif

  template<typename T>
  inline void load_aligned_bin(const std::string& bin_file, T*& data,
                               size_t& npts, size_t& dim, size_t& rounded_dim) {
    diskann::cout << "Reading bin file " << bin_file << " ..." << std::flush;

    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    uint64_t      fsize = reader.tellg();
    reader.seekg(0);

    load_aligned_bin_impl(reader, fsize, data, npts, dim, rounded_dim);
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

  // this function will take in_file of n*d dimensions and save the output as a
  // floating point matrix
  // with n*(d+1) dimensions. All vectors are scaled by a large value M so that
  // the norms are <=1 and the final coordinate is set so that the resulting
  // norm (in d+1 coordinates) is equal to 1 this is a classical transformation
  // from MIPS to L2 search from "On Symmetric and Asymmetric LSHs for Inner
  // Product Search" by Neyshabur and Srebro

  template<typename T>
  float prepare_base_for_inner_products(const std::string in_file,
                                        const std::string out_file) {
    std::cout << "Pre-processing base file by adding extra coordinate"
              << std::endl;
    std::ifstream in_reader(in_file.c_str(), std::ios::binary);
    std::ofstream out_writer(out_file.c_str(), std::ios::binary);
    _u64          npts, in_dims, out_dims;
    float         max_norm = 0;

    _u32 npts32, dims32;
    in_reader.read((char*) &npts32, sizeof(uint32_t));
    in_reader.read((char*) &dims32, sizeof(uint32_t));

    npts = npts32;
    in_dims = dims32;
    out_dims = in_dims + 1;
    _u32 outdims32 = (_u32) out_dims;

    out_writer.write((char*) &npts32, sizeof(uint32_t));
    out_writer.write((char*) &outdims32, sizeof(uint32_t));

    size_t               BLOCK_SIZE = 100000;
    size_t               block_size = npts <= BLOCK_SIZE ? npts : BLOCK_SIZE;
    std::unique_ptr<T[]> in_block_data =
        std::make_unique<T[]>(block_size * in_dims);
    std::unique_ptr<float[]> out_block_data =
        std::make_unique<float[]>(block_size * out_dims);

    std::memset(out_block_data.get(), 0, sizeof(float) * block_size * out_dims);
    _u64 num_blocks = DIV_ROUND_UP(npts, block_size);

    std::vector<float> norms(npts, 0);

    for (_u64 b = 0; b < num_blocks; b++) {
      _u64 start_id = b * block_size;
      _u64 end_id = (b + 1) * block_size < npts ? (b + 1) * block_size : npts;
      _u64 block_pts = end_id - start_id;
      in_reader.read((char*) in_block_data.get(),
                     block_pts * in_dims * sizeof(T));
      for (_u64 p = 0; p < block_pts; p++) {
        for (_u64 j = 0; j < in_dims; j++) {
          norms[start_id + p] +=
              in_block_data[p * in_dims + j] * in_block_data[p * in_dims + j];
        }
        max_norm =
            max_norm > norms[start_id + p] ? max_norm : norms[start_id + p];
      }
    }

    max_norm = std::sqrt(max_norm);

    in_reader.seekg(2 * sizeof(_u32), std::ios::beg);
    for (_u64 b = 0; b < num_blocks; b++) {
      _u64 start_id = b * block_size;
      _u64 end_id = (b + 1) * block_size < npts ? (b + 1) * block_size : npts;
      _u64 block_pts = end_id - start_id;
      in_reader.read((char*) in_block_data.get(),
                     block_pts * in_dims * sizeof(T));
      for (_u64 p = 0; p < block_pts; p++) {
        for (_u64 j = 0; j < in_dims; j++) {
          out_block_data[p * out_dims + j] =
              in_block_data[p * in_dims + j] / max_norm;
        }
        float res = 1 - (norms[start_id + p] / (max_norm * max_norm));
        res = res <= 0 ? 0 : std::sqrt(res);
        out_block_data[p * out_dims + out_dims - 1] = res;
      }
      out_writer.write((char*) out_block_data.get(),
                       block_pts * out_dims * sizeof(float));
    }
    out_writer.close();
    return max_norm;
  }

  // plain saves data as npts X ndims array into filename
  template<typename T>
  void save_Tvecs(const char* filename, T* data, size_t npts, size_t ndims) {
    std::string fname(filename);

    // create cached ofstream with 64MB cache
    cached_ofstream writer(fname, 64 * 1048576);

    unsigned dims_u32 = (unsigned) ndims;

    // start writing
    for (uint64_t i = 0; i < npts; i++) {
      // write dims in u32
      writer.write((char*) &dims_u32, sizeof(unsigned));

      // get cur point in data
      T* cur_pt = data + i * ndims;
      writer.write((char*) cur_pt, ndims * sizeof(T));
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
};  // namespace diskann

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

inline bool file_exists(const std::string& name) {
  struct stat buffer;
  auto        val = stat(name.c_str(), &buffer);
  diskann::cout << " Stat(" << name.c_str() << ") returned: " << val
                << std::endl;
  return (val == 0);
}

inline _u64 get_file_size(const std::string& fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    _u64 end_pos = reader.tellg();
    diskann::cout << " Tellg: " << reader.tellg() << " as u64: " << end_pos
                  << std::endl;
    reader.close();
    return end_pos;
  } else {
    diskann::cout << "Could not open file: " << fname << std::endl;
    return 0;
  }
}

inline bool validate_file_size(const std::string& name) {
  std::ifstream in(std::string(name), std::ios::binary);
  in.seekg(0, in.end);
  size_t actual_file_size = in.tellg();
  in.seekg(0, in.beg);
  size_t expected_file_size;
  in.read((char*) &expected_file_size, sizeof(uint64_t));
  if (actual_file_size != expected_file_size) {
    diskann::cout << "Error loading" << name
                  << ". Expected "
                     "size (metadata): "
                  << expected_file_size
                  << ", actual file size : " << actual_file_size
                  << ". Exitting." << std::endl;
    in.close();
    return false;
  }
  in.close();
  return true;
}

#ifdef _WINDOWS
#include <intrin.h>
#include <Psapi.h>

inline void printProcessMemory(const char* message) {
  PROCESS_MEMORY_COUNTERS counters;
  HANDLE                  h = GetCurrentProcess();
  GetProcessMemoryInfo(h, &counters, sizeof(counters));
  diskann::cout << message << " [Peaking Working Set size: "
                << counters.PeakWorkingSetSize * 1.0 / (1024 * 1024 * 1024)
                << "GB Working set size: "
                << counters.WorkingSetSize * 1.0 / (1024 * 1024 * 1024)
                << "GB Private bytes "
                << counters.PagefileUsage * 1.0 / (1024 * 1024 * 1024) << "GB]"
                << std::endl;
}
#else

// need to check and change this
inline bool avx2Supported() {
  return true;
}

inline void printProcessMemory(const char* message) {
  diskann::cout << message << std::endl;
}
#endif

extern bool AvxSupportedCPU;
extern bool Avx2SupportedCPU;
