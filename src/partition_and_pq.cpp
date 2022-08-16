// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <math_utils.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>

#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#include "logger.h"
#include "exceptions.h"
#include "index.h"
#include "parameters.h"
#include "tsl/robin_set.h"
#include "utils.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <typeinfo>
#include <tsl/robin_map.h>

#include <cassert>
#include "memory_mapper.h"
#include "partition_and_pq.h"
#ifdef _WINDOWS
#include <xmmintrin.h>
#endif

// block size for reading/ processing large files and matrices in blocks
#define BLOCK_SIZE 5000000

//#define SAVE_INFLATED_PQ true

template<typename T>
void gen_random_slice(const std::string base_file,
                      const std::string output_prefix, double sampling_rate) {
  _u64            read_blk_size = 64 * 1024 * 1024;
  cached_ifstream base_reader(base_file.c_str(), read_blk_size);
  std::ofstream sample_writer(std::string(output_prefix + "_data.bin").c_str(),
                              std::ios::binary);
  std::ofstream sample_id_writer(
      std::string(output_prefix + "_ids.bin").c_str(), std::ios::binary);

  std::random_device
               rd;  // Will be used to obtain a seed for the random number engine
  auto         x = rd();
  std::mt19937 generator(
      x);  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distribution(0, 1);

  size_t   npts, nd;
  uint32_t npts_u32, nd_u32;
  uint32_t num_sampled_pts_u32 = 0;
  uint32_t one_const = 1;

  base_reader.read((char *) &npts_u32, sizeof(uint32_t));
  base_reader.read((char *) &nd_u32, sizeof(uint32_t));
  diskann::cout << "Loading base " << base_file << ". #points: " << npts_u32
                << ". #dim: " << nd_u32 << "." << std::endl;
  sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.write((char *) &nd_u32, sizeof(uint32_t));
  sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.write((char *) &one_const, sizeof(uint32_t));

  npts = npts_u32;
  nd = nd_u32;
  std::unique_ptr<T[]> cur_row = std::make_unique<T[]>(nd);

  for (size_t i = 0; i < npts; i++) {
    base_reader.read((char *) cur_row.get(), sizeof(T) * nd);
    float sample = distribution(generator);
    if (sample < sampling_rate) {
      sample_writer.write((char *) cur_row.get(), sizeof(T) * nd);
      uint32_t cur_i_u32 = (_u32) i;
      sample_id_writer.write((char *) &cur_i_u32, sizeof(uint32_t));
      num_sampled_pts_u32++;
    }
  }
  sample_writer.seekp(0, std::ios::beg);
  sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.seekp(0, std::ios::beg);
  sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.close();
  sample_id_writer.close();
  diskann::cout << "Wrote " << num_sampled_pts_u32
                << " points to sample file: " << output_prefix + "_data.bin"
                << std::endl;
}

// streams data from the file, and samples each vector with probability p_val
// and returns a matrix of size slice_size* ndims as floating point type.
// the slice_size and ndims are set inside the function.

/***********************************
 * Reimplement using gen_random_slice(const T* inputdata,...)
 ************************************/

template<typename T>
void gen_random_slice(const std::string data_file, double p_val,
                      float *&sampled_data, size_t &slice_size, size_t &ndims) {
  size_t                          npts;
  uint32_t                        npts32, ndims32;
  std::vector<std::vector<float>> sampled_vectors;

  // amount to read in one shot
  _u64 read_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(data_file.c_str(), read_blk_size);

  // metadata: npts, ndims
  base_reader.read((char *) &npts32, sizeof(unsigned));
  base_reader.read((char *) &ndims32, sizeof(unsigned));
  npts = npts32;
  ndims = ndims32;

  std::unique_ptr<T[]> cur_vector_T = std::make_unique<T[]>(ndims);
  p_val = p_val < 1 ? p_val : 1;

  std::random_device rd;  // Will be used to obtain a seed for the random number
  size_t             x = rd();
  std::mt19937       generator((unsigned) x);
  std::uniform_real_distribution<float> distribution(0, 1);

  for (size_t i = 0; i < npts; i++) {
    base_reader.read((char *) cur_vector_T.get(), ndims * sizeof(T));
    float rnd_val = distribution(generator);
    if (rnd_val < p_val) {
      std::vector<float> cur_vector_float;
      for (size_t d = 0; d < ndims; d++)
        cur_vector_float.push_back(cur_vector_T[d]);
      sampled_vectors.push_back(cur_vector_float);
    }
  }
  slice_size = sampled_vectors.size();
  sampled_data = new float[slice_size * ndims];
  for (size_t i = 0; i < slice_size; i++) {
    for (size_t j = 0; j < ndims; j++) {
      sampled_data[i * ndims + j] = sampled_vectors[i][j];
    }
  }
}

// same as above, but samples from the matrix inputdata instead of a file of
// npts*ndims to return sampled_data of size slice_size*ndims.
template<typename T>
void gen_random_slice(const T *inputdata, size_t npts, size_t ndims,
                      double p_val, float *&sampled_data, size_t &slice_size) {
  std::vector<std::vector<float>> sampled_vectors;
  const T *                       cur_vector_T;

  p_val = p_val < 1 ? p_val : 1;

  std::random_device
               rd;  // Will be used to obtain a seed for the random number engine
  size_t       x = rd();
  std::mt19937 generator(
      (unsigned) x);  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distribution(0, 1);

  for (size_t i = 0; i < npts; i++) {
    cur_vector_T = inputdata + ndims * i;
    float rnd_val = distribution(generator);
    if (rnd_val < p_val) {
      std::vector<float> cur_vector_float;
      for (size_t d = 0; d < ndims; d++)
        cur_vector_float.push_back(cur_vector_T[d]);
      sampled_vectors.push_back(cur_vector_float);
    }
  }
  slice_size = sampled_vectors.size();
  sampled_data = new float[slice_size * ndims];
  for (size_t i = 0; i < slice_size; i++) {
    for (size_t j = 0; j < ndims; j++) {
      sampled_data[i * ndims + j] = sampled_vectors[i][j];
    }
  }
}

// given training data in train_data of dimensions num_train * dim, generate PQ
// pivots using k-means algorithm to partition the co-ordinates into
// num_pq_chunks (if it divides dimension, else rounded) chunks, and runs
// k-means in each chunk to compute the PQ pivots and stores in bin format in
// file pq_pivots_path as a s num_centers*dim floating point binary file
int generate_pq_pivots(const float *passed_train_data, size_t num_train,
                       unsigned dim, unsigned num_centers,
                       unsigned num_pq_chunks, unsigned max_k_means_reps,
                       std::string pq_pivots_path, bool make_zero_mean) {
  if (num_pq_chunks > dim) {
    diskann::cout << " Error: number of chunks more than dimension"
                  << std::endl;
    return -1;
  }

  std::unique_ptr<float[]> train_data =
      std::make_unique<float[]>(num_train * dim);
  std::memcpy(train_data.get(), passed_train_data,
              num_train * dim * sizeof(float));

  for (uint64_t i = 0; i < num_train; i++) {
    for (uint64_t j = 0; j < dim; j++) {
      if (passed_train_data[i * dim + j] != train_data[i * dim + j])
        diskann::cout << "error in copy" << std::endl;
    }
  }

  std::unique_ptr<float[]> full_pivot_data;

  if (file_exists(pq_pivots_path)) {
    size_t file_dim, file_num_centers;
    diskann::load_bin<float>(pq_pivots_path, full_pivot_data, file_num_centers,
                             file_dim, METADATA_SIZE);
    if (file_dim == dim && file_num_centers == num_centers) {
      diskann::cout << "PQ pivot file exists. Not generating again"
                    << std::endl;
      return -1;
    }
  }

  // Calculate centroid and center the training data
  std::unique_ptr<float[]> centroid = std::make_unique<float[]>(dim);
  for (uint64_t d = 0; d < dim; d++) {
    centroid[d] = 0;
  }
  if (make_zero_mean) {  // If we use L2 distance, there is an option to
                         // translate all vectors to make them centered and then
                         // compute PQ. This needs to be set to false when using
                         // PQ for MIPS as such translations dont preserve inner
                         // products.
    for (uint64_t d = 0; d < dim; d++) {
      for (uint64_t p = 0; p < num_train; p++) {
        centroid[d] += train_data[p * dim + d];
      }
      centroid[d] /= num_train;
    }

    for (uint64_t d = 0; d < dim; d++) {
      for (uint64_t p = 0; p < num_train; p++) {
        train_data[p * dim + d] -= centroid[d];
      }
    }
  }

  std::vector<uint32_t> rearrangement;
  std::vector<uint32_t> chunk_offsets;

  size_t low_val = (size_t) std::floor((double) dim / (double) num_pq_chunks);
  size_t high_val = (size_t) std::ceil((double) dim / (double) num_pq_chunks);
  size_t max_num_high = dim - (low_val * num_pq_chunks);
  size_t cur_num_high = 0;
  size_t cur_bin_threshold = high_val;

  std::vector<std::vector<uint32_t>> bin_to_dims(num_pq_chunks);
  tsl::robin_map<uint32_t, uint32_t> dim_to_bin;
  std::vector<float>                 bin_loads(num_pq_chunks, 0);

  // Process dimensions not inserted by previous loop
  for (uint32_t d = 0; d < dim; d++) {
    if (dim_to_bin.find(d) != dim_to_bin.end())
      continue;
    auto  cur_best = num_pq_chunks + 1;
    float cur_best_load = std::numeric_limits<float>::max();
    for (uint32_t b = 0; b < num_pq_chunks; b++) {
      if (bin_loads[b] < cur_best_load &&
          bin_to_dims[b].size() < cur_bin_threshold) {
        cur_best = b;
        cur_best_load = bin_loads[b];
      }
    }
    diskann::cout << " Pushing " << d << " into bin #: " << cur_best
                  << std::endl;
    bin_to_dims[cur_best].push_back(d);
    if (bin_to_dims[cur_best].size() == high_val) {
      cur_num_high++;
      if (cur_num_high == max_num_high)
        cur_bin_threshold = low_val;
    }
  }

  rearrangement.clear();
  chunk_offsets.clear();
  chunk_offsets.push_back(0);

  for (uint32_t b = 0; b < num_pq_chunks; b++) {
    diskann::cout << "[ ";
    for (auto p : bin_to_dims[b]) {
      rearrangement.push_back(p);
      diskann::cout << p << ",";
    }
    diskann::cout << "] " << std::endl;
    if (b > 0)
      chunk_offsets.push_back(chunk_offsets[b - 1] +
                              (unsigned) bin_to_dims[b - 1].size());
  }
  chunk_offsets.push_back(dim);

  diskann::cout << "\nCross-checking rearranged order of coordinates:"
                << std::endl;
  for (auto p : rearrangement)
    diskann::cout << p << " ";
  diskann::cout << std::endl;

  full_pivot_data.reset(new float[num_centers * dim]);

  for (size_t i = 0; i < num_pq_chunks; i++) {
    size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];

    if (cur_chunk_size == 0)
      continue;
    std::unique_ptr<float[]> cur_pivot_data =
        std::make_unique<float[]>(num_centers * cur_chunk_size);
    std::unique_ptr<float[]> cur_data =
        std::make_unique<float[]>(num_train * cur_chunk_size);
    std::unique_ptr<uint32_t[]> closest_center =
        std::make_unique<uint32_t[]>(num_train);

    diskann::cout << "Processing chunk " << i << " with dimensions ["
                  << chunk_offsets[i] << ", " << chunk_offsets[i + 1] << ")"
                  << std::endl;

#pragma omp parallel for schedule(static, 65536)
    for (int64_t j = 0; j < (_s64) num_train; j++) {
      std::memcpy(cur_data.get() + j * cur_chunk_size,
                  train_data.get() + j * dim + chunk_offsets[i],
                  cur_chunk_size * sizeof(float));
    }

    kmeans::kmeanspp_selecting_pivots(cur_data.get(), num_train, cur_chunk_size,
                                      cur_pivot_data.get(), num_centers);

    kmeans::run_lloyds(cur_data.get(), num_train, cur_chunk_size,
                       cur_pivot_data.get(), num_centers, max_k_means_reps,
                       NULL, closest_center.get());

    for (uint64_t j = 0; j < num_centers; j++) {
      std::memcpy(full_pivot_data.get() + j * dim + chunk_offsets[i],
                  cur_pivot_data.get() + j * cur_chunk_size,
                  cur_chunk_size * sizeof(float));
    }
  }

  std::vector<size_t> cumul_bytes(5, 0);
  cumul_bytes[0] = METADATA_SIZE;
  cumul_bytes[1] =
      cumul_bytes[0] +
      diskann::save_bin<float>(pq_pivots_path.c_str(), full_pivot_data.get(),
                               (size_t) num_centers, dim, cumul_bytes[0]);
  cumul_bytes[2] = cumul_bytes[1] + diskann::save_bin<float>(
                                        pq_pivots_path.c_str(), centroid.get(),
                                        (size_t) dim, 1, cumul_bytes[1]);
  cumul_bytes[3] =
      cumul_bytes[2] +
      diskann::save_bin<uint32_t>(pq_pivots_path.c_str(), rearrangement.data(),
                                  rearrangement.size(), 1, cumul_bytes[2]);
  cumul_bytes[4] =
      cumul_bytes[3] +
      diskann::save_bin<uint32_t>(pq_pivots_path.c_str(), chunk_offsets.data(),
                                  chunk_offsets.size(), 1, cumul_bytes[3]);
  diskann::save_bin<_u64>(pq_pivots_path.c_str(), cumul_bytes.data(),
                          cumul_bytes.size(), 1, 0);

  diskann::cout << "Saved pq pivot data to " << pq_pivots_path << " of size "
                << cumul_bytes[cumul_bytes.size() - 1] << "B." << std::endl;

  return 0;
}

// streams the base file (data_file), and computes the closest centers in each
// chunk to generate the compressed data_file and stores it in
// pq_compressed_vectors_path.
// If the numbber of centers is < 256, it stores as byte vector, else as 4-byte
// vector in binary format.
template<typename T>
int generate_pq_data_from_pivots(const std::string data_file,
                                 unsigned num_centers, unsigned num_pq_chunks,
                                 std::string pq_pivots_path,
                                 std::string pq_compressed_vectors_path) {
  _u64            read_blk_size = 64 * 1024 * 1024;
  cached_ifstream base_reader(data_file, read_blk_size);
  _u32            npts32;
  _u32            basedim32;
  base_reader.read((char *) &npts32, sizeof(uint32_t));
  base_reader.read((char *) &basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  size_t dim = basedim32;

  std::unique_ptr<float[]>    full_pivot_data;
  std::unique_ptr<float[]>    centroid;
  std::unique_ptr<uint32_t[]> rearrangement;
  std::unique_ptr<uint32_t[]> chunk_offsets;

  std::string inflated_pq_file = pq_compressed_vectors_path + "_inflated.bin";

  if (!file_exists(pq_pivots_path)) {
    std::cout << "ERROR: PQ k-means pivot file not found" << std::endl;
    throw diskann::ANNException("PQ k-means pivot file not found", -1);
  } else {
    _u64                    nr, nc;
    std::unique_ptr<_u64[]> file_offset_data;

    diskann::load_bin<_u64>(pq_pivots_path.c_str(), file_offset_data, nr, nc,
                            0);

    if (nr != 5) {
      diskann::cout << "Error reading pq_pivots file " << pq_pivots_path
                    << ". Offsets dont contain correct metadata, # offsets = "
                    << nr << ", but expecting 5.";
      throw diskann::ANNException(
          "Error reading pq_pivots file at offsets data.", -1, __FUNCSIG__,
          __FILE__, __LINE__);
    }

    diskann::load_bin<float>(pq_pivots_path.c_str(), full_pivot_data, nr, nc,
                             file_offset_data[0]);

    if ((nr != num_centers) || (nc != dim)) {
      diskann::cout << "Error reading pq_pivots file " << pq_pivots_path
                    << ". file_num_centers  = " << nr << ", file_dim = " << nc
                    << " but expecting " << num_centers << " centers in " << dim
                    << " dimensions.";
      throw diskann::ANNException(
          "Error reading pq_pivots file at pivots data.", -1, __FUNCSIG__,
          __FILE__, __LINE__);
    }

    diskann::load_bin<float>(pq_pivots_path.c_str(), centroid, nr, nc,
                             file_offset_data[1]);

    if ((nr != dim) || (nc != 1)) {
      diskann::cout << "Error reading pq_pivots file " << pq_pivots_path
                    << ". file_dim  = " << nr << ", file_cols = " << nc
                    << " but expecting " << dim << " entries in 1 dimension.";
      throw diskann::ANNException(
          "Error reading pq_pivots file at centroid data.", -1, __FUNCSIG__,
          __FILE__, __LINE__);
    }

    diskann::load_bin<uint32_t>(pq_pivots_path.c_str(), rearrangement, nr, nc,
                                file_offset_data[2]);

    if ((nr != dim) || (nc != 1)) {
      diskann::cout << "Error reading pq_pivots file " << pq_pivots_path
                    << ". file_dim  = " << nr << ", file_cols = " << nc
                    << " but expecting " << dim << " entries in 1 dimension.";
      throw diskann::ANNException(
          "Error reading pq_pivots file at re-arrangement data.", -1,
          __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::load_bin<uint32_t>(pq_pivots_path.c_str(), chunk_offsets, nr, nc,
                                file_offset_data[3]);

    if (nr != (uint64_t) num_pq_chunks + 1 || nc != 1) {
      diskann::cout
          << "Error reading pq_pivots file at chunk offsets; file has nr=" << nr
          << ",nc=" << nc << ", expecting nr=" << num_pq_chunks + 1 << ", nc=1."
          << std::endl;
      throw diskann::ANNException(
          "Error reading pq_pivots file at chunk offsets.", -1, __FUNCSIG__,
          __FILE__, __LINE__);
    }

    diskann::cout << "Loaded PQ pivot information" << std::endl;
  }

  std::ofstream compressed_file_writer(pq_compressed_vectors_path,
                                       std::ios::binary);
  _u32          num_pq_chunks_u32 = num_pq_chunks;

  compressed_file_writer.write((char *) &num_points, sizeof(uint32_t));
  compressed_file_writer.write((char *) &num_pq_chunks_u32, sizeof(uint32_t));

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;

#ifdef SAVE_INFLATED_PQ
  std::ofstream inflated_file_writer(inflated_pq_file, std::ios::binary);
  inflated_file_writer.write((char *) &num_points, sizeof(uint32_t));
  inflated_file_writer.write((char *) &basedim32, sizeof(uint32_t));

  std::unique_ptr<float[]> block_inflated_base =
      std::make_unique<float[]>(block_size * dim);
  std::memset(block_inflated_base.get(), 0, block_size * dim * sizeof(float));
#endif

  std::unique_ptr<_u32[]> block_compressed_base =
      std::make_unique<_u32[]>(block_size * (_u64) num_pq_chunks);
  std::memset(block_compressed_base.get(), 0,
              block_size * (_u64) num_pq_chunks * sizeof(uint32_t));

  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float =
      std::make_unique<float[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_tmp =
      std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read((char *) (block_data_T.get()),
                     sizeof(T) * (cur_blk_size * dim));
    diskann::convert_types<T, float>(block_data_T.get(), block_data_tmp.get(),
                                     cur_blk_size, dim);

    diskann::cout << "Processing points  [" << start_id << ", " << end_id
                  << ").." << std::flush;

    for (uint64_t p = 0; p < cur_blk_size; p++) {
      for (uint64_t d = 0; d < dim; d++) {
        block_data_tmp[p * dim + d] -= centroid[d];
      }
    }

    for (uint64_t p = 0; p < cur_blk_size; p++) {
      for (uint64_t d = 0; d < dim; d++) {
        block_data_float[p * dim + d] =
            block_data_tmp[p * dim + rearrangement[d]];
      }
    }

    for (size_t i = 0; i < num_pq_chunks; i++) {
      size_t cur_chunk_size = chunk_offsets[i + 1] - chunk_offsets[i];
      if (cur_chunk_size == 0)
        continue;

      std::unique_ptr<float[]> cur_pivot_data =
          std::make_unique<float[]>(num_centers * cur_chunk_size);
      std::unique_ptr<float[]> cur_data =
          std::make_unique<float[]>(cur_blk_size * cur_chunk_size);
      std::unique_ptr<uint32_t[]> closest_center =
          std::make_unique<uint32_t[]>(cur_blk_size);

#pragma omp parallel for schedule(static, 8192)
      for (int64_t j = 0; j < (_s64) cur_blk_size; j++) {
        for (uint64_t k = 0; k < cur_chunk_size; k++)
          cur_data[j * cur_chunk_size + k] =
              block_data_float[j * dim + chunk_offsets[i] + k];
      }

#pragma omp parallel for schedule(static, 1)
      for (int64_t j = 0; j < (_s64) num_centers; j++) {
        std::memcpy(cur_pivot_data.get() + j * cur_chunk_size,
                    full_pivot_data.get() + j * dim + chunk_offsets[i],
                    cur_chunk_size * sizeof(float));
      }

      math_utils::compute_closest_centers(cur_data.get(), cur_blk_size,
                                          cur_chunk_size, cur_pivot_data.get(),
                                          num_centers, 1, closest_center.get());

#pragma omp parallel for schedule(static, 8192)
      for (int64_t j = 0; j < (_s64) cur_blk_size; j++) {
        block_compressed_base[j * num_pq_chunks + i] = closest_center[j];
#ifdef SAVE_INFLATED_PQ
        for (uint64_t k = 0; k < cur_chunk_size; k++)
          block_inflated_base[j * dim + chunk_offsets[i] + k] =
              cur_pivot_data[closest_center[j] * cur_chunk_size + k] +
              centroid[chunk_offsets[i] + k];
#endif
      }
    }

    if (num_centers > 256) {
      compressed_file_writer.write(
          (char *) (block_compressed_base.get()),
          cur_blk_size * num_pq_chunks * sizeof(uint32_t));
    } else {
      std::unique_ptr<uint8_t[]> pVec =
          std::make_unique<uint8_t[]>(cur_blk_size * num_pq_chunks);
      diskann::convert_types<uint32_t, uint8_t>(
          block_compressed_base.get(), pVec.get(), cur_blk_size, num_pq_chunks);
      compressed_file_writer.write(
          (char *) (pVec.get()),
          cur_blk_size * num_pq_chunks * sizeof(uint8_t));
    }
#ifdef SAVE_INFLATED_PQ
    inflated_file_writer.write((char *) (block_inflated_base.get()),
                               cur_blk_size * dim * sizeof(float));
#endif
    diskann::cout << ".done." << std::endl;
  }
// Gopal. Splitting diskann_dll into separate DLLs for search and build.
// This code should only be available in the "build" DLL.
#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
  MallocExtension::instance()->ReleaseFreeMemory();
#endif
  compressed_file_writer.close();
#ifdef SAVE_INFLATED_PQ
  inflated_file_writer.close();
#endif
  return 0;
}

int estimate_cluster_sizes(float *test_data_float, size_t num_test,
                           float *pivots, const size_t num_centers,
                           const size_t test_dim, const size_t k_base,
                           std::vector<size_t> &cluster_sizes) {
  cluster_sizes.clear();

  size_t *shard_counts = new size_t[num_centers];

  for (size_t i = 0; i < num_centers; i++) {
    shard_counts[i] = 0;
  }

  size_t block_size = num_test <= BLOCK_SIZE ? num_test : BLOCK_SIZE;
  _u32 * block_closest_centers = new _u32[block_size * k_base];
  float *block_data_float;

  size_t num_blocks = DIV_ROUND_UP(num_test, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_test);
    size_t cur_blk_size = end_id - start_id;

    block_data_float = test_data_float + start_id * test_dim;

    math_utils::compute_closest_centers(block_data_float, cur_blk_size,
                                        test_dim, pivots, num_centers, k_base,
                                        block_closest_centers);

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        shard_counts[shard_id]++;
      }
    }
  }

  diskann::cout << "Estimated cluster sizes: ";
  for (size_t i = 0; i < num_centers; i++) {
    _u32 cur_shard_count = (_u32) shard_counts[i];
    cluster_sizes.push_back((size_t) cur_shard_count);
    diskann::cout << cur_shard_count << " ";
  }
  diskann::cout << std::endl;
  delete[] shard_counts;
  delete[] block_closest_centers;
  return 0;
}

template<typename T>
int shard_data_into_clusters(const std::string data_file, float *pivots,
                             const size_t num_centers, const size_t dim,
                             const size_t k_base, std::string prefix_path) {
  _u64 read_blk_size = 64 * 1024 * 1024;
  //  _u64 write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(data_file, read_blk_size);
  _u32            npts32;
  _u32            basedim32;
  base_reader.read((char *) &npts32, sizeof(uint32_t));
  base_reader.read((char *) &basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  if (basedim32 != dim) {
    diskann::cout << "Error. dimensions dont match for train set and base set"
                  << std::endl;
    return -1;
  }

  std::unique_ptr<size_t[]> shard_counts =
      std::make_unique<size_t[]>(num_centers);
  std::vector<std::ofstream> shard_data_writer(num_centers);
  std::vector<std::ofstream> shard_idmap_writer(num_centers);
  _u32                       dummy_size = 0;
  _u32                       const_one = 1;

  for (size_t i = 0; i < num_centers; i++) {
    std::string data_filename =
        prefix_path + "_subshard-" + std::to_string(i) + ".bin";
    std::string idmap_filename =
        prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
    shard_data_writer[i] =
        std::ofstream(data_filename.c_str(), std::ios::binary);
    shard_idmap_writer[i] =
        std::ofstream(idmap_filename.c_str(), std::ios::binary);
    shard_data_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
    shard_data_writer[i].write((char *) &basedim32, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *) &const_one, sizeof(uint32_t));
    shard_counts[i] = 0;
  }

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<_u32[]> block_closest_centers =
      std::make_unique<_u32[]>(block_size * k_base);
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float =
      std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read((char *) block_data_T.get(),
                     sizeof(T) * (cur_blk_size * dim));
    diskann::convert_types<T, float>(block_data_T.get(), block_data_float.get(),
                                     cur_blk_size, dim);

    math_utils::compute_closest_centers(block_data_float.get(), cur_blk_size,
                                        dim, pivots, num_centers, k_base,
                                        block_closest_centers.get());

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t   shard_id = block_closest_centers[p * k_base + p1];
        uint32_t original_point_map_id = (uint32_t)(start_id + p);
        shard_data_writer[shard_id].write(
            (char *) (block_data_T.get() + p * dim), sizeof(T) * dim);
        shard_idmap_writer[shard_id].write((char *) &original_point_map_id,
                                           sizeof(uint32_t));
        shard_counts[shard_id]++;
      }
    }
  }

  size_t total_count = 0;
  diskann::cout << "Actual shard sizes: " << std::flush;
  for (size_t i = 0; i < num_centers; i++) {
    _u32 cur_shard_count = (_u32) shard_counts[i];
    total_count += cur_shard_count;
    diskann::cout << cur_shard_count << " ";
    shard_data_writer[i].seekp(0);
    shard_data_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
    shard_data_writer[i].close();
    shard_idmap_writer[i].seekp(0);
    shard_idmap_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
    shard_idmap_writer[i].close();
  }

  diskann::cout << "\n Partitioned " << num_points
                << " with replication factor " << k_base << " to get "
                << total_count << " points across " << num_centers << " shards "
                << std::endl;
  return 0;
}

// useful for partitioning large dataset. we first generate only the IDS for
// each shard, and retrieve the actual vectors on demand.
template<typename T>
int shard_data_into_clusters_only_ids(const std::string data_file,
                                      float *pivots, const size_t num_centers,
                                      const size_t dim, const size_t k_base,
                                      std::string prefix_path) {
  _u64 read_blk_size = 64 * 1024 * 1024;
  //  _u64 write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(data_file, read_blk_size);
  _u32            npts32;
  _u32            basedim32;
  base_reader.read((char *) &npts32, sizeof(uint32_t));
  base_reader.read((char *) &basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  if (basedim32 != dim) {
    diskann::cout << "Error. dimensions dont match for train set and base set"
                  << std::endl;
    return -1;
  }

  std::unique_ptr<size_t[]> shard_counts =
      std::make_unique<size_t[]>(num_centers);

  std::vector<std::ofstream> shard_idmap_writer(num_centers);
  _u32                       dummy_size = 0;
  _u32                       const_one = 1;

  for (size_t i = 0; i < num_centers; i++) {
    std::string idmap_filename =
        prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
    shard_idmap_writer[i] =
        std::ofstream(idmap_filename.c_str(), std::ios::binary);
    shard_idmap_writer[i].write((char *) &dummy_size, sizeof(uint32_t));
    shard_idmap_writer[i].write((char *) &const_one, sizeof(uint32_t));
    shard_counts[i] = 0;
  }

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<_u32[]> block_closest_centers =
      std::make_unique<_u32[]>(block_size * k_base);
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
  std::unique_ptr<float[]> block_data_float =
      std::make_unique<float[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read((char *) block_data_T.get(),
                     sizeof(T) * (cur_blk_size * dim));
    diskann::convert_types<T, float>(block_data_T.get(), block_data_float.get(),
                                     cur_blk_size, dim);

    math_utils::compute_closest_centers(block_data_float.get(), cur_blk_size,
                                        dim, pivots, num_centers, k_base,
                                        block_closest_centers.get());

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t   shard_id = block_closest_centers[p * k_base + p1];
        uint32_t original_point_map_id = (uint32_t)(start_id + p);
        shard_idmap_writer[shard_id].write((char *) &original_point_map_id,
                                           sizeof(uint32_t));
        shard_counts[shard_id]++;
      }
    }
  }

  size_t total_count = 0;
  diskann::cout << "Actual shard sizes: " << std::flush;
  for (size_t i = 0; i < num_centers; i++) {
    _u32 cur_shard_count = (_u32) shard_counts[i];
    total_count += cur_shard_count;
    diskann::cout << cur_shard_count << " ";
    shard_idmap_writer[i].seekp(0);
    shard_idmap_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
    shard_idmap_writer[i].close();
  }

  diskann::cout << "\n Partitioned " << num_points
                << " with replication factor " << k_base << " to get "
                << total_count << " points across " << num_centers << " shards "
                << std::endl;
  return 0;
}

template<typename T>
int retrieve_shard_data_from_ids(const std::string data_file,
                                 std::string       idmap_filename,
                                 std::string       data_filename) {
  _u64 read_blk_size = 64 * 1024 * 1024;
  //  _u64 write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(data_file, read_blk_size);
  _u32            npts32;
  _u32            basedim32;
  base_reader.read((char *) &npts32, sizeof(uint32_t));
  base_reader.read((char *) &basedim32, sizeof(uint32_t));
  size_t num_points = npts32;
  size_t dim = basedim32;

  _u32 dummy_size = 0;

  std::ofstream shard_data_writer(data_filename.c_str(), std::ios::binary);
  shard_data_writer.write((char *) &dummy_size, sizeof(uint32_t));
  shard_data_writer.write((char *) &basedim32, sizeof(uint32_t));

  _u32 *shard_ids;
  _u64  shard_size, tmp;
  diskann::load_bin<_u32>(idmap_filename, shard_ids, shard_size, tmp);

  _u32 cur_pos = 0;
  _u32 num_written = 0;
  std::cout << "Shard has " << shard_size << " points" << std::endl;

  size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
  std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);

  size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * block_size;
    size_t end_id = (std::min)((block + 1) * block_size, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read((char *) block_data_T.get(),
                     sizeof(T) * (cur_blk_size * dim));

    for (size_t p = 0; p < cur_blk_size; p++) {
      uint32_t original_point_map_id = (uint32_t)(start_id + p);
      if (cur_pos == shard_size)
        break;
      if (original_point_map_id == shard_ids[cur_pos]) {
        cur_pos++;
        shard_data_writer.write((char *) (block_data_T.get() + p * dim),
                                sizeof(T) * dim);
        num_written++;
      }
    }
    if (cur_pos == shard_size)
      break;
  }

  diskann::cout << "Written file with " << num_written << " points"
                << std::endl;

  shard_data_writer.seekp(0);
  shard_data_writer.write((char *) &num_written, sizeof(uint32_t));
  shard_data_writer.close();
  delete[] shard_ids;
  return 0;
}

// partitions a large base file into many shards using k-means hueristic
// on a random sample generated using sampling_rate probability. After this, it
// assignes each base point to the closest k_base nearest centers and creates
// the shards.
// The total number of points across all shards will be k_base * num_points.

template<typename T>
int partition(const std::string data_file, const float sampling_rate,
              size_t num_parts, size_t max_k_means_reps,
              const std::string prefix_path, size_t k_base) {
  size_t train_dim;
  size_t num_train;
  float *train_data_float;

  gen_random_slice<T>(data_file, sampling_rate, train_data_float, num_train,
                      train_dim);

  float *pivot_data;

  std::string cur_file = std::string(prefix_path);
  std::string output_file;

  // kmeans_partitioning on training data

  //  cur_file = cur_file + "_kmeans_partitioning-" + std::to_string(num_parts);
  output_file = cur_file + "_centroids.bin";

  pivot_data = new float[num_parts * train_dim];

  // Process Global k-means for kmeans_partitioning Step
  diskann::cout << "Processing global k-means (kmeans_partitioning Step)"
                << std::endl;
  kmeans::kmeanspp_selecting_pivots(train_data_float, num_train, train_dim,
                                    pivot_data, num_parts);

  kmeans::run_lloyds(train_data_float, num_train, train_dim, pivot_data,
                     num_parts, max_k_means_reps, NULL, NULL);

  diskann::cout << "Saving global k-center pivots" << std::endl;
  diskann::save_bin<float>(output_file.c_str(), pivot_data, (size_t) num_parts,
                           train_dim);

  // now pivots are ready. need to stream base points and assign them to
  // closest clusters.

  shard_data_into_clusters<T>(data_file, pivot_data, num_parts, train_dim,
                              k_base, prefix_path);
  delete[] pivot_data;
  delete[] train_data_float;
  return 0;
}

template<typename T>
int partition_with_ram_budget(const std::string data_file,
                              const double sampling_rate, double ram_budget,
                              size_t            graph_degree,
                              const std::string prefix_path, size_t k_base) {
  size_t train_dim;
  size_t num_train;
  float *train_data_float;
  size_t max_k_means_reps = 10;

  int  num_parts = 3;
  bool fit_in_ram = false;

  gen_random_slice<T>(data_file, sampling_rate, train_data_float, num_train,
                      train_dim);

  size_t test_dim;
  size_t num_test;
  float *test_data_float;
  gen_random_slice<T>(data_file, sampling_rate, test_data_float, num_test,
                      test_dim);

  float *pivot_data = nullptr;

  std::string cur_file = std::string(prefix_path);
  std::string output_file;

  // kmeans_partitioning on training data

  //  cur_file = cur_file + "_kmeans_partitioning-" + std::to_string(num_parts);
  output_file = cur_file + "_centroids.bin";

  while (!fit_in_ram) {
    fit_in_ram = true;

    double max_ram_usage = 0;
    if (pivot_data != nullptr)
      delete[] pivot_data;

    pivot_data = new float[num_parts * train_dim];
    // Process Global k-means for kmeans_partitioning Step
    diskann::cout << "Processing global k-means (kmeans_partitioning Step)"
                  << std::endl;
    kmeans::kmeanspp_selecting_pivots(train_data_float, num_train, train_dim,
                                      pivot_data, num_parts);

    kmeans::run_lloyds(train_data_float, num_train, train_dim, pivot_data,
                       num_parts, max_k_means_reps, NULL, NULL);

    // now pivots are ready. need to stream base points and assign them to
    // closest clusters.

    std::vector<size_t> cluster_sizes;
    estimate_cluster_sizes(test_data_float, num_test, pivot_data, num_parts,
                           train_dim, k_base, cluster_sizes);

    for (auto &p : cluster_sizes) {
      p = (_u64)(p /
                 sampling_rate);  // to account for the fact that p is the size
                                  // of the shard over the testing sample.
      double cur_shard_ram_estimate =
          diskann::estimate_ram_usage(p, train_dim, sizeof(T), graph_degree);

      if (cur_shard_ram_estimate > max_ram_usage)
        max_ram_usage = cur_shard_ram_estimate;
    }
    diskann::cout << "With " << num_parts << " parts, max estimated RAM usage: "
                  << max_ram_usage / (1024 * 1024 * 1024)
                  << "GB, budget given is " << ram_budget << std::endl;
    if (max_ram_usage > 1024 * 1024 * 1024 * ram_budget) {
      fit_in_ram = false;
      num_parts += 2;
    }
  }

  diskann::cout << "Saving global k-center pivots" << std::endl;
  diskann::save_bin<float>(output_file.c_str(), pivot_data, (size_t) num_parts,
                           train_dim);

  shard_data_into_clusters_only_ids<T>(data_file, pivot_data, num_parts,
                                       train_dim, k_base, prefix_path);
  delete[] pivot_data;
  delete[] train_data_float;
  delete[] test_data_float;
  return num_parts;
}

// Instantations of supported templates

template void DISKANN_DLLEXPORT
                                gen_random_slice<int8_t>(const std::string base_file,
                         const std::string output_prefix, double sampling_rate);
template void DISKANN_DLLEXPORT gen_random_slice<uint8_t>(
    const std::string base_file, const std::string output_prefix,
    double sampling_rate);
template void DISKANN_DLLEXPORT
gen_random_slice<float>(const std::string base_file,
                        const std::string output_prefix, double sampling_rate);

template void DISKANN_DLLEXPORT
                                gen_random_slice<float>(const float *inputdata, size_t npts, size_t ndims,
                        double p_val, float *&sampled_data, size_t &slice_size);
template void DISKANN_DLLEXPORT gen_random_slice<uint8_t>(
    const uint8_t *inputdata, size_t npts, size_t ndims, double p_val,
    float *&sampled_data, size_t &slice_size);
template void DISKANN_DLLEXPORT gen_random_slice<int8_t>(
    const int8_t *inputdata, size_t npts, size_t ndims, double p_val,
    float *&sampled_data, size_t &slice_size);

template void DISKANN_DLLEXPORT gen_random_slice<float>(
    const std::string data_file, double p_val, float *&sampled_data,
    size_t &slice_size, size_t &ndims);
template void DISKANN_DLLEXPORT gen_random_slice<uint8_t>(
    const std::string data_file, double p_val, float *&sampled_data,
    size_t &slice_size, size_t &ndims);
template void DISKANN_DLLEXPORT gen_random_slice<int8_t>(
    const std::string data_file, double p_val, float *&sampled_data,
    size_t &slice_size, size_t &ndims);

template DISKANN_DLLEXPORT int partition<int8_t>(
    const std::string data_file, const float sampling_rate, size_t num_centers,
    size_t max_k_means_reps, const std::string prefix_path, size_t k_base);
template DISKANN_DLLEXPORT int partition<uint8_t>(
    const std::string data_file, const float sampling_rate, size_t num_centers,
    size_t max_k_means_reps, const std::string prefix_path, size_t k_base);
template DISKANN_DLLEXPORT int partition<float>(
    const std::string data_file, const float sampling_rate, size_t num_centers,
    size_t max_k_means_reps, const std::string prefix_path, size_t k_base);

template DISKANN_DLLEXPORT int partition_with_ram_budget<int8_t>(
    const std::string data_file, const double sampling_rate, double ram_budget,
    size_t graph_degree, const std::string prefix_path, size_t k_base);
template DISKANN_DLLEXPORT int partition_with_ram_budget<uint8_t>(
    const std::string data_file, const double sampling_rate, double ram_budget,
    size_t graph_degree, const std::string prefix_path, size_t k_base);
template DISKANN_DLLEXPORT int partition_with_ram_budget<float>(
    const std::string data_file, const double sampling_rate, double ram_budget,
    size_t graph_degree, const std::string prefix_path, size_t k_base);

template DISKANN_DLLEXPORT int retrieve_shard_data_from_ids<float>(
    const std::string data_file, std::string idmap_filename,
    std::string data_filename);
template DISKANN_DLLEXPORT int retrieve_shard_data_from_ids<uint8_t>(
    const std::string data_file, std::string idmap_filename,
    std::string data_filename);
template DISKANN_DLLEXPORT int retrieve_shard_data_from_ids<int8_t>(
    const std::string data_file, std::string idmap_filename,
    std::string data_filename);

template DISKANN_DLLEXPORT int generate_pq_data_from_pivots<int8_t>(
    const std::string data_file, unsigned num_centers, unsigned num_pq_chunks,
    std::string pq_pivots_path, std::string pq_compressed_vectors_path);
template DISKANN_DLLEXPORT int generate_pq_data_from_pivots<uint8_t>(
    const std::string data_file, unsigned num_centers, unsigned num_pq_chunks,
    std::string pq_pivots_path, std::string pq_compressed_vectors_path);
template DISKANN_DLLEXPORT int generate_pq_data_from_pivots<float>(
    const std::string data_file, unsigned num_centers, unsigned num_pq_chunks,
    std::string pq_pivots_path, std::string pq_compressed_vectors_path);
