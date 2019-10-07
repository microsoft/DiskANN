#include <math_utils.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include "exceptions.h"
#include "index_nsg.h"
#include "parameters.h"
#include "tsl/robin_set.h"
#include "utils.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <typeinfo>

#include <cassert>
#include "memory_mapper.h"
#include "partition_and_pq.h"
#ifdef _WINDOWS
#include <xmmintrin.h>
#endif

#define BLOCK_SIZE 5000000

// streams data from the file, and samples each vector with probability p_val
// and returns a matrix of size slice_size* ndims as floating point type.
// the slice_size and ndims are set inside the function.

/***********************************
* Reimplement using gen_random_slice(const T* inputdata,...)
 ************************************/

template<typename T>
void gen_random_slice(const std::string data_file, float p_val,
                      float *&sampled_data, size_t &slice_size, size_t &ndims) {
  size_t                          npts;
  uint32_t                        npts32, ndims32;
  std::vector<std::vector<float>> sampled_vectors;
  T *                             cur_vector_T;

  // amount to read in one shot
  _u64 read_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(data_file.c_str(), read_blk_size);

  // metadata: npts, ndims
  base_reader.read((char *) &npts32, sizeof(unsigned));
  base_reader.read((char *) &ndims32, sizeof(unsigned));
  npts = npts32;
  ndims = ndims32;

  cur_vector_T = new T[ndims];
  p_val = p_val < 1 ? p_val : 1;

  std::random_device rd;  // Will be used to obtain a seed for the random number
  size_t             x = rd();
  std::mt19937       generator(x);
  std::uniform_real_distribution<float> distribution(0, 1);

  for (size_t i = 0; i < npts; i++) {
    base_reader.read((char *) cur_vector_T, ndims * sizeof(T));
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

  delete[] cur_vector_T;
}

// same as above, but samples from the matrix inputdata instead of a file of
// npts*ndims to return sampled_data of size slice_size*ndims.
template<typename T>
void gen_random_slice(const T *inputdata, size_t npts, size_t ndims,
                      float p_val, float *&sampled_data, size_t &slice_size) {
  std::vector<std::vector<float>> sampled_vectors;
  const T *                       cur_vector_T;

  p_val = p_val < 1 ? p_val : 1;

  std::random_device
               rd;  // Will be used to obtain a seed for the random number engine
  size_t       x = rd();
  std::mt19937 generator(
      x);  // Standard mersenne_twister_engine seeded with rd()
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
int generate_pq_pivots(const float *train_data, size_t num_train, size_t dim,
                       size_t num_centers, size_t num_pq_chunks,
                       size_t max_k_means_reps, std::string pq_pivots_path) {
  if (num_pq_chunks > dim) {
    std::cout << " Error: number of chunks more than dimension" << std::endl;
    return -1;
  }

  size_t chunk_size = std::floor((double) dim / (double) num_pq_chunks);
  float *full_pivot_data;
  size_t corrected_num_pq_chunks = DIV_ROUND_UP(dim, chunk_size);

  std::cout << "Corrected number of chunks " << corrected_num_pq_chunks
            << std::endl;

  if (file_exists(pq_pivots_path)) {
    size_t file_dim, file_num_centers;
    diskann::load_bin<float>(pq_pivots_path, full_pivot_data, file_num_centers,
                             file_dim);
    if (file_dim == dim && file_num_centers == num_centers) {
      std::cout << "PQ pivot file exists. Not generating again" << std::endl;
      delete[] full_pivot_data;
      return -1;
    }
  }

  full_pivot_data = new float[num_centers * dim];

  for (size_t i = 0; i < corrected_num_pq_chunks; i++) {
    size_t cur_chunk_size =
        chunk_size < (dim - i * chunk_size) ? chunk_size : dim - i * chunk_size;
    float *   cur_pivot_data = new float[num_centers * cur_chunk_size];
    float *   cur_data = new float[num_train * cur_chunk_size];
    uint32_t *closest_center = new uint32_t[num_train];

    std::cout << "Processing chunk " << i << " with dimensions ["
              << i * chunk_size << ", " << i * chunk_size + cur_chunk_size
              << ")" << std::endl;
#pragma omp parallel for schedule(static, 65536)
    for (int64_t j = 0; j < (_s64) num_train; j++) {
      std::memcpy(cur_data + j * cur_chunk_size,
                  train_data + j * dim + i * chunk_size,
                  cur_chunk_size * sizeof(float));
    }

    kmeans::kmeanspp_selecting_pivots(cur_data, num_train, cur_chunk_size,
                                      cur_pivot_data, num_centers);

    kmeans::run_lloyds(cur_data, num_train, cur_chunk_size, cur_pivot_data,
                       num_centers, max_k_means_reps, NULL, closest_center);

    for (uint64_t j = 0; j < num_centers; j++) {
      std::memcpy(full_pivot_data + j * dim + i * chunk_size,
                  cur_pivot_data + j * cur_chunk_size,
                  cur_chunk_size * sizeof(float));
    }

    delete[] cur_data;
    delete[] cur_pivot_data;
    delete[] closest_center;
  }

  diskann::save_bin<float>(pq_pivots_path.c_str(), full_pivot_data,
                           (size_t) num_centers, dim);
  return 0;
}

// streams the base file (data_file), and computes the closest centers in each
// chunk to generate the compressed data_file and stores it in
// pq_compressed_vectors_path.
// If the numbber of centers is < 256, it stores as byte vector, else as 4-byte
// vector in binary format.
template<typename T>
int generate_pq_data_from_pivots(const std::string data_file,
                                 size_t num_centers, size_t num_pq_chunks,
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

  size_t chunk_size = std::floor((double) dim / (double) num_pq_chunks);
  size_t corrected_num_pq_chunks = DIV_ROUND_UP(dim, chunk_size);

  float *full_pivot_data = new float[num_centers * dim];

  if (!file_exists(pq_pivots_path)) {
    std::cout << "ERROR: PQ k-means pivot file not found" << std::endl;
    return -1;
  } else {
    size_t file_num_centers;
    size_t file_dim;
    diskann::load_bin<float>(pq_pivots_path, full_pivot_data, file_num_centers,
                             file_dim);

    if (file_num_centers != num_centers) {
      std::cout << "ERROR: file number of PQ centers " << file_num_centers
                << " does "
                   "not match input argument "
                << num_centers << std::endl;
      return -1;
    }
    if (file_dim != dim) {
      std::cout << "ERROR: PQ pivot dimension does "
                   "not match base file dimension"
                << std::endl;
      return -1;
    }
    std::cout << "Loaded PQ pivot information" << std::endl;
  }

  std::ofstream compressed_file_writer(pq_compressed_vectors_path,
                                       std::ios::binary);
  _u32 corrected_num_pq_chunks_u32 = corrected_num_pq_chunks;

  compressed_file_writer.write((char *) &num_points, sizeof(uint32_t));
  compressed_file_writer.write((char *) &corrected_num_pq_chunks_u32,
                               sizeof(uint32_t));

  _u32 * block_compressed_base = new _u32[BLOCK_SIZE * corrected_num_pq_chunks];
  T *    block_data_T = new T[BLOCK_SIZE * dim];
  float *block_data_float = new float[BLOCK_SIZE * dim];

  size_t num_blocks = DIV_ROUND_UP(num_points, BLOCK_SIZE);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * BLOCK_SIZE;
    size_t end_id = (std::min)((block + 1) * BLOCK_SIZE, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read((char *) block_data_T, sizeof(T) * (cur_blk_size * dim));
    diskann::convert_types<T, float>(block_data_T, block_data_float,
                                     cur_blk_size, dim);

    std::cout << "Processing points  [" << start_id << ", " << end_id << ").."
              << std::flush;

    for (size_t i = 0; i < corrected_num_pq_chunks; i++) {
      size_t cur_chunk_size = chunk_size < (dim - i * chunk_size)
                                  ? chunk_size
                                  : dim - i * chunk_size;
      float *   cur_pivot_data = new float[num_centers * cur_chunk_size];
      float *   cur_data = new float[cur_blk_size * cur_chunk_size];
      uint32_t *closest_center = new uint32_t[cur_blk_size];

#pragma omp parallel for schedule(static, 8192)
      for (int64_t j = 0; j < (_s64) cur_blk_size; j++) {
        for (uint64_t k = 0; k < cur_chunk_size; k++)
          cur_data[j * cur_chunk_size + k] =
              block_data_float[j * dim + i * chunk_size + k];
      }

#pragma omp parallel for schedule(static, 1)
      for (int64_t j = 0; j < (_s64) num_centers; j++) {
        std::memcpy(cur_pivot_data + j * cur_chunk_size,
                    full_pivot_data + j * dim + i * chunk_size,
                    cur_chunk_size * sizeof(float));
      }

      math_utils::compute_closest_centers(cur_data, cur_blk_size,
                                          cur_chunk_size, cur_pivot_data,
                                          num_centers, 1, closest_center);

#pragma omp parallel for schedule(static, 8192)
      for (int64_t j = 0; j < (_s64) cur_blk_size; j++) {
        block_compressed_base[j * corrected_num_pq_chunks + i] =
            closest_center[j];
      }

      delete[] cur_data;
      delete[] cur_pivot_data;
      delete[] closest_center;
    }

    if (num_centers > 256) {
      compressed_file_writer.write(
          (char *) block_compressed_base,
          cur_blk_size * corrected_num_pq_chunks * sizeof(uint32_t));
    } else {
      uint8_t *pVec = new uint8_t[cur_blk_size * corrected_num_pq_chunks];
      diskann::convert_types<uint32_t, uint8_t>(
          block_compressed_base, pVec, cur_blk_size, corrected_num_pq_chunks);
      compressed_file_writer.write(
          (char *) pVec,
          cur_blk_size * corrected_num_pq_chunks * sizeof(uint8_t));
      delete[] pVec;
    }

    std::cout << ".done." << std::endl;
  }
  delete[] block_data_float;
  delete[] block_data_T;
  delete[] block_compressed_base;
  delete[] full_pivot_data;

  compressed_file_writer.close();
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
  size_t dim;
  size_t train_dim;
  size_t num_points;
  size_t num_train;
  float *train_data_float;

  gen_random_slice<T>(data_file, sampling_rate, train_data_float, num_train,
                      train_dim);

  float *pivot_data;

  std::string cur_file = std::string(prefix_path);
  std::string output_file;

  // kmeans_partitioning on training data

  cur_file = cur_file + "_kmeans_partitioning-" + std::to_string(num_parts);
  output_file = cur_file + "_pivots_float.bin";

  if (!file_exists(output_file)) {
    pivot_data = new float[num_parts * train_dim];

    // Process Global k-means for kmeans_partitioning Step
    std::cout << "Processing global k-means (kmeans_partitioning Step)"
              << std::endl;
    kmeans::kmeanspp_selecting_pivots(train_data_float, num_train, train_dim,
                                      pivot_data, num_parts);

    kmeans::run_lloyds(train_data_float, num_train, train_dim, pivot_data,
                       num_parts, max_k_means_reps, NULL, NULL);

    std::cout << "Saving global k-center pivots" << std::endl;
    diskann::save_bin<float>(output_file.c_str(), pivot_data,
                             (size_t) num_parts, train_dim);
  } else {
    size_t file_num_parts;
    size_t file_dim;
    diskann::load_bin<float>(output_file, pivot_data, file_num_parts, file_dim);
    if (file_num_parts != num_parts || file_dim != train_dim) {
      std::cout << "ERROR: file number of kmeans_partitioning centers does "
                   "not match input argument (or) file "
                   "dimension does not match real dimension"
                << std::endl;
      return -1;
    }
  }

  delete[] train_data_float;

  // now pivots are ready. need to stream base points and assign them to
  // closest clusters.

  _u64 read_blk_size = 64 * 1024 * 1024;
  //  _u64 write_blk_size = 64 * 1024 * 1024;
  // create cached reader + writer
  cached_ifstream base_reader(data_file, read_blk_size);
  _u32            npts32;
  _u32            basedim32;
  base_reader.read((char *) &npts32, sizeof(uint32_t));
  base_reader.read((char *) &basedim32, sizeof(uint32_t));
  num_points = npts32;
  dim = basedim32;
  if (basedim32 != train_dim) {
    std::cout << "Error. dimensions dont match for train set and base set"
              << std::endl;
    return -1;
  }

  size_t *                   shard_counts = new size_t[num_parts];
  std::vector<std::ofstream> shard_data_writer(num_parts);
  std::vector<std::ofstream> shard_idmap_writer(num_parts);
  _u32                       dummy_size = 0;
  _u32                       const_one = 1;

  for (size_t i = 0; i < num_parts; i++) {
    std::string data_filename =
        cur_file + "_subshard-" + std::to_string(i) + ".bin";
    std::string idmap_filename =
        cur_file + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
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

  _u32 * block_closest_centers = new _u32[BLOCK_SIZE * k_base];
  T *    block_data_T = new T[BLOCK_SIZE * dim];
  float *block_data_float = new float[BLOCK_SIZE * dim];

  size_t num_blocks = DIV_ROUND_UP(num_points, BLOCK_SIZE);

  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_id = block * BLOCK_SIZE;
    size_t end_id = (std::min)((block + 1) * BLOCK_SIZE, num_points);
    size_t cur_blk_size = end_id - start_id;

    base_reader.read((char *) block_data_T, sizeof(T) * (cur_blk_size * dim));
    diskann::convert_types<T, float>(block_data_T, block_data_float,
                                     cur_blk_size, dim);

    math_utils::compute_closest_centers(block_data_float, cur_blk_size, dim,
                                        pivot_data, num_parts, k_base,
                                        block_closest_centers);

    for (size_t p = 0; p < cur_blk_size; p++) {
      for (size_t p1 = 0; p1 < k_base; p1++) {
        size_t shard_id = block_closest_centers[p * k_base + p1];
        _u32   original_point_map_id = start_id + p;
        shard_data_writer[shard_id].write((char *) (block_data_T + p * dim),
                                          sizeof(T) * dim);
        shard_idmap_writer[shard_id].write((char *) &original_point_map_id,
                                           sizeof(uint32_t));
        shard_counts[shard_id]++;
      }
    }
  }

  size_t total_count = 0;

  for (size_t i = 0; i < num_parts; i++) {
    _u32 cur_shard_count = shard_counts[i];
    total_count += cur_shard_count;
    shard_data_writer[i].seekp(0);
    shard_data_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
    shard_data_writer[i].close();
    shard_idmap_writer[i].seekp(0);
    shard_idmap_writer[i].write((char *) &cur_shard_count, sizeof(uint32_t));
    shard_idmap_writer[i].close();
  }

  std::cout << "Partitioned " << num_points << " with replication factor "
            << k_base << " to get " << total_count << " points across "
            << num_parts << " shards " << std::endl;
  delete[] pivot_data;
  delete[] shard_counts;
  delete[] block_closest_centers;
  delete[] block_data_T;
  delete[] block_data_float;
  return 0;
}

// Instantations of supported templates

template void DISKANN_DLLEXPORT
gen_random_slice<float>(const float *inputdata, size_t npts, size_t ndims,
                        float p_val, float *&sampled_data, size_t &slice_size);
template void DISKANN_DLLEXPORT gen_random_slice<uint8_t>(
    const uint8_t *inputdata, size_t npts, size_t ndims, float p_val,
    float *&sampled_data, size_t &slice_size);
template void DISKANN_DLLEXPORT
gen_random_slice<int8_t>(const int8_t *inputdata, size_t npts, size_t ndims,
                         float p_val, float *&sampled_data, size_t &slice_size);

template void DISKANN_DLLEXPORT gen_random_slice<float>(
    const std::string data_file, float p_val, float *&sampled_data,
    size_t &slice_size, size_t &ndims);
template void DISKANN_DLLEXPORT gen_random_slice<uint8_t>(
    const std::string data_file, float p_val, float *&sampled_data,
    size_t &slice_size, size_t &ndims);
template void DISKANN_DLLEXPORT gen_random_slice<int8_t>(
    const std::string data_file, float p_val, float *&sampled_data,
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

template DISKANN_DLLEXPORT int generate_pq_data_from_pivots<int8_t>(
    const std::string data_file, size_t num_centers, size_t num_pq_chunks,
    std::string pq_pivots_path, std::string pq_compressed_vectors_path);
template DISKANN_DLLEXPORT int generate_pq_data_from_pivots<uint8_t>(
    const std::string data_file, size_t num_centers, size_t num_pq_chunks,
    std::string pq_pivots_path, std::string pq_compressed_vectors_path);
template DISKANN_DLLEXPORT int generate_pq_data_from_pivots<float>(
    const std::string data_file, size_t num_centers, size_t num_pq_chunks,
    std::string pq_pivots_path, std::string pq_compressed_vectors_path);

// template<typename T>
// int partition(const char *base_file, const char *train_file, size_t
// num_centers,
//              size_t max_k_means_reps, const char *prefix_dir, size_t k_base);

// template<typename T>
// int generate_pq_data_from_pivots(const T *base_data, size_t num_points,
//                                 size_t dim, size_t num_centers,
//                                 size_t      num_pq_chunks,
//                                 std::string pq_pivots_path,
//                                 std::string pq_compressed_vectors_path);
