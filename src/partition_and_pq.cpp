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
#include "util.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <typeinfo>

#include <cassert>
#include "MemoryMapper.h"
#include "partition_and_pq.h"
#ifdef __NSG_WINDOWS__
#include <xmmintrin.h>
#endif

template<typename T>
void gen_random_slice(T *base_data, size_t points_num, size_t dim,
                      const char *outputfile, size_t slice_size) {
  std::cout << "Generating random sample of base data to use as training.."
            << std::flush;

  if (slice_size > points_num) {
    std::cout << "Error generating randomr slice. Slice size is greater than "
                 "number of points"
              << std::endl;
    return;
  }

  float *sampled_data = new float[slice_size * dim];
  size_t counter = 0;

  //  std::ofstream out(outputfile, std::ios::binary | std::ios::out);

  std::random_device
               rd;  // Will be used to obtain a seed for the random number engine
  size_t       x = rd();
  std::mt19937 generator(
      x);  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<size_t> distribution(0, points_num - 1);

  size_t            tmp_pivot;
  std::vector<bool> flag(points_num, 0);

  for (size_t i = 0; i < slice_size; i++) {
    tmp_pivot = distribution(generator);
    while (flag[tmp_pivot] == true) {
      tmp_pivot = distribution(generator);
    }
    flag[tmp_pivot] = true;
    for (size_t iterdim = 0; iterdim < dim; iterdim++)
      sampled_data[counter * dim + iterdim] =
          base_data[tmp_pivot * dim + iterdim];
    counter++;
  }

  NSG::save_bin<float>(outputfile, sampled_data, slice_size, dim);
  std::cout << "done." << std::endl;
}

template<typename T>
int generate_pq_pivots(std::string train_file_path, size_t num_centers,
                       size_t num_pq_chunks, size_t max_k_means_reps,
                       std::string pq_pivots_path) {
  size_t num_train, dim;
  float *train_data;
  NSG::load_bin<float>(train_file_path.c_str(), train_data, num_train, dim);

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
    NSG::load_bin<float>(pq_pivots_path.c_str(), full_pivot_data,
                         file_num_centers, file_dim);
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
    for (int64_t j = 0; j < num_train; j++) {
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

  //  save_Tvecs_plain<float>(pq_pivots_path.c_str(), full_pivot_data,
  //                          (size_t) num_centers, dim);
  NSG::save_bin<float>(pq_pivots_path.c_str(), full_pivot_data,
                       (size_t) num_centers, dim);
  return 0;
}

template<typename T>
int generate_pq_data_from_pivots(const T *base_data, size_t num_points,
                                 size_t dim, size_t num_centers,
                                 size_t      num_pq_chunks,
                                 std::string pq_pivots_path,
                                 std::string pq_compressed_vectors_path) {
  if (num_pq_chunks > dim) {
    std::cout << " Error: number of chunks more than dimension" << std::endl;
    return -1;
  }
  size_t chunk_size = std::floor((double) dim / (double) num_pq_chunks);
  size_t corrected_num_pq_chunks = DIV_ROUND_UP(dim, chunk_size);

  uint32_t *compressed_base;

  std::string lossy_vectors_float_bin_path = "_lossy_vectors_float.bin";

  if (file_exists(pq_compressed_vectors_path)) {
    size_t file_dim, file_num_points;
    NSG::load_bin<uint32_t>(pq_compressed_vectors_path.c_str(), compressed_base,
                            file_num_points, file_dim);
    if (file_dim == corrected_num_pq_chunks && file_num_points == num_points) {
      std::cout << "Compressed base file exists. Not generating again"
                << std::endl;
      delete[] compressed_base;
      return -1;
    }
  }

  float *full_pivot_data = new float[num_centers * dim];
  compressed_base = new uint32_t[num_points * corrected_num_pq_chunks];

  if (!file_exists(pq_pivots_path)) {
    // Process Global k-means for kmeans_partitioning Step
    std::cout << "ERROR: PQ k-means pivot file not found" << std::endl;
    return -1;
  } else {
    size_t file_num_centers;
    size_t file_dim;
    NSG::load_bin<float>(pq_pivots_path.c_str(), full_pivot_data,
                         file_num_centers, file_dim);

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

  for (size_t i = 0; i < corrected_num_pq_chunks; i++) {
    size_t cur_chunk_size =
        chunk_size < (dim - i * chunk_size) ? chunk_size : dim - i * chunk_size;
    float *   cur_pivot_data = new float[num_centers * cur_chunk_size];
    float *   cur_data = new float[num_points * cur_chunk_size];
    uint32_t *closest_center = new uint32_t[num_points];

    std::cout << "Processing chunk " << i << " with dimensions ["
              << i * chunk_size << ", " << i * chunk_size + cur_chunk_size
              << ")" << std::endl;
#pragma omp parallel for schedule(static, 8192)
    for (int64_t j = 0; j < num_points; j++) {
      for (uint64_t k = 0; k < cur_chunk_size; k++)
        cur_data[j * cur_chunk_size + k] =
            base_data[j * dim + i * chunk_size + k];
    }

#pragma omp parallel for schedule(static, 1)
    for (int64_t j = 0; j < num_centers; j++) {
      std::memcpy(cur_pivot_data + j * cur_chunk_size,
                  full_pivot_data + j * dim + i * chunk_size,
                  cur_chunk_size * sizeof(float));
    }

    math_utils::compute_closest_centers(cur_data, num_points, cur_chunk_size,
                                        cur_pivot_data, num_centers, 1,
                                        closest_center);

#pragma omp parallel for schedule(static, 8192)
    for (int64_t j = 0; j < num_points; j++) {
      compressed_base[j * corrected_num_pq_chunks + i] = closest_center[j];
    }

    delete[] cur_data;
    delete[] cur_pivot_data;
    delete[] closest_center;
  }

  NSG::save_bin<uint32_t>(pq_compressed_vectors_path.c_str(), compressed_base,
                          (size_t) num_points, corrected_num_pq_chunks);

  //  save_bin<float>(lossy_vectors_float_bin_path.c_str(), base_data,
  //                          (size_t) num_points, dim);

  return 0;
}

/*template<typename T>
int partition(const char *base_file, const char *train_file, size_t num_centers,
              size_t max_k_means_reps, const char *prefix_dir, size_t k_base) {
  size_t    dim;
  size_t    train_dim;
  size_t    num_points;
  size_t    num_train;
  T *       base_data;
  float *   train_data;
  float *   pivot_data;
  uint32_t *base_closest_centers;

  NSG::load_bin<float>(train_file, train_data, num_train, train_dim);
  NSG::load_bin<T>(base_file, base_data, num_points, dim);

  if (train_dim != dim) {
    std::cout << "Error: training and base dimensions dont match" << std::endl;
    return -1;
  }

  std::string cur_file = std::string(prefix_dir);
  std::string output_file;

  // Random rotation before kmeans_partitioning

  // kmeans_partitioning on training data

  cur_file = cur_file + "_kmeans_partitioning-" + std::to_string(num_centers);
  output_file = cur_file + "_pivots_float.bin";

  if (!file_exists(output_file)) {
    pivot_data = new float[num_centers * dim];

    // Process Global k-means for kmeans_partitioning Step
    std::cout << "Processing global k-means (kmeans_partitioning Step)"
              << std::endl;
    kmeans::kmeanspp_selecting_pivots(train_data, num_train, dim, pivot_data,
                                      num_centers);

    float residual = 0;
    residual = kmeans::run_lloyds(train_data, num_train, dim, pivot_data,
                                  num_centers, max_k_means_reps, NULL, NULL);

    std::cout << "Saving global k-center pivots" << std::endl;
    NSG::save_bin<float>(output_file.c_str(), pivot_data, (size_t) num_centers,
                         dim);
  } else {
    size_t file_num_centers;
    size_t file_dim;
    NSG::load_bin<float>(output_file.c_str(), pivot_data, file_num_centers,
                         file_dim);
    if (file_num_centers != num_centers || file_dim != dim) {
      std::cout << "ERROR: file number of kmeans_partitioning centers does "
                   "not match input argument (or) file "
                   "dimension does not match real dimension"
                << std::endl;
      return -1;
    }
  }

  output_file = cur_file + "_nn-" + std::to_string(k_base) + "_uint32.bin";

  if (!file_exists(output_file)) {
    std::cout << "Closest centers file does not exist. Computing..."
              << std::flush;
    base_closest_centers = new uint32_t[num_points * k_base];
    math_utils::compute_closest_centers(base_data, num_points, dim, pivot_data,
                                        num_centers, k_base,
                                        base_closest_centers);
    std::cout << "done. Now saving file..." << std::flush;
    NSG::save_bin<uint32_t>(output_file.c_str(), base_closest_centers,
                            num_points, k_base);
    std::cout << "done." << std::endl;
  } else {
    size_t file_num;
    size_t file_d;
    NSG::load_bin<uint32_t>(output_file.c_str(), base_closest_centers, file_num,
                            file_d);
    if (file_num != num_points || file_d != k_base) {
      std::cout << "ERROR: kmeans_partitioning nearest neighbor file does "
                   "not match in parameters. "
                << std::endl;
      return -1;
    }
    std::cout << "Loaded kmeans_partitioning nearest neighbors from file."
              << std::endl;
  }

  bool *inverted_index = new bool[num_points * num_centers];

#pragma omp parallel for schedule(static, 8192)
  for (int64_t i = 0; i < num_points; i++)
    for (size_t j = 0; j < num_centers; j++)
      inverted_index[i * num_centers + j] = false;
#pragma omp parallel for schedule(static, 8192)
  for (int64_t i = 0; i < num_points; i++)
    for (size_t j = 0; j < k_base; j++)
      inverted_index[i * num_centers + base_closest_centers[i * k_base + j]] =
          true;

  size_t    total_count = 0;
  T *       cur_base = new float[num_points * dim];
  uint32_t *cur_ids = new uint32_t[num_points];
  //#pragma omp parallel for schedule(static, 1)
  for (size_t i = 0; i < num_centers; i++) {
    size_t cur_count = 0;
    for (size_t j = 0; j < num_points; j++) {
      if (inverted_index[j * num_centers + i] == true) {
        std::memcpy(cur_base + cur_count * dim, base_data + j * dim,
                    dim * sizeof(T));
        cur_ids[cur_count] = j;
        cur_count++;
      }
    }
    total_count += cur_count;

    output_file = cur_file + "_subshard-" + std::to_string(i) + ".bin";
    NSG::save_bin<T>(output_file.c_str(), cur_base, cur_count, dim);
    output_file =
        cur_file + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
    NSG::save_bin<uint32_t>(output_file.c_str(), cur_ids, cur_count, 1);
  }
  std::cout << "saved all files totalling " << total_count << " points out of "
            << num_points << " points" << std::endl;
  delete[] cur_base;
  delete[] cur_ids;
  delete[] base_closest_centers;
  delete[] train_data;
  delete[] base_data;
  delete[] inverted_index;
}
*/

template void gen_random_slice<int8_t>(int8_t *base_data, size_t points_num,
                                       size_t dim, const char *outputfile,
                                       size_t slice_size);
template void gen_random_slice<float>(float *base_data, size_t points_num,
                                      size_t dim, const char *outputfile,
                                      size_t slice_size);
template void gen_random_slice<uint8_t>(uint8_t *base_data, size_t points_num,
                                        size_t dim, const char *outputfile,
                                        size_t slice_size);
/*
template int partition<int8_t> (const char *base_file, const char *train_file,
size_t num_centers,
              size_t max_k_means_reps, const char *prefix_dir, size_t k_base);
template int partition<uint8_t> (const char *base_file, const char *train_file,
size_t num_centers,
              size_t max_k_means_reps, const char *prefix_dir, size_t k_base);
template int partition<float> (const char *base_file, const char *train_file,
size_t num_centers,
              size_t max_k_means_reps, const char *prefix_dir, size_t k_base);
*/

template int generate_pq_pivots<int8_t>(std::string train_file_path,
                                        size_t      num_centers,
                                        size_t      num_pq_chunks,
                                        size_t      max_k_means_reps,
                                        std::string pq_pivots_path);
template int generate_pq_pivots<uint8_t>(std::string train_file_path,
                                         size_t      num_centers,
                                         size_t      num_pq_chunks,
                                         size_t      max_k_means_reps,
                                         std::string pq_pivots_path);
template int generate_pq_pivots<float>(std::string train_file_path,
                                       size_t num_centers, size_t num_pq_chunks,
                                       size_t      max_k_means_reps,
                                       std::string pq_pivots_path);

template int generate_pq_data_from_pivots<int8_t>(
    const int8_t *base_data, size_t num_points, size_t dim, size_t num_centers,
    size_t num_pq_chunks, std::string pq_pivots_path,
    std::string pq_compressed_vectors_path);
template int generate_pq_data_from_pivots<uint8_t>(
    const uint8_t *base_data, size_t num_points, size_t dim, size_t num_centers,
    size_t num_pq_chunks, std::string pq_pivots_path,
    std::string pq_compressed_vectors_path);
template int generate_pq_data_from_pivots<float>(
    const float *base_data, size_t num_points, size_t dim, size_t num_centers,
    size_t num_pq_chunks, std::string pq_pivots_path,
    std::string pq_compressed_vectors_path);
