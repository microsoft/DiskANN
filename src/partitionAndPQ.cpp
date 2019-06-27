#include <math_utils.h>
#include <omp.h>
#include <utils.h>
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
#include "efanna2e/exceptions.h"
#include "efanna2e/index_nsg.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "tsl/robin_set.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include <partitionAndPQ.h>
#include <cassert>
#include "MemoryMapper.h"
#ifdef __NSG_WINDOWS__
#include <xmmintrin.h>
#endif

template <typename T>
void gen_random_slice(T *base_data, size_t points_num, size_t dim,
                      const char *outputfile,
                      size_t      slice_size) {  // load data with fvecs pattern
  std::cout << "Generating random sample of base data to use as training.."
            << std::flush;
  uint32_t unsigned_dim = static_cast<uint32_t>(dim);

  bool *flag = new bool[points_num];
  for (size_t i = 0; i < points_num; i++) {
    flag[i] = false;
  }

  std::ofstream out(outputfile, std::ios::binary | std::ios::out);

  std::random_device
               rd;  // Will be used to obtain a seed for the random number engine
  size_t       x = rd();
  std::mt19937 generator(
      x);  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<size_t> distribution(0, points_num - 1);

  size_t tmp_pivot;
  for (size_t i = 0; i < slice_size; i++) {
    tmp_pivot = distribution(generator);
    while (flag[tmp_pivot] == true) {
      tmp_pivot = distribution(generator);
    }
    flag[tmp_pivot] = true;
    out.write((char *) &unsigned_dim, 4);
    out.write((char *) (base_data + (tmp_pivot * dim)), dim * sizeof(T));
  }
  out.close();
  std::cout << "done." << std::endl;
}

int generate_pq_pivots(const char *train_file, size_t num_centers,
                       size_t num_chunks, size_t max_k_means_reps,
                       const char *working_prefix_file) {
  size_t num_train, dim;
  float *train_data;
  load_file_into_data<float>(train_file, train_data, num_train, dim);

  if (num_chunks > dim) {
    std::cout << " Error: number of chunks more than dimension" << std::endl;
    return -1;
  }

  size_t    chunk_size = std::floor((double) dim / (double) num_chunks);
  uint32_t *ivf_closest_center = new uint32_t[num_train];
  float *   full_pivot_data;
  size_t    corrected_num_chunks = DIV_ROUND_UP(dim, chunk_size);

  std::cout << "Corrected number of chunks " << corrected_num_chunks
            << std::endl;

  std::string pivot_file_path = std::string(working_prefix_file) + "_PQ-" +
                                std::to_string(num_centers) + "-" +
                                std::to_string(num_chunks) + ".piv";

  if (file_exists(pivot_file_path)) {
    size_t file_dim, file_num_centers;
    load_Tvecs_plain<float, float>(pivot_file_path.c_str(), full_pivot_data,
                                   file_num_centers, file_dim);
    if (file_dim == dim && file_num_centers == num_centers) {
      std::cout << "PQ pivot file exists. Not generating again" << std::endl;
      delete[] full_pivot_data;
      return 1;
    }
  }

  full_pivot_data = new float[num_centers * dim];

  for (size_t i = 0; i < corrected_num_chunks; i++) {
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

    float residual = 0;
    residual =
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

  save_Tvecs_plain<float>(pivot_file_path.c_str(), full_pivot_data,
                    (size_t) num_centers, dim);
  return 0;
}

int generate_pq_data_from_pivots(const char *base_file, size_t num_centers,
                                 size_t      num_chunks,
                                 const char *working_prefix_file) {
  size_t dim;
  size_t num_points;
  //  float*      data_load;
  float *base_data;
  load_file_into_data<float>(base_file, base_data, num_points, dim);
  std::cout << "Base Loaded\n";

  if (num_chunks > dim) {
    std::cout << " Error: number of chunks more than dimension" << std::endl;
    return -1;
  }
  size_t chunk_size = std::floor((double) dim / (double) num_chunks);
  size_t corrected_num_chunks = DIV_ROUND_UP(dim, chunk_size);

  uint32_t *compressed_base;

  std::string compressed_file_path =
      std::string(working_prefix_file) + "_PQ-" + std::to_string(num_centers) +
      "-" + std::to_string(num_chunks) + "_compressed.ivecs";

  std::string lossy_fvecs_path = std::string(working_prefix_file) + "_PQ-" +
                                 std::to_string(num_centers) + "-" +
                                 std::to_string(num_chunks) + "_lossy.fvecs";

  std::string pivot_file_path = std::string(working_prefix_file) + "_PQ-" +
                                std::to_string(num_centers) + "-" +
                                std::to_string(num_chunks) + ".piv";

  if (file_exists(compressed_file_path)) {
    size_t file_dim, file_num_points;
    load_Tvecs_plain<uint32_t, uint32_t>(compressed_file_path.c_str(),
                                         compressed_base, file_num_points,
                                         file_dim);
    std::cout << "Here " << file_num_points << " " << file_dim << std::endl;
    if (file_dim == corrected_num_chunks && file_num_points == num_points) {
      std::cout << "Compressed base file exists. Not generating again"
                << std::endl;
      delete[] compressed_base;
      return 1;
    }
  }

  float *full_pivot_data = new float[num_centers * dim];
  compressed_base = new uint32_t[num_points * corrected_num_chunks];

  if (!file_exists(pivot_file_path)) {
    // Process Global k-means for IVF Step
    std::cout << "ERROR: PQ k-means pivot file not found" << std::endl;
    return -1;
  } else {
    size_t file_num_centers;
    size_t file_dim;
    load_Tvecs_plain<float, float>(pivot_file_path.c_str(), full_pivot_data,
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

  for (size_t i = 0; i < corrected_num_chunks; i++) {
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
      std::memcpy(cur_data + j * cur_chunk_size,
                  base_data + j * dim + i * chunk_size,
                  cur_chunk_size * sizeof(float));
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
      std::memcpy(base_data + j * dim + i * chunk_size,
                  cur_pivot_data + closest_center[j] * cur_chunk_size,
                  cur_chunk_size * sizeof(float));
      compressed_base[j * corrected_num_chunks + i] = closest_center[j];
    }

    delete[] cur_data;
    delete[] cur_pivot_data;
    delete[] closest_center;
  }

  save_Tvecs_plain<uint32_t>(compressed_file_path.c_str(), compressed_base,
                             (size_t) num_points, corrected_num_chunks);

  save_Tvecs_plain<float>(lossy_fvecs_path.c_str(), base_data,
                          (size_t) num_points, dim);

  delete[] base_data;
  return 0;
}

int partition(const char *base_file, const char *train_file, size_t num_centers,
              size_t max_k_means_reps, const char *prefix_dir, size_t k_base) {
  size_t    dim;
  size_t    train_dim;
  size_t    num_points;
  size_t    num_train;
  float *   base_data;
  float *   train_data;
  float *   pivot_data;
  uint32_t *base_closest_centers;

  load_file_into_data<float>(train_file, train_data, num_train, train_dim);
  load_file_into_data<float>(base_file, base_data, num_points, dim);

  if (train_dim != dim) {
    std::cout << "Error: training and base dimensions dont match" << std::endl;
    return -1;
  }

  std::string cur_file = std::string(prefix_dir);
  std::string output_file;

  // Random rotation before IVF

  // IVF on training data

  cur_file = cur_file + "_IVF-" + std::to_string(num_centers);
  output_file = cur_file + ".piv";

  if (!file_exists(output_file)) {
    pivot_data = new float[num_centers * dim];

    // Process Global k-means for IVF Step
    std::cout << "Processing global k-means (IVF Step)" << std::endl;
    kmeans::kmeanspp_selecting_pivots(train_data, num_train, dim, pivot_data,
                                      num_centers);

    float residual = 0;
    residual = kmeans::run_lloyds(train_data, num_train, dim, pivot_data,
                                  num_centers, max_k_means_reps, NULL, NULL);

    std::cout << "Saving global k-center pivots" << std::endl;
    save_Tvecs_plain<float>(output_file.c_str(), pivot_data,
                            (size_t) num_centers, dim);
  } else {
    size_t file_num_centers;
    size_t file_dim;
    load_Tvecs_plain<float, float>(output_file.c_str(), pivot_data,
                                   file_num_centers, file_dim);
    if (file_num_centers != num_centers || file_dim != dim) {
      std::cout << "ERROR: file number of IVF centers does "
                   "not match input argument (or) file "
                   "dimension does not match real dimension"
                << std::endl;
      return -1;
    }
  }

  output_file = cur_file + "_nn-" + std::to_string(k_base) + ".ivecs";

  if (!file_exists(output_file)) {
    std::cout << "Closest centers file does not exist. Computing..."
              << std::flush;
    base_closest_centers = new uint32_t[num_points * k_base];
    math_utils::compute_closest_centers(base_data, num_points, dim, pivot_data,
                                        num_centers, k_base,
                                        base_closest_centers);
    std::cout << "done. Now saving file..." << std::flush;
    save_Tvecs_plain<uint32_t>(output_file.c_str(), base_closest_centers,
                               num_points, k_base);
    std::cout << "done." << std::endl;
  } else {
    size_t file_num;
    size_t file_d;
    load_Tvecs_plain<uint32_t, uint32_t>(
        output_file.c_str(), base_closest_centers, file_num, file_d);
    if (file_num != num_points || file_d != k_base) {
      std::cout << "ERROR: IVF nearest neighbor file does "
                   "not match in parameters. "
                << std::endl;
      return -1;
    }
    std::cout << "Loaded IVF nearest neighbors from file." << std::endl;
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
  float *   cur_base = new float[num_points * dim];
  uint32_t *cur_ids = new uint32_t[num_points];
  //#pragma omp parallel for schedule(static, 1)
  for (size_t i = 0; i < num_centers; i++) {
    size_t cur_count = 0;
    for (size_t j = 0; j < num_points; j++) {
      if (inverted_index[j * num_centers + i] == true) {
        std::memcpy(cur_base + cur_count * dim, base_data + j * dim,
                    dim * sizeof(float));
        cur_ids[cur_count] = j;
        cur_count++;
      }
    }
    total_count += cur_count;

    output_file = cur_file + "_subshard-" + std::to_string(i) + ".fvecs";
    save_Tvecs_plain<float>(output_file.c_str(), cur_base, cur_count, dim);
    output_file = cur_file + "_subshard-" + std::to_string(i) + "_ids.ivecs";
    save_Tvecs_plain<uint32_t>(output_file.c_str(), cur_ids, cur_count, 1);
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

template void gen_random_slice<int8_t>(int8_t *base_data, size_t points_num, size_t dim,
                               const char *outputfile, size_t slice_size);
template void gen_random_slice<float>(float *base_data, size_t points_num,
                                       size_t dim, const char *outputfile,
                                       size_t slice_size);
template void gen_random_slice<uint8_t>(uint8_t *base_data, size_t points_num,
                                       size_t dim, const char *outputfile,
                                       size_t slice_size);