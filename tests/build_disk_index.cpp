//#include <distances.h>
//#include <indexing.h>
//
#include <omp.h>

#include <index.h>
#include <math_utils.h>
#include "partition_and_pq.h"
#include "utils.h"

#define TRAINING_SET_SIZE 1500000

template<typename T>
bool build_disk_index(const char* dataFilePath, const char* indexFilePath,
                      const char*  indexBuildParameters,
                      const Metric compareMetric) {
  std::stringstream parser;
  parser << std::string(indexBuildParameters);
  std::string              cur_param;
  std::vector<std::string> param_list;
  while (parser >> cur_param)
    param_list.push_back(cur_param);

  if (param_list.size() != 5) {
    std::cout << "Correct usage of parameters is L (indexing search list size) "
                 "R (max degree) B (RAM limit of final index in "
                 "GB) M (memory limit while indexing) T (number of threads for "
                 "indexing)"
              << std::endl;
    return 1;
  }

  std::string index_prefix_path(indexFilePath);
  std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
  std::string pq_compressed_vectors_path =
      index_prefix_path + "_compressed.bin";
  std::string mem_index_path = index_prefix_path + "_mem.index";
  std::string disk_index_path = index_prefix_path + "_disk.index";

  unsigned L = (unsigned) atoi(param_list[0].c_str());
  unsigned R = (unsigned) atoi(param_list[1].c_str());
  double   final_index_ram_limit =
      (((double) atof(param_list[2].c_str())) - 0.25) * 1024.0 * 1024.0 *
      1024.0;
  double indexing_ram_budget = (float) atof(param_list[3].c_str());
  _u32   num_threads = (_u32) atoi(param_list[4].c_str());

  auto s = std::chrono::high_resolution_clock::now();

  std::cout << "loading data.." << std::endl;
  T* data_load = NULL;

  size_t points_num, dim;

  diskann::get_bin_metadata(dataFilePath, points_num, dim);

  size_t num_pq_chunks = (std::floor)(_u64(final_index_ram_limit / points_num));
  std::cout << "Going to compress " << dim << "-dimensional data into "
            << num_pq_chunks << " bytes per vector." << std::endl;

  auto s = std::chrono::high_resolution_clock::now();

  size_t train_size, train_dim;
  float* train_data;

  double p_val = ((double) TRAINING_SET_SIZE / (double) points_num);
  // generates random sample and sets it to train_data and updates train_size
  gen_random_slice<T>(dataFilePath, p_val, train_data, train_size, train_dim);

  std::cout << "Training data loaded of size " << train_size << std::endl;

  generate_pq_pivots(train_data, train_size, dim, 256, num_pq_chunks, 15,
                     pq_pivots_path);
  generate_pq_data_from_pivots<T>(dataFilePath, 256, num_pq_chunks,
                                  pq_pivots_path, pq_compressed_vectors_path);

  delete[] train_data;

  train_data = nullptr;

  diskann::build_merged_vamana_index<T>(dataFilePath, compareMetric, L, R,
                                        p_val, indexing_ram_budget,
                                        mem_index_path);

  double sample_sampling_rate = (150000.0 / points_num);

  gen_random_slice<T>(dataFilePath, sample_base_prefix, sample_sampling_rate);

  auto                          e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  std::cout << "Indexing time: " << diff.count() << "\n";

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cout << "Usage: " << argv[0]
              << "  <data_type [float/uint8/int8]>   <data_file [.bin]>  "
                 "<index_prefix_path>  <L "
                 "< [ram limit of final index in GB]> <M [indexing time memory "
                 "limit in GB]> <T [number of threads during indexing>"
              << std::endl;
  } else {
    std::string params = std::string(argv[4]) + " " + std::string(argv[5]) +
                         " " + std::string(argv[6]) + " " +
                         std::string(argv[7]) + " " + std::string(argv[8]);
    if (std::string(argv[1]) == std::string("float"))
      build_disk_index<float>(argv[2], argv[3], params.c_str());
    else if (std::string(argv[1]) == std::string("int8"))
      build_disk_index<int8_t>(argv[2], argv[3], params.c_str());
    else if (std::string(argv[1]) == std::string("uint8"))
      build_disk_index<uint8_t>(argv[2], argv[3], params.c_str());
    else
      std::cout << "Error. wrong file type" << std::endl;
  }
}
