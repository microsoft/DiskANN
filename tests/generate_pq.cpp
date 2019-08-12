#include "math_utils.h"
#include "partition_and_pq.h"

#define KMEANS_ITERS_FOR_PQ 15

template<typename T>
bool generate_pq(const std::string& data_path,
                 const std::string& index_prefix_path,
                 const size_t num_pq_centers, const size_t num_pq_chunks,
                 const float sampling_rate) {
  std::string train_file_path = index_prefix_path + "_training_set_float.bin";
  std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
  std::string pq_compressed_vectors_path =
      index_prefix_path + "_compressed.bin";

  std::cout << "Loading data.." << std::endl;
  T*     data_load = NULL;
  size_t points_num, dim;
  NSG::load_bin<T>(data_path, data_load, points_num, dim);
  std::cout << "done." << std::endl;

  // generates random sample and sets it to train_data and updates train_size
  size_t train_size, train_dim;
  float* train_data;
 gen_random_slice<float>(data_path, sampling_rate, train_data,
                      train_size, train_dim);
  std::cout << "For computing pivots, loaded sample data of size " << train_size
            << std::endl;

  generate_pq_pivots(train_data, train_size, dim, num_pq_centers, num_pq_chunks,
                     KMEANS_ITERS_FOR_PQ, pq_pivots_path);
  generate_pq_data_from_pivots<T>(data_load, points_num, dim, num_pq_centers,
                                  num_pq_chunks, pq_pivots_path,
                                  pq_compressed_vectors_path);

  delete[] data_load;
  delete[] train_data;

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << "Usage: \n"
              << argv[0]
              << "  <data_type[float/uint8/int8]>   <data_file[.bin]>"
                 "  <PQ_prefix_path>  <target-bytes/data-point>  <sample_size>"
              << std::endl;
  } else {
    const std::string data_path(argv[2]);
    const std::string index_prefix_path(argv[3]);
    const size_t      num_pq_centers = 256;
    const size_t      num_pq_chunks = (size_t) atoi(argv[4]);
    const size_t      sample_size = (size_t) atoi(argv[5]);

    if (std::string(argv[1]) == std::string("float"))
      generate_pq<float>(data_path, index_prefix_path, num_pq_centers,
                         num_pq_chunks, sample_size);
    else if (std::string(argv[1]) == std::string("int8"))
      generate_pq<int8_t>(data_path, index_prefix_path, num_pq_centers,
                          num_pq_chunks, sample_size);
    else if (std::string(argv[1]) == std::string("uint8"))
      generate_pq<uint8_t>(data_path, index_prefix_path, num_pq_centers,
                           num_pq_chunks, sample_size);
    else
      std::cout << "Error. wrong file type" << std::endl;
  }
}
