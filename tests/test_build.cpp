//#include <distances.h>
//#include <indexing.h>
#include <index_nsg.h>
#include <math_utils.h>
#include <partitionAndPQ.h>
#include "util.h"

// #define TRAINING_SET_SIZE 2000000
//#define TRAINING_SET_SIZE 2000000

template<typename T>
bool testBuildIndex(const char* dataFilePath, const char* indexFilePath,
                    const char* indexBuildParameters) {
  std::stringstream parser;
  parser << std::string(indexBuildParameters);
  std::string              cur_param;
  std::vector<std::string> param_list;
  while (parser >> cur_param)
    param_list.push_back(cur_param);

  if (param_list.size() != 5) {
    std::cout << "Correct usage of parameters is L (indexing search list size) "
                 "R (max degree) C (visited list maximum size) B (approximate "
                 "compressed number of bytes per datapoint to store in "
                 "memory) TRAINING_SIZE"
              << std::endl;
    return 1;
  }

  std::string index_prefix_path(indexFilePath);
  std::string index_params_path = index_prefix_path + "_params.bin";
  std::string train_file_path = index_prefix_path + "_training_set_float.bin";
  std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
  std::string pq_compressed_vectors_path =
      index_prefix_path + "_compressed_uint32.bin";
  std::string randnsg_path = index_prefix_path + "_unopt.rnsg";
  std::string diskopt_path = index_prefix_path + "_diskopt.rnsg";

  unsigned L = (unsigned) atoi(param_list[0].c_str());
  unsigned R = (unsigned) atoi(param_list[1].c_str());
  unsigned C = (unsigned) atoi(param_list[2].c_str());
  size_t   num_pq_chunks = (size_t) atoi(param_list[3].c_str());
  size_t   TRAINING_SET_SIZE = (size_t) atoi(param_list[4].c_str());

  T* data_load = NULL;

  size_t points_num, dim;

  NSG::load_bin<T>(dataFilePath, data_load, points_num, dim);
  std::cout << "Data loaded" << std::endl;


  auto s = std::chrono::high_resolution_clock::now();

  if (points_num > 2 * TRAINING_SET_SIZE) {
    gen_random_slice(data_load, points_num, dim, train_file_path.c_str(),
                     (size_t) TRAINING_SET_SIZE);
  } else {
    float* float_data = new float[points_num * dim];
    for (size_t i = 0; i < points_num; i++) {
      for (size_t j = 0; j < dim; j++) {
        float_data[i * dim + j] = data_load[i * dim + j];
      }
    }

    NSG::save_bin<float>(train_file_path.c_str(), float_data, points_num, dim);
    delete[] float_data;
  }

  //  unsigned    nn_graph_deg = (unsigned) atoi(argv[3]);

  generate_pq_pivots<T>(train_file_path, 256, num_pq_chunks, 15,
                        pq_pivots_path);
  generate_pq_data_from_pivots<T>(data_load, points_num, dim, 256,
                                  num_pq_chunks, pq_pivots_path,
                                  pq_compressed_vectors_path);

  NSG::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", 1.2f);
  paras.Set<unsigned>("num_rnds", 2);
  paras.Set<std::string>("save_path", randnsg_path);

  data_load = NSG::data_align(data_load, points_num, dim);
  std::cout << "Base data aligned for optimized Rand-NSG execution."
            << std::endl;
  NSG::IndexNSG<T>* _pNsgIndex =
      new NSG::IndexNSG<T>(dim, points_num, NSG::L2, nullptr);
  if(file_exists(randnsg_path.c_str())) {
	  _pNsgIndex->SetData(data_load);
	  _pNsgIndex->Load(randnsg_path.c_str());
  }
  else {
  _pNsgIndex->BuildRandomHierarchical(data_load, paras);
  _pNsgIndex->Save(randnsg_path.c_str());
  }

  _pNsgIndex->save_disk_opt_graph(diskopt_path.c_str());


  uint32_t* params_array = new uint32_t[5];
  params_array[0] = (uint32_t) L;
  params_array[1] = (uint32_t) R;
  params_array[2] = (uint32_t) C;
  params_array[3] = (uint32_t) dim;
  params_array[4] = (uint32_t) num_pq_chunks;
  NSG::save_bin<uint32_t>(index_params_path.c_str(), params_array, 5, 1);
  std::cout << "Saving params to " << index_params_path << "\n";

  auto                          e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  std::cout << "Indexing time: " << diff.count() << "\n";

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cout << "Usage: " << argv[0]
              << " data_file[bin] data_type [float/uint8/int8] index_prefix L "
                 "R C N_CHUNKS TRAINING_SIZE"
              << std::endl;
  } else {
    std::string params = std::string(argv[4]) + " " + std::string(argv[5]) +
                         " " + std::string(argv[6]) + " " +
                         std::string(argv[7]) + " " + std::string(argv[8]);
    if (std::string(argv[2]) == std::string("float"))
      testBuildIndex<float>(argv[1], argv[3], params.c_str());
    else if (std::string(argv[2]) == std::string("int8"))
      testBuildIndex<int8_t>(argv[1], argv[3], params.c_str());
    else if (std::string(argv[2]) == std::string("uint8"))
      testBuildIndex<uint8_t>(argv[1], argv[3], params.c_str());
    else
      std::cout << "Error. wrong file type" << std::endl;
  }
}
