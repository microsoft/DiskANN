#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include "util.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <typeinfo>

template<typename T>
int aux_main(int argc, char **argv) {
  _u64            read_blk_size = 64 * 1024 * 1024;
  cached_ifstream base_reader(argv[2], read_blk_size);
  std::ofstream   sample_writer(
      std::string(std::string(argv[3]) + "_data.bin").c_str(),
      std::ios::binary);
  std::ofstream sample_id_writer(
      std::string(std::string(argv[3]) + "_ids.bin").c_str(), std::ios::binary);
  float sampling_rate = atof(argv[4]);

  std::random_device
               rd;  // Will be used to obtain a seed for the random number engine
  size_t       x = rd();
  std::mt19937 generator(
      x);  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distribution(0, 1);

  T *      cur_row;
  size_t   npts, nd;
  uint32_t npts_u32, nd_u32;
  uint32_t num_sampled_pts_u32 = 0;
  uint32_t one_const = 1;

  base_reader.read((char *) &npts_u32, sizeof(uint32_t));
  base_reader.read((char *) &nd_u32, sizeof(uint32_t));
  std::cout << "Loading base " << argv[2] << ". #points: " << npts_u32
            << ". #dim: " << nd_u32 << "." << std::endl;
  sample_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_writer.write((char *) &nd_u32, sizeof(uint32_t));
  sample_id_writer.write((char *) &num_sampled_pts_u32, sizeof(uint32_t));
  sample_id_writer.write((char *) &one_const, sizeof(uint32_t));

  npts = npts_u32;
  nd = nd_u32;
  cur_row = new T[nd];

  for (size_t i = 0; i < npts; i++) {
    base_reader.read((char *) cur_row, sizeof(T) * nd);
    float sample = distribution(generator);
    if (sample < sampling_rate) {
      sample_writer.write((char *) cur_row, sizeof(T) * nd);
      uint32_t cur_i_u32 = i;
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
  delete[] cur_row;
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << argv[0] << " data_type [fliat/int8/uint8] base_bin_file "
                            "sample_output_prefix sampling_probability"
              << std::endl;
    exit(-1);
  }

  if (std::string(argv[1]) == std::string("float")) {
    aux_main<float>(argc, argv);
  } else if (std::string(argv[1]) == std::string("int8")) {
    aux_main<int8_t>(argc, argv);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    aux_main<uint8_t>(argc, argv);
  } else
    std::cout << "Unsupported type. Use float/int8/uint8." << std::endl;
  return 0;
}
