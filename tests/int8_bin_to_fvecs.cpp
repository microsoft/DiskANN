#include <iostream>
#include <string>
#include "util.h"

int main(int argc, char** argv) {
  if (argc != 4 && argc != 3) {
    std::cout << argv[0]
              << "input_file output_file scaling_factor (default = 1)"
              << std::endl;
    exit(-1);
  }
  int8_t* srcdata;
  size_t  num, dim;
  float   scaling_factor = 1;
  if (argc == 4)
    scaling_factor = (float) std::atof(argv[3]);
  NSG::load_bin<int8_t>(argv[1], srcdata, num, dim);
  float* data_float = new float[num * dim];
  NSG::convert_types<int8_t, float>(srcdata, data_float, num, dim);
  for (size_t i = 0; i < num; i++) {
    for (size_t j = 0; j < dim; j++)
      data_float[i * dim + j] /= scaling_factor;
  }
  NSG::save_Tvecs<float>(argv[2], data_float, num, dim);
  delete[] data_float;
  delete[] srcdata;
}
