#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include "efanna2e/util.h"

void write_low_prec(char* filename, float* data, unsigned num, unsigned dim) {
  std::ofstream writer(filename, std::ios::binary | std::ios::out);
  float         abs_max = std::numeric_limits<float>::min();
  float         scale_factor = 127.0f;
  for (size_t i = 0; i < (size_t) num * (size_t) dim; i++) {
    abs_max = std::max(std::abs(data[i]), abs_max);
  }
  scale_factor /= abs_max;
  std::cout << "Comptuted scale factor = " << scale_factor << std::endl;
#pragma omp parallel for schedule(static, 524288)
  for (size_t i = 0; i < (size_t) num * (size_t) dim; i++) {
    int8_t low_prec_val = (int8_t)(data[i] * scale_factor);
    // handle overflow
    low_prec_val =
        low_prec_val < 0 && data[i] > 0 ? (int8_t) 127 : low_prec_val;
    // handle underflow
    low_prec_val =
        low_prec_val > 0 && data[i] < 0 ? (int8_t) -127 : low_prec_val;
    // convert back to float
    data[i] = ((float) low_prec_val) / scale_factor;
  }
  std::cout << "Converted fp32 -> int8 -> fp32" << std::endl;
  // write to disk
  for (unsigned i = 0; i < num; i++) {
    writer.write((char*) &dim, sizeof(unsigned));
    writer.write((char*) (data + (size_t) dim * (size_t) i),
                 (size_t)(dim * sizeof(float)));
  }
  writer.close();
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << argv[0] << " fp32_file int8_file" << std::endl;
    exit(-1);
  }

  float*   data = NULL;
  unsigned npts, ndims;
  efanna2e::load_Tvecs<float>(argv[1], data, npts, ndims);
  std::cout << "Data loaded\n";
  write_low_prec(argv[2], data, npts, ndims);
  std::cout << "Output file written\n";
  return 0;
}
