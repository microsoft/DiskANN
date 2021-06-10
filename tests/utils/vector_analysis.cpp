// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

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
#include "partition_and_pq.h"
#include "utils.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <typeinfo>

template<typename T>
int analyze_norm(std::string base_file) {
  std::cout<<"Analyzing data norms" << std::endl;
  T* data;
  _u64 npts, ndims;
  diskann::load_bin<T>(base_file, data, npts, ndims);
  std::vector<float> norms(npts, 0);
  #pragma omp parallel for schedule(dynamic)
  for (_u32 i = 0; i<npts; i++) {
    for (_u32 d = 0; d < ndims; d++) 
    norms[i] += data[i*ndims + d]* data[i* ndims + d];
  }
  std::sort(norms.begin(), norms.end());
  for (_u32 p = 0; p < 100; p+=5) 
  std::cout<<"percentile "<<p<<": " << norms[std::floor((p/100.0)*npts)] << std::endl;
  std::cout<<"percentile 100"<<": " << norms[npts-1] << std::endl;
  return 0;
}



template<typename T>
int aux_main(int argc, char** argv) {

  std::string base_file(argv[2]);
  _u32 option = atoi(argv[3]);
  if (option == 1) 
  analyze_norm<T>(base_file);
  return 0;
}

int main(int argc, char** argv) {

    if (argc != 4) {
    std::cout << argv[0] << " data_type [float/int8/uint8] base_bin_file "
                            "[option: 1-norm analysis]"
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
