// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << argv[0] << " input_bin   output_bin    npoints" << std::endl;
    exit(-1);
  }
  std::ifstream readr(argv[1], std::ios::binary);
  int           npts_s32;
  int           ndims_s32;
  readr.read((char*) &npts_s32, sizeof(_s32));
  readr.read((char*) &ndims_s32, sizeof(_s32));
  size_t npts = npts_s32;
  size_t ndims = ndims_s32;
  readr.seekg(0, std::ios::beg);

  std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
            << std::endl;

  _s32 rpts = atoi(argv[3]);
  std::cout << "Adding " << rpts << " random points to new file" << std::endl; 

  std::vector<_s32> random_ids(npts);
  std::iota(random_ids.begin(), random_ids.end(), 0);
  std::random_shuffle(random_ids.begin(), random_ids.end());
  random_ids.resize(rpts);
  std::cout << random_ids.size() << std::endl;

  std::ofstream writer(argv[2], std::ios::binary);

  writer.write((char*) &rpts, sizeof(_s32));
  writer.write((char*) &ndims_s32, sizeof(_s32));
  float* read_buf = new float[npts * ndims];
  float* write_buf = new float[rpts * ndims];
  readr.read((char*) read_buf, npts*ndims*(sizeof(float)));
  for(_s32 i=0; i<(_s32) random_ids.size(); i++){
    auto id = random_ids[i];
  	memcpy(write_buf+i*ndims, read_buf+id*(ndims), ndims*(sizeof(float)));
  }
  for(size_t i=0; i<ndims; i++){
    std::cout << *(write_buf+i) << std::endl; 
  }
  writer.write((char*) write_buf, rpts*ndims*sizeof(float));

  delete[] read_buf;
  delete[] write_buf;

  readr.close();
  writer.close();
}