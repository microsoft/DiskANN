// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "v2/index_merger.h"

#include <numeric>
#include <omp.h>
#include <cstring>
#include <ctime>
#include <timer.h>
#include <iomanip>

#include "aux_utils.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

template<typename T>
void run(const char* disk_in, const std::vector<std::string>& mem_in,
         const char* disk_out, const char* deleted_tags, const uint32_t ndims,
         diskann::Distance<T>* dist, const uint32_t beam_width,
         const uint32_t range, const uint32_t l_index, const float alpha,
         const uint32_t maxc) {
  diskann::cout << "Instantiating StreamingMerger\n";
  diskann::StreamingMerger<T> merger(disk_in, mem_in, disk_out, deleted_tags,
                                     ndims, dist, beam_width, range, l_index,
                                     alpha, maxc);

  diskann::cout << "Starting merge\n";
  merger.merge();

  diskann::cout << "Finished merging\n";
}

int main(int argc, char** argv) {
  if (argc < 13) {
    diskann::cout
        << "Correct usage: " << argv[0]
        << " <type[int8/uint8/float]> <disk_in> <disk_out> <deleted_tags_file>"
        << " <ndims> <beam_width> <range> <L_index> <alpha> <maxc> "
           "<tmp_working_folder> <mem_index_1> <mem_index_2> .."
        << std::endl;
    exit(-1);
  }
  diskann::cout.setf(std::ios::unitbuf);

  int         arg_no = 1;
  std::string index_type = argv[arg_no++];
  // assert(index_type == std::string("float"));
  const char* disk_in = argv[arg_no++];
  diskann::cout << "Input SSD-DiskANN: " << disk_in << "\n";
  const char* disk_out = argv[arg_no++];
  diskann::cout << "Output SSD-DiskANN: " << disk_out << "\n";
  const char* delete_list_file = argv[arg_no++];
  diskann::cout << "Deleted tags file : " << delete_list_file << "\n";
  uint32_t data_dim = (uint32_t) atoi(argv[arg_no++]);
  diskann::cout << "Data dim : " << data_dim << "\n";
  uint32_t beam_width = (uint32_t) atoi(argv[arg_no++]);
  diskann::cout << "Beam Width : " << beam_width << "\n";
  uint32_t range = (uint32_t) atoi(argv[arg_no++]);
  diskann::cout << "Range (max out-degree) : " << range << "\n";
  uint32_t L_index = (uint32_t) atoi(argv[arg_no++]);
  diskann::cout << "L-index : " << L_index << "\n";
  float alpha = (float) atof(argv[arg_no++]);
  diskann::cout << "alpha : " << alpha << "\n";
  uint32_t maxc = (uint32_t) atoi(argv[arg_no++]);
  diskann::cout << "max-C : " << maxc << "\n";
  TMP_FOLDER = argv[arg_no++];
  diskann::cout << "Working folder : " << TMP_FOLDER << "\n";
  std::vector<std::string> mem_in;
  while (arg_no < argc) {
    const char* cur_mem_in = argv[arg_no++];
    diskann::cout << "Mem-DiskANN index #" << mem_in.size() + 1 << " : "
                  << cur_mem_in << "\n";
    mem_in.emplace_back(cur_mem_in);
  }

  if (index_type == std::string("float")) {
    diskann::DistanceL2 dist_cmp;
    run<float>(disk_in, mem_in, disk_out, delete_list_file, data_dim, &dist_cmp,
               beam_width, range, L_index, alpha, maxc);
  } else if (index_type == std::string("uint8")) {
    diskann::DistanceL2UInt8 dist_cmp;
    run<uint8_t>(disk_in, mem_in, disk_out, delete_list_file, data_dim,
                 &dist_cmp, beam_width, range, L_index, alpha, maxc);
  } else if (index_type == std::string("int8")) {
    diskann::DistanceL2Int8 dist_cmp;
    run<int8_t>(disk_in, mem_in, disk_out, delete_list_file, data_dim,
                &dist_cmp, beam_width, range, L_index, alpha, maxc);
  } else {
    diskann::cout << "Unsupported type : " << index_type << "\n";
  }
  diskann::cout << "Exiting\n";
}
