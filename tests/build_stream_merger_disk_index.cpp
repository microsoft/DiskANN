// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "omp.h"

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

template<typename T>
bool build_index(const char* dataFilePath, const char* indexFilePath,
                 const char* indexBuildParameters, const char* tag_file,
                 int single_file_index) {
  std::string tag_filename = std::string(tag_file);

  bool save_index_as_one_file;
  if (single_file_index == 0)
    save_index_as_one_file = false;
  else
    save_index_as_one_file = true;

  if (tag_filename == "null")
    return diskann::build_disk_index<T>(
        dataFilePath, indexFilePath, indexBuildParameters, diskann::Metric::L2,
        save_index_as_one_file, nullptr);
  return diskann::build_disk_index<T>(dataFilePath, indexFilePath,
                                      indexBuildParameters, diskann::Metric::L2,
                                      save_index_as_one_file, tag_file);
}

int main(int argc, char** argv) {
  if (argc != 11) {
    diskann::cout << "Usage: " << argv[0]
                  << "  [data_type<float/int8/uint8>]  [data_file.bin]  "
                     "[index_prefix_path]  "
                     "[R]  [L]  [B]  [M]  [T] [<tags.bin (use \"null\" for "
                     "none)>] <single_file_index(0/1)>. See README for more "
                     "information on "
                     "parameters."
                  << std::endl;
  } else {
    std::string params = std::string(argv[4]) + " " + std::string(argv[5]) +
                         " " + std::string(argv[6]) + " " +
                         std::string(argv[7]) + " " + std::string(argv[8]);
    int single_file_index = atoi(argv[10]);
    if (std::string(argv[1]) == std::string("float"))
      build_index<float>(argv[2], argv[3], params.c_str(), argv[9],
                         single_file_index);
    else if (std::string(argv[1]) == std::string("int8"))
      build_index<int8_t>(argv[2], argv[3], params.c_str(), argv[9],
                          single_file_index);
    else if (std::string(argv[1]) == std::string("uint8"))
      build_index<uint8_t>(argv[2], argv[3], params.c_str(), argv[9],
                           single_file_index);
    else
      diskann::cout << "Error. wrong file type" << std::endl;
  }
}
