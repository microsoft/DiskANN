//#include <distances.h>
//#include <indexing.h>
//
#include "omp.h"

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

template<typename T>
bool build_index(const char* dataFilePath, const char* indexFilePath,
                 const char* indexBuildParameters) {
  return diskann::build_disk_index<T>(
      dataFilePath, indexFilePath, indexBuildParameters, diskann::Metric::L2);
}

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cout << "Usage: " << argv[0]
              << "  <data_type [float/uint8/int8]>   <data_file [.bin]>  "
                 "<index_prefix_path>  <L candidate list size during indexing> "
                 "<R [degree of final "
                 "index]> <B [memory footprint of disk index]> <M [indexing "
                 "time memory "
                 "limit in GB]> <T [number of threads during indexing>"
              << std::endl;
  } else {
    std::string params = std::string(argv[4]) + " " + std::string(argv[5]) +
                         " " + std::string(argv[6]) + " " +
                         std::string(argv[7]) + " " + std::string(argv[8]);
    if (std::string(argv[1]) == std::string("float"))
      build_index<float>(argv[2], argv[3], params.c_str());
    else if (std::string(argv[1]) == std::string("int8"))
      build_index<int8_t>(argv[2], argv[3], params.c_str());
    else if (std::string(argv[1]) == std::string("uint8"))
      build_index<uint8_t>(argv[2], argv[3], params.c_str());
    else
      std::cout << "Error. wrong file type" << std::endl;
  }
}
