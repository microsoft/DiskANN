#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
//#include <parallel/algorithm>
#include <string>
#include <vector>
#include <set>
#include "cached_io.h"
#include "utils.h"
#include <boost/dynamic_bitset.hpp>




int main(int argc, char **argv) {
  if (argc != 8) {
    std::cout
        << argv[0]
        << " vamana_index_prefix[1] vamana_index_suffix[2] idmaps_prefix[3] "
           "idmaps_suffix[4] n_shards[5] max_degree[6] output_vamana_path[7]"
        << std::endl;
    exit(-1);
  }

  std::string nsg_prefix(argv[1]);
  std::string nsg_suffix(argv[2]);
  std::string idmaps_prefix(argv[3]);
  std::string idmaps_suffix(argv[4]);
  _u64        nshards = (_u64) std::atoi(argv[5]);
  _u32        max_degree = (_u64) std::atoi(argv[6]);
  std::string output_nsg(argv[7]);

  return diskann::merge_shards(nsg_prefix, nsg_suffix, idmaps_prefix,
                                idmaps_suffix, nshards, output_nsg,
                                max_degree);
}
