//#include <distances.h>
//#include <indexing.h>

#include <index.h>
#include <math_utils.h>

#include "partition_and_pq.h"
#include "util.h"

// DEPRECATED: NEED TO REPROGRAM

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cout << argv[0] << "format: base_set "
                            "num_clusters_per_chunk number_chunks "
                            "prefix_for_working_file "
              << std::endl;
    exit(-1);
  }

  size_t num_centers = (size_t) strtol(argv[2], NULL, 10);
  size_t num_chunks = (size_t) strtol(argv[3], NULL, 10);
  generate_pq_data_from_pivots(argv[1], num_centers, num_chunks, argv[4]);
}
