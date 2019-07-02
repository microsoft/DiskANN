//#include <distances.h>
//#include <indexing.h>
#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <math_utils.h>
#include <partitionAndPQ.h>
#include <utils.h>

int main(int argc, char** argv) {
  auto s = std::chrono::high_resolution_clock::now();

  if (argc != 7) {
    std::cout << argv[0] << " format: base_set train_set "
                            "num_clusters "
                            "max_reps prefix_for_working_directory k_base "
              << std::endl;
    exit(-1);
  }
  size_t num_clusters = std::atoi(argv[3]);
  size_t max_reps = std::atoi(argv[4]);
  size_t k_base = std::atoi(argv[6]);
  partition(argv[1], argv[2], num_clusters, max_reps, argv[5], k_base);
}
