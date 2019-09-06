#include <index_nsg.h>
#include <timer.h>

#include <omp.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "util.h"

int main(int argc, char** argv) {
  if (argc != 10) {
    std::cout << "Correct usage: " << argv[0]
              << " data_file L R C alpha num_rounds "
              << "save_graph_file #incr_points #fake_points" << std::endl;
    exit(-1);
  }

  float* data_load = NULL;
  size_t num_points, dim, num_new;

  NSG::load_Tvecs<float>(argv[1], data_load, num_points, dim);
  data_load = NSG::data_align(data_load, num_points, dim);
  std::cout << "Data loaded and aligned" << std::endl;

  unsigned    L = (unsigned) atoi(argv[2]);
  unsigned    R = (unsigned) atoi(argv[3]);
  unsigned    C = (unsigned) atoi(argv[4]);
  float       alpha = (float) std::atof(argv[5]);
  unsigned    num_rnds = (unsigned) std::atoi(argv[6]);
  std::string save_path(argv[7]);
  unsigned    num_incr = (unsigned) atoi(argv[8]);
  unsigned    num_fake = (unsigned) atoi(argv[9]);

  NSG::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<unsigned>("num_rnds", num_rnds);

  num_new = num_points + num_fake;

  auto data_copy = new float[num_new * dim];
  memcpy((void*) (data_copy + num_fake * dim), (void*) data_load,
         num_points * dim * sizeof(float));
  auto data_copy_copy = new float[num_new * dim];

  typedef unsigned TagT;

  NSG::IndexNSG<float, TagT> index(dim, num_new - num_incr, NSG::L2, num_new,
                                   true);
  {
    std::vector<TagT> tags(num_new - num_incr);
    std::iota(tags.begin(), tags.end(), 0);

    index.gen_fake_point(num_fake, data_copy);
    NSG::Timer timer;
    index.build(data_copy, paras, tags);
    memcpy((void*) data_copy_copy, (void*) data_copy,
           num_new * dim * sizeof(float));
    std::cout << "Index time: " << timer.elapsed() / 1000 << "ms\n";
  }

  std::vector<NSG::Neighbor>       pool, tmp;
  tsl::robin_set<unsigned>         visited;
  std::vector<NSG::SimpleNeighbor> cut_graph;

  {
    NSG::Timer timer;
    for (size_t i = num_new - num_incr; i < num_new; ++i)
      index.insert_point(data_copy_copy + i * dim, paras, pool, tmp, visited,
                         cut_graph, i);
    std::cout << "Incremental time: " << timer.elapsed() / 1000 << "ms\n";
  }
  index.save(save_path.c_str());

  tsl::robin_set<unsigned> delete_list;
  while (delete_list.size() < num_incr)
    delete_list.insert(((rand() * rand() * rand()) % num_points) + num_fake);
  std::cout << "Deleting " << delete_list.size() << " elements" << std::endl;

  {
    NSG::Timer timer;
    index.enable_delete();
    unsigned              count = 0;
    std::vector<unsigned> new_location;
    new_location.resize(num_new, num_new);

    for (auto p : delete_list)
      // if (index.delete_point(p) != 0)
      if (index.eager_delete(p, paras, new_location) != 0)
        std::cerr << "Delete tag " << p << " not found" << std::endl;
      else {
        count++;
        if (count % 1000 == 0)
          std::cout << count << std::endl;
      }

    if (index.disable_delete(paras, true) != 0) {
      std::cerr << "Disable delete failed" << std::endl;
      return -1;
    }
    std::cout << "Delete time: " << timer.elapsed() / 1000 << "ms\n";
  }

  {
    NSG::Timer timer;
    for (auto p : delete_list)
      index.insert_point(data_copy_copy + (size_t) p * (size_t) dim, paras,
                         pool, tmp, visited, cut_graph, p);
    std::cout << "Re-incremental time: " << timer.elapsed() / 1000 << "ms\n";
  }

  auto save_path_reinc = save_path + ".reinc";
  index.save(save_path_reinc.c_str());

  delete[] data_copy_copy;
  delete[] data_copy;
  delete[] data_load;

  return 0;
}
