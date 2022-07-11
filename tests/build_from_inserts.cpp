// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <index.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <boost/program_options.hpp>
#include <future>

#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "aux_utils.h"
#include "index.h"
#include "utils.h"
#include "memory_mapper.h"

namespace po = boost::program_options;


// build index via insertion, then delete and reinsert every point
// in batches of 10% graph size
template<typename T>
void build_from_inserts(const std::string& data_path, const unsigned L,
                        const unsigned R, const float alpha,
                        const unsigned            thread_count,
                        const std::string& save_path) {
  const unsigned C = 500;
  const bool     saturate_graph = false;

  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", saturate_graph);
  paras.Set<unsigned>("num_rnds", 1);
  paras.Set<unsigned>("num_threads", thread_count);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path, data_load, num_points, dim,
                               aligned_dim);

  using TagT = uint32_t;
  unsigned   num_frozen = 1;
  const bool enable_tags = true;
  const bool support_eager_delete = false;

  auto num_frozen_str = getenv("TTS_NUM_FROZEN");

  if (num_frozen_str != nullptr) {
    num_frozen = std::atoi(num_frozen_str);
    std::cout << "Overriding num_frozen to" << num_frozen << std::endl;
  }

  diskann::Index<T, TagT> index(diskann::L2, dim, num_points, true, paras,
                                paras, enable_tags, support_eager_delete);

  std::vector<TagT> one_tag;
  one_tag.push_back(0);

  index.build(&data_load[0], 1, paras, one_tag);

  std::cout << "Inserting every point into the index" << std::endl;

  diskann::Timer index_timer;

  std::vector<int> num_pruned(num_points - 1, 0);

#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
  for (int64_t j = 1; j < (int64_t) num_points; j++) {
    num_pruned[j-1] = index.insert_point(&data_load[j * aligned_dim], static_cast<TagT>(j)).first;
  }

  double seconds = index_timer.elapsed() / 1000000.0;

  std::cout << "Inserted points in " << seconds << " seconds" << std::endl;
  std::cout << "Total number of prunes: " << std::accumulate(num_pruned.begin(), num_pruned.end(), 0) << std::endl; 

  index.save(save_path.c_str());
  
}

int main(int argc, char** argv) {
  std::string data_type, data_path, save_path;
  unsigned    num_threads, R, L;
  float       alpha;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("data_path",
                       po::value<std::string>(&data_path)->required(),
                       "Input data file in bin format");
    desc.add_options()("save_path",
                       po::value<std::string>(&save_path)->required(),
                       "Path prefix for saving index file components");
    desc.add_options()("max_degree,R",
                       po::value<uint32_t>(&R)->default_value(64),
                       "Maximum graph degree");
    desc.add_options()(
        "Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
        "Build complexity, higher value results in better graphs");
    desc.add_options()(
        "alpha", po::value<float>(&alpha)->default_value(1.2f),
        "alpha controls density and diameter of graph, set 1 for sparse graph, "
        "1.2 or 1.4 for denser graphs with lower diameter");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  try {
    if (data_type == std::string("int8"))
      build_from_inserts<int8_t>(data_path, L, R, alpha, num_threads,
                                 save_path);
    else if (data_type == std::string("uint8"))
      build_from_inserts<uint8_t>(data_path, L, R, alpha, num_threads,
                                  save_path);
    else if (data_type == std::string("float"))
      build_from_inserts<float>(data_path, L, R, alpha, num_threads, 
                                save_path);
    else
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "Caught unknown exception" << std::endl;
    exit(-1);
  }

  return 0;
}