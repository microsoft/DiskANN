// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"

namespace po = boost::program_options;

template<typename T, typename TagT = uint32_t>
int build_in_memory_index(const diskann::Metric& metric,
                          const std::string& data_path, const unsigned R,
                          const unsigned L, const float alpha,
                          const std::string& save_path,
                          const unsigned num_threads, const bool use_pq_build,
                          const size_t num_pq_bytes, const bool use_opq) {
  diskann::Parameters paras;
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>(
      "C", 750);  // maximum candidate set size during pruning procedure
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", 0);
  paras.Set<unsigned>("num_threads", num_threads);

  _u64 data_num, data_dim;
  diskann::get_bin_metadata(data_path, data_num, data_dim);

  diskann::Index<T, TagT> index(metric, data_dim, data_num, false, false, false,
                                use_pq_build, num_pq_bytes, use_opq);
  auto                    s = std::chrono::high_resolution_clock::now();
  index.build(data_path.c_str(), data_num, paras);

  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;

  std::cout << "Indexing time: " << diff.count() << "\n";
  index.save(save_path.c_str());

  return 0;
}

int main(int argc, char** argv) {
  std::string data_type, dist_fn, data_path, index_path_prefix;
  unsigned    num_threads, R, L, build_PQ_bytes;
  float       alpha;
  bool        use_pq_build, use_opq;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips>");
    desc.add_options()("data_path",
                       po::value<std::string>(&data_path)->required(),
                       "Input data file in bin format");
    desc.add_options()("index_path_prefix",
                       po::value<std::string>(&index_path_prefix)->required(),
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
    desc.add_options()(
        "build_PQ_bytes", po::value<uint32_t>(&build_PQ_bytes)->default_value(0),
        "Number of PQ bytes to build the index; 0 for full precision build");
    desc.add_options()(
        "use_opq", po::bool_switch()->default_value(false),
        "Set true for OPQ compression while using PQ distance comparisons for "
        "building the index, and false for PQ compression");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
    use_pq_build = (build_PQ_bytes > 0);
    use_opq = vm["use_opq"].as<bool>();
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  diskann::Metric metric;
  if (dist_fn == std::string("mips")) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (dist_fn == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (dist_fn == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else {
    std::cout << "Unsupported distance function. Currently only L2/ Inner "
                 "Product/Cosine are supported."
              << std::endl;
    return -1;
  }

  try {
    diskann::cout << "Starting index build with R: " << R << "  Lbuild: " << L
                  << "  alpha: " << alpha << "  #threads: " << num_threads
                  << std::endl;
    if (data_type == std::string("int8"))
      return build_in_memory_index<int8_t>(metric, data_path, R, L, alpha,
                                           index_path_prefix, num_threads,
                                           use_pq_build, build_PQ_bytes, use_opq);
    else if (data_type == std::string("uint8"))
      return build_in_memory_index<uint8_t>(
          metric, data_path, R, L, alpha, index_path_prefix, num_threads,
          use_pq_build, build_PQ_bytes, use_opq);
    else if (data_type == std::string("float"))
      return build_in_memory_index<float>(metric, data_path, R, L, alpha,
                                          index_path_prefix, num_threads,
                                          use_pq_build, build_PQ_bytes, use_opq);
    else {
      std::cout << "Unsupported type. Use one of int8, uint8 or float."
                << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Index build failed." << std::endl;
    return -1;
  }
}
