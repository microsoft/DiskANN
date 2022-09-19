// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <boost/program_options.hpp>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string data_type, dist_fn, data_path, index_path_prefix;
  unsigned    num_threads, R, L, disk_PQ;
  float       B, M;
  bool        append_reorder_data = false;
  bool        use_opq = false;

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
    desc.add_options()("search_DRAM_budget,B", po::value<float>(&B)->required(),
                       "DRAM budget in GB for searching the index to set the "
                       "compressed level for data while search happens");
    desc.add_options()("build_DRAM_budget,M", po::value<float>(&M)->required(),
                       "DRAM budget in GB for building the index");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()("PQ_disk_bytes",
                       po::value<uint32_t>(&disk_PQ)->default_value(0),
                       "Number of bytes to which vectors should be compressed "
                       "on SSD; 0 for no compression");
    desc.add_options()("append_reorder_data",
                       po::bool_switch()->default_value(false),
                       "Include full precision data in the index. Use only in "
                       "conjuction with compressed data on SSD.");

    desc.add_options()("use_opq", po::bool_switch()->default_value(false),
                       "Use Optimized Product Quantization (OPQ).");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
    if (vm["append_reorder_data"].as<bool>())
      append_reorder_data = true;
    if (vm["use_opq"].as<bool>())
      use_opq = true;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  diskann::Metric metric;
  if (dist_fn == std::string("l2"))
    metric = diskann::Metric::L2;
  else if (dist_fn == std::string("mips"))
    metric = diskann::Metric::INNER_PRODUCT;
  else {
    std::cout << "Error. Only l2 and mips distance functions are supported"
              << std::endl;
    return -1;
  }

  if (append_reorder_data) {
    if (disk_PQ == 0) {
      std::cout << "Error: It is not necessary to append data for reordering "
                   "when vectors are not compressed on disk."
                << std::endl;
      return -1;
    }
    if (data_type != std::string("float")) {
      std::cout << "Error: Appending data for reordering currently only "
                   "supported for float data type."
                << std::endl;
      return -1;
    }
  }

  std::string params = std::string(std::to_string(R)) + " " +
                       std::string(std::to_string(L)) + " " +
                       std::string(std::to_string(B)) + " " +
                       std::string(std::to_string(M)) + " " +
                       std::string(std::to_string(num_threads)) + " " +
                       std::string(std::to_string(disk_PQ)) + " " +
                       std::string(std::to_string(append_reorder_data));

  try {
    if (data_type == std::string("int8"))
      return diskann::build_disk_index<int8_t>(data_path.c_str(),
                                               index_path_prefix.c_str(),
                                               params.c_str(), metric, use_opq);
    else if (data_type == std::string("uint8"))
      return diskann::build_disk_index<uint8_t>(
          data_path.c_str(), index_path_prefix.c_str(), params.c_str(), metric,
          use_opq);
    else if (data_type == std::string("float"))
      return diskann::build_disk_index<float>(data_path.c_str(),
                                              index_path_prefix.c_str(),
                                              params.c_str(), metric, use_opq);
    else {
      diskann::cerr << "Error. Unsupported data type" << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Index build failed." << std::endl;
    return -1;
  }
}
