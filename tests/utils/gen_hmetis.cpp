// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>

#include "utils.h"

namespace po = boost::program_options;


int aux_main(const std::string &gt_file,
             const std::string &output_file,
             const unsigned K, const unsigned num_pts_in_dataset) {
    uint32_t* gt = nullptr;
    float* dists = nullptr;
    size_t num_queries, gt_dim;
    diskann::load_truthset(gt_file, gt, dists, num_queries, gt_dim);

    if (gt_dim < K) {
	  std::cout << "Ground truth dimension " << gt_dim
				<< " smaller than K " << K << std::endl;
	  return -1;
	}

    std::ofstream hmetis(output_file);
    hmetis << num_queries << " " << num_pts_in_dataset << std::endl;
    for (size_t i = 0; i < num_queries; i++) {
      for (size_t j = 0; j < K; j++) {
        if (gt[i * gt_dim + j] >= num_pts_in_dataset) {
          diskann::cout << "Ground truth contains point with ID larger (or equal) than "
                           "num_pts_in_dataset"
                        << std::endl;
          return -1;
        }
        // hMetis vertex IDs are 1-based!
        hmetis << gt[i * gt_dim + j] + 1 << " ";
	  }
	  hmetis << std::endl;
	}

    delete[] gt;
    if (dists != nullptr) {
	  delete[] dists;
	}
        
    return 0;
}



int main(int argc, char** argv) {
    std::string gt_file, output_file;
    unsigned    K, num_pts_in_dataset;
    po::options_description desc{"Arguments"};
    try {
      desc.add_options()("help,h", "Print information on arguments");
      desc.add_options()("gt_file",
                         po::value<std::string>(&gt_file)->required(),
                         "Path to the ground truth .bin file");
      desc.add_options()("output_file",
                         po::value<std::string>(&output_file)->required(),
                         "Path to the output hMetis file");
      desc.add_options()("K,recall_at", po::value<unsigned>(&K)->required(),
                         "Points returned per query");
      desc.add_options()("num_pts_in_dataset", po::value<unsigned>(&num_pts_in_dataset)->required(),
                         "Number of points in dataset");

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
      aux_main(gt_file, output_file, K, num_pts_in_dataset);
    } catch (const std::exception& e) {
      std::cout << std::string(e.what()) << std::endl;
      std::cerr << "Partitioning failed." << std::endl;
      return -1;
    }
}
