// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <random>
#include <boost/program_options.hpp>

#include "utils.h"

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string output_file;
  _u64        num_labels, num_points;

  try {
    po::options_description desc{"Arguments"};

    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("output_file,O",
                       po::value<std::string>(&output_file)->required(),
                       "Filename for saving the label file");
    desc.add_options()("num_points,N",
                       po::value<uint64_t>(&num_points)->required(),
                       "Number of points in dataset");
    desc.add_options()("num_labels,L",
                       po::value<uint64_t>(&num_labels)->required(),
                       "Number of unique labels, up to 100");

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

  if (num_labels > 100) {
    std::cerr << "Error: num_labels must be 100 or less" << '\n';
    return -1;
  }

  if (num_points <= 0) {
    std::cerr << "Error: num_points must be greater than 0" << '\n';
    return -1;
  }

  std::cout << "Generating synthetic labels for " << num_points
            << " points with " << num_labels << " unique labels" << '\n';

  try {
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
      std::cerr << "Error: could not open output file " << output_file << '\n';
      return -1;
    }

    for (int i = 0; i < num_points; i++) {
      bool label_written = false;
      for (int j = 1; j <= num_labels; j++) {
        // 50% chance to assign each label
        if (rand() > (RAND_MAX / 2)) {
          if (label_written) {
            outfile << ',';
          }
          outfile << j;
          label_written = true;
        }
      }

      if (i < num_points - 1) {
        outfile << '\n';
      }
    }
    outfile.close();

    std::cout << "Labels written to " << output_file << '\n';

  } catch (const std::exception& ex) {
    std::cerr << "Label generation failed: " << ex.what() << '\n';
    return -1;
  }

  return 0;
}