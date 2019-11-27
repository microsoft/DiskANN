// convert.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//

#include <iostream>
#include "util.h"

template<typename source_type, typename dest_type>
int aux_main(int argc, char** argv) {
  source_type* sourceVec = nullptr;
  size_t       num_points, num_dims;
  std::cout << "Loading source vector from file " << argv[2] << std::endl;
  diskann::load_bin<source_type>(argv[2], sourceVec, num_points, num_dims);

  dest_type* destVec = new dest_type[num_points * num_dims];
  diskann::convert_types<source_type, dest_type>(sourceVec, destVec, num_points,
                                             num_dims);
  std::cout << "Converted vector from type " << argv[1] << "to type " << argv[3]
            << std::endl;
  diskann::save_bin<dest_type>(argv[4], destVec, num_points, num_dims);
  std::cout << "Saved converted vector to file " << argv[4] << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cout
        << "Usage: convert <source_type> <source_file> <dest_type> <dest_file> "
           "where type is one of [float/uint8/int8], case-sensitive. "
        << std::endl;
  }

  std::string sourceType(argv[1]);
  std::string destType(argv[3]);

  if (sourceType == destType) {
    std::cout << "Nothing to do. source and destination types are the same."
              << std::endl;
    return 0;
  }

  if (sourceType == "int8" && destType == "uint8") {
    return aux_main<int8_t, uint8_t>(argc, argv);
  } else if (sourceType == "int8" && destType == "float") {
    return aux_main<int8_t, float>(argc, argv);
  } else if (sourceType == "uint8" && destType == "int8") {
    return aux_main<uint8_t, int8_t>(argc, argv);
  } else if (sourceType == "uint8" && destType == "float") {
    return aux_main<uint8_t, float>(argc, argv);
  } else if (sourceType == "float" && destType == "int8") {
    return aux_main<float, int8_t>(argc, argv);
  } else if (sourceType == "float" && destType == "uint8") {
    return aux_main<float, uint8_t>(argc, argv);
  } else {
    std::cout << "One of the source or destination types is unknown: "
              << argv[0] << "," << argv[2] << std::endl;
    return 0;
  }
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add
//   Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project
//   and select the .sln file
