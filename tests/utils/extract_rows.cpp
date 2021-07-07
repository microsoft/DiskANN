// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include "utils.h"
#include "extract_rows.h"

using namespace std;

template<typename T>
void extract_rows(const std::string& input_file, _u64 nrows, _u64 ndims,
                  _u64 offset, _u64 nrows_to_extract,
                  const std::string& output_file, bool has_id,
                  bool replace_content) {
  std::cout << "Trying to extract: " << nrows_to_extract << " rows of "
            << typeid(T).name() << " data in " << ndims
            << " dimensions from offset: " << offset
            << " and saving to file: " << output_file << "...";
  std::ifstream fin;
  fin.open(input_file, std::ios::binary);

  _u32          rows32 = (_u32) nrows_to_extract, dims32 = (_u32) ndims;
  std::ofstream fout(output_file, std::ios::binary);
  fout.write((const char*) &rows32, sizeof(uint32_t));
  fout.write((const char*) &dims32, sizeof(uint32_t));

  T* data = new T[nrows_to_extract * ndims];
  if (replace_content) {
    std::cout << std::endl << "Replacing file content with random values...";
    srand((unsigned int) time((time_t*) nullptr));
    for (int i = 0; i < nrows_to_extract * ndims; i++) {
      // since we won't allow replace_content if datatype != float, we can
      // assume that T is float always.
      data[i] = (T)(rand() / RAND_MAX);
    }
    std::cout << nrows_to_extract * ndims << " values replaced. " << std::endl;
  } else {
    fin.seekg(2 * sizeof(_u32) + offset * ndims * sizeof(T));
    fin.read((char*) data, nrows_to_extract * ndims * sizeof(T));
  }

  fout.write((const char*) data, nrows_to_extract * ndims * sizeof(T));

  if (has_id) {
    _u64* tags = new _u64[nrows_to_extract];
    _u64  tag_start =
        2 * sizeof(_u32) + nrows * ndims * sizeof(T) + offset * sizeof(_u64);
    fin.seekg(tag_start);
    fin.read((char*) tags, nrows_to_extract * sizeof(_u64));
    fout.write((const char*) tags, nrows_to_extract * sizeof(_u64));
    delete[] tags;
  }
  fout.close();
  fin.close();
  delete[] data;

  std::cout << " done." << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 6 && argc != 7 && argc != 8) {
    std::cout
        << "Usage: <program> <input_file> <datatype> (uint8|int8|float) "
           "<#_rows_to_extract> <start_from_row_#> "
           "(zero if beginning) <output_file> [has_tags(true/false) default is "
           "false, file doesn't have ids.] [replace_random (true/false) "
           "default is false, i.e.don't replace file content with random data]"
        << std::endl;
  }

  int         count = 1;
  std::string input_file = argv[count++];
  std::string datatype = argv[count++];
  uint64_t    nrows_to_extract = atoi(argv[count++]);
  uint64_t    offset = atoi(argv[count++]);
  std::string output_file = argv[count++];
  bool        has_id = false;
  bool        replace_content = false;

  if (argc >= 7) {
    if (std::string(argv[count++]) == "true") {
      has_id = true;
    }
  }
  if (argc >= 8) {
    if (std::string(argv[count++]) == "true") {
      if (!has_id || datatype != "float") {
        std::cout << "Can replace content ONLY if has_id is true and datatype "
                     "== float. "
                  << std::endl;
        return -6;
      } else {
        replace_content = true;
      }
    }
  }

  if (!file_exists(input_file)) {
    std::cerr << "Input file: " << input_file << " does not exist. Terminating."
              << std::endl;
    return -1;
  }
  uint64_t nrows, ncols;
  diskann::get_bin_metadata(input_file, nrows, ncols, 0);

  if (nrows_to_extract > nrows) {
    std::cerr << "Number of rows in file: " << nrows
              << " is less than num rows to extract: " << nrows_to_extract
              << std::endl;
    return -2;
  }
  if (offset > nrows) {
    std::cerr << "Start offset " << offset
              << " is greater than or equal to rows in file: " << nrows
              << std::endl;
    return -3;
  }
  if (offset + nrows_to_extract > nrows) {
    std::cerr << "Sum of start offset: " << offset
              << " and # rows to extract: " << nrows_to_extract
              << " is greater than or equal to rows in file: " << nrows
              << std::endl;
    return -4;
  }

  if (datatype == "uint8") {
    extract_rows<uint8_t>(input_file, nrows, ncols, offset, nrows_to_extract,
                          output_file, has_id, replace_content);
  } else if (datatype == "int8") {
    extract_rows<int8_t>(input_file, nrows, ncols, offset, nrows_to_extract,
                         output_file, has_id, replace_content);
  } else if (datatype == "float") {
    extract_rows<float>(input_file, nrows, ncols, offset, nrows_to_extract,
                        output_file, has_id, replace_content);

  } else {
    std::cerr << "Unknown datatype: " << datatype;
    return -5;
  }

  return 0;
}