// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "filter_brute_force_index.h"

namespace diskann {

  template<typename T>
  FilterBruteForceIndex<T>::FilterBruteForceIndex(const std::string& disk_index_file) {
    _disk_index_file = disk_index_file;
    _filter_bf_data_file = _disk_index_file + "_brute_force.txt";
  }
  template<typename T>
  bool FilterBruteForceIndex<T>::brute_force_index_available() const {}

  template<typename T>
  bool FilterBruteForceIndex<T>::brute_forceable_filter(const std::string& filter) const {}

  template<typename T>
  int FilterBruteForceIndex<T>::load() {
    if (false == file_exists(_filter_bf_data_file)) {
      diskann::cerr << "Index does not have brute force support." << std::endl;
      return 1;
    }
    std::ifstream bf_in(_filter_bf_data_file);
    if (!bf_in.is_open()) {
      std::stringstream ss;
      ss << "Could not open " << _filter_bf_data_file << " for reading. " << std::endl;
      diskann::cerr << ss.str() << std::endl;
      throw diskann::ANNException(ss.str(), -1);
    }

    std::string line;
    std::vector<std::string> label_and_points;
    label_and_points.reserve(2);
    std::unordered_set<location_t> points;
 
    size_t linenum = 0;
    while (getline(bf_in, line)) {
      split_string(line, '\t', label_and_points);
      if (label_and_points.size() == 2) {
        
        std::istringstream iss(label_and_points[1]);
        std::string pt_str; 
        while (getline(iss, pt_str, ',')) {
          points.insert(strtoul(pt_str));
        }
        assert(points.size() > 0);
        _bf_filter_index.insert(label_and_points[0], points);
        points.clear();
      } else {
        std::stringstream ss;
        ss << "Error reading brute force data at line: " << line_num 
          << " found " << label_and_points.size() << " tab separated entries instead of 2" 
          << std::endl;
        diskann::cerr << ss.str();
        throw diskann::ANNException(ss.str(), -1);
      }
      line_num++;
    }
  }
}