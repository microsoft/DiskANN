// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once
#include "common_includes.h"
#include "windows_customizations.h"
#include "filter_utils.h"

namespace diskann {

  template<typename T>
  class FilterBruteForceIndex {
  public :
    DISKANN_DLLEXPORT FilterBruteForceIndex(const std::string& disk_index_file);
    DISKANN_DLLEXPORT bool brute_force_index_available() const; 
    DISKANN_DLLEXPORT bool brute_forceable_filter(const std::string& filter) const;
    DISKANN_DLLEXPORT int load();

  private :
    diskann::inverted_index_t _bf_filter_index;
    bool _is_loaded;
    std::string _disk_index_file;
  };
}