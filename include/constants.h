// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <string>

namespace diskann {

  class Constants {
   public:
    const static std::string extra_start_points_data_file_suffix;
    const static std::string extra_start_points_id_file_suffix;
    const static std::string random;
    const static std::string closest;
    const static std::string num_extra_start_points;
    const static std::string selection_strategy_of_extra_start_points;
  };

}  // namespace diskann
