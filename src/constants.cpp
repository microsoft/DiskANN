#include "constants.h"
#include <windows_customizations.h>

namespace diskann {

  const std::string Constants::extra_start_points_data_file_suffix =
      "_extra_start_points_data.bin";
  const std::string Constants::extra_start_points_id_file_suffix =
      "_extra_start_points_ids.bin";
  const std::string Constants::random = "random";
  const std::string Constants::closest = "closest";
  const std::string Constants::num_extra_start_points = "num_extra_start_points";
  const std::string Constants::selection_strategy_of_extra_start_points =
      "selection_strategy_of_extra_starting_points";
}  // namespace diskann
