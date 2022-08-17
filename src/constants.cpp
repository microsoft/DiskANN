#include "constants.h"
#include <windows_customizations.h>

namespace diskann {

const std::string Constants::starting_points_data_file_suffix =
        "_starting_points_data.bin";
const std::string Constants::starting_points_id_file_suffix =
        "_starting_points_ids.bin";
const std::string Constants::random = "random";
const std::string Constants::closest = "closest";
const std::string Constants::num_starting_points = "num_starting_points";
const std::string Constants::selection_strategy_of_starting_points = "selection_strategy_of_starting_points";
}  // namespace diskann