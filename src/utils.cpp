
#pragma once

#include <math_utils.h>
#include <utils.h>
#include <cstring>

void debug_code(size_t* closest_center, size_t* test_set, size_t num_points,
                size_t dim, float* cur_pivot_data, float* cur_data) {
  for (size_t i = 0; i < num_points; i++)
    if (closest_center[i] != test_set[i]) {
      std::cout << "debug_cod: differing in point " << i << " "
                << closest_center[i] << " " << test_set[i] << "." << std::flush;
      print_test_vec<float>(cur_pivot_data + closest_center[i] * dim, dim, 1);
      float d = math_utils::calc_distance(
          cur_data + i * dim, cur_pivot_data + closest_center[i] * dim, dim);
      std::cout << "distnces are " << d << std::flush;
      print_test_vec<float>(cur_pivot_data + test_set[i] * dim, dim, 1);
      d = math_utils::calc_distance(cur_data + i * dim,
                                    cur_pivot_data + test_set[i] * dim, dim);
      std::cout << " and " << d << std::endl;
      break;
    }
}
