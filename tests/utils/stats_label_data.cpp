// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <omp.h>
#include <string.h>
#include <atomic>
#include <cstring>
#include <iomanip>
#include <set>

#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#else
#include <Windows.h>
#endif

void stats_analysis(const std::string labels_file, std::string univeral_label,
                    _u32 density = 10) {
  std::string   token, line;
  std::ifstream labels_stream(labels_file);
  std::unordered_map<std::string, _u32> label_counts;
  std::string                           label_with_max_points;
  _u32                                  max_points = 0;
  long long                             sum = 0;
  long long                             point_cnt = 0;
  float avg_labels_per_pt, avg_labels_per_pt_incl_0, mean_label_size,
      mean_label_size_incl_0;

  std::vector<_u32> labels_per_point;
  _u32              dense_pts = 0;
  if (labels_stream.is_open()) {
    while (getline(labels_stream, line)) {
      point_cnt++;
      std::stringstream iss(line);
      _u32              lbl_cnt = 0;
      while (getline(iss, token, ',')) {
        lbl_cnt++;
        token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
        token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
        if (label_counts.find(token) == label_counts.end())
          label_counts[token] = 0;
        label_counts[token]++;
      }
      if (lbl_cnt >= density) {
        dense_pts++;
      }
      labels_per_point.emplace_back(lbl_cnt);
    }
  }

  std::cout << "fraction of dense points with >= " << density << " labels = "
            << (float) dense_pts / (float) labels_per_point.size() << std::endl;
  std::sort(labels_per_point.begin(), labels_per_point.end());

  std::vector<std::pair<std::string, _u32>> label_count_vec;

  for (auto it = label_counts.begin(); it != label_counts.end(); it++) {
    auto& lbl = *it;
    label_count_vec.emplace_back(std::make_pair(lbl.first, lbl.second));
    if (lbl.second > max_points) {
      max_points = lbl.second;
      label_with_max_points = lbl.first;
    }
    sum += lbl.second;
  }

  sort(label_count_vec.begin(), label_count_vec.end(),
       [](const std::pair<std::string, _u32>& lhs,
          const std::pair<std::string, _u32>& rhs) {
         return lhs.second < rhs.second;
       });

  for (float p = 0; p < 1; p += 0.05) {
    std::cout << "Percentile " << (100 * p) << "\t"
              << label_count_vec[(_u32)(p * label_count_vec.size())].first
              << " with count="
              << label_count_vec[(_u32)(p * label_count_vec.size())].second
              << std::endl;
  }

  std::cout << "Most common label "
            << "\t" << label_count_vec[label_count_vec.size() - 1].first
            << " with count="
            << label_count_vec[label_count_vec.size() - 1].second << std::endl;
  if (label_count_vec.size() > 1)
    std::cout << "Second common label "
              << "\t" << label_count_vec[label_count_vec.size() - 2].first
              << " with count="
              << label_count_vec[label_count_vec.size() - 2].second
              << std::endl;
  if (label_count_vec.size() > 2)
    std::cout << "Third common label "
              << "\t" << label_count_vec[label_count_vec.size() - 3].first
              << " with count="
              << label_count_vec[label_count_vec.size() - 3].second
              << std::endl;
  avg_labels_per_pt = (sum) / (float) point_cnt;
  mean_label_size = (sum) / label_counts.size();
  std::cout << "Total number of points = " << point_cnt
            << ", number of labels = " << label_counts.size() << std::endl;
  std::cout << "Average number of labels per point = " << avg_labels_per_pt
            << std::endl;
  std::cout << "Mean label size excluding 0 = " << mean_label_size << std::endl;
  std::cout << "Most popular label is " << label_with_max_points << " with "
            << max_points << " pts" << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cout << "Usage:\n"
              << argv[0]
              << " [labels_file] [universal_label (use \"null\") if none] "
                 "[optional: density threshold]\n";
    exit(-1);
  }

  const std::string labels_file(argv[1]);
  const std::string universal_label(argv[2]);
  _u32              density = 1;
  if (argc == 4) {
    density = std::atoi(argv[3]);
  }
  stats_analysis(labels_file, universal_label, density);
}
