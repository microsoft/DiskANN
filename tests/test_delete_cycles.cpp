// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <index.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <boost/program_options.hpp>
#include <future>

#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

namespace po = boost::program_options;

std::string get_save_filename(const std::string& save_path,
                              size_t             points_to_skip,
                              size_t             last_point_threshold) {
  std::string final_path = save_path;
  final_path += std::to_string(points_to_skip) + "-";
  final_path += std::to_string(last_point_threshold);
  return final_path;
}

// build index via insertion, then delete and reinsert every point
// in batches of 10% graph size
template<typename T>
void test_batch_deletes(const std::string& data_path, const unsigned L,
                        const unsigned R, const float alpha,
                        const unsigned            thread_count,
                        const std::vector<size_t> Dvec,
                        const std::string& save_path, const int rounds) {
  const unsigned C = 500;
  const bool     saturate_graph = false;

  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", saturate_graph);
  paras.Set<unsigned>("num_rnds", 1);
  paras.Set<unsigned>("num_threads", thread_count);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path, data_load, num_points, dim,
                               aligned_dim);

  using TagT = uint32_t;
  unsigned   num_frozen = 1;
  const bool enable_tags = true;
  const bool support_eager_delete = false;

  auto num_frozen_str = getenv("TTS_NUM_FROZEN");

  if (num_frozen_str != nullptr) {
    num_frozen = std::atoi(num_frozen_str);
    std::cout << "Overriding num_frozen to" << num_frozen << std::endl;
  }

  // auto                    s = std::chrono::high_resolution_clock::now();
  diskann::Index<T, TagT> index(diskann::L2, dim, num_points, true, paras,
                                paras, enable_tags, support_eager_delete);

  std::vector<TagT> one_tag;
  one_tag.push_back(0);

  index.build(&data_load[0], 1, paras, one_tag);

  std::cout << "Inserting every point into the index" << std::endl;
  // std::chrono::duration<double> diff =
  // std::chrono::high_resolution_clock::now() - s;

  diskann::Timer index_timer;

#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
  for (int64_t j = 1; j < (int64_t) num_points; j++) {
    index.insert_point(&data_load[j * aligned_dim], static_cast<TagT>(j));
  }

  double seconds = index_timer.elapsed() / 1000000.0;

  std::cout << "Inserted points in " << seconds << " seconds" << std::endl;

  index.save(save_path.c_str());
  std::cout << std::endl;
  std::cout << std::endl;

  // CYCLING START

  for (const int delete_policy : Dvec) {
    std::cout << "Beginning index build for Delete Policy " << delete_policy
              << std::endl;
    std::cout << std::endl;
    diskann::Index<T, TagT> indexCycle(diskann::L2, dim, num_points, true,
                                       paras, paras, enable_tags,
                                       support_eager_delete);
    indexCycle.load(save_path.c_str(), thread_count, L);
    std::cout << "Index loaded" << std::endl;
    std::cout << std::endl;

    int parts = 10;
    int points_in_part;

    for (int i = 0; i < rounds; i++) {
      std::cout << "ROUND " << i + 1 << std::endl;
      std::cout << std::endl;

      std::vector<int64_t> indices(num_points);
      std::iota(indices.begin(), indices.end(), 0);
      std::random_shuffle(indices.begin(), indices.end());

      int points_seen = 0;
      for (int j = 0; j < parts; j++) {
        if (j == parts - 1)
          points_in_part = num_points - points_seen;
        else
          points_in_part = num_points / parts;

        // DELETIONS
        std::cout << "Deleting " << points_in_part
                  << " points from the index..." << std::endl;
        indexCycle.enable_delete();
        tsl::robin_set<TagT> deletes;
        // size_t               deletes_start = num_points - num_deletes;
        for (int k = points_seen; k < points_seen + points_in_part; k++) {
          deletes.insert(static_cast<TagT>(indices[k]));
        }
        std::vector<TagT> failed_deletes;
        indexCycle.lazy_delete(deletes, failed_deletes);
        omp_set_num_threads(thread_count);
        diskann::Timer delete_timer;
        // not using concurrent deletes
        indexCycle.disable_delete(paras, true, false, delete_policy);
        double elapsedSeconds = delete_timer.elapsed() / 1000000.0;

        std::cout << "Deleted " << points_in_part << " points in "
                  << elapsedSeconds << " seconds" << std::endl;

        // RE-INSERTIONS
        std::cout << "Re-inserting the same " << points_in_part
                  << " points from the index..." << std::endl;
        diskann::Timer insert_timer;
#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
        for (int64_t k = points_seen;
             k < (int64_t) points_seen + points_in_part; k++) {
          indexCycle.insert_point(&data_load[indices[k] * aligned_dim],
                                  static_cast<TagT>(indices[k]));
        }
        elapsedSeconds = insert_timer.elapsed() / 1000000.0;

        std::cout << "Inserted " << points_in_part << " points in "
                  << elapsedSeconds << " seconds" << std::endl;
        std::cout << std::endl;

        points_seen += points_in_part;
      }

      const auto save_path_inc =
          get_save_filename(save_path + ".after-cycle-", i, delete_policy);
      indexCycle.save(save_path_inc.c_str());
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  std::string         data_type, data_path, save_path;
  unsigned            num_threads, R, L;
  float               alpha;
  std::vector<size_t> Dvec;
  int                 rounds;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("data_path",
                       po::value<std::string>(&data_path)->required(),
                       "Input data file in bin format");
    desc.add_options()("save_path",
                       po::value<std::string>(&save_path)->required(),
                       "Path prefix for saving index file components");
    desc.add_options()("max_degree,R",
                       po::value<uint32_t>(&R)->default_value(64),
                       "Maximum graph degree");
    desc.add_options()(
        "Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
        "Build complexity, higher value results in better graphs");
    desc.add_options()(
        "alpha", po::value<float>(&alpha)->default_value(1.2f),
        "alpha controls density and diameter of graph, set 1 for sparse graph, "
        "1.2 or 1.4 for denser graphs with lower diameter");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()("delete_policies",
                       po::value<std::vector<size_t>>(&Dvec)->multitoken(),
                       "List of delete policies to iterate over");
    desc.add_options()("rounds", po::value<int>(&rounds)->default_value(1),
                       "Number of rounds");

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

  try {
    if (data_type == std::string("int8"))
      test_batch_deletes<int8_t>(data_path, L, R, alpha, num_threads, Dvec,
                                 save_path, rounds);
    else if (data_type == std::string("uint8"))
      test_batch_deletes<uint8_t>(data_path, L, R, alpha, num_threads, Dvec,
                                  save_path, rounds);
    else if (data_type == std::string("float"))
      test_batch_deletes<float>(data_path, L, R, alpha, num_threads, Dvec,
                                save_path, rounds);
    else
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "Caught unknown exception" << std::endl;
    exit(-1);
  }

  return 0;
}
