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

// load_aligned_bin modified to read pieces of the file, but using ifstream
// instead of cached_ifstream.
template<typename T>
inline void load_aligned_bin_part(const std::string& bin_file, T* data,
                                  size_t offset_points, size_t points_to_read) {
  diskann::Timer timer;
  std::ifstream  reader;
  reader.exceptions(std::ios::failbit | std::ios::badbit);
  reader.open(bin_file, std::ios::binary | std::ios::ate);
  size_t actual_file_size = reader.tellg();
  reader.seekg(0, std::ios::beg);

  int npts_i32, dim_i32;
  reader.read((char*) &npts_i32, sizeof(int));
  reader.read((char*) &dim_i32, sizeof(int));
  size_t npts = (unsigned) npts_i32;
  size_t dim = (unsigned) dim_i32;

  size_t expected_actual_file_size =
      npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
  if (actual_file_size != expected_actual_file_size) {
    std::stringstream stream;
    stream << "Error. File size mismatch. Actual size is " << actual_file_size
           << " while expected size is  " << expected_actual_file_size
           << " npts = " << npts << " dim = " << dim
           << " size of <T>= " << sizeof(T) << std::endl;
    std::cout << stream.str();
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
  }

  if (offset_points + points_to_read > npts) {
    std::stringstream stream;
    stream << "Error. Not enough points in file. Requested " << offset_points
           << "  offset and " << points_to_read << " points, but have only "
           << npts << " points" << std::endl;
    std::cout << stream.str();
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
  }

  reader.seekg(2 * sizeof(uint32_t) + offset_points * dim * sizeof(T));

  const size_t rounded_dim = ROUND_UP(dim, 8);

  for (size_t i = 0; i < points_to_read; i++) {
    reader.read((char*) (data + i * rounded_dim), dim * sizeof(T));
    memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
  }
  reader.close();

  const double elapsedSeconds = timer.elapsed() / 1000000.0;
  std::cout << "Read " << points_to_read << " points using non-cached reads in "
            << elapsedSeconds << std::endl;
}

std::string get_save_filename(const std::string& save_path,
                              size_t             points_to_skip,
                              size_t             last_point_threshold) {
  std::string final_path = save_path;
  if (points_to_skip > 0) {
    final_path += std::to_string(points_to_skip) + "-";
  }
  final_path += std::to_string(last_point_threshold);
  return final_path;
}

template<typename T>
void build_incremental_index(
    const std::string& data_path, const unsigned L, const unsigned R,
    const float alpha, const unsigned thread_count, size_t points_to_skip,
    size_t max_points_to_insert, size_t beginning_index_size,
    size_t points_per_checkpoint, size_t checkpoints_per_snapshot,
    const std::string& save_path, size_t points_to_delete_from_beginning,
    bool concurrent) {
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

  size_t dim, aligned_dim;
  size_t num_points;

  diskann::get_bin_metadata(data_path, num_points, dim);
  aligned_dim = ROUND_UP(dim, 8);

  if (points_to_skip > num_points) {
    throw diskann::ANNException("Asked to skip more points than in data file",
                                -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  if (max_points_to_insert == 0) {
    max_points_to_insert = num_points;
  }

  if (points_to_skip + max_points_to_insert > num_points) {
    max_points_to_insert = num_points - points_to_skip;
    std::cerr << "WARNING: Reducing max_points_to_insert to "
              << max_points_to_insert
              << " points since the data file has only that many" << std::endl;
  }

  using TagT = uint32_t;
  unsigned   num_frozen = 1;
  const bool enable_tags = true;
  const bool support_eager_delete = false;

  auto num_frozen_str = getenv("TTS_NUM_FROZEN");

  if (num_frozen_str != nullptr) {
    num_frozen = std::atoi(num_frozen_str);
    std::cout << "Overriding num_frozen to" << num_frozen << std::endl;
  }

  diskann::Index<T, TagT> index(diskann::L2, dim, max_points_to_insert, true,
                                paras, paras, enable_tags,
                                support_eager_delete);

  size_t       current_point_offset = points_to_skip;
  const size_t last_point_threshold = points_to_skip + max_points_to_insert;

  if (beginning_index_size > max_points_to_insert) {
    beginning_index_size = max_points_to_insert;
    std::cerr << "WARNING: Reducing beginning index size to "
              << beginning_index_size
              << " points since the data file has only that many" << std::endl;
  }
  if (checkpoints_per_snapshot > 0 &&
      beginning_index_size > points_per_checkpoint) {
    beginning_index_size = points_per_checkpoint;
    std::cerr << "WARNING: Reducing beginning index size to "
              << beginning_index_size << std::endl;
  }

  size_t allocSize = beginning_index_size * aligned_dim * sizeof(T);
  T*     data_part = nullptr;

  diskann::alloc_aligned((void**) &data_part, allocSize, 8 * sizeof(T));

  std::vector<TagT> tags(beginning_index_size);
  std::iota(tags.begin(), tags.end(), static_cast<TagT>(current_point_offset));

  load_aligned_bin_part(data_path, data_part, current_point_offset,
                        beginning_index_size);
  std::cout << "load aligned bin succeeded" << std::endl;
  diskann::Timer timer;

  if (beginning_index_size > 0) {
    index.build(data_part, beginning_index_size, paras, tags);
  } else if (getenv("TTS_FAKE_FROZEN_POINT") != nullptr) {
    std::cout << "Adding a fake point for build() and deleting it" << std::endl;

    std::vector<TagT> one_tag;
    one_tag.push_back(UINT32_MAX);

    std::vector<T> fake_coords(aligned_dim);
    for (size_t i = 0; i < dim; i++) {
      fake_coords[i] = static_cast<T>(i);
    }

    index.build(fake_coords.data(), 1, paras, one_tag);
    index.enable_delete();
    index.lazy_delete(one_tag[0]);
  }

  const double elapsedSeconds = timer.elapsed() / 1000000.0;
  std::cout << "Initial non-incremental index build time for "
            << beginning_index_size << " points took " << elapsedSeconds
            << " seconds (" << beginning_index_size / elapsedSeconds
            << " points/second)\n ";

  current_point_offset = beginning_index_size;

  if (concurrent) {
    int sub_threads = (thread_count + 1) / 2;
    {
      diskann::Timer timer;
      index.enable_delete();

      auto inserts = std::async(std::launch::async, [&]() {
        size_t last_snapshot_points_threshold = 0;
        size_t num_checkpoints_till_snapshot = checkpoints_per_snapshot;

        for (size_t i = current_point_offset; i < last_point_threshold;
             i += points_per_checkpoint,
                    current_point_offset += points_per_checkpoint) {
          std::cout << i << std::endl << std::endl;

          const size_t j_threshold =
              std::min(i + points_per_checkpoint, last_point_threshold);

          load_aligned_bin_part(data_path, data_part, i, j_threshold - i);

          diskann::Timer insert_timer;

#pragma omp parallel for num_threads(sub_threads) schedule(dynamic)
          for (int64_t j = i; j < (int64_t) j_threshold; j++) {
            index.insert_point(&data_part[(j - i) * aligned_dim],
                               static_cast<TagT>(j));
          }
          const double elapsedSeconds = insert_timer.elapsed() / 1000000.0;
          std::cout << "Insertion time " << elapsedSeconds << " seconds ("
                    << (j_threshold - i) / elapsedSeconds
                    << " points/second overall, "
                    << (j_threshold - i) / elapsedSeconds / thread_count
                    << " per thread)\n ";
        }
      });

      auto deletes = std::async(std::launch::async, [&]() {
        std::cout << "Deleting " << points_to_delete_from_beginning
                  << " points from the beginning of the index..." << std::endl;

        tsl::robin_set<TagT> deletes;
        // std::vector<TagT> deletes(points_to_delete_from_beginning);
        for (size_t i = 0; i < points_to_delete_from_beginning; i++) {
          deletes.insert(static_cast<TagT>(points_to_skip + i));
        }
        std::vector<TagT> failed_deletes;
        index.lazy_delete(deletes, failed_deletes);
        omp_set_num_threads(sub_threads);
        diskann::Timer delete_timer;
        index.disable_delete(paras, true, true);
        const double elapsedSeconds = delete_timer.elapsed() / 1000000.0;

        std::cout << "Deleted " << points_to_delete_from_beginning
                  << " points in " << elapsedSeconds << " seconds ("
                  << points_to_delete_from_beginning / elapsedSeconds
                  << " points/second overall, "
                  << points_to_delete_from_beginning / elapsedSeconds /
                         sub_threads
                  << " per thread)\n ";
      });

      inserts.wait();
      deletes.wait();

      std::cout << "Time Elapsed" << timer.elapsed() / 1000 << "ms\n";
      const auto save_path_inc = save_path + ".after_concurrent_change";
      index.save(save_path_inc.c_str());
    }
  } else {
    current_point_offset += beginning_index_size;

    size_t last_snapshot_points_threshold = 0;
    size_t num_checkpoints_till_snapshot = checkpoints_per_snapshot;

    for (size_t i = current_point_offset; i < last_point_threshold;
         i += points_per_checkpoint,
                current_point_offset += points_per_checkpoint) {
      std::cout << i << std::endl << std::endl;

      const size_t j_threshold =
          std::min(i + points_per_checkpoint, last_point_threshold);

      load_aligned_bin_part(data_path, data_part, i, j_threshold - i);

      diskann::Timer insert_timer;

#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
      for (int64_t j = i; j < (int64_t) j_threshold; j++) {
        index.insert_point(&data_part[(j - i) * aligned_dim],
                           static_cast<TagT>(j));
      }
      const double elapsedSeconds = insert_timer.elapsed() / 1000000.0;
      std::cout << "Insertion time " << elapsedSeconds << " seconds ("
                << (j_threshold - i) / elapsedSeconds
                << " points/second overall, "
                << (j_threshold - i) / elapsedSeconds / thread_count
                << " per thread)\n ";

      if (checkpoints_per_snapshot > 0 &&
          --num_checkpoints_till_snapshot == 0) {
        diskann::Timer save_timer;

        const auto save_path_inc =
            get_save_filename(save_path + ".inc-", points_to_skip, j_threshold);
        index.save(save_path_inc.c_str());
        const double elapsedSeconds = save_timer.elapsed() / 1000000.0;
        const size_t points_saved = j_threshold - points_to_skip;

        std::cout << "Saved " << points_saved << " points in " << elapsedSeconds
                  << " seconds (" << points_saved / elapsedSeconds
                  << " points/second)\n ";

        num_checkpoints_till_snapshot = checkpoints_per_snapshot;
        last_snapshot_points_threshold = j_threshold;
      }

      std::cout << "Number of points in the index post insertion "
                << j_threshold << std::endl;
    }

    if (checkpoints_per_snapshot >= 0 &&
        last_snapshot_points_threshold != last_point_threshold) {
      const auto save_path_inc = get_save_filename(
          save_path + ".inc-", points_to_skip, last_point_threshold);
      index.save(save_path_inc.c_str());
    }

    if (points_to_delete_from_beginning > 0) {
      if (points_to_delete_from_beginning > max_points_to_insert) {
        points_to_delete_from_beginning =
            static_cast<unsigned>(max_points_to_insert);
        std::cerr << "WARNING: Reducing points to delete from beginning to "
                  << points_to_delete_from_beginning
                  << " points since the data file has only that many"
                  << std::endl;
      }

      std::cout << "Deleting " << points_to_delete_from_beginning
                << " points from the beginning of the index..." << std::endl;
      index.enable_delete();

      tsl::robin_set<TagT> deletes;
      for (size_t i = 0; i < points_to_delete_from_beginning; i++) {
        deletes.insert(static_cast<TagT>(points_to_skip + i));
      }
      std::vector<TagT> failed_deletes;

      diskann::Timer request_timer;
      index.lazy_delete(deletes, failed_deletes);
      std::cout << "Prepared request in " << request_timer.elapsed() / 1000000.0
                << " seconds (" << failed_deletes.size() << " failed)\n";

      omp_set_num_threads(thread_count);

      diskann::Timer delete_timer;
      index.disable_delete(paras, true);
      const double elapsedSeconds = delete_timer.elapsed() / 1000000.0;

      std::cout << "Deleted " << points_to_delete_from_beginning
                << " points in " << elapsedSeconds << " seconds ("
                << points_to_delete_from_beginning / elapsedSeconds
                << " points/second overall, "
                << points_to_delete_from_beginning / elapsedSeconds /
                       thread_count
                << " per thread)\n ";

      if (checkpoints_per_snapshot >= 0) {
        const auto save_path_inc =
            get_save_filename(save_path + ".after-delete-",
                              points_to_skip + points_to_delete_from_beginning,
                              last_point_threshold);
        index.save(save_path_inc.c_str());
      }
    }
  }

  diskann::aligned_free(data_part);
}

int main(int argc, char** argv) {
  std::string data_type, dist_fn, data_path, index_path_prefix;
  unsigned    num_threads, R, L;
  float       alpha;
  size_t      points_to_skip, max_points_to_insert, beginning_index_size,
      points_per_checkpoint, checkpoints_per_snapshot,
      points_to_delete_from_beginning;
  bool concurrent;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips>");
    desc.add_options()("data_path",
                       po::value<std::string>(&data_path)->required(),
                       "Input data file in bin format");
    desc.add_options()("index_path_prefix",
                       po::value<std::string>(&index_path_prefix)->required(),
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
    desc.add_options()("points_to_skip",
                       po::value<uint64_t>(&points_to_skip)->required(), "");
    desc.add_options()(
        "max_points_to_insert",
        po::value<uint64_t>(&max_points_to_insert)->default_value(0), "");
    desc.add_options()("beginning_index_size",
                       po::value<uint64_t>(&beginning_index_size)->required(),
                       "");
    desc.add_options()("points_per_checkpoint",
                       po::value<uint64_t>(&points_per_checkpoint)->required(),
                       "");
    desc.add_options()(
        "checkpoints_per_snapshot",
        po::value<uint64_t>(&checkpoints_per_snapshot)->required(), "");
    desc.add_options()(
        "points_to_delete_from_beginning",
        po::value<uint64_t>(&points_to_delete_from_beginning)->required(), "");
    desc.add_options()("do_concurrent",
                       po::value<bool>(&concurrent)->required(), "");

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
      build_incremental_index<int8_t>(
          data_path, L, R, alpha, num_threads, points_to_skip,
          max_points_to_insert, beginning_index_size, points_per_checkpoint,
          checkpoints_per_snapshot, index_path_prefix,
          points_to_delete_from_beginning, concurrent);
    else if (data_type == std::string("uint8"))
      build_incremental_index<uint8_t>(
          data_path, L, R, alpha, num_threads, points_to_skip,
          max_points_to_insert, beginning_index_size, points_per_checkpoint,
          checkpoints_per_snapshot, index_path_prefix,
          points_to_delete_from_beginning, concurrent);
    else if (data_type == std::string("float"))
      build_incremental_index<float>(
          data_path, L, R, alpha, num_threads, points_to_skip,
          max_points_to_insert, beginning_index_size, points_per_checkpoint,
          checkpoints_per_snapshot, index_path_prefix,
          points_to_delete_from_beginning, concurrent);
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
