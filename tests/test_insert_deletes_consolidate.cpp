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

using std::random_shuffle;
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
                              size_t points_to_skip, float points_deleted,
                              float  points_inserted,
                              size_t last_point_threshold) {
  std::string final_path = save_path;
  if (points_to_skip > 0) {
    final_path += "skip" + std::to_string(points_to_skip) + "-";
  }
  if (points_deleted > 0 && points_inserted > 0) {
    final_path += "insert" + std::to_string(points_inserted) + "-" + "delete" +
                  std::to_string(points_deleted);
  } else if (points_deleted == 0 && points_inserted > 0) {
    final_path += "insert" + std::to_string(points_inserted);
  } else if (points_deleted > 0 && points_inserted == 0) {
    final_path += "delete" + std::to_string(points_deleted);
  }
  return final_path;
}

template<typename T, typename TagT>
void insert_till_next_checkpoint(diskann::Index<T, TagT>& index,
                                 size_t max_points, float insert_percentage,
                                 float build_percentage, size_t thread_count,
                                 T* data, size_t aligned_dim,
                                 float                delete_percentage,
                                 diskann::Parameters& delete_params) {
  diskann::Timer insert_timer;
  int64_t        insert_points = max_points * insert_percentage;
  int64_t        delete_points = max_points * delete_percentage;
  int64_t        build_points = max_points * build_percentage;
  if (delete_percentage == insert_percentage) {
    std::cout << " random insertion and deletion" << std::endl;
    std::vector<int> streaming_numbers;
    for (int64_t j = 0; j < (int64_t) build_points - 1; j++) {
      streaming_numbers.push_back(j);
    }
    random_shuffle(streaming_numbers.begin(), streaming_numbers.end());

#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
    for (int64_t i1 = 0; i1 < insert_points; i1++) {
      int64_t k1 = streaming_numbers[i1];
      index.insert_point(&data[k1 * aligned_dim], static_cast<TagT>(k1));
    }
    const double elapsedSeconds = insert_timer.elapsed() / 1000000.0;
    std::cout << "Insertion time " << elapsedSeconds << " seconds ("
              << insert_points / elapsedSeconds << " points/second overall, "
              << insert_points / elapsedSeconds / thread_count
              << " per thread)\n ";

    for (int64_t i2 = 0; i2 < delete_points; i2++) {
      int64_t k2 = streaming_numbers[i2];
      index.lazy_delete(k2);
    }
    auto report = index.consolidate_deletes(delete_params);
    std::cout << "#active points: " << report._active_points << std::endl
              << "max points: " << report._max_points << std::endl
              << "empty slots: " << report._empty_slots << std::endl
              << "deletes processed: " << report._slots_released << std::endl
              << "latest delete size: " << report._delete_set_size << std::endl
              << "rate: (" << delete_points / report._time
              << " points/second overall, "
              << delete_points / report._time /
                     delete_params.Get<unsigned>("num_threads")
              << " per thread)" << std::endl;
  } else if (insert_percentage > 0 && delete_percentage == 0) {
    std::cout << " only insert randomly" << std::endl;
#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
    for (int64_t i = build_points; i < insert_points + build_points; i++) {
      index.insert_point(&data[i * aligned_dim], static_cast<TagT>(i));
    }
    const double elapsedSeconds = insert_timer.elapsed() / 1000000.0;
    std::cout << "Insertion time " << elapsedSeconds << " seconds ("
              << (max_points * insert_percentage) / elapsedSeconds
              << " points/second overall, "
              << (max_points * insert_percentage) / elapsedSeconds /
                     thread_count
              << " per thread)\n ";
  }
}

template<typename T, typename TagT>
void delete_from_beginning(diskann::Index<T, TagT>& index,
                           diskann::Parameters&     delete_params,
                           size_t max_points, float build_percentage,
                           size_t points_to_skip, float delete_percentage) {
  int64_t build_points = max_points * build_percentage;
  try {
    std::cout << std::endl
              << "Lazy deleting points "
              << " randomly ";
    std::vector<int> delete_numbers;
    for (int64_t j = 0; j < (int64_t) build_points - 1; j++) {
      delete_numbers.push_back(j);
    }
    random_shuffle(delete_numbers.begin(), delete_numbers.end());

    for (size_t i = points_to_skip;
         i < (points_to_skip + max_points * delete_percentage); ++i) {
      int64_t m = delete_numbers[i];
      index.lazy_delete(m);
    }
    auto report = index.consolidate_deletes(delete_params);
    std::cout << "#active points: " << report._active_points << std::endl
              << "max points: " << report._max_points << std::endl
              << "empty slots: " << report._empty_slots << std::endl
              << "deletes processed: " << report._slots_released << std::endl
              << "latest delete size: " << report._delete_set_size << std::endl
              << "rate: (" << (max_points * delete_percentage) / report._time
              << " points/second overall, "
              << (max_points * delete_percentage) / report._time /
                     delete_params.Get<unsigned>("num_threads")
              << " per thread)" << std::endl;
  } catch (std::system_error& e) {
    std::cout << "Exception caught in deletion thread: " << e.what()
              << std::endl;
  }
}

template<typename T>
void build_incremental_index(const std::string& data_path, const unsigned L,
                             const unsigned R, const float alpha,
                             const unsigned thread_count, size_t points_to_skip,
                             size_t max_points, float insert_percentage,
                             float              build_percentage,
                             const std::string& save_path,
                             float              delete_percentage,
                             size_t start_deletes_after, bool concurrent) {
  const unsigned      C = 500;
  const bool          saturate_graph = false;
  diskann::Parameters params;
  params.Set<unsigned>("L", L);
  params.Set<unsigned>("R", R);
  params.Set<unsigned>("C", C);
  params.Set<float>("alpha", alpha);
  params.Set<bool>("saturate_graph", saturate_graph);
  params.Set<unsigned>("num_rnds", 1);
  params.Set<unsigned>("num_threads", thread_count);
  size_t dim, aligned_dim;
  size_t num_points;

  diskann::get_bin_metadata(data_path, num_points, dim);
  aligned_dim = ROUND_UP(dim, 8);
  std::cout << "num_points = " << num_points << " "
            << "dim = " << dim << std::endl;
  std::cout << "aligned_dim = " << aligned_dim << std::endl;

  if (points_to_skip > num_points) {
    throw diskann::ANNException("Asked to skip more points than in data file",
                                -1, __FUNCSIG__, __FILE__, __LINE__);
  }
  if (max_points == 0) {
    max_points = num_points;
  }
  if (points_to_skip + max_points > num_points) {
    max_points = num_points - points_to_skip;
    std::cerr << "WARNING: Reducing max_points to " << max_points
              << " points since the data file has only that many" << std::endl;
  }

  using TagT = uint32_t;
  unsigned   num_frozen = 1;
  const bool enable_tags = true;
  const bool support_eager_delete = false;
  auto       num_frozen_str = getenv("TTS_NUM_FROZEN");
  if (num_frozen_str != nullptr) {
    num_frozen = std::atoi(num_frozen_str);
    std::cout << "Overriding num_frozen to" << num_frozen << std::endl;
  }

  diskann::Index<T, TagT> index(diskann::L2, dim, max_points, true, params,
                                params, enable_tags, support_eager_delete,
                                concurrent);

  const size_t last_point_threshold = points_to_skip + max_points;

  if (max_points * insert_percentage > max_points) {
    insert_percentage = 1;
    std::cerr << "WARNING: Reducing beginning index size to "
              << max_points * insert_percentage
              << " points since the data file has only that many" << std::endl;
  }
  int64_t build_point = max_points * build_percentage;
  if (build_point == max_points) {
    build_point = max_points * build_percentage - 1;
  } else {
    build_point = max_points * build_percentage;
  }

  T* data = nullptr;
  diskann::alloc_aligned((void**) &data, build_point * aligned_dim * sizeof(T),
                         8 * sizeof(T));

  std::vector<TagT> tags(build_point);
  std::iota(tags.begin(), tags.end(), static_cast<TagT>(points_to_skip));
  load_aligned_bin_part(data_path, data, points_to_skip, build_point);
  std::cout << "load aligned bin succeeded" << std::endl;
  diskann::Timer timer;
  if (build_point > 0) {
    index.build(data, build_point, params, tags);
    index.enable_delete();
  } else {
    index.build_with_zero_points();
    index.enable_delete();
  }
  const double elapsedSeconds = timer.elapsed() / 1000000.0;
  std::cout << "Initial non-incremental index build time for " << build_point
            << " points took " << elapsedSeconds << " seconds ("
            << build_point / elapsedSeconds << " points/second)\n ";

  int64_t insert_index_size = max_points * insert_percentage;
  if (insert_percentage > 0) {
    if (delete_percentage > 0) {
      load_aligned_bin_part(data_path, data, points_to_skip, build_point);
    } else {
      diskann::alloc_aligned(
          (void**) &data,
          (build_point + insert_index_size) * aligned_dim * sizeof(T),
          8 * sizeof(T));
      std::vector<TagT> tags(build_point + max_points * insert_percentage);
      std::iota(tags.begin(), tags.end(), static_cast<TagT>(points_to_skip));
      load_aligned_bin_part(data_path, data, points_to_skip,
                            (build_point + insert_index_size));
    }
    insert_till_next_checkpoint(index, max_points, insert_percentage,
                                build_percentage, thread_count, data,
                                aligned_dim, delete_percentage, params);
  } else {
    if (delete_percentage > 0) {
      delete_from_beginning(index, params, max_points, build_percentage,
                            points_to_skip, delete_percentage);
    }
  }
  const auto save_path_inc = get_save_filename(
      save_path + ".after-", points_to_skip, delete_percentage,
      insert_percentage, last_point_threshold);
  index.save(save_path_inc.c_str(), true);
  diskann::aligned_free(data);
}

int main(int argc, char** argv) {
  std::string data_type, dist_fn, data_path, index_path_prefix;
  unsigned    num_threads, R, L;
  float       alpha, insert_percentage, build_percentage, delete_percentage;
  size_t      points_to_skip, max_points, start_deletes_after;
  bool        concurrent;
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
                       po::value<uint64_t>(&points_to_skip)->required(),
                       "Skip these first set of points from file");
    desc.add_options()(
        "max_points", po::value<uint64_t>(&max_points)->default_value(0),
        "These number of points from the file are inserted after "
        "points_to_skip");
    desc.add_options()("insert_percentage",
                       po::value<float>(&insert_percentage)->required(),
                       "Batch build will be called on these set of points");
    desc.add_options()("build_percentage",
                       po::value<float>(&build_percentage)->required(),
                       "build will be called on these set of points");
    desc.add_options()("delete_percentage",
                       po::value<float>(&delete_percentage)->required(), "");
    desc.add_options()("do_concurrent",
                       po::value<bool>(&concurrent)->default_value(false), "");
    desc.add_options()(
        "start_deletes_after",
        po::value<uint64_t>(&start_deletes_after)->default_value(0), "");
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
          data_path, L, R, alpha, num_threads, points_to_skip, max_points,
          insert_percentage, build_percentage, index_path_prefix,
          delete_percentage, start_deletes_after, concurrent);
    else if (data_type == std::string("uint8"))
      build_incremental_index<uint8_t>(
          data_path, L, R, alpha, num_threads, points_to_skip, max_points,
          insert_percentage, build_percentage, index_path_prefix,
          delete_percentage, start_deletes_after, concurrent);
    else if (data_type == std::string("float"))
      build_incremental_index<float>(
          data_path, L, R, alpha, num_threads, points_to_skip, max_points,
          insert_percentage, build_percentage, index_path_prefix,
          delete_percentage, start_deletes_after, concurrent);
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
