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
void test_batch_deletes(const std::string& data_path, const unsigned L,
                        const unsigned R, const float alpha,
                        const unsigned            thread_count,
                        const std::vector<size_t> Dvec,
                        const std::string& save_path, const int delete_policy) {
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

  size_t dim;
  size_t num_points;

  diskann::get_bin_metadata(data_path, num_points, dim);

  using TagT = uint32_t;
  unsigned   num_frozen = 1;
  const bool enable_tags = true;
  const bool support_eager_delete = false;

  auto num_frozen_str = getenv("TTS_NUM_FROZEN");

  if (num_frozen_str != nullptr) {
    num_frozen = std::atoi(num_frozen_str);
    std::cout << "Overriding num_frozen to" << num_frozen << std::endl;
  }

  diskann::Index<T, TagT> index(diskann::L2, dim, num_points, true, paras,
                                paras, enable_tags, support_eager_delete);

  std::vector<TagT> tags(num_points);
  std::iota(tags.begin(), tags.end(), 0);

  auto s = std::chrono::high_resolution_clock::now();
  index.build(data_path.c_str(), num_points, paras, tags);

  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;

  std::cout << "Indexing time: " << diff.count() << "\n";
  index.save(save_path.c_str());

  for (size_t num_deletes : Dvec) {
    diskann::Index<T, TagT> indexDel(diskann::L2, dim, num_points, true, paras,
                                     paras, enable_tags, support_eager_delete);
    indexDel.load(save_path.c_str(), thread_count, L);
    std::cout << "Index loaded" << std::endl;
    std::cout << "Deleting " << num_deletes
              << " points from the end of the index..." << std::endl;
    indexDel.enable_delete();
    tsl::robin_set<TagT> deletes;
    size_t               deletes_start = num_points - num_deletes;
    for (size_t i = deletes_start; i < deletes_start + num_deletes; i++) {
      deletes.insert(static_cast<TagT>(i));
    }
    std::vector<TagT> failed_deletes;
    indexDel.lazy_delete(deletes, failed_deletes);
    omp_set_num_threads(thread_count);
    diskann::Timer delete_timer;
    // not using concurrent deletes
    indexDel.disable_delete(paras, true, false, delete_policy);
    const double elapsedSeconds = delete_timer.elapsed() / 1000000.0;

    std::cout << "Deleted " << num_deletes << " points in " << elapsedSeconds
              << " seconds (" << num_deletes / elapsedSeconds
              << " points/second overall, "
              << num_deletes / elapsedSeconds / thread_count
              << " per thread)\n ";

    const auto save_path_inc =
        get_save_filename(save_path + ".after-delete-", num_deletes, 0);
    indexDel.save(save_path_inc.c_str());
  }
}

int main(int argc, char** argv) {
  std::string         data_type, data_path, save_path;
  unsigned            num_threads, R, L;
  float               alpha;
  std::vector<size_t> Dvec;
  int                 delete_policy;

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
    desc.add_options()("deletes",
                       po::value<std::vector<size_t>>(&Dvec)->multitoken(),
                       "List of batch sizes for deletions");
    desc.add_options()("delete_policy",
                       po::value<int>(&delete_policy)->default_value(0),
                       "Delete policy: 0 for all, 1 for closest, 2 for random");

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
                                 save_path, delete_policy);
    else if (data_type == std::string("uint8"))
      test_batch_deletes<uint8_t>(data_path, L, R, alpha, num_threads, Dvec,
                                  save_path, delete_policy);
    else if (data_type == std::string("float"))
      test_batch_deletes<float>(data_path, L, R, alpha, num_threads, Dvec,
                                save_path, delete_policy);
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
