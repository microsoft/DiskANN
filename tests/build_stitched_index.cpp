// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/program_options.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <tuple>

#include <omp.h>
#ifndef _WINDOWS
#include <sys/uio.h>
#endif

#include "index.h"
#include "memory_mapper.h"
#include "parameters.h"
#include "utils.h"

namespace po = boost::program_options;

// macros
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

// custom types (for readability)
typedef unsigned              label;
typedef tsl::robin_set<label> label_set;
typedef std::string           path;

// structs for returning multiple items from a function
typedef std::tuple<std::vector<label_set>, tsl::robin_map<label, _u32>,
                   label_set>
    parse_label_file_return_values;
typedef std::tuple<std::vector<std::vector<_u32>>, _u64>
    load_label_index_return_values;
typedef std::tuple<std::vector<std::vector<_u32>>, _u64>
    stitch_indices_return_values;

/*
 * Inline function to display progress bar.
 */
inline void print_progress(double percentage) {
  int val = (int) (percentage * 100);
  int lpad = (int) (percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}

/*
 * Inline function to generate a random integer in a range.
 */
inline size_t random(size_t range_from, size_t range_to) {
  std::random_device                    rand_dev;
  std::mt19937                          generator(rand_dev());
  std::uniform_int_distribution<size_t> distr(range_from, range_to);
  return distr(generator);
}

/*
 * function to handle command line parsing.
 *
 * Arguments are merely the inputs from the command line.
 */
size_t handle_args(int argc, char **argv, std::string &data_type,
                   path &input_data_path, path &final_index_path_prefix,
                   path &label_data_path, label &universal_label,
                   unsigned &num_threads, unsigned &R, unsigned &L,
                   unsigned &stitched_R, float &alpha) {
  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("data_path",
                       po::value<path>(&input_data_path)->required(),
                       "Input data file in bin format");
    desc.add_options()("index_path_prefix",
                       po::value<path>(&final_index_path_prefix)->required(),
                       "Path prefix for saving index file components");
    desc.add_options()("max_degree,R",
                       po::value<uint32_t>(&R)->default_value(64),
                       "Maximum graph degree");
    desc.add_options()(
        "Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
        "Build complexity, higher value results in better graphs");
    desc.add_options()("stitched_R",
                       po::value<uint32_t>(&stitched_R)->default_value(100),
                       "Degree to prune final graph down to");
    desc.add_options()(
        "alpha", po::value<float>(&alpha)->default_value(1.2f),
        "alpha controls density and diameter of graph, set 1 for sparse graph, "
        "1.2 or 1.4 for denser graphs with lower diameter");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()("label_file",
                       po::value<path>(&label_data_path)->default_value(""),
                       "Input label file in txt format if present");
    desc.add_options()(
        "universal_label",
        po::value<label>(&universal_label)->default_value(UINT32_MAX),
        "If a point comes with the specified universal label (and only the "
        "univ. "
        "label), then the point is considered to have every possible label");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      exit(0);
    }
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    throw;
  }
  return 1;
}

/*
 * Parses the label datafile, which has comma-separated labels on
 * each line. Line i corresponds to point id i.
 *
 * Returns three objects via std::tuple:
 * 1. map: key is point id, value is vector of labels said point has
 * 2. map: key is label, value is number of points with the label
 * 3. the label universe as a set
 */
parse_label_file_return_values parse_label_file(path  label_data_path,
                                                label universal_label) {
  std::ifstream label_data_stream(label_data_path);
  std::string   line, token;
  unsigned      line_cnt = 0;

  // allows us to reserve space for the points_to_labels vector
  while (std::getline(label_data_stream, line))
    line_cnt++;
  label_data_stream.clear();
  label_data_stream.seekg(0, std::ios::beg);

  // values to return
  std::vector<label_set>      point_ids_to_labels(line_cnt);
  tsl::robin_map<label, _u32> labels_to_number_of_points;
  label_set                   all_labels;

  std::vector<_u32> points_with_universal_label;
  line_cnt = 0;
  while (std::getline(label_data_stream, line)) {
    std::istringstream current_labels_comma_separated(line);
    label_set          current_labels;

    // get point id
    _u32 point_id = line_cnt;

    // parse comma separated labels
    bool current_universal_label_check = false;
    while (getline(current_labels_comma_separated, token, ',')) {
      token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
      token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());

      // if token is empty, there's no labels for the point
      unsigned token_as_num = std::stoul(token);
      if (token_as_num == universal_label) {
        points_with_universal_label.push_back(point_id);
        current_universal_label_check = true;
      } else {
        all_labels.insert(token_as_num);
        current_labels.insert(token_as_num);
        labels_to_number_of_points[token_as_num]++;
      }
    }

    if (current_labels.size() <= 0 && !current_universal_label_check) {
      std::cerr << "Error: " << point_id << " has no labels." << std::endl;
      exit(-1);
    }
    point_ids_to_labels[point_id] = current_labels;
    line_cnt++;
  }

  // for every point with universal label, set its label set to all labels
  // also, increment the count for number of points a label has
  for (const auto &point_id : points_with_universal_label) {
    point_ids_to_labels[point_id] = all_labels;
    for (const auto &label : all_labels)
      labels_to_number_of_points[label]++;
  }

  std::cout << "Identified " << all_labels.size() << " distinct label(s) for "
            << point_ids_to_labels.size() << " points\n"
            << std::endl;

  return std::make_tuple(point_ids_to_labels, labels_to_number_of_points,
                         all_labels);
}

/*
 * For each label, generates a file containing all vectors that have said label.
 * Also copies data from original bin file to new dimension-aligned file.
 *
 * Utilizes POSIX functions mmap and writev in order to minimize memory
 * overhead, so we include an STL version as well.
 *
 * Each data file is saved under the following format:
 *    input_data_path + "_" + label
 */
template<typename T>
tsl::robin_map<label, std::vector<_u32>> generate_label_specific_vector_files(
    path                        input_data_path,
    tsl::robin_map<label, _u32> labels_to_number_of_points,
    std::vector<label_set> point_ids_to_labels, label_set all_labels) {
  auto file_writing_timer = std::chrono::high_resolution_clock::now();
  diskann::MemoryMapper input_data(input_data_path);
  char *                input_start = input_data.getBuf();

  _u32 number_of_points, dimension;
  std::memcpy(&number_of_points, input_start, sizeof(_u32));
  std::memcpy(&dimension, input_start + sizeof(_u32), sizeof(_u32));
  const _u32   VECTOR_SIZE = dimension * sizeof(T);
  const size_t METADATA = 2 * sizeof(_u32);
  if (number_of_points != point_ids_to_labels.size()) {
    std::cerr << "Error: number of points in labels file and data file differ."
              << std::endl;
    throw;
  }

  tsl::robin_map<label, iovec *>           label_to_iovec_map;
  tsl::robin_map<label, _u32>              label_to_curr_iovec;
  tsl::robin_map<label, std::vector<_u32>> label_id_to_orig_id;

  // setup iovec list for each label
  for (const auto &label : all_labels) {
    iovec *label_iovecs =
        (iovec *) malloc(labels_to_number_of_points[label] * sizeof(iovec));
    if (label_iovecs == nullptr) {
      throw;
    }
    label_to_iovec_map[label] = label_iovecs;
    label_to_curr_iovec[label] = 0;
    label_id_to_orig_id[label].reserve(labels_to_number_of_points[label]);
  }

  // each point added to corresponding per-label iovec list
  for (_u32 point_id = 0; point_id < number_of_points; point_id++) {
    char *curr_point = input_start + METADATA + (VECTOR_SIZE * point_id);
    iovec curr_iovec;

    curr_iovec.iov_base = curr_point;
    curr_iovec.iov_len = VECTOR_SIZE;
    for (const auto &label : point_ids_to_labels[point_id]) {
      *(label_to_iovec_map[label] + label_to_curr_iovec[label]) = curr_iovec;
      label_to_curr_iovec[label]++;
      label_id_to_orig_id[label].push_back(point_id);
    }
  }

  // write each label iovec to resp. file
  for (const auto &label : all_labels) {
    int  label_input_data_fd;
    path curr_label_input_data_path(input_data_path + "_" +
                                    std::to_string(label));
    _u32 curr_num_pts = labels_to_number_of_points[label];

    label_input_data_fd =
        open(curr_label_input_data_path.c_str(),
             O_CREAT | O_WRONLY | O_TRUNC | O_APPEND, (mode_t) 0644);
    if (label_input_data_fd == -1)
      throw;

    // write metadata
    _u32 metadata[2] = {curr_num_pts, dimension};
    int  return_value = write(label_input_data_fd, metadata, sizeof(_u32) * 2);
    if (return_value == -1) {
      throw;
    }

    // limits on number of iovec structs per writev means we need to perform
    // multiple writevs
    size_t i = 0;
    while (curr_num_pts > IOV_MAX) {
      return_value =
          writev(label_input_data_fd,
                 (label_to_iovec_map[label] + (IOV_MAX * i)), IOV_MAX);
      if (return_value == -1) {
        close(label_input_data_fd);
        throw;
      }
      curr_num_pts -= IOV_MAX;
      i += 1;
    }
    return_value =
        writev(label_input_data_fd, (label_to_iovec_map[label] + (IOV_MAX * i)),
               curr_num_pts);
    if (return_value == -1) {
      close(label_input_data_fd);
      throw;
    }

    free(label_to_iovec_map[label]);
    close(label_input_data_fd);
  }

  std::chrono::duration<double> file_writing_time =
      std::chrono::high_resolution_clock::now() - file_writing_timer;
  std::cout << "generated " << all_labels.size()
            << " label-specific vector files for index building in time "
            << file_writing_time.count() << "\n"
            << std::endl;

  return label_id_to_orig_id;
}

// for use on systems without writev (i.e. Windows)
template<typename T>
tsl::robin_map<label, std::vector<_u32>>
generate_label_specific_vector_files_compat(
    path                        input_data_path,
    tsl::robin_map<label, _u32> labels_to_number_of_points,
    std::vector<label_set> point_ids_to_labels, label_set all_labels) {
  auto          file_writing_timer = std::chrono::high_resolution_clock::now();
  std::ifstream input_data_stream(input_data_path);

  _u32 number_of_points, dimension;
  input_data_stream.read((char *) &number_of_points, sizeof(_u32));
  input_data_stream.read((char *) &dimension, sizeof(_u32));
  const _u32 VECTOR_SIZE = dimension * sizeof(T);
  if (number_of_points != point_ids_to_labels.size()) {
    std::cerr << "Error: number of points in labels file and data file differ."
              << std::endl;
    throw;
  }

  tsl::robin_map<label, char *>            labels_to_vectors;
  tsl::robin_map<label, _u32>              labels_to_curr_vector;
  tsl::robin_map<label, std::vector<_u32>> label_id_to_orig_id;

  for (const auto &label : all_labels) {
    _u32  number_of_label_pts = labels_to_number_of_points[label];
    char *vectors = (char *) malloc(number_of_label_pts * VECTOR_SIZE);
    if (vectors == nullptr) {
      throw;
    }
    labels_to_vectors[label] = vectors;
    labels_to_curr_vector[label] = 0;
    label_id_to_orig_id[label].reserve(number_of_label_pts);
  }

  for (_u32 point_id = 0; point_id < number_of_points; point_id++) {
    char *curr_vector = (char *) malloc(VECTOR_SIZE);
    input_data_stream.read(curr_vector, VECTOR_SIZE);
    for (const auto &label : point_ids_to_labels[point_id]) {
      char *curr_label_vector_ptr =
          labels_to_vectors[label] +
          (labels_to_curr_vector[label] * VECTOR_SIZE);
      memcpy(curr_label_vector_ptr, curr_vector, VECTOR_SIZE);
      labels_to_curr_vector[label]++;
      label_id_to_orig_id[label].push_back(point_id);
    }
    free(curr_vector);
  }

  for (const auto &label : all_labels) {
    path curr_label_input_data_path(input_data_path + "_" +
                                    std::to_string(label));
    _u32 number_of_label_pts = labels_to_number_of_points[label];

    std::ofstream label_file_stream;
    label_file_stream.exceptions(std::ios::badbit | std::ios::failbit);
    label_file_stream.open(curr_label_input_data_path);

    label_file_stream.write((char *) &number_of_label_pts, sizeof(_u32));
    label_file_stream.write((char *) &dimension, sizeof(_u32));
    label_file_stream.write((char *) labels_to_vectors[label],
                            number_of_label_pts * VECTOR_SIZE);

    free(labels_to_vectors[label]);
    label_file_stream.close();
  }

  std::chrono::duration<double> file_writing_time =
      std::chrono::high_resolution_clock::now() - file_writing_timer;
  std::cout << "generated " << all_labels.size()
            << " label-specific vector files for index building in time "
            << file_writing_time.count() << "\n"
            << std::endl;

  return label_id_to_orig_id;
}

/*
 * Using passed in parameters and files generated from step 3,
 * builds a vanilla diskANN index for each label.
 *
 * Each index is saved under the following path:
 *  final_index_path_prefix + "_" + label
 */
template<typename T>
void generate_label_indices(path input_data_path, path final_index_path_prefix,
                            label_set all_labels, unsigned R, unsigned L,
                            float alpha, unsigned num_threads) {
  diskann::Parameters label_index_build_parameters;
  label_index_build_parameters.Set<unsigned>("R", R);
  label_index_build_parameters.Set<unsigned>("L", L);
  label_index_build_parameters.Set<unsigned>("C", 750);
	label_index_build_parameters.Set<unsigned>("Lf", 0);
  label_index_build_parameters.Set<bool>("saturate_graph", 0);
  label_index_build_parameters.Set<float>("alpha", alpha);
  label_index_build_parameters.Set<unsigned>("num_threads", num_threads);

  std::cout << "Generating indices per label..." << std::endl;
  // for each label, build an index on resp. points
  double total_indexing_time = 0.0, indexing_percentage = 0.0;
  std::cout.setstate(std::ios_base::failbit);
  diskann::cout.setstate(std::ios_base::failbit);
  for (const auto &label : all_labels) {
    path curr_label_input_data_path(input_data_path + "_" +
                                    std::to_string(label));
    path curr_label_index_path(final_index_path_prefix + "_" +
                               std::to_string(label));

    size_t number_of_label_points, dimension;
    diskann::get_bin_metadata(curr_label_input_data_path,
                              number_of_label_points, dimension);
    diskann::Index<T> index(diskann::Metric::L2, dimension,
                            number_of_label_points, false, false);

    auto index_build_timer = std::chrono::high_resolution_clock::now();
    index.build(curr_label_input_data_path.c_str(), number_of_label_points,
                label_index_build_parameters);
    std::chrono::duration<double> current_indexing_time =
        std::chrono::high_resolution_clock::now() - index_build_timer;

    total_indexing_time += current_indexing_time.count();
    indexing_percentage += (1 / (double) all_labels.size());
    print_progress(indexing_percentage);

    index.save(curr_label_index_path.c_str());
  }
  std::cout.clear();
  diskann::cout.clear();

  std::cout << "\nDone. Generated per-label indices in " << total_indexing_time
            << " seconds\n"
            << std::endl;
}

/*
 * Manually loads a graph index in from a given file.
 *
 * Returns both the graph index and the size of the file in bytes.
 */
load_label_index_return_values load_label_index(path label_index_path,
                                                _u32 label_number_of_points) {
  std::ifstream label_index_stream;
  label_index_stream.exceptions(std::ios::badbit | std::ios::failbit);
  label_index_stream.open(label_index_path, std::ios::binary);

  _u64         index_file_size, index_num_frozen_points;
  _u32         index_max_observed_degree, index_entry_point;
  const size_t INDEX_METADATA = 2 * sizeof(_u64) + 2 * sizeof(_u32);
  label_index_stream.read((char *) &index_file_size, sizeof(_u64));
  label_index_stream.read((char *) &index_max_observed_degree, sizeof(_u32));
  label_index_stream.read((char *) &index_entry_point, sizeof(_u32));
  label_index_stream.read((char *) &index_num_frozen_points, sizeof(_u64));
  size_t bytes_read = INDEX_METADATA;

  std::vector<std::vector<_u32>> label_index(label_number_of_points);
  _u32                           nodes_read = 0;
  while (bytes_read != index_file_size) {
    _u32 current_node_num_neighbors;
    label_index_stream.read((char *) &current_node_num_neighbors, sizeof(_u32));
    nodes_read++;

    std::vector<_u32> current_node_neighbors(current_node_num_neighbors);
    label_index_stream.read((char *) current_node_neighbors.data(),
                            current_node_num_neighbors * sizeof(_u32));
    label_index[nodes_read - 1].swap(current_node_neighbors);
    bytes_read += sizeof(_u32) * (current_node_num_neighbors + 1);
  }

  return std::make_tuple(label_index, index_file_size);
}

/*
 * Custom index save to write the in-memory index to disk.
 * Also writes required files for diskANN API -
 *  1. labels_to_medoids
 *  2. universal_label
 *  3. data (redundant for static indices)
 *  4. labels (redundant for static indices)
 */
void save_full_index(path final_index_path_prefix, path input_data_path,
                     _u64                           final_index_size,
                     std::vector<std::vector<_u32>> stitched_graph,
                     tsl::robin_map<label, _u32>    entry_points,
                     label universal_label, path label_data_path) {
  // aux. file 1
  auto          saving_index_timer = std::chrono::high_resolution_clock::now();
  std::ifstream original_label_data_stream;
  original_label_data_stream.exceptions(std::ios::badbit | std::ios::failbit);
  original_label_data_stream.open(label_data_path, std::ios::binary);
  std::ofstream new_label_data_stream;
  new_label_data_stream.exceptions(std::ios::badbit | std::ios::failbit);
  new_label_data_stream.open(final_index_path_prefix + "_labels.txt",
                             std::ios::binary);
  new_label_data_stream << original_label_data_stream.rdbuf();
  original_label_data_stream.close();
  new_label_data_stream.close();

  // aux. file 2
  std::ifstream original_input_data_stream;
  original_input_data_stream.exceptions(std::ios::badbit | std::ios::failbit);
  original_input_data_stream.open(input_data_path, std::ios::binary);
  std::ofstream new_input_data_stream;
  new_input_data_stream.exceptions(std::ios::badbit | std::ios::failbit);
  new_input_data_stream.open(final_index_path_prefix + ".data",
                             std::ios::binary);
  new_input_data_stream << original_input_data_stream.rdbuf();
  original_input_data_stream.close();
  new_input_data_stream.close();

  // aux. file 3
  std::ofstream labels_to_medoids_writer;
  labels_to_medoids_writer.exceptions(std::ios::badbit | std::ios::failbit);
  labels_to_medoids_writer.open(final_index_path_prefix +
                                "_labels_to_medoids.txt");
  for (auto iter : entry_points)
    labels_to_medoids_writer << iter.first << ", " << iter.second << std::endl;
  labels_to_medoids_writer.close();

  // aux. file 4 (only if we're using a universal label)
  if (universal_label != UINT32_MAX) {
    std::ofstream universal_label_writer;
    universal_label_writer.exceptions(std::ios::badbit | std::ios::failbit);
    universal_label_writer.open(final_index_path_prefix +
                                "_universal_label.txt");
    universal_label_writer << universal_label << std::endl;
    universal_label_writer.close();
  }

  // main index
  _u64         index_num_frozen_points = 0, index_num_edges = 0;
  _u32         index_max_observed_degree = 0, index_entry_point = 0;
  const size_t METADATA = 2 * sizeof(_u64) + 2 * sizeof(_u32);
  for (auto &point_neighbors : stitched_graph) {
    index_max_observed_degree =
        std::max(index_max_observed_degree, (_u32) point_neighbors.size());
  }

  std::ofstream stitched_graph_writer;
  stitched_graph_writer.exceptions(std::ios::badbit | std::ios::failbit);
  stitched_graph_writer.open(final_index_path_prefix);

  stitched_graph_writer.write((char *) &final_index_size, sizeof(_u64));
  stitched_graph_writer.write((char *) &index_max_observed_degree,
                              sizeof(_u32));
  stitched_graph_writer.write((char *) &index_entry_point, sizeof(_u32));
  stitched_graph_writer.write((char *) &index_num_frozen_points, sizeof(_u64));

  size_t bytes_written = METADATA;
  for (_u32 node_point = 0; node_point < stitched_graph.size(); node_point++) {
    _u32 current_node_num_neighbors = stitched_graph[node_point].size();
    std::vector<_u32> current_node_neighbors = stitched_graph[node_point];
    stitched_graph_writer.write((char *) &current_node_num_neighbors,
                                sizeof(_u32));
    bytes_written += sizeof(_u32);
    for (const auto &current_node_neighbor : current_node_neighbors) {
      stitched_graph_writer.write((char *) &current_node_neighbor,
                                  sizeof(_u32));
      bytes_written += sizeof(_u32);
    }
    index_num_edges += current_node_num_neighbors;
  }

  if (bytes_written != final_index_size) {
    std::cerr << "Error: written bytes does not match allocated space"
              << std::endl;
    throw;
  }

  stitched_graph_writer.close();

  std::chrono::duration<double> saving_index_time =
      std::chrono::high_resolution_clock::now() - saving_index_timer;
  std::cout << "Stitched graph written in " << saving_index_time.count()
            << " seconds" << std::endl;
  std::cout << "Stitched graph average degree: "
            << ((float) index_num_edges) / ((float) (stitched_graph.size()))
            << std::endl;
  std::cout << "Stitched graph max degree: " << index_max_observed_degree
            << std::endl
            << std::endl;
}

/*
 * Unions the per-label graph indices together via the following policy:
 *  - any two nodes can only have at most one edge between them -
 *
 * Returns the "stitched" graph and its expected file size.
 */
template<typename T>
stitch_indices_return_values stitch_label_indices(
    path final_index_path_prefix, _u32 total_number_of_points,
    label_set                                all_labels,
    tsl::robin_map<label, _u32>              labels_to_number_of_points,
    tsl::robin_map<label, _u32> &            label_entry_points,
    tsl::robin_map<label, std::vector<_u32>> label_id_to_orig_id_map) {
  size_t                         final_index_size = 0;
  std::vector<std::vector<_u32>> stitched_graph(total_number_of_points);

  auto stitching_index_timer = std::chrono::high_resolution_clock::now();
  for (const auto &label : all_labels) {
    path curr_label_index_path(final_index_path_prefix + "_" +
                               std::to_string(label));
    std::vector<std::vector<_u32>> curr_label_index;
    _u64                           curr_label_index_size;
    _u32                           curr_label_entry_point;

    std::tie(curr_label_index, curr_label_index_size) = load_label_index(
        curr_label_index_path, labels_to_number_of_points[label]);
    curr_label_entry_point = random(0, curr_label_index.size());
    label_entry_points[label] =
        label_id_to_orig_id_map[label][curr_label_entry_point];

    for (_u32 node_point = 0; node_point < curr_label_index.size();
         node_point++) {
      _u32 original_point_id = label_id_to_orig_id_map[label][node_point];
      for (auto &node_neighbor : curr_label_index[node_point]) {
        _u32 original_neighbor_id =
            label_id_to_orig_id_map[label][node_neighbor];
        std::vector<_u32> curr_point_neighbors =
            stitched_graph[original_point_id];
        if (std::find(curr_point_neighbors.begin(), curr_point_neighbors.end(),
                      original_neighbor_id) == curr_point_neighbors.end()) {
          stitched_graph[original_point_id].push_back(original_neighbor_id);
          final_index_size += sizeof(_u32);
        }
      }
    }
  }

  const size_t METADATA = 2 * sizeof(_u64) + 2 * sizeof(_u32);
  final_index_size += (total_number_of_points * sizeof(_u32) + METADATA);

  std::chrono::duration<double> stitching_index_time =
      std::chrono::high_resolution_clock::now() - stitching_index_timer;
  std::cout << "stitched graph generated in memory in "
            << stitching_index_time.count() << " seconds" << std::endl;

  return std::make_tuple(stitched_graph, final_index_size);
}

/*
 * Applies the prune_neighbors function from src/index.cpp to
 * every node in the stitched graph.
 *
 * This is an optional step, hence the saving of both the full
 * and pruned graph.
 */
template<typename T>
void prune_and_save(path final_index_path_prefix, path input_data_path,
                    std::vector<std::vector<_u32>> stitched_graph,
                    unsigned                       stitched_R,
                    tsl::robin_map<label, _u32>    label_entry_points,
                    label universal_label, path label_data_path,
                    unsigned num_threads) {
  size_t dimension, number_of_label_points;
  auto   diskann_cout_buffer = diskann::cout.rdbuf(nullptr);
  auto   std_cout_buffer = std::cout.rdbuf(nullptr);
  auto   pruning_index_timer = std::chrono::high_resolution_clock::now();

  diskann::get_bin_metadata(input_data_path, number_of_label_points, dimension);
  diskann::Index<T> index(diskann::Metric::L2, dimension,
                          number_of_label_points, false, false);

  // not searching this index, set search_l to 0
  index.load(final_index_path_prefix.c_str(), num_threads, 1);

  diskann::Parameters paras;
  paras.Set<unsigned>("R", stitched_R);
  paras.Set<unsigned>(
      "C", 750);  // maximum candidate set size during pruning procedure
  paras.Set<float>("alpha", 1.2);
  paras.Set<bool>("saturate_graph", 1);
  std::cout << "parsing labels" << std::endl;

  index.prune_all_nbrs(paras);
  index.save((final_index_path_prefix + "_pruned").c_str());

  diskann::cout.rdbuf(diskann_cout_buffer);
  std::cout.rdbuf(std_cout_buffer);
  std::chrono::duration<double> pruning_index_time =
      std::chrono::high_resolution_clock::now() - pruning_index_timer;
  std::cout << "pruning performed in " << pruning_index_time.count()
            << " seconds\n"
            << std::endl;
}

/*
 * Delete all temporary artifacts.
 * In the process of creating the stitched index, some temporary artifacts are
 * created:
 * 1. the separate bin files for each labels' points
 * 2. the separate diskANN indices built for each label
 * 3. the '.data' file created while generating the indices
 */
void clean_up_artifacts(path input_data_path, path final_index_path_prefix,
                        label_set all_labels) {
  for (const auto &label : all_labels) {
    path curr_label_input_data_path(input_data_path + "_" +
                                    std::to_string(label));
    path curr_label_index_path(final_index_path_prefix + "_" +
                               std::to_string(label));
    path curr_label_index_path_data(curr_label_index_path + ".data");

    if (std::remove(curr_label_index_path.c_str()) != 0)
      throw;
    if (std::remove(curr_label_input_data_path.c_str()) != 0)
      throw;
    if (std::remove(curr_label_index_path_data.c_str()) != 0)
      throw;
  }
}

int main(int argc, char **argv) {
  // 1. handle cmdline inputs
  std::string data_type;
  path        input_data_path, final_index_path_prefix, label_data_path;
  label       universal_label;
  unsigned    num_threads, R, L, stitched_R;
  float       alpha;

  auto index_timer = std::chrono::high_resolution_clock::now();
  handle_args(argc, argv, data_type, input_data_path, final_index_path_prefix,
              label_data_path, universal_label, num_threads, R, L, stitched_R,
              alpha);

  // 2. parse label file and create necessary data structures
  std::vector<label_set>      point_ids_to_labels;
  tsl::robin_map<label, _u32> labels_to_number_of_points;
  label_set                   all_labels;

  std::tie(point_ids_to_labels, labels_to_number_of_points, all_labels) =
      parse_label_file(label_data_path, universal_label);

  // 3. for each label, make a separate data file
  tsl::robin_map<label, std::vector<_u32>> label_id_to_orig_id_map;
  _u32 total_number_of_points = point_ids_to_labels.size();

#ifndef _WINDOWS
  if (data_type == "uint8")
    label_id_to_orig_id_map = generate_label_specific_vector_files<uint8_t>(
        input_data_path, labels_to_number_of_points, point_ids_to_labels,
        all_labels);
  else if (data_type == "int8")
    label_id_to_orig_id_map = generate_label_specific_vector_files<int8_t>(
        input_data_path, labels_to_number_of_points, point_ids_to_labels,
        all_labels);
  else if (data_type == "float")
    label_id_to_orig_id_map = generate_label_specific_vector_files<float>(
        input_data_path, labels_to_number_of_points, point_ids_to_labels,
        all_labels);
  else
    throw;
#else
  if (data_type == "uint8")
    label_id_to_orig_id_map =
        generate_label_specific_vector_files_compat<uint8_t>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels,
            all_labels);
  else if (data_type == "int8")
    label_id_to_orig_id_map =
        generate_label_specific_vector_files_compat<int8_t>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels,
            all_labels);
  else if (data_type == "float")
    label_id_to_orig_id_map =
        generate_label_specific_vector_files_compat<float>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels,
            all_labels);
  else
    throw;
#endif

  // 4. for each created data file, create a vanilla diskANN index
  if (data_type == "uint8")
    generate_label_indices<uint8_t>(input_data_path, final_index_path_prefix,
                                    all_labels, R, L, alpha, num_threads);
  else if (data_type == "int8")
    generate_label_indices<int8_t>(input_data_path, final_index_path_prefix,
                                   all_labels, R, L, alpha, num_threads);
  else if (data_type == "float")
    generate_label_indices<float>(input_data_path, final_index_path_prefix,
                                  all_labels, R, L, alpha, num_threads);
  else
    throw;

  // 5. "stitch" the indices together
  std::vector<std::vector<_u32>> stitched_graph;
  tsl::robin_map<label, _u32>    label_entry_points;
  _u64                           stitched_graph_size;

  if (data_type == "uint8")
    std::tie(stitched_graph, stitched_graph_size) =
        stitch_label_indices<uint8_t>(
            final_index_path_prefix, total_number_of_points, all_labels,
            labels_to_number_of_points, label_entry_points,
            label_id_to_orig_id_map);
  else if (data_type == "int8")
    std::tie(stitched_graph, stitched_graph_size) =
        stitch_label_indices<int8_t>(
            final_index_path_prefix, total_number_of_points, all_labels,
            labels_to_number_of_points, label_entry_points,
            label_id_to_orig_id_map);
  else if (data_type == "float")
    std::tie(stitched_graph, stitched_graph_size) = stitch_label_indices<float>(
        final_index_path_prefix, total_number_of_points, all_labels,
        labels_to_number_of_points, label_entry_points,
        label_id_to_orig_id_map);
  else
    throw;

  // 5a. save the stitched graph to disk
  save_full_index(final_index_path_prefix, input_data_path, stitched_graph_size,
                  stitched_graph, label_entry_points, universal_label,
                  label_data_path);

  // 6. run a prune on the stitched index, and save to disk
  if (data_type == "uint8")
    prune_and_save<uint8_t>(final_index_path_prefix, input_data_path,
                            stitched_graph, stitched_R, label_entry_points,
                            universal_label, label_data_path, num_threads);
  else if (data_type == "int8")
    prune_and_save<int8_t>(final_index_path_prefix, input_data_path,
                           stitched_graph, stitched_R, label_entry_points,
                           universal_label, label_data_path, num_threads);
  else if (data_type == "float")
    prune_and_save<float>(final_index_path_prefix, input_data_path,
                          stitched_graph, stitched_R, label_entry_points,
                          universal_label, label_data_path, num_threads);
  else
    throw;

  std::chrono::duration<double> index_time =
      std::chrono::high_resolution_clock::now() - index_timer;
  std::cout << "pruned/stitched graph generated in " << index_time.count()
            << " seconds" << std::endl;

  clean_up_artifacts(input_data_path, final_index_path_prefix, all_labels);
}
