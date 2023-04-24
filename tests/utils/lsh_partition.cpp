// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <unordered_map>
#include <unordered_set>
#include <boost/program_options.hpp>

#include "disk_utils.h"

namespace po = boost::program_options;

template<typename T1, typename T2>
float dot_product(const T1* a, const T2* b, const size_t dim) {
  float res = 0.0;
  for (int i = 0; i < dim; ++i) {
    res += a[i] * b[i];
  }
  return res;
}

float sample_random_number(bool normal) {
  constexpr unsigned                           seed = 3500; // lucky seed
  static std::mt19937                          generator(seed);
  static std::uniform_real_distribution<float> uniform_distribution(0, 1);
  static std::normal_distribution<float>       normal_distribution(0, 1);
  if (!normal)
    return uniform_distribution(generator);
  else
    return normal_distribution(generator);
}

float sample_uniform_number() {
  return sample_random_number(false);
}

float sample_gaussian_number() {
  return sample_random_number(true);
}

std::unique_ptr<float[]> sample_gaussian_vector(size_t dim) {
  std::unique_ptr<float[]> res = std::make_unique<float[]>(dim);
  for (int i = 0; i < dim; ++i)
    res[i] = sample_gaussian_number();
  return res;
}

size_t lsh_nodes_created = 0;

template<typename T>
class LSHTreeNode {
    std::unique_ptr<float[]> direction;
    float                    offset;
    float                    width;
    std::unordered_map<int64_t, std::pair<LSHTreeNode*, size_t>> children;
    // TODO change to unique_ptr or something

    template <typename T2>
    int64_t bucket (const size_t dim, const T2* point) const {
      float dp = dot_product(point, direction.get(), dim);
      return static_cast<int64_t>((offset + dp) / width); // round down
    }

    public:

    // will mess up ids_to_partition (maybe should pass as && to make it clearer)
    void build(std::vector<uint32_t>& ids_to_partition, const size_t dim,
                const T* points, std::vector<std::vector<uint32_t>>& pieces,
        const size_t max_piece_size, const float _width) {

      width = _width;

      // offset = random point in (0,width)
      offset = width * sample_uniform_number();

      // direction = random point on unit sphere
      direction = sample_gaussian_vector(dim);
      float norm = sqrt(dot_product(direction.get(), direction.get(), dim));
      for (int i = 0; i < dim; ++i) {
        direction[i] /= norm;
      }

      std::unordered_map<int64_t, std::vector<uint32_t>> initial_partition;
      for (const uint32_t point_id : ids_to_partition) {
        const int64_t bucket_id = bucket(dim, points + point_id * dim);
        initial_partition[bucket_id].push_back(point_id);
      }

      // release memory for ids_to_partition
      std::vector<uint32_t>().swap(ids_to_partition);

      constexpr float width_multiplier = 0.8;

      for (auto& p : initial_partition) {
        if (p.second.size() <= max_piece_size) {
          const size_t piece_id = pieces.size();
          pieces.emplace_back(std::move(p.second));
          children.emplace(p.first, std::make_pair(nullptr, piece_id));
        } else {

          // optimization: contract nodes with just one child (trivial partitions)
          if (initial_partition.size() == 1) {
            build(p.second, dim, points, pieces, max_piece_size,
                  width * width_multiplier);
            return; // w/o incrementing lsh_nodes_created
          }

          LSHTreeNode* child = new LSHTreeNode;
          child->build(p.second, dim, points, pieces, max_piece_size,
                       width * width_multiplier);
          children.emplace(p.first, std::make_pair(child, 0));
        }
      }
      lsh_nodes_created++;
    }

    // returns -1 if it would get routed to an empty piece
    // (maybe this should be done differently)
    template <typename T2>
    int64_t route_to_piece(const size_t dim, const T2* point) const {
      const int64_t bucket_id = bucket(dim, point);
      auto          it = children.find(bucket_id);
      if (it == children.end()) {
        return -1;
      } else if (it->second.first == nullptr) {
        return it->second.second;
      } else {
        return it->second.first->route_to_piece(dim, point);
      }
    }
};

std::vector<size_t> bin_packing(
    const std::vector<std::vector<uint32_t>>& pieces, size_t max_shard_size) {
    // simple greedy sequential bin-packing for now
    std::vector<size_t> piece_to_shard(pieces.size());
    if (pieces.empty())
      return piece_to_shard;
    piece_to_shard[0] = 0;
    size_t current_size = pieces[0].size();
    for (int i = 1; i < pieces.size(); ++i) {
        // can we assign i to current shard?
      current_size += pieces[i].size();
      if (current_size <= max_shard_size) {
        // yes
        piece_to_shard[i] = piece_to_shard[i - 1];
      } else {
        // no
        current_size = pieces[i].size();
        piece_to_shard[i] = piece_to_shard[i - 1] + 1;
      }
    }
    return piece_to_shard;
}

template <typename T>
int write_shards_to_disk(const std::string& output_file_prefix, const size_t num_shards,
                         const bool writing_queries, T* points, const size_t dim,
                         const std::vector<std::vector<uint32_t>>& pieces,
                         const std::vector<size_t>& piece_to_shard,
                         const bool write_hmetis_file) {
    std::unique_ptr<size_t[]> shard_counts =
        std::make_unique<size_t[]>(num_shards);
    std::vector<std::ofstream> shard_data_writer(num_shards);
    std::vector<std::ofstream> shard_idmap_writer(num_shards);
    const uint32_t             dim32 = dim;
    const uint32_t             dummy_size = 0;
    const uint32_t             const_one = 1;
    for (size_t i = 0; i < num_shards; ++i) {
      const std::string data_filename =
          output_file_prefix + "_subshard-" + std::to_string(i) + ".bin";
      const std::string idmap_filename =
          output_file_prefix + "_subshard-" + std::to_string(i) +
          (writing_queries ? "_query" : "") + "_ids_uint32.bin";
      if (!writing_queries)
        shard_data_writer[i] =
            std::ofstream(data_filename.c_str(), std::ios::binary);
      shard_idmap_writer[i] =
          std::ofstream(idmap_filename.c_str(), std::ios::binary);
      if (shard_data_writer[i].fail() || shard_idmap_writer[i].fail()) {
        diskann::cout << "Error: failed to open shard file for writing. Check "
                         "limit for max number for open files (on Linux, run "
                         "ulimit -n to check and ulimit -n 12000 to set)"
                      << std::endl;
        return -1;
      }
      if (!writing_queries) {
        shard_data_writer[i].write((char*) &dummy_size, sizeof(uint32_t));
        shard_data_writer[i].write((char*) &dim32, sizeof(uint32_t));
      }
      shard_idmap_writer[i].write((char*) &dummy_size, sizeof(uint32_t));
      shard_idmap_writer[i].write((char*) &const_one, sizeof(uint32_t));
    }
    
    // for write_hmetis_file
    std::unordered_map<size_t, int> shard_of_point;

    for (size_t piece_id = 0; piece_id < pieces.size(); ++piece_id) {
      const size_t shard_id = piece_to_shard[piece_id];
      if (!writing_queries) {
        for (const size_t point_id : pieces[piece_id]) {
          // write point
          shard_data_writer[shard_id].write((char*) (points + point_id * dim),
                                            sizeof(T) * dim);
        }
      }
      if (write_hmetis_file) {
        for (const size_t point_id : pieces[piece_id]) {
          shard_of_point[point_id] = shard_id;
        }
      }
      // write ids
      shard_idmap_writer[shard_id].write(
          (char*) pieces[piece_id].data(),
          sizeof(uint32_t) * pieces[piece_id].size());
      shard_counts[shard_id] += pieces[piece_id].size();
    }

    size_t total_count = 0;
    if (writing_queries)
      diskann::cout << "Queries: ";
    diskann::cout << "Actual shard sizes: " << std::flush;
    for (size_t i = 0; i < num_shards; ++i) {
      uint32_t cur_shard_count = (uint32_t) shard_counts[i];
      total_count += cur_shard_count;
      diskann::cout << cur_shard_count << " ";
      if (!writing_queries) {
        shard_data_writer[i].seekp(0);
        shard_data_writer[i].write((char*) &cur_shard_count, sizeof(uint32_t));
        shard_data_writer[i].close();
      }
      shard_idmap_writer[i].seekp(0);
      shard_idmap_writer[i].write((char*) &cur_shard_count, sizeof(uint32_t));
      shard_idmap_writer[i].close();
    }
    diskann::cout << "Total count: " << total_count << std::endl;

    if (write_hmetis_file) {
      std::string   hmetis_filename = output_file_prefix + "_partition.hmetis";
      std::ofstream hmetis(hmetis_filename);
      diskann::cout << "writing .hmetis file..." << std::endl;
      size_t next_expected_id = 0;
      for (const std::pair<size_t, int>& p : shard_of_point) {
        if (p.first != next_expected_id) {
          diskann::cout << "the partitioned points are not contiguous?"
                        << std::endl;
          return -1;
        }
        ++next_expected_id;
		hmetis << p.second << std::endl;
	  }
      diskann::cout << "done writing .hmetis file." << std::endl;
    }

    return 0;
}


template <typename T>
int aux_main(const std::string &input_file,
             const std::string &output_file_prefix,
             const std::string& query_file,
             const unsigned max_shard_size,
             const unsigned query_fanout,
             const bool write_hmetis_file) {

    // load dataset
    // TODO for later: handle datasets that don't fit in memory
    // (maybe first do LSH on a subsampled subset)
    size_t num_points, dim;
    std::unique_ptr<T[]> points;
    diskann::cout << "Reading the dataset..." << std::endl;
    diskann::load_bin<T>(input_file, points, num_points, dim);

    std::vector<uint32_t> all_ids(num_points);
    for (uint32_t i = 0; i < num_points; ++i) {
      all_ids[i] = i;
    }

    std::vector<std::vector<uint32_t>> pieces;
    LSHTreeNode<T>                     lsh_tree;
    constexpr float                    initial_width = 1e30;
    // run the LSH partitioning into pieces
    lsh_tree.build(all_ids, dim, points.get(), pieces, max_shard_size, initial_width);
    diskann::cout << lsh_nodes_created << " LSH tree nodes created"
                  << std::endl;

    // bin-pack pieces into shards
    diskann::cout << "LSH partitioning finished. Now bin-packing pieces into shards..." << std::endl;
    const std::vector<size_t> piece_to_shard = bin_packing(pieces, max_shard_size);
    const size_t num_shards = piece_to_shard.back() + 1;

    // write shards to disk
    diskann::cout << "Writing shards to disk..." << std::endl;
    int ret = write_shards_to_disk<T>(output_file_prefix, num_shards, false,
                                      points.get(), dim, pieces, piece_to_shard,
                                      write_hmetis_file);
    if (ret != 0)
      return ret;

    if (query_file != "") {
      // also partition the query set
      if (query_fanout > num_shards) {
        diskann::cout << "query fanout is larger than number of shards"
                      << std::endl;
        return -1;
      }

      size_t num_queries, query_dim;
      std::unique_ptr<T[]> queries;
      diskann::cout << "Reading the query set..." << std::endl;
      diskann::load_bin<T>(query_file, queries, num_queries, query_dim);
      if (query_dim != dim) {
        diskann::cout << "dimension mismatch between dataset and query file"
                      << std::endl;
        return -1;
      }

      std::vector<std::vector<uint32_t>> query_pieces(pieces.size());
      for (size_t query_id = 0; query_id < num_queries; ++query_id) {
        if (query_id % 100 == 0)
          diskann::cout << "query_id = " << query_id << std::endl;
        // a variant of multi-probe LSH:
        // until you have `query_fanout` different shards,
        // add some random error to the query, and route to a shard

        // TODO: replace with something that routes a query to *at most*
        // `query_fanout` different shards (but then, would need to have some
        // reasonable guess on the error magnitude)

        float error_magnitude = 0.0; // first we ask the exact query point
        std::unordered_set<size_t> shards_for_this_query;
        std::unique_ptr<float[]> noised_query = std::make_unique<float[]>(dim);
        while (shards_for_this_query.size() < query_fanout) {
          for (int d = 0; d < dim; ++d) {
            noised_query[d] = (float) queries[query_id * dim + d] +
                              sample_gaussian_number() * error_magnitude;
          }
          const int64_t piece_id =
              lsh_tree.route_to_piece(dim, noised_query.get());
          if (piece_id != -1) {
            const size_t shard_id = piece_to_shard[piece_id];
            // is this a new shard?
            if (shards_for_this_query.insert(shard_id).second == true) {
              query_pieces[piece_id].push_back(query_id);
            }
          }
          // increase magnitude of error for the next try
          if (error_magnitude == 0.0) {
            error_magnitude = 1e-15; // some initial value
          } else {
            if (error_magnitude < 1e-1)
              error_magnitude *= 2.0;
            else
              error_magnitude *= 1.01;
            if (error_magnitude > 1e15) {
              // time to give up
              break;
            }
          }
        }
      }

      // write routed queries to disk
      diskann::cout << "Writing query assignments to disk..." << std::endl;
      int ret =
          write_shards_to_disk<T>(output_file_prefix, num_shards, true, nullptr,
                                  dim, query_pieces, piece_to_shard, false);
      if (ret != 0)
        return ret;
    }

    diskann::cout << "Produced " << num_shards << " shards" << std::endl;
    return 0;
}

// Applies a partitioning scheme inspired by LSH to a dataset.
// Can also partition a query set.
//
// Output files will be: output_file_prefix_subshard-X.bin
//                   and output_file_prefix_subshard-X_ids_uint32.bin
//                   and output_file_prefix_subshard-X_query_ids_uint32.bin (optionally)
// where X = 0,1,2,...

int main(int argc, char** argv) {
  std::string input_file, output_file_prefix, query_file;
  unsigned max_shard_size, query_fanout;
  bool        write_hmetis_file;

  std::string data_type;

  po::options_description desc{ "Arguments" };
  try {
      desc.add_options()("help,h", "Print information on arguments");
      desc.add_options()("data_type",
                         po::value<std::string>(&data_type)->required(),
                         "data type <int8/uint8/float>");
      desc.add_options()("input_file",
                         po::value<std::string>(&input_file)->required(),
                         "Path to the dataset .bin file");
      desc.add_options()(
          "query_file",
          po::value<std::string>(&query_file)->default_value(std::string("")),
          "Path to the query .bin file (optional)");
      desc.add_options()(
          "output_file_prefix",
          po::value<std::string>(&output_file_prefix)->required(),
          "Output file prefix. Will generate files like this_subshard-0.bin "
          "and "
          "this_subshard-0_ids_uint32.bin");
      desc.add_options()(
          "max_shard_size", po::value<unsigned>(&max_shard_size)->required(),
          "Maximum allowed size (number of points) of each shard");
      desc.add_options()("query_fanout",
                         po::value<unsigned>(&query_fanout)->default_value(0),
                         "The fanout of each query (multi-probe LSH)");
      desc.add_options()(
          "write_hmetis_file",
          po::value<bool>(&write_hmetis_file)->default_value(false),
          "Also output the partition as a .hmetis format file (optional)");
    
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

  if (query_file != "" && query_fanout == 0) {
    diskann::cout
        << "query_fanout must be given if a query file is to be partitioned"
        << std::endl;
    return -1;
  }

  try {
    if (data_type == std::string("float")) {
      return aux_main<float>(input_file, output_file_prefix, query_file,
                             max_shard_size, query_fanout, write_hmetis_file);
    } else if (data_type == std::string("int8")) {
      return aux_main<int8_t>(input_file, output_file_prefix, query_file,
                              max_shard_size, query_fanout, write_hmetis_file);
    } else if (data_type == std::string("uint8")) {
      return aux_main<uint8_t>(input_file, output_file_prefix, query_file,
                               max_shard_size, query_fanout, write_hmetis_file);
    } else {
      std::cerr << "Unsupported data type. Use float or int8 or uint8"
                << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    std::cerr << "Partitioning failed." << std::endl;
    return -1;
  }
}
