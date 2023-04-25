// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include "math_utils.h"
#include <unordered_map>
#include <unordered_set>
#include <boost/program_options.hpp>

#include "disk_utils.h"

namespace po = boost::program_options;

void assign_junk(const size_t dim, float* centroid) {
  for (int i = 0; i < dim; ++i) {
	centroid[i] = 1e15;  // give it some stupid centroid
  }
}

template<typename T>
void compute_centroid(const size_t dim, const T* points,
                      const std::vector<uint32_t>& point_ids, float* centroid) {
  if (point_ids.empty()) {
    assign_junk(dim, centroid);
  } else {
    for (int i = 0; i < dim; ++i) {
      centroid[i] = 0.0;
    }
    for (const uint32_t point_id : point_ids) {
      for (int i = 0; i < dim; ++i) {
        centroid[i] += points[point_id * dim + i];
      }
    }
    for (int i = 0; i < dim; ++i) {
      centroid[i] /= point_ids.size();
    }
  }
}

template<typename T>
void compute_subcentroids(const size_t dim, const T* points,
                          const std::vector<uint32_t>& point_ids,
                          float* subcentroids, const unsigned num_subcentroids) {
  if (point_ids.empty()) {
    for (int i = 0; i < num_subcentroids; ++i) {
      assign_junk(dim, subcentroids + i * dim);
    }
  } else {
    // run kmeans
    std::unique_ptr<float[]> train_data_float =
        std::make_unique<float[]>(point_ids.size() * dim);
    for (size_t i = 0; i < point_ids.size(); ++i) {
      for (int j = 0; j < dim; ++j) {
		train_data_float[i * dim + j] = points[point_ids[i] * dim + j];
	  }
    }

    // hope it won't crash if point_ids.size() < num_subcentroids...

    kmeans::kmeanspp_selecting_pivots(train_data_float.get(), point_ids.size(),
                                      dim, subcentroids, num_subcentroids);

    constexpr size_t max_reps = 15;
    kmeans::run_lloyds(train_data_float.get(), point_ids.size(), dim,
                       subcentroids, num_subcentroids, max_reps, NULL, NULL);
  }
}

template<typename T>
void compute_geomedian(const size_t dim, const T* points,
                      const std::vector<uint32_t>& point_ids, float* geomedian) {
  if (point_ids.empty()) {
    assign_junk(dim, geomedian);
  } else if (point_ids.size() == 1) {
    for (int i = 0; i < dim; ++i) {
	  geomedian[i] = points[point_ids[0] * dim + i];
	}
  } else { // at least two points
    // Weiszfeld algorithm
    std::unique_ptr<float[]> second_vector = std::make_unique<float[]>(dim);
    float*                   A[2] = {geomedian, second_vector.get()};
    constexpr int            iterations = 100; // must be even

    // initialization
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < dim; ++i) {
        A[j][i] =
            (points[point_ids[0] * dim + i] + points[point_ids[1] * dim + i]) /
            2;
      } 
    }

    for (int iteration = 0; iteration < iterations; ++iteration) {
      std::vector<float> numerator(dim, 0.0);
      float              denominator = 0.0;
      for (const uint32_t point_id : point_ids) {
		float dist = 0.0;
        for (int i = 0; i < dim; ++i) {
		  dist += (points[point_id * dim + i] - A[iteration % 2][i]) *
				  (points[point_id * dim + i] - A[iteration % 2][i]);
		}
		dist = sqrt(dist);
        if (dist > 1e-9) {
          for (int i = 0; i < dim; ++i) {
			numerator[i] += points[point_id * dim + i] / dist;
		  }
		  denominator += 1.0 / dist;
		}
	  }
      for (int i = 0; i < dim; ++i) {
        A[1 - iteration % 2][i] = numerator[i] / denominator;
      }
    }
  }
}

template<typename T>
int write_shards_to_disk(const std::string& output_file_prefix,
                         const bool writing_queries, T* points, const size_t dim,
                         const std::vector<std::vector<uint32_t>>& points_routed_to_shard) {
    const size_t              num_shards = points_routed_to_shard.size();
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

    for (size_t shard_id = 0; shard_id < num_shards; ++shard_id) {
      if (!writing_queries) {
        for (const uint32_t point_id : points_routed_to_shard[shard_id]) {
          // write point
          shard_data_writer[shard_id].write((char*) (points + point_id * dim),
                                            sizeof(T) * dim);
        }
      }
      // write ids
      shard_idmap_writer[shard_id].write(
          (char*) points_routed_to_shard[shard_id].data(),
          sizeof(uint32_t) * points_routed_to_shard[shard_id].size());
      shard_counts[shard_id] += points_routed_to_shard[shard_id].size();
	}

    /*for (uint32_t point_id = 0; point_id < shard_of_point.size();
              ++point_id) {
	  const size_t shard_id = shard_of_point[point_id];
      if (!writing_queries) {
		// write point
		shard_data_writer[shard_id].write((char*) (points + point_id * dim),
            										  sizeof(T) * dim);
	  }
	  // write id
	  shard_idmap_writer[shard_id].write((char*) &point_id, sizeof(uint32_t));
	  shard_counts[shard_id]++;
	}*/

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
    return 0;
}


template <typename T>
int aux_main(const std::string &input_file,
             const std::string &output_file_prefix,
             const std::string& query_file, const std::string& gt_file,
             const std::string& hmetis_file, const std::string& mode,
             const unsigned K, const unsigned query_fanout,
             const unsigned num_subcentroids) {

    // load dataset
    // TODO for later: handle datasets that don't fit in memory
    size_t num_points, dim;
    std::unique_ptr<T[]> points;
    diskann::cout << "Reading the dataset..." << std::endl;
    diskann::load_bin<T>(input_file, points, num_points, dim);


    // load hmetis partitioning
    std::ifstream hmetis(hmetis_file);
    if (hmetis.fail()) {
	  diskann::cout << "Error: failed to open hmetis file for reading"
					<< std::endl;
	  return -1;
	}
    size_t                num_shards = 0;
    std::vector<size_t> shard_of_point;
    constexpr size_t      definitely_no_more_shards_than_this = 10'000;
    for (size_t i = 0; i < num_points; ++i) {
      size_t shard_id;
	  hmetis >> shard_id;
      if (shard_id < 0 || shard_id >= definitely_no_more_shards_than_this) {
        diskann::cout << "Error: hmetis file contains invalid shard id"
				  << std::endl;
		return -1;
      }
      num_shards = std::max(num_shards, shard_id + 1);
	  shard_of_point.push_back(shard_id); // metis partition IDs are 0-based!
    }
    std::vector<std::vector<uint32_t>> points_routed_to_shard(num_shards);
    for (size_t point_id = 0; point_id < num_points; ++point_id) {
	  points_routed_to_shard[shard_of_point[point_id]].push_back(point_id);
	}


    // write shards to disk
    diskann::cout << "Writing shards to disk..." << std::endl;
    int ret = write_shards_to_disk<T>(output_file_prefix, false, points.get(),
                                      dim, points_routed_to_shard);
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

      // compute centroids/geomedians for each shard
      std::unique_ptr<float[]> centroids =
          std::make_unique<float[]>(num_shards * dim);
      for (size_t shard_id = 0; shard_id < num_shards; ++shard_id) {
        if (mode == "geomedian") {
          compute_geomedian<T>(dim, points.get(),
                               points_routed_to_shard[shard_id],
                               centroids.get() + shard_id * dim);
        } else {
          // mode can be from_ground_truth or centroid; in both cases we save centroids
          compute_centroid<T>(dim, points.get(),
                              points_routed_to_shard[shard_id],
                              centroids.get() + shard_id * dim);
        }
      }
      if (mode == "geomedian") {
        diskann::cout << "Saving geomedians (as _centroids.bin)" << std::endl;
      } else {
        // mode can be from_ground_truth or centroid; in both cases we save
        // centroids
        diskann::cout << "Saving centroids" << std::endl;
      }
      const std::string centroids_filename =
          output_file_prefix + "_centroids.bin";
      diskann::save_bin<float>(centroids_filename, centroids.get(), num_shards,
                               dim);
      // done computing centroids/geomedians

      std::unique_ptr<float[]> subcentroids;
      if (mode == "multicentroids") {
        subcentroids =
            std::make_unique<float[]>(num_shards * num_subcentroids * dim);
        for (size_t shard_id = 0; shard_id < num_shards; ++shard_id) {
          compute_subcentroids<T>(
              dim, points.get(), points_routed_to_shard[shard_id],
              subcentroids.get() + shard_id * num_subcentroids * dim,
              num_subcentroids);
        }
        diskann::cout << "computed subcentroids" << std::endl;
      }
      // subcentroids do not get saved to a file

      std::vector<std::unordered_map<size_t, size_t>> shard_to_count_of_GT_pts(
          num_queries);
      if (gt_file != "") {
        // load ground truth
        uint32_t* gt = nullptr;
        float*    dists = nullptr;
        size_t    num_queries, gt_dim;
        diskann::load_truthset(gt_file, gt, dists, num_queries, gt_dim);
        if (dists != nullptr)
          delete[] dists;

        if (gt_dim < K) {
          std::cout << "Ground truth dimension " << gt_dim << " smaller than K "
                    << K << std::endl;
          return -1;
        }
        // ground truth loaded

        for (size_t query_id = 0; query_id < num_queries; ++query_id) {
          for (size_t gt_id = 0; gt_id < K; ++gt_id) {
            size_t gt_point_id = gt[query_id * gt_dim + gt_id];
            size_t gt_shard_id = shard_of_point[gt_point_id];
            shard_to_count_of_GT_pts[query_id][gt_shard_id]++;
          }
        }

        delete[] gt;
      } // else: shard_to_count_of_GT_pts[query_id] will be empty

      std::vector<std::vector<std::pair<size_t, size_t>>> query_to_shards;
      // query_to_shards[query_id] is a vector of pairs (shard_id, # GT points
      // in shard) sorted by the order (preference) in which the shards will be
      // asked (the # GT points can be 0 if the order is suboptimal)

      // fill query_to_shards[]
      if (mode == "from_ground_truth") {
        for (size_t query_id = 0; query_id < num_queries; ++query_id) {
          query_to_shards.emplace_back(
              shard_to_count_of_GT_pts[query_id].begin(),
              shard_to_count_of_GT_pts[query_id].end());

          // sort query_to_shards[query_id] by decreasing second
          std::sort(query_to_shards[query_id].begin(),
                    query_to_shards[query_id].end(),
                    [](const std::pair<size_t, size_t>& a,
                       const std::pair<size_t, size_t>& b) {
                      return a.second > b.second;
                    });
        }
        diskann::cout << "Computed the query -> shard assignment using ground "
                         "truth (optimistically)"
                      << std::endl;
      } else if (mode == "centroids" || mode == "geomedian") {
        std::unique_ptr<float[]> queries_float =
            std::make_unique<float[]>(num_queries * dim);
        diskann::convert_types<T, float>(queries.get(), queries_float.get(),
                                         num_queries, dim);
        // need to order all shards if we want to print statistics
        // using the GT, otherwise query_fanout will be enough
        const size_t num_shards_to_order =
            (gt_file != "") ? num_shards : query_fanout;
        std::unique_ptr<uint32_t[]> closest_centroids_ivf =
            std::make_unique<uint32_t[]>(num_queries * num_shards_to_order);
        math_utils::compute_closest_centers(
            queries_float.get(), num_queries, dim, centroids.get(), num_shards,
            num_shards_to_order, closest_centroids_ivf.get());
        for (size_t query_id = 0; query_id < num_queries; ++query_id) {
          query_to_shards.emplace_back();
          for (int i = 0; i < num_shards_to_order; ++i) {
            const size_t shard_id =
                closest_centroids_ivf[query_id * num_shards_to_order + i];
            query_to_shards[query_id].emplace_back(
                shard_id, shard_to_count_of_GT_pts[query_id][shard_id]);
            // shard_to_count_of_GT_pts[query_id][shard_id] will be(come) 0 if
            // wasn't present
          }
        }

        diskann::cout << "Computed the query -> shard assignment using "
                         "approximation by centroids"
                      << std::endl;
      } else if (mode == "multicentroids") {

        std::unique_ptr<float[]> queries_float =
            std::make_unique<float[]>(num_queries * dim);
        diskann::convert_types<T, float>(queries.get(), queries_float.get(),
                                         num_queries, dim);
        const size_t num_subcenters = num_shards * num_subcentroids;
        std::unique_ptr<uint32_t[]> closest_centroids_ivf =
            std::make_unique<uint32_t[]>(num_queries * num_subcenters);
        math_utils::compute_closest_centers(
            queries_float.get(), num_queries, dim, subcentroids.get(),
            num_subcenters, num_subcenters, closest_centroids_ivf.get());
        for (size_t query_id = 0; query_id < num_queries; ++query_id) {
          query_to_shards.emplace_back();
          std::unordered_set<size_t> seen_shards;
          for (int i = 0; i < num_subcenters; ++i) {
            const size_t shard_id =
                closest_centroids_ivf[query_id * num_subcenters + i] /
                num_subcentroids;
            if (seen_shards.insert(shard_id).second == true) {
              query_to_shards[query_id].emplace_back(
                  shard_id, shard_to_count_of_GT_pts[query_id][shard_id]);
              // shard_to_count_of_GT_pts[query_id][shard_id] will be(come) 0 if
              // wasn't present
            }
          }
        }

        diskann::cout << "Computed the query -> shard assignment using "
                         "approximation by MULTIcentroids"
                      << std::endl;

      } else {
        diskann::cout << "unsupported mode?" << std::endl;
        return -1;
      }
      // filled query_to_shards[]


      if (gt_file != "") {
        // compute and display some statistics
        diskann::cout << "\nStatistics on fanout:" << std::endl;

        // fanout means: how many shards, in the order that they will be asked
        // (maybe suboptimal), do you need to ask to get 100% coverage?

        // so, to compute this, we first adjust query_to_shards[]
        // because there could be trailing empty shards
        for (size_t query_id = 0; query_id < num_queries; ++query_id) {
          if (query_to_shards[query_id].empty()) {
            diskann::cout << "implementation error?" << std::endl;
            return -1;
          }
          while (query_to_shards[query_id].back().second == 0) {
            query_to_shards[query_id].pop_back();
          }
        }
        // now we have: fanout == query_to_shards[query_id].size()

        // 1. average fanout
        float avg_fanout = 0.0;
        for (size_t query_id = 0; query_id < num_queries; ++query_id) {
          avg_fanout += query_to_shards[query_id].size();
        }
        avg_fanout /= num_queries;
        diskann::cout << "Average fanout: " << avg_fanout << std::endl
                      << std::endl;

        // 1.5. "weighted average fanout"
        float weighted_avg_fanout = 0.0;
        for (size_t query_id = 0; query_id < num_queries; ++query_id) {
          for (int i = 0; i < query_to_shards[query_id].size(); ++i) {
            weighted_avg_fanout +=
                (uint64_t) i * query_to_shards[query_id][i].second;
          }
        }
        weighted_avg_fanout /= num_queries * K;
        diskann::cout << "\"Weighted average\" fanout: " << weighted_avg_fanout
                      << std::endl
                      << std::endl;

        // 2. histogram of fanouts
        const size_t max_interesting_fanout =
            (mode == "from_ground_truth") ? K : 1.5 * K;
        std::vector<size_t> num_queries_with_fanout(max_interesting_fanout + 1,
                                                    0);
        for (size_t query_id = 0; query_id < num_queries; ++query_id) {
          num_queries_with_fanout[std::min(query_to_shards[query_id].size(),
                                           max_interesting_fanout)]++;
        }
        diskann::cout << "Histogram of fanouts:" << std::endl;
        for (size_t fanout = 1; fanout <= max_interesting_fanout; ++fanout) {
          diskann::cout << std::setw(2) << fanout;
          if (fanout < max_interesting_fanout)
            diskann::cout << " ";
          else
            diskann::cout << "+";
          diskann::cout << " -- " << std::setprecision(2) << std::fixed
                        << 100.0 * num_queries_with_fanout[fanout] / num_queries
                        << "%\n";
        }
        diskann::cout << std::endl;

        // 3. for F=1,2,..., what recall if taking top F shards
        diskann::cout << "Coverage (best possible recall) if taking top F "
                         "shards for every query:"
                      << std::endl;
        std::vector<size_t> coverage_of_query(num_queries, 0);
        for (size_t fanout = 1; fanout <= max_interesting_fanout; ++fanout) {
          size_t total_recalled_points = 0;
          // take the fanout-th shard for every query
          for (size_t query_id = 0; query_id < num_queries; ++query_id) {
            if (query_to_shards[query_id].size() >= fanout) {
              coverage_of_query[query_id] +=
                  query_to_shards[query_id][fanout - 1].second;
            }
            total_recalled_points += coverage_of_query[query_id];
          }
          diskann::cout << std::setw(2) << fanout << " -- "
                        << std::setprecision(2) << std::fixed
                        << 100.0 * total_recalled_points / (K * num_queries)
                        << "%\n";
        }
        diskann::cout << std::endl;
        // done computing statistics
      }      

      // now we route queries
      std::vector<std::vector<uint32_t>> queries_routed_to_shard(num_shards);
      // we have computed query_to_shards[] above
      // now, each query goes to the top `query_fanout` many shards
      for (size_t query_id = 0; query_id < num_queries; ++query_id) {
        for (size_t j = 0;
             j < query_fanout && j < query_to_shards[query_id].size(); ++j) {
          const size_t shard_id = query_to_shards[query_id][j].first;
          queries_routed_to_shard[shard_id].push_back(query_id);
        }
      }      

      // write routed queries to disk
      diskann::cout << "Writing query assignments to disk..." << std::endl;
      int ret = write_shards_to_disk<T>(output_file_prefix, true,
                                        nullptr, dim, queries_routed_to_shard);
      if (ret != 0)
        return ret;
    }

    diskann::cout << "Produced " << num_shards << " shards" << std::endl;
    return 0;
}

// Applies the partitioning computed by a hypergraph partitioner.
// Can also partition a query set.
//
// Output files will be: output_file_prefix_subshard-X.bin
//                   and output_file_prefix_subshard-X_ids_uint32.bin
//                   and output_file_prefix_subshard-X_query_ids_uint32.bin (optionally)
// where X = 0,1,2,...
// and also output_file_prefix_centroids.bin

int main(int argc, char** argv) {
  std::string input_file, output_file_prefix, query_file, gt_file, hmetis_file, mode;
  unsigned K, query_fanout, num_subcentroids;

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
      desc.add_options()("hmetis_file",
                         po::value<std::string>(&hmetis_file)->required(),
                         "Path to the hmetis file (where i-th line = 1-based "
                         "partition ID of point i)");
      desc.add_options()(
          "gt_file",
          po::value<std::string>(&gt_file)->default_value(std::string("")),
          "Path to the ground truth .bin file (optional)");
      desc.add_options()("mode,query_routing_mode",
                         po::value<std::string>(&mode)->default_value(
                             std::string("centroids")),
                         "How to route queries to shards (from_ground_truth / "
                         "centroids / multicentroids / geomedian)");
      desc.add_options()("K,recall_at", po::value<unsigned>(&K)->default_value(0),
                         "Number of points returned per query");
      desc.add_options()(
          "query_file",
          po::value<std::string>(&query_file)->default_value(std::string("")),
          "Path to the query .bin file (optional)");
      desc.add_options()(
          "output_file_prefix",
          po::value<std::string>(&output_file_prefix)->required(),
          "Output file prefix. Will generate files like this_subshard-0.bin "
          "and this_subshard-0_ids_uint32.bin");
      desc.add_options()("query_fanout",
                         po::value<unsigned>(&query_fanout)->default_value(0),
                         "The fanout of each query");
      desc.add_options()("num_subcentroids",
                         po::value<unsigned>(&num_subcentroids)->default_value(0),
                         "The number of subcentroids (for multicentroids mode)");
    
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

  if (mode != "centroids" && mode != "multicentroids" && mode != "geomedian" &&
      mode != "from_ground_truth") {
    diskann::cout << "mode must be centroids, multicentroids, geomedian or "
                     "from_ground_truth"
                  << std::endl;
    return -1;
  }

  if (mode == "from_ground_truth" && gt_file == "") {
    diskann::cout
        << "using from_ground_truth mode but no ground truth file given"
        << std::endl;
    return -1;
  }

  if (gt_file != "" && K == 0) {
    diskann::cout << "if ground truth given, must also specify K" << std::endl;
    return -1;
  }

  if (mode == "multicentroids" && num_subcentroids == 0) {
    diskann::cout << "if multicentroids mode, must specify num_subcentroids"
                  << std::endl;
	return -1;
  }

  try {
    if (data_type == std::string("float")) {
      return aux_main<float>(input_file, output_file_prefix, query_file,
                             gt_file, hmetis_file, mode, K, query_fanout,
                             num_subcentroids);
    } else if (data_type == std::string("int8")) {
      return aux_main<int8_t>(input_file, output_file_prefix, query_file,
                              gt_file, hmetis_file, mode, K, query_fanout,
                              num_subcentroids);
    } else if (data_type == std::string("uint8")) {
      return aux_main<uint8_t>(input_file, output_file_prefix, query_file,
                               gt_file, hmetis_file, mode, K, query_fanout,
                               num_subcentroids);
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
