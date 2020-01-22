#include "utils.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
//#include <parallel/algorithm>
#include <string>
#include <vector>
#include <set>
#include "cached_io.h"
#include "utils.h"
#include <boost/dynamic_bitset.hpp>
#include <set>

namespace diskann {
  float calc_recall_set(unsigned num_queries, unsigned *gold_std,
                        float *gs_dist, unsigned dim_gs, unsigned *our_results,
                        unsigned dim_or, unsigned recall_at) {
    unsigned           total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t    tie_breaker = recall_at;
      if (gs_dist != nullptr) {
        tie_breaker = recall_at - 1;
        float *gt_dist_vec = gs_dist + dim_gs * i;
        while (tie_breaker < dim_gs &&
               gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);
      unsigned cur_recall = 0;
      for (auto &v : gt) {
        if (res.find(v) != res.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return ((float) total_recall) / ((float) num_queries) *
           (100.0 / ((float) recall_at));
  }

  /***************************************************
      Support for Merging Many Vamana Indices
   ***************************************************/

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs) {
    uint32_t      npts32, dim;
    size_t        actual_file_size = get_file_size(fname);
    std::ifstream reader(fname.c_str(), std::ios::binary);
    reader.read((char *) &npts32, sizeof(uint32_t));
    reader.read((char *) &dim, sizeof(uint32_t));
    if (dim != 1 ||
        actual_file_size !=
            ((size_t) npts32) * sizeof(uint32_t) + 2 * sizeof(uint32_t)) {
      std::cout
          << "Error reading idmap file. Check if the file is bin file with "
             "1 dimensional data. Actual: "
          << actual_file_size
          << ", expected: " << (size_t) npts32 + 2 * sizeof(uint32_t)
          << std::endl;
      exit(-1);
    }
    ivecs.resize(npts32);
    reader.read((char *) ivecs.data(), ((size_t) npts32) * sizeof(uint32_t));
    reader.close();
  }

  int merge_shards(const std::string &nsg_prefix, const std::string &nsg_suffix,
                   const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards,
                   const std::string &output_nsg, unsigned max_degree) {
    // Read ID maps
    std::vector<std::string>           nsg_names(nshards);
    std::vector<std::vector<unsigned>> idmaps(nshards);
    for (_u64 shard = 0; shard < nshards; shard++) {
      nsg_names[shard] = nsg_prefix + std::to_string(shard) + nsg_suffix;
      read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix,
                 idmaps[shard]);
    }

    // find max node id
    _u64 nnodes = 0;
    _u64 nelems = 0;
    for (auto &idmap : idmaps) {
      for (auto &id : idmap) {
        nnodes = std::max(nnodes, (_u64) id);
      }
      nelems += idmap.size();
    }
    nnodes++;
    std::cout << "# nodes: " << nnodes << ", max. degree: " << max_degree
              << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<unsigned, unsigned>> node_shard;
    node_shard.reserve(nelems);
    for (_u64 shard = 0; shard < nshards; shard++) {
      std::cout << "Creating inverse map -- shard #" << shard << "\n";
      for (_u64 idx = 0; idx < idmaps[shard].size(); idx++) {
        _u64 node_id = idmaps[shard][idx];
        node_shard.push_back(std::make_pair(node_id, shard));
      }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left,
                                                       const auto &right) {
      return left.first < right.first ||
             (left.first == right.first && left.second < right.second);
    });
    std::cout << "Finished computing node -> shards map\n";

    // create cached nsg readers
    std::vector<cached_ifstream> nsg_readers(nshards);
    for (_u64 i = 0; i < nshards; i++) {
      nsg_readers[i].open(nsg_names[i], 1024 * 1048576);
      size_t actual_file_size = get_file_size(nsg_names[i]);
      size_t expected_file_size;
      nsg_readers[i].read((char *) &expected_file_size, sizeof(uint64_t));
      if (actual_file_size != expected_file_size) {
        std::cout << "Error in Vamana Index file " << nsg_names[i] << std::endl;
        exit(-1);
      }
    }

    size_t merged_index_size = 16;
    // create cached nsg writers
    cached_ofstream nsg_writer(output_nsg, 1024 * 1048576);
    nsg_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    unsigned output_width = max_degree;
    unsigned max_input_width = 0;
    // read width from each nsg to advance buffer by sizeof(unsigned) bytes
    for (auto &reader : nsg_readers) {
      unsigned input_width;
      reader.read((char *) &input_width, sizeof(unsigned));
      max_input_width =
          input_width > max_input_width ? input_width : max_input_width;
    }

    std::cout << "Max input width: " << max_input_width
              << ", output width: " << output_width << std::endl;
    //  _u64 rep_factor = (_u64)(std::round((float) nelems / (float) nnodes));
    //  std::cout << "Input width: " << width
    //            << ", output width: " << width * rep_factor << "\n";

    //  width *= rep_factor;
    nsg_writer.write((char *) &output_width, sizeof(unsigned));
    std::string   medoid_file = output_nsg + "_medoids.bin";
    std::ofstream medoid_writer(medoid_file.c_str(), std::ios::binary);
    _u32          nshards_u32 = nshards;
    _u32          one_val = 1;
    medoid_writer.write((char *) &nshards_u32, sizeof(uint32_t));
    medoid_writer.write((char *) &one_val, sizeof(uint32_t));

    for (_u64 shard = 0; shard < nshards; shard++) {
      unsigned medoid;
      // read medoid
      nsg_readers[shard].read((char *) &medoid, sizeof(unsigned));
      // rename medoid
      medoid = idmaps[shard][medoid];

      medoid_writer.write((char *) &medoid, sizeof(uint32_t));
      // write renamed medoid
      if (shard == (nshards - 1))  //--> uncomment if running hierarchical
        nsg_writer.write((char *) &medoid, sizeof(unsigned));
    }
    medoid_writer.close();

    std::cout << "Starting merge\n";
    //  std::set<unsigned>    nhood_set;
    boost::dynamic_bitset<> nhood_set(nnodes);
    std::vector<unsigned>   final_nhood;

    unsigned nnbrs = 0, shard_nnbrs = 0;
    unsigned cur_id = 0;
    for (const auto &id_shard : node_shard) {
      unsigned node_id = id_shard.first;
      unsigned shard_id = id_shard.second;
      if (cur_id < node_id) {
        std::random_shuffle(final_nhood.begin(), final_nhood.end());
        nnbrs = (std::min)(final_nhood.size(), (uint64_t) max_degree);
        // write into merged ofstream
        nsg_writer.write((char *) &nnbrs, sizeof(unsigned));
        nsg_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
        merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
        if (cur_id % 499999 == 1) {
          std::cout << "." << std::flush;
        }
        cur_id = node_id;
        nnbrs = 0;
        for (auto &p : final_nhood)
          nhood_set[p] = 0;
        final_nhood.clear();
      }
      // read from shard_id ifstream
      nsg_readers[shard_id].read((char *) &shard_nnbrs, sizeof(unsigned));
      std::vector<unsigned> shard_nhood(shard_nnbrs);
      nsg_readers[shard_id].read((char *) shard_nhood.data(),
                                 shard_nnbrs * sizeof(unsigned));

      // rename nodes
      for (_u64 j = 0; j < shard_nnbrs; j++) {
        if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0) {
          nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
          final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
        }
      }
    }

    std::random_shuffle(final_nhood.begin(), final_nhood.end());
    nnbrs = (std::min)(final_nhood.size(), (uint64_t) max_degree);
    // write into merged ofstream
    nsg_writer.write((char *) &nnbrs, sizeof(unsigned));
    nsg_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
    merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
    for (auto &p : final_nhood)
      nhood_set[p] = 0;
    final_nhood.clear();

    std::cout << "Expected size: " << merged_index_size << std::endl;
    //  merged_index_size = nsg_writer.get_file_size();
    nsg_writer.reset();
    nsg_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    std::cout << "Finished merge\n";
    return 0;
  }
};
