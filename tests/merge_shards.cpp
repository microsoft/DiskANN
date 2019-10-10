#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
//#include <parallel/algorithm>
#include <string>
#include <vector>
#include "cached_io.h"
#include "utils.h"

void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs) {
  uint32_t      npts32, dim;
  size_t        actual_file_size = get_file_size(fname);
  std::ifstream reader(fname.c_str(), std::ios::binary);
  reader.read((char *) &npts32, sizeof(uint32_t));
  reader.read((char *) &dim, sizeof(uint32_t));
  if (dim != 1 ||
      actual_file_size !=
          ((size_t) npts32) * sizeof(uint32_t) + 2 * sizeof(uint32_t)) {
    std::cout << "Error reading idmap file. Check if the file is bin file with "
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
                 const std::string &output_nsg) {
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
  std::cout << "# nodes: " << nnodes << "\n";

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
  std::sort(node_shard.begin(), node_shard.end(),
            [](const auto &left, const auto &right) {
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
      std::cout << "Error in Rand-NSG file " << nsg_names[i] << std::endl;
      exit(-1);
    }
  }

  size_t merged_index_size = 0;
  // create cached nsg writers
  cached_ofstream nsg_writer(output_nsg, 1024 * 1048576);
  nsg_writer.write((char *) &merged_index_size, sizeof(uint64_t));

  unsigned width;
  // read width from each nsg to advance buffer by sizeof(unsigned) bytes
  for (auto &reader : nsg_readers) {
    reader.read((char *) &width, sizeof(unsigned));
  }

  _u64 rep_factor = (_u64)(std::round((float) nelems / (float) nnodes));
  std::cout << "Input width: " << width
            << ", output width: " << width * rep_factor << "\n";

  width *= rep_factor;
  nsg_writer.write((char *) &width, sizeof(unsigned));
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
  unsigned *nhood = new unsigned[32768];
  unsigned *shard_nhood = nhood;

  unsigned nnbrs = 0, shard_nnbrs;
  unsigned cur_id = 0;
  for (const auto &id_shard : node_shard) {
    unsigned node_id = id_shard.first;
    unsigned shard_id = id_shard.second;
    if (cur_id < node_id) {
      // write into merged ofstream
      nsg_writer.write((char *) &nnbrs, sizeof(unsigned));
      nsg_writer.write((char *) nhood, nnbrs * sizeof(unsigned));
      if (cur_id % 999999 == 1) {
        std::cout << "." << std::flush;
      }
      cur_id = node_id;
      nnbrs = 0;
      shard_nhood = nhood;
    }
    // read from shard_id ifstream
    nsg_readers[shard_id].read((char *) &shard_nnbrs, sizeof(unsigned));
    nsg_readers[shard_id].read((char *) shard_nhood,
                               shard_nnbrs * sizeof(unsigned));

    // rename nodes
    for (_u64 j = 0; j < shard_nnbrs; j++) {
      shard_nhood[j] = idmaps[shard_id][shard_nhood[j]];
    }

    nnbrs += shard_nnbrs;
    shard_nhood += shard_nnbrs;
  }
  nsg_writer.write((char *) &nnbrs, sizeof(unsigned));
  nsg_writer.write((char *) nhood, nnbrs * sizeof(unsigned));
  nsg_writer.flush_cache();
  merged_index_size = nsg_writer.get_file_size();
  nsg_writer.reset();
  nsg_writer.write((char *) &merged_index_size, sizeof(uint64_t));

  std::cout << "Finished merge\n";
  delete[] nhood;
  return 0;
}
int main(int argc, char **argv) {
  if (argc != 7) {
    std::cout << argv[0] << " nsg_prefix[1] nsg_suffix[2] idmaps_prefix[3] "
                            "idmaps_suffix[4] n_shards[5] output_nsg[6]"
              << std::endl;
    exit(-1);
  }

  std::string nsg_prefix(argv[1]);
  std::string nsg_suffix(argv[2]);
  std::string idmaps_prefix(argv[3]);
  std::string idmaps_suffix(argv[4]);
  _u64        nshards = (_u64) std::atoi(argv[5]);
  std::string output_nsg(argv[6]);

  return merge_shards(nsg_prefix, nsg_suffix, idmaps_prefix, idmaps_suffix,
                      nshards, output_nsg);
}
