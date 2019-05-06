#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include "cached_io.h"
#include "efanna2e/util.h"

_u64 get_file_size(const std::string &fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  _u64          end_pos = reader.tellg();
  reader.close();
  return end_pos;
}

void read_nsg(const std::string &fname, std::vector<unsigned> &nsg) {
  _u64 fsize = get_file_size(fname);
  std::cout << "Reading file: " << fname << ", size: " << fsize << "B\n";
  nsg.resize(fsize / sizeof(unsigned));
  std::ifstream reader(fname, std::ios::binary);
  reader.read((char *) nsg.data(), fsize);
  reader.close();
}

void read_bad_ivecs(const std::string &fname, std::vector<unsigned> &ivecs) {
  _u64 fsize = get_file_size(fname);
  std::cout << "Reading bad ivecs: " << fname << ", size: " << fsize << "B\n";
  std::vector<unsigned> local_vec;
  local_vec.resize(fsize / sizeof(unsigned));
  std::ifstream reader(fname, std::ios::binary);
  reader.read((char *) local_vec.data(), fsize);
  reader.close();
  ivecs.resize(local_vec.size() / 2);
  for (_u64 i = 0; i < local_vec.size() / 2; i++) {
    ivecs[i] = local_vec[2 * i + 1];
  }
}

void read_unsigned_ivecs(const std::string &    fname,
                         std::vector<unsigned> &ivecs) {
  std::ifstream reader(fname, std::ios::binary);
  unsigned      nvals;
  reader.read((char *) &nvals, sizeof(unsigned));
  _u64 fsize = (nvals + 1) * sizeof(unsigned);
  std::cout << "Reading ivecs: " << fname << ", size: " << fsize << "B\n";
  ivecs.resize(nvals);

  reader.read((char *) ivecs.data(), nvals * sizeof(unsigned));
  reader.close();
}

void read_shard_id_maps(const std::vector<std::string> &    fnames,
                        std::vector<std::vector<unsigned>> &id_maps) {
  for (_u64 i = 0; i < fnames.size(); i++) {
    read_bad_ivecs(fnames[i], id_maps[i]);
  }
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cout << argv[0] << " nsg_prefix[1] nsg_suffix[2] idmaps_prefix[3] "
                            "idmaps_suffix[4] n_shards[5] output_nsg[6]"
              << std::endl;
    exit(-1);
  }

  _u64                     nshards = (_u64) std::atoi(argv[5]);
  std::vector<std::string> nsg_names(nshards);
  std::vector<std::string> idmaps_names(nshards);
  std::string              nsg_prefix(argv[1]);
  std::string              nsg_suffix(argv[2]);
  std::string              idmaps_prefix(argv[3]);
  std::string              idmaps_suffix(argv[4]);

  for (_u64 shard = 0; shard < nshards; shard++) {
    nsg_names[shard] = nsg_prefix + std::to_string(shard) + nsg_suffix;
    idmaps_names[shard] = idmaps_prefix + std::to_string(shard) + idmaps_suffix;
  }

  std::vector<std::vector<unsigned>> idmaps(nshards);

  // read all id maps
  read_shard_id_maps(idmaps_names, idmaps);

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
  std::vector<std::vector<uint8_t>> node_shard_map(nnodes,
                                                   std::vector<uint8_t>());
  for (_u64 shard = 0; shard < nshards; shard++) {
#pragma omp parallel for num_threads(16)
    for (_u64 idx = 0; idx < idmaps[shard].size(); idx++) {
      _u64 node_id = idmaps[shard][idx];
      node_shard_map[node_id].push_back(shard);
    }
  }
  std::cout << "Finished computing node -> shards map\n";

  // compute replication factor
  _u64 rep_factor = 1;
  for (auto &map : node_shard_map) {
    rep_factor = std::max(rep_factor, (_u64) map.size());
  }

  // create cached nsg readers
  std::vector<cached_ifstream> nsg_readers(nshards);
  for (_u64 i = 0; i < nshards; i++) {
    nsg_readers[i].open(nsg_names[i], 1024 * 1048576);
  }

  // create cached nsg writers
  std::string     final_nsg_name(argv[6]);
  cached_ofstream nsg_writer(final_nsg_name, 1024 * 1048576);

  unsigned width;
  // read width from each nsg to advance buffer by sizeof(unsigned) bytes
  for (auto &reader : nsg_readers) {
    reader.read((char *) &width, sizeof(unsigned));
  }
  std::cout << "Input width: " << width
            << ", output width: " << width * rep_factor << "\n";
  width *= rep_factor;
  nsg_writer.write((char *) &width, sizeof(unsigned));
  for (_u64 shard = 0; shard < nshards; shard++) {
    unsigned medoid;
    // read medoid
    nsg_readers[shard].read((char *) &medoid, sizeof(unsigned));
    // rename medoid
    medoid = idmaps[shard][medoid];
    // write renamed medoid
    //if (shard == (nshards - 1)) --> only uncomment if running hierarchical merge
    nsg_writer.write((char *) &medoid, sizeof(unsigned));
  }

  std::cout << "Starting merge\n";
  unsigned *nhood = new unsigned[32768];
  unsigned  nnbrs, shard_nnbrs;
  unsigned  id = 0;
  for (const auto &map : node_shard_map) {
    nnbrs = 0;
    // read all nbrs of shard
    for (const auto &shard_id : map) {
      unsigned *shard_nhood = nhood + nnbrs;
      // read from shard_id ifstream
      nsg_readers[shard_id].read((char *) &shard_nnbrs, sizeof(unsigned));
      nsg_readers[shard_id].read((char *) shard_nhood,
                                 shard_nnbrs * sizeof(unsigned));

      // rename nodes
      for (_u64 j = 0; j < shard_nnbrs; j++) {
        shard_nhood[j] = idmaps[shard_id][shard_nhood[j]];
      }

      nnbrs += shard_nnbrs;
    }
    
    // sort nhood
    std::sort(nhood, nhood + nnbrs);

    // write into merged ofstream
    nsg_writer.write((char *) &nnbrs, sizeof(unsigned));
    nsg_writer.write((char *) nhood, nnbrs * sizeof(unsigned));
    if (id % 999999 == 1) {
      std::cout << "Finished merging " << id << " nodes\n";
    }
    id++;
  }
  std::cout << "Finished merge\n";
  delete[] nhood;
  return 0;
}
