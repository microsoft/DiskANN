#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <parallel/algorithm>
#include <vector>
#include "cached_io.h"
#include "utils.h"

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
  uint32_t      npts32, dummy;
  std::ifstream reader(fname.c_str(), std::ios::binary);
  reader.read((char *) &npts32, sizeof(uint32_t));
  reader.read((char *) &dummy, sizeof(uint32_t));
  npts32 = fsize / sizeof(unsigned) - 2;
  std::cout << "Points = " << npts32 << std::endl;
  ivecs.resize(npts32);
  reader.read((char *) ivecs.data(), npts32 * sizeof(uint32_t));
  reader.close();
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
  std::string              output_nsg(argv[6]);
  std::string              medoid_file = output_nsg + "_medoids.bin";

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
  std::vector<std::pair<unsigned, unsigned>> node_shard;
  node_shard.reserve(nelems);
  for (_u64 shard = 0; shard < nshards; shard++) {
    std::cout << "Creating inverse map -- shard #" << shard << "\n";
    for (_u64 idx = 0; idx < idmaps[shard].size(); idx++) {
      _u64 node_id = idmaps[shard][idx];
      node_shard.push_back(std::make_pair(node_id, shard));
    }
  }
  __gnu_parallel::sort(
      node_shard.begin(), node_shard.end(),
      [](const auto &left, const auto &right) {
        return left.first < right.first ||
               (left.first == right.first && left.second < right.second);
      });
  std::cout << "Finished computing node -> shards map\n";

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

  _u64 rep_factor = (_u64)(std::round((float) nelems / (float) nnodes));
  std::cout << "Input width: " << width
            << ", output width: " << width * rep_factor << "\n";

  width *= rep_factor;
  nsg_writer.write((char *) &width, sizeof(unsigned));
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
        std::cout << "Finished merging " << cur_id << " nodes\n";
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

  std::cout << "Finished merge\n";
  delete[] nhood;
  return 0;
}
