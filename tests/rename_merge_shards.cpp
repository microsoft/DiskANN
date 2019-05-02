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

  // create cached nsg readers
  std::vector<cached_ifstream> nsg_readers(nshards);
  for (_u64 i = 0; i < nshards; i++) {
    nsg_readers[i].open(nsg_names[i], 256 * 1048576);
  }

  // create cached nsg writers
  std::string     final_nsg_name(argv[6]);
  cached_ofstream nsg_writer(final_nsg_name, 256 * 1048576);

  // compute nnodes
  _u64 nnodes = 0;
  for (auto &idmap : idmaps) {
    nnodes += idmap.size();
  }
  std::cout << "# nodes : " << nnodes << "\n";

  unsigned width;
  // read width from each nsg to advance buffer by sizeof(unsigned) bytes
  for (auto &reader : nsg_readers) {
    reader.read((char *) &width, sizeof(unsigned));
  }
  nsg_writer.write((char *) &width, sizeof(unsigned));
  for (auto &reader : nsg_readers) {
    unsigned medoid;
    reader.read((char *) &medoid, sizeof(unsigned));
    nsg_writer.write((char *) &medoid, sizeof(unsigned));
  }

  // compute shard ids
  std::cout << "Computing shard<->id ordering\n";
  std::vector<std::pair<unsigned, unsigned>> shard_ids;
  shard_ids.reserve(nnodes);
  {
    unsigned shard = 0;
    for (auto &idmap : idmaps) {
      for (auto id : idmap) {
        shard_ids.push_back(std::make_pair(shard, id));
      }
      shard++;
    }
  }
  // sort by ids
  std::sort(shard_ids.begin(), shard_ids.end(),
            [](const auto &left, const auto &right) {
              return left.second < right.second;
            });
  std::cout << "Starting merge\n";
  unsigned *nhood = new unsigned[32768];
  unsigned  nnbrs;
  for (const auto &shard_id : shard_ids) {
    unsigned shard = shard_id.first;
    unsigned id = shard_id.second;

    // read from shard ifstream
    nsg_readers[shard].read((char *) &nnbrs, sizeof(unsigned));
    nsg_readers[shard].read((char *) nhood, nnbrs * sizeof(unsigned));

    // rename nodes
    for(_u64 j=0;j<nnbrs;j++){
      nhood[j] = idmaps[shard][nhood[j]];
    }

    // write into merged ofstream
    nsg_writer.write((char *) &nnbrs, sizeof(unsigned));
    nsg_writer.write((char *) nhood, nnbrs * sizeof(unsigned));
    if (id % 999999 == 1) {
      std::cout << "Finished merging " << id << " nodes\n";
    }
  }
  std::cout << "Finished merge\n";
  return 0;
}
