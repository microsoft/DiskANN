#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
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
 for(_u64 i=0; i < local_vec.size() / 2;i++){
 ivecs[i] = local_vec[2*i + 1]; 
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

void read_nsgs(const std::vector<std::string> &    fnames,
               std::vector<std::vector<unsigned>> &nsgs) {
  for (_u64 i = 0; i < fnames.size(); i++) {
    read_nsg(fnames[i], nsgs[i]);
  }
}

void read_shard_id_maps(const std::vector<std::string> &    fnames,
                        std::vector<std::vector<unsigned>> &id_maps) {
  for (_u64 i = 0; i < fnames.size(); i++) {
    read_bad_ivecs(fnames[i], id_maps[i]);
  }
}

void rename_nsgs(std::vector<std::vector<unsigned>> &nsgs, std::vector<std::vector<unsigned>> &id_maps, std::vector<std::vector<_u64>> &node_sizes){
  node_sizes.resize(nsgs.size());
  #pragma omp parallel for schedule(dynamic, 1)
  for(_u64 shard=0;shard < nsgs.size();shard++){
    // rename medoid
    auto &nsg = nsgs[shard];
    auto &id_map = id_maps[shard];
    auto &node_size = node_sizes[shard];
    nsg[1] = id_map[nsg[1]];
    // start going through the nsg
    _u64 cur_off = 1;
    while(cur_off < nsg.size()){
      // obtain nnbrs
      unsigned nnbrs = nsg[cur_off];
      cur_off++;
      node_size.push_back(nnbrs);
      // rename next nnbrs values
      for(_u64 j=0;j<nnbrs;j++){
        nsg[cur_off] = id_map[nsg[cur_off]];
        cur_off++;
      }
    }
  }
}

void merge_nsgs(const std::vector<std::vector<unsigned>> &nsgs,
                const std::vector<std::vector<unsigned>> &id_maps,
                std::vector<unsigned> &                   final_nsg) {
  // num shards
  _u64 nshards = nsgs.size();
  // 1 element for width
  _u64 final_size = 1;
  for (auto &nsg : nsgs) {
    // keep all unsigned values except width
    final_size += (nsg.size() - 1);
  }

  std::cout << "Final NSG size: " << final_size * sizeof(unsigned) << "B\n";
  final_nsg.resize(final_size);

  _u64 header_size = 0;

  // copy nsg width
  final_nsg[0] = nsgs[0][0];
  header_size++;

  // copy nsg medoid ids in each shard + rename medoids
  for (_u64 shard = 0; shard < nshards; shard++) {
    // copy medoid
    final_nsg[shard + 1] = nsgs[shard][1];
    // rename medoid
    final_nsg[shard + 1] = id_maps[shard][final_nsg[shard + 1]];
    header_size++;
  }

  // compute offsets of each shard data
  std::vector<_u64> shard_offsets(nshards, 0);
  shard_offsets[0] = header_size;
  for (_u64 shard = 1; shard < nshards; shard++) {
    // ignore width & medoid from each nsg
    shard_offsets[shard] =
        shard_offsets[shard - 1] + (nsgs[shard - 1].size() - 2);
  }

// copy nsg data from each shard in parallel for maximum memory bandwidth
#pragma omp parallel for schedule(dynamic, 1)
  for (_u64 shard = 0; shard < nsgs.size(); shard++) {
    memcpy(final_nsg.data() + shard_offsets[shard], nsgs[shard].data() + 2,
           (nsgs[shard].size() - 2) * sizeof(unsigned));
#pragma omp critical
    { std::cout << "Copied shard #" << shard << "\n"; }
  }

  // compute nnnodes
  _u64 nnodes = 0;
  for (auto &id_map : id_maps) {
    nnodes += id_map.size();
  }
  std::cout << "# of nodes: " << nnodes << "\n";

  // assign nodes to shards
  std::cout << "Assigning nodes to shards\n";
  std::vector<_u64> node_shard_id(nnodes, 0);
  _u64              cur_pos = 0;
  for (_u64 shard = 0; shard < nshards; shard++) {
    std::fill(node_shard_id.data() + cur_pos,
              node_shard_id.data() + cur_pos + id_maps[shard].size(), shard);
    cur_pos += id_maps[shard].size();
  }
  assert(node_shard_id[nshards-1] == nshards-1);

  // compute node sizes
  std::cout << "Computing node sizes\n";
  std::vector<_u64> node_sizes(nnodes, 0);
  unsigned *        scan_array = final_nsg.data() + header_size;
  for (_u64 i = 0; i < nnodes; i++) {
    node_sizes[i] = *scan_array;
    assert(node_sizes[i] <= final_nsg[0]);
    scan_array += (node_sizes[i] + 1);
  }

  // compute node offests in final nsg
  std::vector<_u64> node_offsets(nnodes, 0);
  node_offsets[0] = header_size;
  for (_u64 i = 1; i < nnodes; i++) {
    node_offsets[i] = node_offsets[i - 1] + node_sizes[i - 1];
  }

  std::cout << "Starting node re-name for nnodes: "<< nnodes << "\n";
  std::atomic<_u64> nnodes_done;
  nnodes_done.store(0);
#pragma omp parallel for schedule(dynamic, 262144)
  for (_u64 i = 0; i < nnodes; i++) {
    unsigned *      node_nhood = (final_nsg.data() + node_offsets[i] + 1);
    unsigned        nnbrs = node_sizes[i];
    _u64            shard_id = node_shard_id[i];
    const unsigned *shard_id_map = id_maps[shard_id].data();
    for (unsigned nbr = 0; nbr < nnbrs; nbr++) {
      assert(node_nhood[nbr] <= id_maps[shard_id].size());
      node_nhood[nbr] = shard_id_map[node_nhood[nbr]];
    }
    nnodes_done++;
    if (nnodes_done.load() % 999999 == 1) {
      std::cout << nnodes_done.load() << " nodes renamed\n";
    }
  }
  std::cout << "Done renaming nodes\n";
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

  std::vector<std::vector<unsigned>> nsgs(nshards);
  std::vector<std::vector<unsigned>> idmaps(nshards);

  // read all id maps
  read_shard_id_maps(idmaps_names, idmaps);

  // read all nsgs
  read_nsgs(nsg_names, nsgs);

  std::string           final_nsg_name(argv[6]);
  std::vector<unsigned> final_nsg;
  std::cout << "Starting merge\n";
  merge_nsgs(nsgs, idmaps, final_nsg);
  std::cout << "Finished merge\n";
  std::cout << "Writing merged NSG to " << final_nsg_name << "\n";
  std::ofstream writer(final_nsg_name, std::ios::binary);
  writer.write((char *) final_nsg.data(), final_nsg.size() * sizeof(unsigned));
  std::cout << "Finished writing merged NSG\n";

  return 0;
}
