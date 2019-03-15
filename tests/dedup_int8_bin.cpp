#include <efanna2e/util.h>
#include <fcntl.h>
#include <tsl/robin_map.h>
#include <unistd.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

// hash_fn: https://github.com/Microsoft/BLAS-on-flash/blob/master/src/utils.cpp
_u64 fnv64a(const char* str, const _u64 n_bytes) {
  const _u64 fnv64Offset = 14695981039346656037u;
  const _u64 fnv64Prime = 0x100000001b3u;
  _u64       hash = fnv64Offset;

  for (_u64 i = 0; i < n_bytes; i++) {
    hash = hash ^ (_u64) *str++;
    hash *= fnv64Prime;
  }

  return hash;
}

void compute_unique_idxs(_s8* data, _u64 npts, _u64 ndims,
                         std::vector<_u64>& unique_idxs) {
  _u64*     hash_values = new _u64[npts];
#pragma omp parallel for
  for (_u64 i = 0; i < npts; i++) {
    hash_values[i] = fnv64a((char*) (data + i * ndims), ndims * sizeof(_s8));
  }

  tsl::robin_map<_u64, std::vector<_u64>> hash_to_index_map;

  // build map of values
  for (_u64 i = 0; i < npts; i++) {
    // if hash not added to map, add hash to map
    if (hash_to_index_map.find(hash_values[i]) == hash_to_index_map.end()) {
      hash_to_index_map.insert(
          std::make_pair(hash_values[i], std::vector<_u64>()));
      std::vector<_u64>& idx_vec = hash_to_index_map[hash_values[i]];
      idx_vec.push_back(i);
    } else {
      // get all vecs hashed to same hash value
      std::vector<_u64>& idx_vec = hash_to_index_map[hash_values[i]];
      _s8*               vec = (data + i * ndims);
      bool               dup_found = false;
      for (const auto& id : idx_vec) {
        // check if data[i, :] is duplicate of data[id, :]
        _s8* cmp_vec = (data + id * ndims);
        int  ret = memcmp((void*) vec, (void*) cmp_vec, ndims * sizeof(_s8));
        if (ret == 0) {
          dup_found = true;
          std::cout << "Duplicate: " << i << " <--> " << id << std::endl;
          break;
        }
      }
      if (!dup_found) {
        idx_vec.push_back(i);
      }
    }
  }
  for (auto k_v : hash_to_index_map) {
    for (auto& id : k_v.second) {
      unique_idxs.push_back(id);
    }
  }
  std::cout << "Input: " << npts << ", unique: " << unique_idxs.size()
            << ", duplicates removed: " << npts - unique_idxs.size()
            << std::endl;

  // cleanup
  hash_to_index_map.clear();
  delete[] hash_values;
}

void write_unique_bin(char* filename, _s8* data, std::vector<_u64> unique_idxs,
                      _u64 ndims) {
  std::ofstream writer(filename, std::ios::binary);

  std::sort(unique_idxs.begin(), unique_idxs.end());

  _u64 blk_size = 1048576;
  _u64 npts = unique_idxs.size();
  _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
  std::cout << "Total # blocks = " << nblks << std::endl;

  // write npts, ndims
  _s32 npts_s32 = (_s32) npts;
  _s32 ndims_s32 = (_s32) ndims;
  writer.write((char*) &npts_s32, sizeof(_s32));
  writer.write((char*) &ndims_s32, sizeof(_s32));

  // buf to write to disk
  char* write_buf = (char*) malloc(blk_size * ndims * sizeof(_s8));
  for (_u64 i = 0; i < nblks; i++) {
    // # pts in blk
    _u64 n_blk_pts = std::min(npts - i * blk_size, blk_size);

// copy each vector from data to write_buf
#pragma omp parallel for schedule(static, 32768)
    for (_u64 j = 0; j < n_blk_pts; j++) {
      _u64 idx = unique_idxs[i * blk_size + j];
      memcpy((void*) (write_buf + j * ndims * sizeof(_s8)),
             (void*) (data + idx * ndims), ndims * sizeof(_s8));
    }

    writer.write(write_buf, n_blk_pts * ndims * sizeof(_s8));
    std::cout << "Block #" << i << " written" << std::endl;
  }

  std::cout << "Finished writing fvecs " << std::endl;
  // free mem
  free(write_buf);
  writer.close();
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << argv[0] << " int8_bin_in unique_int8_bin_out" << std::endl;
    exit(-1);
  }

  // read in bin
  _s8* data = nullptr;
  _u32 npts, ndims;
  NSG::load_bin<_s8>(argv[1], data, npts, ndims);

  // compute indices of unqiue points
  std::vector<_u64> unique_idxs;
  std::cout << "Computing unique ids" << std::endl;
  compute_unique_idxs(data, npts, ndims, unique_idxs);

  // write all unique data
  std::cout << "Writing unique data to " << argv[2] << std::endl;
  write_unique_bin(argv[2], data, unique_idxs, ndims);

  delete[] data;
}
