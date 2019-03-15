#include <efanna2e/util.h>
#include <fcntl.h>
#include <tsl/robin_map.h>
#include <unistd.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

// hash_fn: https://github.com/Microsoft/BLAS-on-flash/blob/master/src/utils.cpp
uint64_t fnv64a(const char* str, const uint64_t n_bytes) {
  const uint64_t fnv64Offset = 14695981039346656037u;
  const uint64_t fnv64Prime = 0x100000001b3u;
  uint64_t       hash = fnv64Offset;

  for (uint64_t i = 0; i < n_bytes; i++) {
    hash = hash ^ (uint64_t) *str++;
    hash *= fnv64Prime;
  }

  return hash;
}

void compute_unique_idxs(float* data, uint64_t npts, uint64_t ndims,
                         std::vector<uint64_t>& unique_idxs) {
  uint64_t* hash_values = new uint64_t[npts];
#pragma omp parallel for
  for (uint64_t i = 0; i < npts; i++) {
    hash_values[i] = fnv64a((char*) (data + i * ndims), ndims * sizeof(float));
  }

  tsl::robin_map<uint64_t, std::vector<uint64_t>> hash_to_index_map;

  // build map of values
  for (uint64_t i = 0; i < npts; i++) {
    // if hash not added to map, add hash to map
    if (hash_to_index_map.find(hash_values[i]) == hash_to_index_map.end()) {
      hash_to_index_map.insert(
          std::make_pair(hash_values[i], std::vector<uint64_t>()));
      std::vector<uint64_t>& idx_vec = hash_to_index_map[hash_values[i]];
      idx_vec.push_back(i);
    } else {
      // get all vecs hashed to same hash value
      std::vector<uint64_t>& idx_vec = hash_to_index_map[hash_values[i]];
      float*                 vec = (data + i * ndims);
      bool                   dup_found = false;
      for (const auto& id : idx_vec) {
        // check if data[i, :] is duplicate of data[id, :]
        float* cmp_vec = (data + id * ndims);
        int ret = memcmp((void*) vec, (void*) cmp_vec, ndims * sizeof(float));
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

void write_unique(char* filename, float* data,
                  std::vector<uint64_t> unique_idxs, uint64_t ndims) {
  std::ofstream writer(filename, std::ios::binary);

  std::sort(unique_idxs.begin(), unique_idxs.end());

  uint64_t blk_size = 1048576;
  uint64_t npts = unique_idxs.size();
  uint64_t nblks = ROUND_UP(npts, blk_size) / blk_size;
  std::cout << "Total # blocks = " << nblks << std::endl;
  // buf to write to disk
  char* write_buf = (char*) malloc(blk_size * ndims * sizeof(float));
  for (uint64_t i = 0; i < nblks; i++) {
    // # pts in blk
    uint64_t n_blk_pts = std::min(npts - i * blk_size, blk_size);

    // copy each vector from data to write_buf
    for (uint64_t j = 0; j < n_blk_pts; j++) {
      uint64_t idx = unique_idxs[i * blk_size + j];
      memcpy((void*) (write_buf + j * ndims * sizeof(float)),
             (void*) (data + idx * ndims), ndims * sizeof(float));
    }

    writer.write(write_buf, n_blk_pts * ndims * sizeof(float));
    std::cout << "Block #" << i << " written" << std::endl;
  }

  std::cout << "Finished writing fvecs " << std::endl;
  // free mem
  free(write_buf);
  writer.close();
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << argv[0] << " fp32_fvecs_in unique_fp32_fvecs_out ndims\b "
                            "IMPORTANT: SET ndims = data_dim + 1"
              << std::endl;
    exit(-1);
  }

  uint64_t ndims = (uint64_t) std::atoi(argv[3]);

  // read in fvecs
  float*        data = nullptr;
  std::ifstream reader(argv[1], std::ios::ate | std::ios::binary);
  uint64_t      fsize = reader.tellg();
  reader.seekg(0, std::ios::beg);
  uint64_t npts = fsize / (ndims * sizeof(float));
  std::cout << "Reading " << npts << " points, each with " << ndims << " dims"
            << std::endl;
  data = new float[npts * ndims];
  reader.read((char*) data, fsize);
  reader.close();

  // compute indices of unqiue points
  std::vector<uint64_t> unique_idxs;
  std::cout << "Computing unique ids" << std::endl;
  compute_unique_idxs(data, npts, ndims, unique_idxs);

  // write all unique data
  std::cout << "Writing unique data to " << argv[2] << std::endl;
  write_unique(argv[2], data, unique_idxs, ndims);

  delete[] data;
}
