// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"

#include <algorithm>
#include <numeric>
#include <cassert>
#include <random>
#include <thread>
#include <vector>
#include "tsl/robin_set.h"

#define ENTRY_POINT 52292725
//#define ENTRY_POINT 123742

template<typename T, typename TagT = uint32_t>
void dump_to_disk(const T *all_pts, const uint64_t ndims,
                  const std::string &          filename,
                  const std::vector<uint32_t> &tags) {
  T *   new_data = new T[ndims * tags.size()];
  TagT *new_tags = new TagT[tags.size()];

  std::string tag_filename = filename + ".tags";
  std::string data_filename = filename + ".data";
  diskann::cout << "# points : " << tags.size() << "\n";
  diskann::cout << "Tag file : " << tag_filename << "\n";
  diskann::cout << "Data file : " << data_filename << "\n";

  std::ofstream tag_writer(tag_filename);
  for (uint64_t i = 0; i < tags.size(); i++) {
    //    tag_writer << tags[i] << std::endl;
    *(new_tags + i) = tags[i];
    memcpy(new_data + (i * ndims), all_pts + (tags[i] * ndims),
           ndims * sizeof(float));
  }
  //  tag_writer.close();
  diskann::save_bin<TagT>(tag_filename, new_tags, tags.size(), 1);
  diskann::save_bin<T>(data_filename, new_data, tags.size(), ndims);
  delete new_data;
  delete new_tags;
}

template<typename T>
void run(const uint32_t base_count, const uint32_t num_mem_indices,
         const uint32_t delete_count, const uint32_t incr_count,
         const uint32_t num_cycles, const std::string &in_file,
         const std::string &out_prefix, const std::string &deleted_tags_file) {
  // random number generator
  std::random_device dev;
  std::mt19937       rng(dev());

  T *      all_points = nullptr;
  uint64_t npts, ndims;
  diskann::load_bin(in_file, all_points, npts, ndims);
  diskann::cout << "Loaded " << npts << " pts x " << ndims << " dims\n";
  //  assert(npts >= base_count + num_mem_indices);

  std::vector<uint32_t> tags(npts);
  std::iota(tags.begin(), tags.end(), 0);

  diskann::cout << "Base Index : choosing " << base_count << " points\n";
  std::vector<uint32_t> base_tags(tags.begin(), tags.begin() + base_count);

  tsl::robin_set<uint32_t> active_tags;
  tsl::robin_set<uint32_t> inactive_tags;

  tsl::robin_set<uint32_t> new_active_tags;
  tsl::robin_set<uint32_t> new_inactive_tags;

  for (uint32_t i = 0; i < base_count; i++) {
    active_tags.insert(i);
  }

  for (uint32_t i = base_count; i < npts; i++) {
    inactive_tags.insert(i);
  }

  diskann::cout << "Dumping base set \n";
  // write base
  dump_to_disk(all_points, ndims, out_prefix + "_base", base_tags);

  std::vector<uint32_t> delete_vec;
  std::vector<uint32_t> insert_vec;
  uint32_t              count = 0;
  while (count++ < num_cycles) {
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dis(0, 1);

    new_active_tags.clear();
    new_inactive_tags.clear();

    delete_vec.clear();
    insert_vec.clear();

    float active_tags_sampling_rate = (float) ((std::min)(
        (1.0 * delete_count) / (1.0 * ((double) active_tags.size())), 1.0));

    for (auto iter = active_tags.begin(); iter != active_tags.end(); iter++) {
      if (dis(gen) < active_tags_sampling_rate && *iter != ENTRY_POINT) {
        delete_vec.emplace_back(*iter);
        new_inactive_tags.insert(*iter);
      } else
        new_active_tags.insert(*iter);
    }

    float inactive_tags_sampling_rate = (float) ((std::min)(
        (1.0 * incr_count) / (1.0 * ((double) inactive_tags.size())), 1.0));

    for (auto iter = inactive_tags.begin(); iter != inactive_tags.end();
         iter++) {
      if (dis(gen) < inactive_tags_sampling_rate) {
        insert_vec.emplace_back(*iter);
        new_active_tags.insert(*iter);
      } else
        new_inactive_tags.insert(*iter);
    }

    diskann::cout << "Merge program will insert " << insert_vec.size()
                  << " points and delete  " << delete_vec.size()
                  << " points in round  " << count << std::endl;

    active_tags.swap(new_active_tags);
    inactive_tags.swap(new_inactive_tags);

    // TODO (correct) :: enable shuffling tags for better randomness
    // std::shuffle(tags.begin(), tags.end(), rng);

    // split tags

    const uint64_t mem_count =
        ROUND_UP(insert_vec.size(), num_mem_indices) / num_mem_indices;

    std::vector<std::vector<uint32_t>> mem_tags(num_mem_indices);
    uint64_t                           cur_start = 0;
    for (uint64_t i = 0; i < num_mem_indices; i++) {
      std::vector<uint32_t> &ith_tags = mem_tags[i];
      uint64_t new_start = std::min(cur_start + mem_count, insert_vec.size());
      diskann::cout << "Index #" << i + 1 << " : choosing "
                    << new_start - cur_start << " points\n";
      ith_tags.insert(ith_tags.end(), insert_vec.begin() + cur_start,
                      insert_vec.begin() + new_start);
      cur_start = new_start;
    }

    // write mem
    for (uint64_t i = 0; i < num_mem_indices; i++) {
      diskann::cout << "Dumping mem set #" << i + 1 << "\n";
      dump_to_disk(all_points, ndims,
                   out_prefix + "_cycle_" + std::to_string(count) + "_mem_" +
                       std::to_string(i + 1),
                   mem_tags[i]);
    }

    // re-shuffle tags to get delete list
    //  std::shuffle(tags.begin(), tags.end(), rng);
    std::ofstream deleted_tags_writer(deleted_tags_file + "_cycle_" +
                                      std::to_string(count));
    for (uint64_t i = 0; i < delete_vec.size(); i++) {
      deleted_tags_writer << delete_vec[i] << std::endl;
    }
    deleted_tags_writer.close();

    // add remaining tags to final list
    //  std::vector<uint64_t> rem_tags(tags.begin() + delete_count, tags.end());
    //  std::sort(rem_tags.begin(), rem_tags.end());
    // write mem
    //  diskann::cout << "Dumping {all} \\ {deleted} set\n";
    //  dump_to_disk(all_points, ndims, out_prefix + "_oneshot", rem_tags);
  }
  // free all points
  delete all_points;
}

int main(int argc, char **argv) {
  if (argc != 10) {
    diskann::cout << "Correct usage: " << argv[0]
                  << " <type[int8/uint8/float]> <base_count> <num_mem_indices>"
                  << " <delete_count> <incr count> <num_cycles> <in_file> "
                     "<out_prefix> <deleted_tags_file>"
                  << std::endl;
    exit(-1);
  }
  diskann::cout.setf(std::ios::unitbuf);

  int            arg_no = 1;
  std::string    index_type = argv[arg_no++];
  const uint32_t base_count = std::atoi(argv[arg_no++]);
  diskann::cout << "# base points : " << base_count << "\n";
  const uint32_t num_mem_indices = std::atoi(argv[arg_no++]);
  diskann::cout << "# mem indices : " << num_mem_indices << "\n";
  const uint32_t delete_count = std::atoi(argv[arg_no++]);
  diskann::cout << "# deleted tags per cycle : " << delete_count << "\n";
  const uint32_t incr_count = std::atoi(argv[arg_no++]);
  diskann::cout << "# inserted tags per cycle : " << incr_count << "\n";
  const uint32_t num_cycles = std::atoi(argv[arg_no++]);
  diskann::cout << "# cycles : " << num_cycles << "\n";
  const std::string in_file = argv[arg_no++];
  diskann::cout << "In file : " << in_file << "\n";
  const std::string out_prefix = argv[arg_no++];
  diskann::cout << "Out prefix : " << out_prefix << "\n";
  const std::string deleted_tags_file = argv[arg_no++];
  diskann::cout << "Deleted tags file prefix : " << deleted_tags_file << "\n";

  if (index_type == std::string("float")) {
    run<float>(base_count, num_mem_indices, delete_count, incr_count,
               num_cycles, in_file, out_prefix, deleted_tags_file);
  } else if (index_type == std::string("uint8")) {
    run<uint8_t>(base_count, num_mem_indices, delete_count, incr_count,
                 num_cycles, in_file, out_prefix, deleted_tags_file);
  } else if (index_type == std::string("int8")) {
    run<int8_t>(base_count, num_mem_indices, delete_count, incr_count,
                num_cycles, in_file, out_prefix, deleted_tags_file);
  } else {
    diskann::cout << "Unsupported type : " << index_type << "\n";
  }
  diskann::cout << "Exiting\n";
}
