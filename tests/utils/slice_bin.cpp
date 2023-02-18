// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

template <typename T>
int aux_main(const std::string &input_file,
             const std::string &output_file_prefix,
             unsigned num_shards,
             unsigned num_pts_per_shard) {
  std::ifstream reader(input_file, std::ios::binary | std::ios::ate);
  const uint64_t fsize = reader.tellg();
  reader.seekg(0, std::ios::beg);
  int32_t nptstotal32;
  reader.read((char*)&nptstotal32, sizeof(int32_t));
  const uint64_t nptstotal = (uint64_t)nptstotal32;
  int32_t ndims32;
  reader.read((char*)&ndims32, sizeof(int32_t));
  const uint64_t ndims = (uint64_t)ndims32;

  if (fsize != 8 + ndims * nptstotal * sizeof(T)) {
    std::cout << "wrong bin file size" << std::endl;
    return -1;
  }

  std::cout << "Dataset: #pts = " << nptstotal << ", # dims = " << ndims
      << std::endl;

  if (num_shards != 0 && (num_shards - 1) * 1ULL * num_pts_per_shard > nptstotal) {
    std::cout << "too few points in .bin file" << std::endl;
    return -1;
  }
  if (num_pts_per_shard > nptstotal) {
    std::cout << "slice larger than entire input .bin file" << std::endl;
    return -1;
  }

  uint64_t pts_remain = nptstotal;
  uint32_t next_pt_id = 0;

  for (int slice = 1; pts_remain > 0; ++slice) {
    if (num_shards != 0 && slice > num_shards) {
      break;
    }
    const std::string output_filename = output_file_prefix + "_subshard-" + std::to_string(slice) + ".bin";
    std::ofstream writer(output_filename, std::ios::binary);

    int32_t pts_to_copy32 = std::min((int32_t)pts_remain, (int32_t)num_pts_per_shard);
    pts_remain -= pts_to_copy32;

    writer.write((char*)&pts_to_copy32, sizeof(int32_t));
    writer.write((char*)&ndims32, sizeof(int32_t));

    constexpr size_t chunk = 200'000;
    char buf[chunk];
    size_t to_copy = (size_t)pts_to_copy32 * ndims32 * sizeof(T);
    while (to_copy > 0) {
        const size_t curchunk = std::min(chunk, to_copy);
        to_copy -= curchunk;
        reader.read(buf, curchunk);
        writer.write(buf, curchunk);
    }
    writer.close();

    // now the indices

    const std::string output_filename_ids = output_file_prefix + "_subshard-" + std::to_string(slice) + "_ids_uint32.bin";
    std::ofstream writer_ids(output_filename_ids, std::ios::binary);
    constexpr uint32_t const_one = 1;
    writer_ids.write((char*)&pts_to_copy32, sizeof(int32_t));
    writer_ids.write((char*)&const_one, sizeof(int32_t));

    uint32_t pts_remain_for_slice = pts_to_copy32;

    constexpr uint32_t pts_per_iteration = 500'000;
    uint32_t bufids[pts_per_iteration];
    while (pts_remain_for_slice > 0) {
        const uint32_t pts_this_iteration = std::min(pts_remain_for_slice, pts_per_iteration);
        for (uint32_t i = 0; i < pts_this_iteration; ++i) {
            bufids[i] = next_pt_id++;
        }
        writer_ids.write((char*)bufids, pts_this_iteration * sizeof(uint32_t));
        pts_remain_for_slice -= pts_this_iteration;
    }
    writer_ids.close();

    if (next_pt_id + pts_remain != nptstotal) {
      std::cout << "implementation issue?" << std::endl;
      return -1;
    }
  }

  reader.close();
  return 0;
}

// Slices a bin file into slices of predefined size.
// Can also be used to truncate a bin file (by generating one slice).
//
// Output files will be: output_file_prefix_subshard-X.bin
//                   and output_file_prefix_subshard-X_ids_uint32.bin
// where X = 1,2,3,...

int main(int argc, char** argv) {
  std::string input_file, output_file_prefix;
  unsigned num_shards, shard_size;

  std::string data_type;

  po::options_description desc{ "Arguments" };
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("input_file",
                       po::value<std::string>(&input_file)->required(),
                       "Path to the .bin file");
    desc.add_options()("output_file_prefix",
                       po::value<std::string>(&output_file_prefix)->required(),
                       "Output file prefix. Will generate files like this_subshard-5.bin and this_subshard-X_ids_uint32.bin");
    desc.add_options()("num_shards",
                       po::value<unsigned>(&num_shards)->default_value(0),
                       "Number of shards. Default (0) is to partition the entire .bin file");
    desc.add_options()("shard_size",
                       po::value<unsigned>(&shard_size)->required(),
                       "Size (number of points) of each shard");
    
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

  try {
    if (data_type == std::string("float")) {
      return aux_main<float>(input_file, output_file_prefix, num_shards, shard_size);
    } else if (data_type == std::string("float")) {
      return aux_main<float>(input_file, output_file_prefix, num_shards, shard_size);
    } else if (data_type == std::string("float")) {
      return aux_main<float>(input_file, output_file_prefix, num_shards, shard_size);
    } else {
      std::cerr << "Unsupported data type. Use float or int8 or uint8"
                << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    std::cerr << "Slicing failed." << std::endl;
    return -1;
  }
}
