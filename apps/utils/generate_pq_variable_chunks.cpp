// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// This utility trains a PQ codebook using variable (explicit) chunk offsets
// provided via a text file. The chunk offsets file must contain a single line
// with space separated non-negative integers specifying the cumulative
// dimension boundaries. Example for a 96-dim space partitioned into
// 16|24|40|16 would be: 0 16 40 80 96
// Requirements:
//   * first value must be 0
//   * last value must equal the vector dimension (validated after sampling)
//   * values must be strictly increasing
// The tool samples the input data (binary format: uint32 npts, uint32 dim, then
// row-major data) using gen_random_slice (sampling_rate in (0,1]) and trains a
// 256-center PQ codebook per chunk. It then encodes the full dataset producing
// <PQ_prefix>_pq_pivots.bin and <PQ_prefix>_pq_compressed.bin.

#include "math_utils.h"
#include "pq.h"
#include "partition.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

#define KMEANS_ITERS_FOR_PQ 15

template <typename T>
bool generate_pq_with_variable_chunks(const std::string &data_path, const std::string &index_prefix_path,
                                      const size_t num_pq_centers, const std::string &chunk_offsets_file,
                                      const float sampling_rate)
{
    std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";

    // Read chunk offsets (single line of space separated integers)
    std::ifstream fin(chunk_offsets_file);
    if (!fin.is_open())
    {
        std::cerr << "Failed to open chunk offsets file: " << chunk_offsets_file << std::endl;
        return false;
    }
    std::string line;
    if (!std::getline(fin, line))
    {
        std::cerr << "Chunk offsets file is empty: " << chunk_offsets_file << std::endl;
        return false;
    }
    fin.close();
    std::istringstream iss(line);
    std::vector<uint32_t> chunk_offsets;
    uint64_t val64;
    while (iss >> val64)
    {
        if (val64 > std::numeric_limits<uint32_t>::max())
        {
            std::cerr << "Chunk offset value exceeds uint32 range: " << val64 << std::endl;
            return false;
        }
        chunk_offsets.push_back(static_cast<uint32_t>(val64));
    }
    if (chunk_offsets.size() < 2)
    {
        std::cerr << "Need at least two chunk offsets (0 and dim)." << std::endl;
        return false;
    }
    if (!std::is_sorted(chunk_offsets.begin(), chunk_offsets.end()))
    {
        std::cerr << "Chunk offsets must be non-decreasing." << std::endl;
        return false;
    }
    for (size_t i = 1; i < chunk_offsets.size(); i++)
    {
        if (chunk_offsets[i] == chunk_offsets[i - 1])
        {
            std::cerr << "Chunk offsets must be strictly increasing." << std::endl;
            return false;
        }
    }

    // Sample data for training
    size_t train_size, train_dim;
    float *train_data;
    gen_random_slice<T>(data_path, sampling_rate, train_data, train_size, train_dim);
    std::cout << "Loaded sample of size " << train_size << " with dimension " << train_dim << std::endl;

    // Validate first and last offsets relative to dimension
    if (chunk_offsets.front() != 0)
    {
        std::cerr << "First chunk offset must be 0." << std::endl;
        delete[] train_data;
        return false;
    }
    if (chunk_offsets.back() != train_dim)
    {
        std::cerr << "Last chunk offset (" << chunk_offsets.back() << ") must equal train dimension (" << train_dim
                  << ")." << std::endl;
        delete[] train_data;
        return false;
    }

    const size_t num_pq_chunks = chunk_offsets.size() - 1;
    std::cout << "Training PQ with " << num_pq_chunks << " variable chunks." << std::endl;

    // Train pivots using provided offsets (no OPQ path in this driver)
    if (diskann::generate_pq_pivots_with_offsets(train_data, train_size, (uint32_t)train_dim,
                                                 (uint32_t)num_pq_centers, chunk_offsets, KMEANS_ITERS_FOR_PQ,
                                                 pq_pivots_path) != 0)
    {
        std::cerr << "Error during PQ pivot training with provided offsets." << std::endl;
        delete[] train_data;
        return false;
    }

    // Encode full dataset
    if (diskann::generate_pq_data_from_pivots<T>(data_path, (uint32_t)num_pq_centers, (uint32_t)num_pq_chunks,
                                                 pq_pivots_path, pq_compressed_vectors_path, false) != 0)
    {
        std::cerr << "Error during PQ data generation." << std::endl;
        delete[] train_data;
        return false;
    }

    delete[] train_data;
    return true;
}

int main(int argc, char **argv)
{
    // New interface:
    //  <data_type[float/uint8/int8]> <data_file.bin> <PQ_prefix_path> <chunk_offsets_file.txt> <sampling_rate>
    if (argc != 6)
    {
        std::cout << "Usage:\n  " << argv[0]
                  << " <data_type[float|uint8|int8]> <data_file.bin> <PQ_prefix_path> <chunk_offsets_file.txt> <sampling_rate>\n";
        std::cout << "Notes:\n  * chunk_offsets_file: single line of space separated integers (0 ... dim) defining chunk boundaries\n";
        return -1;
    }

    const std::string dtype(argv[1]);
    const std::string data_path(argv[2]);
    const std::string index_prefix_path(argv[3]);
    const std::string chunk_offsets_file(argv[4]);
    const float sampling_rate = (float)atof(argv[5]);
    const size_t num_pq_centers = 256;
    bool ok = false;

    if (dtype == "float")
        ok = generate_pq_with_variable_chunks<float>(data_path, index_prefix_path, num_pq_centers, chunk_offsets_file,
                                                     sampling_rate);
    else if (dtype == "int8")
        ok = generate_pq_with_variable_chunks<int8_t>(data_path, index_prefix_path, num_pq_centers,
                                                      chunk_offsets_file, sampling_rate);
    else if (dtype == "uint8")
        ok = generate_pq_with_variable_chunks<uint8_t>(data_path, index_prefix_path, num_pq_centers,
                                                       chunk_offsets_file, sampling_rate);
    else
        std::cerr << "Unsupported data type: " << dtype << std::endl;

    return ok ? 0 : -1;
}
