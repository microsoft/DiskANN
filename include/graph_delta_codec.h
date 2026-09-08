// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "neighbor_list.h"

namespace diskann
{
struct StreamVByteNodeEncoding
{
    uint32_t degree = 0;
    size_t control_bytes = 0;
    size_t data_bytes = 0;
    size_t encoded_size = 0;
};

uint8_t graph_id_bytes(size_t num_points);
uint8_t graph_degree_bytes(uint32_t max_degree);

StreamVByteNodeEncoding prepare_stream_vbyte_node(
    const NeighborList& neighbors,
    size_t num_points,
    uint8_t id_bytes,
    uint8_t degree_bytes,
    std::vector<uint32_t>& sorted_neighbors);

void encode_stream_vbyte_node(
    const std::vector<uint32_t>& sorted_neighbors,
    const StreamVByteNodeEncoding& encoding,
    uint8_t id_bytes,
    uint8_t degree_bytes,
    uint8_t* destination,
    size_t destination_size);

size_t decode_stream_vbyte_node(
    const uint8_t* encoded,
    size_t encoded_size,
    size_t num_points,
    uint8_t id_bytes,
    uint8_t degree_bytes,
    uint32_t* destination,
    size_t destination_capacity);
} // namespace diskann
