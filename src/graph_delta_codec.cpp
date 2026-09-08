// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "graph_delta_codec.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

namespace diskann
{
namespace
{
uint8_t bit_width(uint32_t value)
{
    uint8_t width = 0;
    while (value != 0)
    {
        ++width;
        value >>= 1;
    }
    return width;
}

uint32_t read_little_endian(const uint8_t* source, uint8_t bytes)
{
    uint32_t value = 0;
    for (uint8_t i = 0; i < bytes; ++i)
        value |= static_cast<uint32_t>(source[i]) << (i * 8);
    return value;
}

void write_little_endian(uint8_t* destination, uint32_t value, uint8_t bytes)
{
    for (uint8_t i = 0; i < bytes; ++i)
        destination[i] = static_cast<uint8_t>(value >> (i * 8));
}

uint8_t byte_width(uint32_t value)
{
    return std::max<uint8_t>(1, static_cast<uint8_t>((bit_width(value) + 7) / 8));
}

uint8_t stream_vbyte_control(
    const std::vector<uint32_t>& sorted_neighbors,
    size_t delta_begin,
    size_t delta_count,
    size_t& data_bytes)
{
    uint8_t control = 0;
    data_bytes = 0;
    for (size_t lane = 0; lane < delta_count; ++lane)
    {
        const size_t index = delta_begin + lane;
        const uint32_t delta = sorted_neighbors[index] - sorted_neighbors[index - 1];
        const uint8_t bytes = byte_width(delta);
        control |= static_cast<uint8_t>((bytes - 1) << (lane * 2));
        data_bytes += bytes;
    }
    return control;
}

#ifdef USE_AVX2
const std::array<std::array<uint8_t, 16>, 256>& stream_vbyte_shuffle_table()
{
    static const auto table = [] {
        std::array<std::array<uint8_t, 16>, 256> result{};
        for (size_t control = 0; control < result.size(); ++control)
        {
            uint8_t input_offset = 0;
            for (size_t lane = 0; lane < 4; ++lane)
            {
                const uint8_t bytes =
                    static_cast<uint8_t>(((control >> (lane * 2)) & 0x3) + 1);
                for (size_t byte = 0; byte < 4; ++byte)
                {
                    result[control][lane * 4 + byte] =
                        byte < bytes ? static_cast<uint8_t>(input_offset + byte) : 0x80;
                }
                input_offset = static_cast<uint8_t>(input_offset + bytes);
            }
        }
        return result;
    }();
    return table;
}

void decode_stream_vbyte_group_simd(
    const uint8_t* input,
    size_t input_bytes,
    uint8_t control,
    uint32_t previous,
    uint32_t* output)
{
    alignas(16) uint8_t padded_input[16] = {};
    std::memcpy(padded_input, input, input_bytes);

    const auto& mask_bytes = stream_vbyte_shuffle_table()[control];
    const __m128i packed = _mm_load_si128(reinterpret_cast<const __m128i*>(padded_input));
    const __m128i mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(mask_bytes.data()));
    __m128i values = _mm_shuffle_epi8(packed, mask);
    values = _mm_add_epi32(values, _mm_slli_si128(values, 4));
    values = _mm_add_epi32(values, _mm_slli_si128(values, 8));
    values = _mm_add_epi32(values, _mm_set1_epi32(static_cast<int>(previous)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output), values);
}
#endif
} // namespace

uint8_t graph_id_bytes(size_t num_points)
{
    if (num_points == 0 || num_points > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Graph node count must be in [1, UINT32_MAX]");
    return std::max<uint8_t>(
        1,
        static_cast<uint8_t>((bit_width(static_cast<uint32_t>(num_points - 1)) + 7) / 8));
}

uint8_t graph_degree_bytes(uint32_t max_degree)
{
    return max_degree <= std::numeric_limits<uint8_t>::max() ? 1 : 2;
}

StreamVByteNodeEncoding prepare_stream_vbyte_node(
    const NeighborList& neighbors,
    size_t num_points,
    uint8_t id_bytes,
    uint8_t degree_bytes,
    std::vector<uint32_t>& sorted_neighbors)
{
    if (id_bytes == 0 || id_bytes > sizeof(uint32_t))
        throw std::invalid_argument("Invalid graph ID byte width");
    if (degree_bytes != 1 && degree_bytes != 2)
        throw std::invalid_argument("Invalid graph degree byte width");
    if (degree_bytes == 1 && neighbors.size() > std::numeric_limits<uint8_t>::max())
        throw std::invalid_argument("Graph degree exceeds one-byte encoding");
    if (neighbors.size() > std::numeric_limits<uint16_t>::max())
        throw std::invalid_argument("Graph degree exceeds two-byte encoding");

    sorted_neighbors.clear();
    sorted_neighbors.reserve(neighbors.size());
    for (uint32_t neighbor : neighbors)
    {
        if (neighbor >= num_points)
            throw std::invalid_argument("Graph contains an out-of-range neighbor ID");
        sorted_neighbors.push_back(neighbor);
    }
    std::sort(sorted_neighbors.begin(), sorted_neighbors.end());

    StreamVByteNodeEncoding encoding;
    encoding.degree = static_cast<uint32_t>(sorted_neighbors.size());
    encoding.encoded_size = degree_bytes;
    if (sorted_neighbors.empty())
        return encoding;

    const size_t delta_count = sorted_neighbors.size() - 1;
    encoding.control_bytes = (delta_count + 3) / 4;
    encoding.encoded_size += id_bytes + encoding.control_bytes;
    for (size_t delta_begin = 1; delta_begin < sorted_neighbors.size(); delta_begin += 4)
    {
        const size_t group_count = std::min<size_t>(4, sorted_neighbors.size() - delta_begin);
        size_t group_bytes = 0;
        stream_vbyte_control(sorted_neighbors, delta_begin, group_count, group_bytes);
        encoding.data_bytes += group_bytes;
    }
    encoding.encoded_size += encoding.data_bytes;
    return encoding;
}

void encode_stream_vbyte_node(
    const std::vector<uint32_t>& sorted_neighbors,
    const StreamVByteNodeEncoding& encoding,
    uint8_t id_bytes,
    uint8_t degree_bytes,
    uint8_t* destination,
    size_t destination_size)
{
    if (destination == nullptr || destination_size != encoding.encoded_size ||
        sorted_neighbors.size() != encoding.degree)
    {
        throw std::invalid_argument("Invalid Stream VByte destination");
    }

    write_little_endian(destination, encoding.degree, degree_bytes);
    if (sorted_neighbors.empty())
        return;

    uint8_t* controls = destination + degree_bytes + id_bytes;
    uint8_t* data = controls + encoding.control_bytes;
    write_little_endian(destination + degree_bytes, sorted_neighbors[0], id_bytes);

    size_t control_index = 0;
    for (size_t delta_begin = 1; delta_begin < sorted_neighbors.size(); delta_begin += 4)
    {
        const size_t group_count = std::min<size_t>(4, sorted_neighbors.size() - delta_begin);
        size_t group_bytes = 0;
        const uint8_t control =
            stream_vbyte_control(sorted_neighbors, delta_begin, group_count, group_bytes);
        controls[control_index++] = control;

        for (size_t lane = 0; lane < group_count; ++lane)
        {
            const size_t index = delta_begin + lane;
            const uint32_t delta = sorted_neighbors[index] - sorted_neighbors[index - 1];
            const uint8_t bytes = byte_width(delta);
            write_little_endian(data, delta, bytes);
            data += bytes;
        }
    }
}

size_t decode_stream_vbyte_node(
    const uint8_t* encoded,
    size_t encoded_size,
    size_t num_points,
    uint8_t id_bytes,
    uint8_t degree_bytes,
    uint32_t* destination,
    size_t destination_capacity)
{
    if (encoded == nullptr || encoded_size < degree_bytes)
        throw std::invalid_argument("Truncated Stream VByte node");

    const uint32_t degree = read_little_endian(encoded, degree_bytes);
    if (degree > destination_capacity)
        throw std::invalid_argument("Stream VByte destination is too small");
    if (degree == 0)
    {
        if (encoded_size != degree_bytes)
            throw std::invalid_argument("Invalid empty Stream VByte node size");
        return 0;
    }
    if (destination == nullptr || id_bytes == 0 || id_bytes > sizeof(uint32_t))
        throw std::invalid_argument("Invalid Stream VByte encoding");

    const size_t delta_count = degree - 1;
    const size_t control_bytes = (delta_count + 3) / 4;
    const size_t header_size = degree_bytes + id_bytes + control_bytes;
    if (encoded_size < header_size)
        throw std::invalid_argument("Truncated Stream VByte header");

    uint32_t previous = read_little_endian(encoded + degree_bytes, id_bytes);
    if (previous >= num_points)
        throw std::invalid_argument("Decoded graph ID is out of range");
    destination[0] = previous;

    const uint8_t* controls = encoded + degree_bytes + id_bytes;
    const uint8_t* data = controls + control_bytes;
    const uint8_t* const end = encoded + encoded_size;
    size_t output_index = 1;

    for (size_t control_index = 0; control_index < control_bytes; ++control_index)
    {
        const uint8_t control = controls[control_index];
        const size_t group_count = std::min<size_t>(4, degree - output_index);
        size_t group_bytes = 0;
        std::array<uint8_t, 4> lengths{};
        for (size_t lane = 0; lane < group_count; ++lane)
        {
            lengths[lane] = static_cast<uint8_t>(((control >> (lane * 2)) & 0x3) + 1);
            group_bytes += lengths[lane];
        }
        if (static_cast<size_t>(end - data) < group_bytes)
            throw std::invalid_argument("Truncated Stream VByte payload");

#ifdef USE_AVX2
        alignas(16) uint32_t decoded_group[4] = {};
        decode_stream_vbyte_group_simd(data, group_bytes, control, previous, decoded_group);
        for (size_t lane = 0; lane < group_count; ++lane)
        {
            if (decoded_group[lane] >= num_points || decoded_group[lane] < previous)
                throw std::invalid_argument("Decoded graph ID is out of range");
            destination[output_index++] = decoded_group[lane];
        }
        previous = decoded_group[group_count - 1];
        data += group_bytes;
#else
        for (size_t lane = 0; lane < group_count; ++lane)
        {
            const uint32_t delta = read_little_endian(data, lengths[lane]);
            data += lengths[lane];
            if (delta > std::numeric_limits<uint32_t>::max() - previous)
                throw std::invalid_argument("Decoded graph ID overflow");
            previous += delta;
            if (previous >= num_points)
                throw std::invalid_argument("Decoded graph ID is out of range");
            destination[output_index++] = previous;
        }
#endif
    }

    if (data != end)
        throw std::invalid_argument("Invalid Stream VByte payload size");
    return degree;
}
} // namespace diskann
