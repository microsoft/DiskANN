#include "in_mem_compressed_graph_store.h"
#include "utils.h"
#include <algorithm>
#include <iostream>
#include <queue>
#include "streamvbytedelta.h"
#include "streamvbyte.h"
#include "varintencode.h"
#include "varintdecode.h"
#include "lz4.h"

namespace diskann
{

// ============================================================================
// InMemCompressedGraphStore (Delta + Varint)
// ============================================================================

size_t InMemCompressedGraphStore::encode_varint(uint32_t value, uint8_t *buf)
{
    size_t bytes = 0;
    while (value >= 0x80)
    {
        buf[bytes++] = (uint8_t)(value | 0x80);
        value >>= 7;
    }
    buf[bytes++] = (uint8_t)value;
    return bytes;
}

uint32_t InMemCompressedGraphStore::decode_varint(const uint8_t *buf, size_t &pos)
{
    uint32_t result = 0;
    uint32_t shift = 0;
    while (true)
    {
        uint8_t byte = buf[pos++];
        result |= (uint32_t)(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0)
            break;
        shift += 7;
    }
    return result;
}

std::tuple<uint32_t, uint32_t, size_t> InMemCompressedGraphStore::load(const std::string &index_path_prefix,
                                                                       const size_t num_points)
{
    auto result = InMemStaticGraphReformatStore::load(index_path_prefix, num_points);
    compress_graph();
    return result;
}

void InMemCompressedGraphStore::compress_graph()
{
    _num_nodes = _node_index.size() - 1;
    _node_degree.resize(_num_nodes);
    _compressed_node_index.resize(_num_nodes + 1);

    size_t estimated_size = 0;
    for (size_t i = 0; i < _num_nodes; i++)
    {
        size_t degree = _node_index[i + 1] - _node_index[i];
        _node_degree[i] = (uint32_t)degree;
        estimated_size += degree * 5;
    }

    _compressed_graph.resize(estimated_size);

    size_t write_pos = 0;
    std::vector<uint32_t> sorted_neighbors;

    for (size_t i = 0; i < _num_nodes; i++)
    {
        _compressed_node_index[i] = write_pos;
        size_t start = _node_index[i];
        size_t end = _node_index[i + 1];
        size_t degree = end - start;

        if (degree == 0)
            continue;

        sorted_neighbors.assign(_graph.data() + start, _graph.data() + end);
        std::sort(sorted_neighbors.begin(), sorted_neighbors.end());

        write_pos += encode_varint(sorted_neighbors[0], _compressed_graph.data() + write_pos);
        for (size_t j = 1; j < degree; j++)
        {
            uint32_t delta = sorted_neighbors[j] - sorted_neighbors[j - 1];
            write_pos += encode_varint(delta, _compressed_graph.data() + write_pos);
        }
    }
    _compressed_node_index[_num_nodes] = write_pos;

    _compressed_graph.resize(write_pos);
    _compressed_graph.shrink_to_fit();
    _compressed_size = write_pos;

    size_t original_size = _graph.size() * sizeof(uint32_t);
    double ratio = (double)_compressed_size / (double)original_size * 100.0;
    diskann::cout << "Graph compression (delta+varint): " << original_size / (1024.0 * 1024.0) << " MB -> "
                  << _compressed_size / (1024.0 * 1024.0) << " MB (" << ratio << "%)" << std::endl;

    _graph.clear();
    _graph.shrink_to_fit();
    _node_index.clear();
    _node_index.shrink_to_fit();
}

const NeighborList InMemCompressedGraphStore::get_neighbours(const location_t i) const
{
    thread_local std::vector<uint32_t> decode_buf;
    assert(i < _num_nodes);
    uint32_t degree = _node_degree[i];
    if (degree == 0)
        return NeighborList(nullptr, 0);

    if (decode_buf.size() < degree)
        decode_buf.resize(degree);

    size_t pos = _compressed_node_index[i];
    const uint8_t *data = _compressed_graph.data();

    decode_buf[0] = decode_varint(data, pos);
    for (uint32_t j = 1; j < degree; j++)
        decode_buf[j] = decode_buf[j - 1] + decode_varint(data, pos);

    return NeighborList(decode_buf.data(), degree);
}

size_t InMemCompressedGraphStore::get_graph_size()
{
    return _compressed_size + _compressed_node_index.size() * sizeof(size_t) + _node_degree.size() * sizeof(uint32_t);
}

// ============================================================================
// InMemBitpackGraphStore (Delta + Bit-Packing)
// ============================================================================

static inline uint8_t bits_needed(uint32_t value)
{
    if (value == 0)
        return 0;
    uint8_t bits = 0;
    while (value > 0)
    {
        bits++;
        value >>= 1;
    }
    return bits;
}

std::tuple<uint32_t, uint32_t, size_t> InMemBitpackGraphStore::load(const std::string &index_path_prefix,
                                                                     const size_t num_points)
{
    auto result = InMemStaticGraphReformatStore::load(index_path_prefix, num_points);
    compress_graph();
    return result;
}

void InMemBitpackGraphStore::compress_graph()
{
    _num_nodes = _node_index.size() - 1;
    _node_degree.resize(_num_nodes);
    _compressed_node_index.resize(_num_nodes + 1);

    // Per node: 1 byte (bits_per_delta) + 4 bytes (first_value) + ceil(degree * bits / 8)
    size_t estimated_size = 0;
    for (size_t i = 0; i < _num_nodes; i++)
    {
        size_t degree = _node_index[i + 1] - _node_index[i];
        _node_degree[i] = (uint32_t)degree;
        // 1 (bits header) + 4 (first value) + worst case degree * 4 bytes
        estimated_size += 5 + degree * 4;
    }

    _compressed_graph.resize(estimated_size);

    size_t write_pos = 0;
    std::vector<uint32_t> sorted_neighbors;
    std::vector<uint32_t> deltas;

    for (size_t i = 0; i < _num_nodes; i++)
    {
        _compressed_node_index[i] = write_pos;
        size_t start = _node_index[i];
        size_t end = _node_index[i + 1];
        size_t degree = end - start;

        if (degree == 0)
            continue;

        sorted_neighbors.assign(_graph.data() + start, _graph.data() + end);
        std::sort(sorted_neighbors.begin(), sorted_neighbors.end());

        // Compute deltas and find max
        deltas.resize(degree > 0 ? degree - 1 : 0);
        uint32_t max_delta = 0;
        for (size_t j = 1; j < degree; j++)
        {
            deltas[j - 1] = sorted_neighbors[j] - sorted_neighbors[j - 1];
            if (deltas[j - 1] > max_delta)
                max_delta = deltas[j - 1];
        }

        uint8_t bpd = bits_needed(max_delta);

        // Write: bits_per_delta (1 byte)
        _compressed_graph[write_pos++] = bpd;

        // Write: first value (4 bytes, little-endian)
        memcpy(_compressed_graph.data() + write_pos, &sorted_neighbors[0], sizeof(uint32_t));
        write_pos += sizeof(uint32_t);

        // Write: packed deltas
        if (bpd > 0 && degree > 1)
        {
            uint64_t bit_buf = 0;
            int buf_bits = 0;

            for (size_t j = 0; j < deltas.size(); j++)
            {
                bit_buf |= ((uint64_t)deltas[j]) << buf_bits;
                buf_bits += bpd;

                while (buf_bits >= 8)
                {
                    _compressed_graph[write_pos++] = (uint8_t)(bit_buf & 0xFF);
                    bit_buf >>= 8;
                    buf_bits -= 8;
                }
            }
            // Flush remaining bits
            if (buf_bits > 0)
            {
                _compressed_graph[write_pos++] = (uint8_t)(bit_buf & 0xFF);
            }
        }
    }
    _compressed_node_index[_num_nodes] = write_pos;

    _compressed_graph.resize(write_pos);
    _compressed_graph.shrink_to_fit();
    _compressed_size = write_pos;

    size_t original_size = _graph.size() * sizeof(uint32_t);
    double ratio = (double)_compressed_size / (double)original_size * 100.0;
    diskann::cout << "Graph compression (delta+bitpack): " << original_size / (1024.0 * 1024.0) << " MB -> "
                  << _compressed_size / (1024.0 * 1024.0) << " MB (" << ratio << "%)" << std::endl;

    _graph.clear();
    _graph.shrink_to_fit();
    _node_index.clear();
    _node_index.shrink_to_fit();
}

const NeighborList InMemBitpackGraphStore::get_neighbours(const location_t i) const
{
    thread_local std::vector<uint32_t> decode_buf;
    assert(i < _num_nodes);
    uint32_t degree = _node_degree[i];
    if (degree == 0)
        return NeighborList(nullptr, 0);

    if (decode_buf.size() < degree)
        decode_buf.resize(degree);

    size_t pos = _compressed_node_index[i];
    const uint8_t *data = _compressed_graph.data();

    uint8_t bpd = data[pos++];

    // Read first value
    memcpy(&decode_buf[0], data + pos, sizeof(uint32_t));
    pos += sizeof(uint32_t);

    if (degree > 1 && bpd > 0)
    {
        uint32_t mask = (1u << bpd) - 1;
        uint64_t bit_buf = 0;
        int buf_bits = 0;

        for (uint32_t j = 1; j < degree; j++)
        {
            while (buf_bits < (int)bpd)
            {
                bit_buf |= ((uint64_t)data[pos++]) << buf_bits;
                buf_bits += 8;
            }
            uint32_t delta = (uint32_t)(bit_buf & mask);
            bit_buf >>= bpd;
            buf_bits -= bpd;

            decode_buf[j] = decode_buf[j - 1] + delta;
        }
    }
    else if (degree > 1 && bpd == 0)
    {
        // All deltas are 0, all neighbors are the same value
        for (uint32_t j = 1; j < degree; j++)
            decode_buf[j] = decode_buf[0];
    }

    return NeighborList(decode_buf.data(), degree);
}

size_t InMemBitpackGraphStore::get_graph_size()
{
    return _compressed_size + _compressed_node_index.size() * sizeof(size_t) + _node_degree.size() * sizeof(uint32_t);
}

// ============================================================================
// InMemReorderCompressedGraphStore (BFS Reorder + Delta + Varint)
// ============================================================================

size_t InMemReorderCompressedGraphStore::encode_varint(uint32_t value, uint8_t *buf)
{
    size_t bytes = 0;
    while (value >= 0x80)
    {
        buf[bytes++] = (uint8_t)(value | 0x80);
        value >>= 7;
    }
    buf[bytes++] = (uint8_t)value;
    return bytes;
}

uint32_t InMemReorderCompressedGraphStore::decode_varint(const uint8_t *buf, size_t &pos)
{
    uint32_t result = 0;
    uint32_t shift = 0;
    while (true)
    {
        uint8_t byte = buf[pos++];
        result |= (uint32_t)(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0)
            break;
        shift += 7;
    }
    return result;
}

std::tuple<uint32_t, uint32_t, size_t> InMemReorderCompressedGraphStore::load(const std::string &index_path_prefix,
                                                                               const size_t num_points)
{
    auto [nodes_read, start_node, frozen_pts] = InMemStaticGraphReformatStore::load(index_path_prefix, num_points);
    reorder_and_compress_graph(start_node);

    // Return original start_node — the Index uses original IDs for data store access.
    // get_neighbours() handles the mapping internally.
    return std::make_tuple(nodes_read, start_node, frozen_pts);
}

void InMemReorderCompressedGraphStore::reorder_and_compress_graph(uint32_t start_node)
{
    _num_nodes = _node_index.size() - 1;

    diskann::cout << "BFS reordering " << _num_nodes << " nodes from start=" << start_node << "..." << std::flush;

    _old_to_new.resize(_num_nodes, UINT32_MAX);
    _new_to_old.resize(_num_nodes);

    // BFS to assign new IDs
    uint32_t next_id = 0;
    std::queue<uint32_t> bfs_queue;

    // Start from the entry point
    _old_to_new[start_node] = next_id;
    _new_to_old[next_id] = start_node;
    next_id++;
    bfs_queue.push(start_node);

    while (!bfs_queue.empty())
    {
        uint32_t old_id = bfs_queue.front();
        bfs_queue.pop();

        size_t start = _node_index[old_id];
        size_t end = _node_index[old_id + 1];
        for (size_t j = start; j < end; j++)
        {
            uint32_t neighbor = _graph[j];
            if (_old_to_new[neighbor] == UINT32_MAX)
            {
                _old_to_new[neighbor] = next_id;
                _new_to_old[next_id] = neighbor;
                next_id++;
                bfs_queue.push(neighbor);
            }
        }
    }

    // Assign remaining unreached nodes
    for (size_t i = 0; i < _num_nodes; i++)
    {
        if (_old_to_new[i] == UINT32_MAX)
        {
            _old_to_new[i] = next_id;
            _new_to_old[next_id] = (uint32_t)i;
            next_id++;
        }
    }

    diskann::cout << "done. " << next_id << " nodes reordered." << std::endl;

    // Now build compressed graph in new ID order
    _node_degree.resize(_num_nodes);
    _compressed_node_index.resize(_num_nodes + 1);

    size_t estimated_size = 0;
    for (size_t new_id = 0; new_id < _num_nodes; new_id++)
    {
        uint32_t old_id = _new_to_old[new_id];
        size_t degree = _node_index[old_id + 1] - _node_index[old_id];
        _node_degree[new_id] = (uint32_t)degree;
        estimated_size += degree * 5;
    }

    _compressed_graph.resize(estimated_size);

    size_t write_pos = 0;
    std::vector<uint32_t> remapped_neighbors;

    for (size_t new_id = 0; new_id < _num_nodes; new_id++)
    {
        _compressed_node_index[new_id] = write_pos;
        uint32_t old_id = _new_to_old[new_id];
        size_t start = _node_index[old_id];
        size_t end = _node_index[old_id + 1];
        size_t degree = end - start;

        if (degree == 0)
            continue;

        // Remap neighbor IDs and sort
        remapped_neighbors.resize(degree);
        for (size_t j = 0; j < degree; j++)
        {
            remapped_neighbors[j] = _old_to_new[_graph[start + j]];
        }
        std::sort(remapped_neighbors.begin(), remapped_neighbors.end());

        // Delta + varint encode
        write_pos += encode_varint(remapped_neighbors[0], _compressed_graph.data() + write_pos);
        for (size_t j = 1; j < degree; j++)
        {
            uint32_t delta = remapped_neighbors[j] - remapped_neighbors[j - 1];
            write_pos += encode_varint(delta, _compressed_graph.data() + write_pos);
        }
    }
    _compressed_node_index[_num_nodes] = write_pos;

    _compressed_graph.resize(write_pos);
    _compressed_graph.shrink_to_fit();
    _compressed_size = write_pos;

    size_t original_size = _graph.size() * sizeof(uint32_t);
    double ratio = (double)_compressed_size / (double)original_size * 100.0;
    diskann::cout << "Graph compression (reorder+delta+varint): " << original_size / (1024.0 * 1024.0) << " MB -> "
                  << _compressed_size / (1024.0 * 1024.0) << " MB (" << ratio << "%)" << std::endl;

    _graph.clear();
    _graph.shrink_to_fit();
    _node_index.clear();
    _node_index.shrink_to_fit();
}

const NeighborList InMemReorderCompressedGraphStore::get_neighbours(const location_t i) const
{
    thread_local std::vector<uint32_t> decode_buf;

    // i is an original ID. Map to the reordered ID to find compressed data.
    uint32_t reordered_id = _old_to_new[i];
    assert(reordered_id < _num_nodes);
    uint32_t degree = _node_degree[reordered_id];
    if (degree == 0)
        return NeighborList(nullptr, 0);

    if (decode_buf.size() < degree)
        decode_buf.resize(degree);

    size_t pos = _compressed_node_index[reordered_id];
    const uint8_t *data = _compressed_graph.data();

    // Decode remapped IDs
    decode_buf[0] = decode_varint(data, pos);
    for (uint32_t j = 1; j < degree; j++)
        decode_buf[j] = decode_buf[j - 1] + decode_varint(data, pos);

    // Convert back to original IDs
    for (uint32_t j = 0; j < degree; j++)
        decode_buf[j] = _new_to_old[decode_buf[j]];

    return NeighborList(decode_buf.data(), degree);
}

size_t InMemReorderCompressedGraphStore::get_graph_size()
{
    return _compressed_size + _compressed_node_index.size() * sizeof(size_t) + _node_degree.size() * sizeof(uint32_t);
}

// ============================================================================
// InMemStreamVByteGraphStore (StreamVByte SIMD Delta)
// ============================================================================

std::tuple<uint32_t, uint32_t, size_t> InMemStreamVByteGraphStore::load(const std::string &index_path_prefix,
                                                                         const size_t num_points)
{
    auto result = InMemStaticGraphReformatStore::load(index_path_prefix, num_points);
    compress_graph();
    return result;
}

void InMemStreamVByteGraphStore::compress_graph()
{
    _num_nodes = _node_index.size() - 1;
    _node_degree.resize(_num_nodes);
    _compressed_node_index.resize(_num_nodes + 1);

    // Estimate: StreamVByte max compressed size per node
    size_t estimated_size = 0;
    for (size_t i = 0; i < _num_nodes; i++)
    {
        size_t degree = _node_index[i + 1] - _node_index[i];
        _node_degree[i] = (uint32_t)degree;
        if (degree > 0)
            estimated_size += streamvbyte_max_compressedbytes((uint32_t)degree);
    }

    _compressed_graph.resize(estimated_size);

    size_t write_pos = 0;
    std::vector<uint32_t> sorted_neighbors;

    for (size_t i = 0; i < _num_nodes; i++)
    {
        _compressed_node_index[i] = write_pos;
        size_t start = _node_index[i];
        size_t end = _node_index[i + 1];
        size_t degree = end - start;

        if (degree == 0)
            continue;

        sorted_neighbors.assign(_graph.data() + start, _graph.data() + end);
        std::sort(sorted_neighbors.begin(), sorted_neighbors.end());

        // StreamVByte delta encode: computes deltas starting from prev=0 internally
        size_t bytes_written = streamvbyte_delta_encode(
            sorted_neighbors.data(), (uint32_t)degree,
            _compressed_graph.data() + write_pos, 0);
        write_pos += bytes_written;
    }
    _compressed_node_index[_num_nodes] = write_pos;

    _compressed_graph.resize(write_pos + STREAMVBYTE_PADDING); // extra padding for safe decode
    _compressed_graph.shrink_to_fit();
    _compressed_size = write_pos;

    size_t original_size = _graph.size() * sizeof(uint32_t);
    double ratio = (double)_compressed_size / (double)original_size * 100.0;
    diskann::cout << "Graph compression (StreamVByte delta): " << original_size / (1024.0 * 1024.0) << " MB -> "
                  << _compressed_size / (1024.0 * 1024.0) << " MB (" << ratio << "%)" << std::endl;

    _graph.clear();
    _graph.shrink_to_fit();
    _node_index.clear();
    _node_index.shrink_to_fit();
}

const NeighborList InMemStreamVByteGraphStore::get_neighbours(const location_t i) const
{
    thread_local std::vector<uint32_t> decode_buf;
    assert(i < _num_nodes);
    uint32_t degree = _node_degree[i];
    if (degree == 0)
        return NeighborList(nullptr, 0);

    if (decode_buf.size() < degree)
        decode_buf.resize(degree);

    streamvbyte_delta_decode(
        _compressed_graph.data() + _compressed_node_index[i],
        decode_buf.data(), degree, 0);

    return NeighborList(decode_buf.data(), degree);
}

size_t InMemStreamVByteGraphStore::get_graph_size()
{
    return _compressed_size + _compressed_node_index.size() * sizeof(size_t) + _node_degree.size() * sizeof(uint32_t);
}

// ============================================================================
// InMemReorderStreamVByteGraphStore (BFS Reorder + StreamVByte SIMD Delta)
// ============================================================================

std::tuple<uint32_t, uint32_t, size_t> InMemReorderStreamVByteGraphStore::load(const std::string &index_path_prefix,
                                                                                const size_t num_points)
{
    auto [nodes_read, start_node, frozen_pts] = InMemStaticGraphReformatStore::load(index_path_prefix, num_points);
    reorder_and_compress_graph(start_node);
    return std::make_tuple(nodes_read, start_node, frozen_pts);
}

void InMemReorderStreamVByteGraphStore::reorder_and_compress_graph(uint32_t start_node)
{
    _num_nodes = _node_index.size() - 1;

    diskann::cout << "BFS reordering " << _num_nodes << " nodes from start=" << start_node << "..." << std::flush;

    _old_to_new.resize(_num_nodes, UINT32_MAX);
    _new_to_old.resize(_num_nodes);

    uint32_t next_id = 0;
    std::queue<uint32_t> bfs_queue;

    _old_to_new[start_node] = next_id;
    _new_to_old[next_id] = start_node;
    next_id++;
    bfs_queue.push(start_node);

    while (!bfs_queue.empty())
    {
        uint32_t old_id = bfs_queue.front();
        bfs_queue.pop();

        size_t start = _node_index[old_id];
        size_t end = _node_index[old_id + 1];
        for (size_t j = start; j < end; j++)
        {
            uint32_t neighbor = _graph[j];
            if (_old_to_new[neighbor] == UINT32_MAX)
            {
                _old_to_new[neighbor] = next_id;
                _new_to_old[next_id] = neighbor;
                next_id++;
                bfs_queue.push(neighbor);
            }
        }
    }

    for (size_t i = 0; i < _num_nodes; i++)
    {
        if (_old_to_new[i] == UINT32_MAX)
        {
            _old_to_new[i] = next_id;
            _new_to_old[next_id] = (uint32_t)i;
            next_id++;
        }
    }

    diskann::cout << "done. " << next_id << " nodes reordered." << std::endl;

    // Compress in reordered ID space using StreamVByte
    _node_degree.resize(_num_nodes);
    _compressed_node_index.resize(_num_nodes + 1);

    size_t estimated_size = 0;
    for (size_t new_id = 0; new_id < _num_nodes; new_id++)
    {
        uint32_t old_id = _new_to_old[new_id];
        size_t degree = _node_index[old_id + 1] - _node_index[old_id];
        _node_degree[new_id] = (uint32_t)degree;
        if (degree > 0)
            estimated_size += streamvbyte_max_compressedbytes((uint32_t)degree);
    }

    _compressed_graph.resize(estimated_size);

    size_t write_pos = 0;
    std::vector<uint32_t> remapped_neighbors;

    for (size_t new_id = 0; new_id < _num_nodes; new_id++)
    {
        _compressed_node_index[new_id] = write_pos;
        uint32_t old_id = _new_to_old[new_id];
        size_t start = _node_index[old_id];
        size_t end = _node_index[old_id + 1];
        size_t degree = end - start;

        if (degree == 0)
            continue;

        remapped_neighbors.resize(degree);
        for (size_t j = 0; j < degree; j++)
            remapped_neighbors[j] = _old_to_new[_graph[start + j]];
        std::sort(remapped_neighbors.begin(), remapped_neighbors.end());

        size_t bytes_written = streamvbyte_delta_encode(
            remapped_neighbors.data(), (uint32_t)degree,
            _compressed_graph.data() + write_pos, 0);
        write_pos += bytes_written;
    }
    _compressed_node_index[_num_nodes] = write_pos;

    _compressed_graph.resize(write_pos + STREAMVBYTE_PADDING);
    _compressed_graph.shrink_to_fit();
    _compressed_size = write_pos;

    size_t original_size = _graph.size() * sizeof(uint32_t);
    double ratio = (double)_compressed_size / (double)original_size * 100.0;
    diskann::cout << "Graph compression (reorder+StreamVByte): " << original_size / (1024.0 * 1024.0) << " MB -> "
                  << _compressed_size / (1024.0 * 1024.0) << " MB (" << ratio << "%)" << std::endl;

    _graph.clear();
    _graph.shrink_to_fit();
    _node_index.clear();
    _node_index.shrink_to_fit();
}

const NeighborList InMemReorderStreamVByteGraphStore::get_neighbours(const location_t i) const
{
    thread_local std::vector<uint32_t> decode_buf;

    uint32_t reordered_id = _old_to_new[i];
    assert(reordered_id < _num_nodes);
    uint32_t degree = _node_degree[reordered_id];
    if (degree == 0)
        return NeighborList(nullptr, 0);

    if (decode_buf.size() < degree)
        decode_buf.resize(degree);

    streamvbyte_delta_decode(
        _compressed_graph.data() + _compressed_node_index[reordered_id],
        decode_buf.data(), degree, 0);

    // Convert back to original IDs
    for (uint32_t j = 0; j < degree; j++)
        decode_buf[j] = _new_to_old[decode_buf[j]];

    return NeighborList(decode_buf.data(), degree);
}

size_t InMemReorderStreamVByteGraphStore::get_graph_size()
{
    return _compressed_size + _compressed_node_index.size() * sizeof(size_t) + _node_degree.size() * sizeof(uint32_t) +
           _old_to_new.size() * sizeof(uint32_t) + _new_to_old.size() * sizeof(uint32_t);
}

// ============================================================================
// InMemMaskedVByteGraphStore (MaskedVByte BMI2 Delta)
// ============================================================================

std::tuple<uint32_t, uint32_t, size_t> InMemMaskedVByteGraphStore::load(const std::string &index_path_prefix,
                                                                         const size_t num_points)
{
    auto result = InMemStaticGraphReformatStore::load(index_path_prefix, num_points);
    compress_graph();
    return result;
}

void InMemMaskedVByteGraphStore::compress_graph()
{
    _num_nodes = _node_index.size() - 1;
    _node_degree.resize(_num_nodes);
    _compressed_node_index.resize(_num_nodes + 1);

    // MaskedVByte worst case: 5 bytes per uint32
    size_t estimated_size = 0;
    for (size_t i = 0; i < _num_nodes; i++)
    {
        size_t degree = _node_index[i + 1] - _node_index[i];
        _node_degree[i] = (uint32_t)degree;
        estimated_size += degree * 5;
    }

    _compressed_graph.resize(estimated_size);

    size_t write_pos = 0;
    std::vector<uint32_t> sorted_neighbors;

    for (size_t i = 0; i < _num_nodes; i++)
    {
        _compressed_node_index[i] = write_pos;
        size_t start = _node_index[i];
        size_t end = _node_index[i + 1];
        size_t degree = end - start;

        if (degree == 0)
            continue;

        sorted_neighbors.assign(_graph.data() + start, _graph.data() + end);
        std::sort(sorted_neighbors.begin(), sorted_neighbors.end());

        size_t bytes_written = vbyte_encode_delta(
            sorted_neighbors.data(), degree,
            _compressed_graph.data() + write_pos, 0);
        write_pos += bytes_written;
    }
    _compressed_node_index[_num_nodes] = write_pos;

    _compressed_graph.resize(write_pos + 16); // padding for safe SIMD decode
    _compressed_graph.shrink_to_fit();
    _compressed_size = write_pos;

    size_t original_size = _graph.size() * sizeof(uint32_t);
    double ratio = (double)_compressed_size / (double)original_size * 100.0;
    diskann::cout << "Graph compression (MaskedVByte delta): " << original_size / (1024.0 * 1024.0) << " MB -> "
                  << _compressed_size / (1024.0 * 1024.0) << " MB (" << ratio << "%)" << std::endl;

    _graph.clear();
    _graph.shrink_to_fit();
    _node_index.clear();
    _node_index.shrink_to_fit();
}

const NeighborList InMemMaskedVByteGraphStore::get_neighbours(const location_t i) const
{
    thread_local std::vector<uint32_t> decode_buf;
    assert(i < _num_nodes);
    uint32_t degree = _node_degree[i];
    if (degree == 0)
        return NeighborList(nullptr, 0);

    if (decode_buf.size() < degree)
        decode_buf.resize(degree);

    masked_vbyte_decode_delta(
        _compressed_graph.data() + _compressed_node_index[i],
        decode_buf.data(), degree, 0);

    return NeighborList(decode_buf.data(), degree);
}

size_t InMemMaskedVByteGraphStore::get_graph_size()
{
    return _compressed_size + _compressed_node_index.size() * sizeof(size_t) + _node_degree.size() * sizeof(uint32_t);
}

// ============================================================================
// InMemLz4GraphStore (LZ4 per-node compression)
// ============================================================================

std::tuple<uint32_t, uint32_t, size_t> InMemLz4GraphStore::load(const std::string &index_path_prefix,
                                                                  const size_t num_points)
{
    auto result = InMemStaticGraphReformatStore::load(index_path_prefix, num_points);
    compress_graph();
    return result;
}

void InMemLz4GraphStore::compress_graph()
{
    _num_nodes = _node_index.size() - 1;
    _node_degree.resize(_num_nodes);
    _compressed_node_index.resize(_num_nodes + 1);

    // LZ4 worst case bound
    size_t estimated_size = 0;
    for (size_t i = 0; i < _num_nodes; i++)
    {
        size_t degree = _node_index[i + 1] - _node_index[i];
        _node_degree[i] = (uint32_t)degree;
        int src_size = (int)(degree * sizeof(uint32_t));
        estimated_size += (size_t)LZ4_compressBound(src_size) + 1; // +1 for empty nodes
    }

    _compressed_graph.resize(estimated_size);

    size_t write_pos = 0;
    std::vector<uint32_t> sorted_neighbors;
    // Pre-compute deltas as uint32 for better LZ4 compression
    std::vector<uint32_t> deltas;

    for (size_t i = 0; i < _num_nodes; i++)
    {
        _compressed_node_index[i] = write_pos;
        size_t start = _node_index[i];
        size_t end = _node_index[i + 1];
        size_t degree = end - start;

        if (degree == 0)
            continue;

        sorted_neighbors.assign(_graph.data() + start, _graph.data() + end);
        std::sort(sorted_neighbors.begin(), sorted_neighbors.end());

        // Delta encode first for better LZ4 compression
        deltas.resize(degree);
        deltas[0] = sorted_neighbors[0];
        for (size_t j = 1; j < degree; j++)
            deltas[j] = sorted_neighbors[j] - sorted_neighbors[j - 1];

        int src_size = (int)(degree * sizeof(uint32_t));
        int max_dst = LZ4_compressBound(src_size);
        int compressed = LZ4_compress_default(
            (const char *)deltas.data(),
            (char *)(_compressed_graph.data() + write_pos),
            src_size, max_dst);

        write_pos += (size_t)compressed;
    }
    _compressed_node_index[_num_nodes] = write_pos;

    _compressed_graph.resize(write_pos);
    _compressed_graph.shrink_to_fit();
    _compressed_size = write_pos;

    size_t original_size = _graph.size() * sizeof(uint32_t);
    double ratio = (double)_compressed_size / (double)original_size * 100.0;
    diskann::cout << "Graph compression (LZ4 delta per-node): " << original_size / (1024.0 * 1024.0) << " MB -> "
                  << _compressed_size / (1024.0 * 1024.0) << " MB (" << ratio << "%)" << std::endl;

    _graph.clear();
    _graph.shrink_to_fit();
    _node_index.clear();
    _node_index.shrink_to_fit();
}

const NeighborList InMemLz4GraphStore::get_neighbours(const location_t i) const
{
    thread_local std::vector<uint32_t> decode_buf;
    assert(i < _num_nodes);
    uint32_t degree = _node_degree[i];
    if (degree == 0)
        return NeighborList(nullptr, 0);

    if (decode_buf.size() < degree)
        decode_buf.resize(degree);

    size_t comp_start = _compressed_node_index[i];
    size_t comp_end = _compressed_node_index[i + 1];
    int comp_size = (int)(comp_end - comp_start);
    int dst_size = (int)(degree * sizeof(uint32_t));

    LZ4_decompress_safe(
        (const char *)(_compressed_graph.data() + comp_start),
        (char *)decode_buf.data(),
        comp_size, dst_size);

    // Undo delta encoding (prefix sum)
    for (uint32_t j = 1; j < degree; j++)
        decode_buf[j] += decode_buf[j - 1];

    return NeighborList(decode_buf.data(), degree);
}

size_t InMemLz4GraphStore::get_graph_size()
{
    return _compressed_size + _compressed_node_index.size() * sizeof(size_t) + _node_degree.size() * sizeof(uint32_t);
}

} // namespace diskann
