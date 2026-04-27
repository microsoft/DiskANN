#pragma once

#include "in_mem_static_graph_reformat_store.h"
#include <vector>
#include <cstdint>

namespace diskann
{

// Compressed graph store using delta encoding + varint compression.
class InMemCompressedGraphStore : public InMemStaticGraphReformatStore
{
  public:
    InMemCompressedGraphStore(const size_t total_pts, const size_t reserve_graph_degree)
        : InMemStaticGraphReformatStore(total_pts, reserve_graph_degree)
    {
    }

    const NeighborList get_neighbours(const location_t i) const override;
    size_t get_graph_size() override;
    std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                 const size_t num_points) override;

  private:
    static size_t encode_varint(uint32_t value, uint8_t *buf);
    static uint32_t decode_varint(const uint8_t *buf, size_t &pos);
    void compress_graph();

    std::vector<uint8_t> _compressed_graph;
    std::vector<size_t> _compressed_node_index;
    std::vector<uint32_t> _node_degree;
    size_t _compressed_size = 0;
    size_t _num_nodes = 0;
};

// Compressed graph store using delta encoding + bit-packing.
// Per-node: stores bits_per_delta (uint8) + first value (uint32) + packed deltas.
class InMemBitpackGraphStore : public InMemStaticGraphReformatStore
{
  public:
    InMemBitpackGraphStore(const size_t total_pts, const size_t reserve_graph_degree)
        : InMemStaticGraphReformatStore(total_pts, reserve_graph_degree)
    {
    }

    const NeighborList get_neighbours(const location_t i) const override;
    size_t get_graph_size() override;
    std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                 const size_t num_points) override;

  private:
    void compress_graph();

    std::vector<uint8_t> _compressed_graph;
    std::vector<size_t> _compressed_node_index;
    std::vector<uint32_t> _node_degree;
    size_t _compressed_size = 0;
    size_t _num_nodes = 0;
};

// Compressed graph with BFS node ID reordering + delta + varint.
// Reorders node IDs so graph-local neighbors get nearby IDs, making deltas smaller.
class InMemReorderCompressedGraphStore : public InMemStaticGraphReformatStore
{
  public:
    InMemReorderCompressedGraphStore(const size_t total_pts, const size_t reserve_graph_degree)
        : InMemStaticGraphReformatStore(total_pts, reserve_graph_degree)
    {
    }

    const NeighborList get_neighbours(const location_t i) const override;
    size_t get_graph_size() override;
    std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                 const size_t num_points) override;

    // Returns the reorder map (old_id -> new_id) for data store reordering
    const std::vector<uint32_t> &get_old_to_new_map() const
    {
        return _old_to_new;
    }
    const std::vector<uint32_t> &get_new_to_old_map() const
    {
        return _new_to_old;
    }

  private:
    static size_t encode_varint(uint32_t value, uint8_t *buf);
    static uint32_t decode_varint(const uint8_t *buf, size_t &pos);
    void reorder_and_compress_graph(uint32_t start_node);

    std::vector<uint32_t> _old_to_new; // mapping from original ID to reordered ID
    std::vector<uint32_t> _new_to_old; // mapping from reordered ID to original ID

    std::vector<uint8_t> _compressed_graph;
    std::vector<size_t> _compressed_node_index;
    std::vector<uint32_t> _node_degree;
    size_t _compressed_size = 0;
    size_t _num_nodes = 0;
};

// Compressed graph using StreamVByte SIMD-accelerated delta encoding.
// Uses lemire/streamvbyte library for 4-8x faster decode than scalar varint.
class InMemStreamVByteGraphStore : public InMemStaticGraphReformatStore
{
  public:
    InMemStreamVByteGraphStore(const size_t total_pts, const size_t reserve_graph_degree)
        : InMemStaticGraphReformatStore(total_pts, reserve_graph_degree)
    {
    }

    const NeighborList get_neighbours(const location_t i) const override;
    size_t get_graph_size() override;
    std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                 const size_t num_points) override;

  private:
    void compress_graph();

    std::vector<uint8_t> _compressed_graph;
    std::vector<size_t> _compressed_node_index;
    std::vector<uint32_t> _node_degree;
    size_t _compressed_size = 0;
    size_t _num_nodes = 0;
};

// Compressed graph with BFS reordering + StreamVByte SIMD delta encoding.
// Best combination: BFS makes deltas small, StreamVByte decodes them fast.
class InMemReorderStreamVByteGraphStore : public InMemStaticGraphReformatStore
{
  public:
    InMemReorderStreamVByteGraphStore(const size_t total_pts, const size_t reserve_graph_degree)
        : InMemStaticGraphReformatStore(total_pts, reserve_graph_degree)
    {
    }

    const NeighborList get_neighbours(const location_t i) const override;
    size_t get_graph_size() override;
    std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                 const size_t num_points) override;

  private:
    void reorder_and_compress_graph(uint32_t start_node);

    std::vector<uint32_t> _old_to_new;
    std::vector<uint32_t> _new_to_old;

    std::vector<uint8_t> _compressed_graph;
    std::vector<size_t> _compressed_node_index;
    std::vector<uint32_t> _node_degree;
    size_t _compressed_size = 0;
    size_t _num_nodes = 0;
};

// Compressed graph using MaskedVByte (BMI2 SIMD varint decode).
class InMemMaskedVByteGraphStore : public InMemStaticGraphReformatStore
{
  public:
    InMemMaskedVByteGraphStore(const size_t total_pts, const size_t reserve_graph_degree)
        : InMemStaticGraphReformatStore(total_pts, reserve_graph_degree)
    {
    }

    const NeighborList get_neighbours(const location_t i) const override;
    size_t get_graph_size() override;
    std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                 const size_t num_points) override;

  private:
    void compress_graph();

    std::vector<uint8_t> _compressed_graph;
    std::vector<size_t> _compressed_node_index;
    std::vector<uint32_t> _node_degree;
    size_t _compressed_size = 0;
    size_t _num_nodes = 0;
};

// Compressed graph using LZ4 per-node compression (upper bound experiment).
class InMemLz4GraphStore : public InMemStaticGraphReformatStore
{
  public:
    InMemLz4GraphStore(const size_t total_pts, const size_t reserve_graph_degree)
        : InMemStaticGraphReformatStore(total_pts, reserve_graph_degree)
    {
    }

    const NeighborList get_neighbours(const location_t i) const override;
    size_t get_graph_size() override;
    std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                 const size_t num_points) override;

  private:
    void compress_graph();

    std::vector<uint8_t> _compressed_graph;
    std::vector<size_t> _compressed_node_index;
    std::vector<uint32_t> _node_degree;
    size_t _compressed_size = 0;
    size_t _num_nodes = 0;
};

} // namespace diskann
