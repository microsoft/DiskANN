// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "unified_index_format.h"
#include "windows_customizations.h"

namespace diskann
{

// Streaming writer for the unified index container.
//
// Caller drives the writer in this order:
//   1) begin(npts, dim, aligned_dim, max_degree, data_type, metric, start_node)
//   2) begin_graph_region()
//      for each node N in [0, npts): write_node(coords_ptr, neighbors_ptr, degree)
//      end_graph_region()
//   3) write_medoids(medoid_ids, num_medoids)              // always called
//   4) (optional) write_pq(pivots_bytes, ..., codes_bytes, ...)
//   5) (optional) write_max_base_norm(value)
//   6) (optional) write_labels(...)
//   7) finalize() — seeks back, writes offset table and header
//
// Writer assumes nodes are appended in strict id order 0..npts-1.
class UnifiedIndexWriter
{
  public:
    explicit UnifiedIndexWriter(const std::string &path);
    ~UnifiedIndexWriter();

    void begin(uint64_t npts, uint64_t dim, uint64_t aligned_dim, uint32_t max_degree,
                                 DataTypeTag data_type, MetricTag metric, uint64_t start_node);

    void begin_graph_region();
    void write_node(const void *coords, const uint32_t *neighbors, uint32_t degree);
    void end_graph_region();

    void write_medoids(const uint32_t *medoid_ids, uint64_t num_medoids);
    void write_pq(const void *pivots_bytes, uint64_t pivots_len, const void *codes_bytes,
                                    uint64_t codes_len);
    void write_max_base_norm(float value);

    // Bitmask encoding: bitmask_bytes = packed rows of `bitmask_size_words * 8` bytes each, npts rows.
    void write_labels_bitmask(uint64_t total_labels, uint64_t universal_label,
                                                const void *dictionary_bytes, uint64_t dictionary_len,
                                                const void *bitmask_bytes, uint64_t bitmask_bytes_len);

    // Integer encoding: per_point_offsets is uint64[npts+1] into per_point_data.
    void write_labels_integer(uint64_t total_labels, uint64_t universal_label,
                                                const void *dictionary_bytes, uint64_t dictionary_len,
                                                const void *per_point_data, uint64_t per_point_data_len,
                                                const uint64_t *per_point_offsets);

    void finalize();

  private:
    void pad_to_4k();
    void write_raw(const void *bytes, uint64_t len);
    uint64_t cur_offset();

    std::string _path;
    std::ofstream _out;
    UnifiedIndexHeader _header{};
    std::vector<uint64_t> _node_offsets; // size npts+1, byte offsets within graph region
    uint64_t _graph_region_start = 0;
    uint64_t _written_nodes = 0;
    bool _graph_open = false;
    bool _finalized = false;
};

// Read-only view over a unified container file.
//
// Holds the parsed header and provides byte ranges for each region. Does not
// own the file — callers re-open as needed (e.g. AlignedFileReader for SSD path).
class UnifiedIndexReader
{
  public:
    explicit UnifiedIndexReader(const std::string &path);

    const UnifiedIndexHeader &header() const
    {
        return _header;
    }
    const std::string &path() const
    {
        return _path;
    }

    // Load and return the uint64[npts+1] offset table.
    std::vector<uint64_t> load_offset_table();

    // Load a region's bytes into a freshly-allocated buffer.
    std::vector<uint8_t> load_region(uint64_t off, uint64_t len);

    // Load a region's bytes directly into a caller-owned buffer. Caller is
    // responsible for sizing the buffer to at least `len` bytes. Avoids the
    // intermediate allocation+copy that the vector-returning overload incurs;
    // intended for hot load paths that already own (or can size) the final
    // destination storage.
    void load_region(uint64_t off, uint64_t len, uint8_t *dst);

  private:
    void parse_header();

    std::string _path;
    UnifiedIndexHeader _header{};
};

} // namespace diskann
