// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "unified_index_io.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "logger.h"

namespace diskann
{

namespace
{
constexpr uint8_t ZERO_SECTOR[UNIFIED_FORMAT_ALIGN] = {};
}

UnifiedIndexWriter::UnifiedIndexWriter(const std::string &path) : _path(path)
{
    _out.exceptions(std::ios::badbit | std::ios::failbit);
    _out.open(_path, std::ios::binary | std::ios::out | std::ios::trunc);
}

UnifiedIndexWriter::~UnifiedIndexWriter()
{
    if (_out.is_open() && !_finalized)
    {
        // Caller forgot to finalize. Close anyway to flush OS handle; file will
        // be unusable but we don't throw from a destructor.
        _out.close();
    }
}

uint64_t UnifiedIndexWriter::cur_offset()
{
    return static_cast<uint64_t>(_out.tellp());
}

void UnifiedIndexWriter::write_raw(const void *bytes, uint64_t len)
{
    if (len == 0)
        return;
    _out.write(static_cast<const char *>(bytes), static_cast<std::streamsize>(len));
}

void UnifiedIndexWriter::pad_to_4k()
{
    uint64_t cur = cur_offset();
    uint64_t aligned = align_up_4k(cur);
    if (aligned == cur)
        return;
    write_raw(ZERO_SECTOR, aligned - cur);
}

void UnifiedIndexWriter::begin(uint64_t npts, uint64_t dim, uint64_t aligned_dim, uint32_t max_degree,
                               DataTypeTag data_type, MetricTag metric, uint64_t start_node)
{
    if (npts == 0)
        throw std::invalid_argument("UnifiedIndexWriter::begin: npts must be > 0");

    _header.magic = UNIFIED_FORMAT_MAGIC;
    _header.version = UNIFIED_FORMAT_VERSION;
    _header.data_type = data_type;
    _header.metric = metric;
    _header.npts = npts;
    _header.dim = dim;
    _header.aligned_dim = aligned_dim;
    _header.max_degree = max_degree;
    _header.flags = 0;
    _header.start_node = start_node;
    _header.label_encoding = LabelEncoding::None;

    _node_offsets.assign(npts + 1, 0);

    // Reserve header sector — written for real in finalize().
    write_raw(ZERO_SECTOR, UNIFIED_FORMAT_ALIGN);

    // Reserve offset table region — filled in finalize().
    _header.offset_table_off = cur_offset();
    _header.offset_table_len = (npts + 1) * sizeof(uint64_t);
    const uint64_t table_padded = align_up_4k(_header.offset_table_len);
    for (uint64_t written = 0; written < table_padded; written += UNIFIED_FORMAT_ALIGN)
    {
        const uint64_t chunk = std::min<uint64_t>(UNIFIED_FORMAT_ALIGN, table_padded - written);
        write_raw(ZERO_SECTOR, chunk);
    }
}

void UnifiedIndexWriter::begin_graph_region()
{
    if (_graph_open)
        throw std::logic_error("UnifiedIndexWriter: graph region already open");
    _graph_open = true;
    _header.graph_region_off = cur_offset();
    _graph_region_start = _header.graph_region_off;
}

void UnifiedIndexWriter::write_node(const void *coords, const uint32_t *neighbors, uint32_t degree)
{
    if (!_graph_open)
        throw std::logic_error("UnifiedIndexWriter: write_node before begin_graph_region");
    if (_written_nodes >= _header.npts)
        throw std::logic_error("UnifiedIndexWriter: too many nodes written");

    const uint64_t coords_bytes = _header.dim * [&]() -> uint64_t {
        switch (_header.data_type)
        {
        case DataTypeTag::Float:
            return sizeof(float);
        case DataTypeTag::Uint8:
            return sizeof(uint8_t);
        case DataTypeTag::Int8:
            return sizeof(int8_t);
        }
        throw std::logic_error("UnifiedIndexWriter: unknown data type");
    }();

    const uint64_t id = _written_nodes;
    _node_offsets[id] = cur_offset() - _graph_region_start;
    write_raw(coords, coords_bytes);
    write_raw(neighbors, static_cast<uint64_t>(degree) * sizeof(uint32_t));
    _node_offsets[id + 1] = cur_offset() - _graph_region_start;
    ++_written_nodes;
}

void UnifiedIndexWriter::end_graph_region()
{
    if (!_graph_open)
        throw std::logic_error("UnifiedIndexWriter: end_graph_region without begin_graph_region");
    _header.graph_region_len = cur_offset() - _header.graph_region_off;
    pad_to_4k();
    _graph_open = false;
}

void UnifiedIndexWriter::write_medoids(const uint32_t *medoid_ids, uint64_t num_medoids)
{
    _header.medoids_off = cur_offset();
    _header.medoids_len = num_medoids * sizeof(uint32_t);
    write_raw(medoid_ids, _header.medoids_len);
    pad_to_4k();
}

void UnifiedIndexWriter::write_pq(const void *pivots_bytes, uint64_t pivots_len, const void *codes_bytes,
                                  uint64_t codes_len)
{
    _header.flags |= HAS_PQ;
    _header.pq_pivots_off = cur_offset();
    _header.pq_pivots_len = pivots_len;
    write_raw(pivots_bytes, pivots_len);
    pad_to_4k();

    _header.pq_codes_off = cur_offset();
    _header.pq_codes_len = codes_len;
    write_raw(codes_bytes, codes_len);
    pad_to_4k();
}

void UnifiedIndexWriter::write_max_base_norm(float value)
{
    _header.flags |= HAS_MAX_BASE_NORM;
    _header.max_base_norm_off = cur_offset();
    _header.max_base_norm_len = sizeof(float);
    write_raw(&value, sizeof(float));
    pad_to_4k();
}

void UnifiedIndexWriter::write_labels_bitmask(uint64_t total_labels, uint64_t universal_label,
                                              const void *dictionary_bytes, uint64_t dictionary_len,
                                              const void *bitmask_bytes, uint64_t bitmask_bytes_len)
{
    _header.flags |= HAS_LABELS;
    _header.label_encoding = LabelEncoding::Bitmask;
    _header.total_labels = total_labels;
    _header.universal_label = universal_label;

    _header.label_dictionary_off = cur_offset();
    _header.label_dictionary_len = dictionary_len;
    write_raw(dictionary_bytes, dictionary_len);
    pad_to_4k();

    _header.per_point_labels_off = cur_offset();
    _header.per_point_labels_len = bitmask_bytes_len;
    write_raw(bitmask_bytes, bitmask_bytes_len);
    pad_to_4k();
}

void UnifiedIndexWriter::write_labels_integer(uint64_t total_labels, uint64_t universal_label,
                                              const void *dictionary_bytes, uint64_t dictionary_len,
                                              const void *per_point_data, uint64_t per_point_data_len,
                                              const uint64_t *per_point_offsets)
{
    _header.flags |= HAS_LABELS;
    _header.label_encoding = LabelEncoding::Integer;
    _header.total_labels = total_labels;
    _header.universal_label = universal_label;

    _header.label_dictionary_off = cur_offset();
    _header.label_dictionary_len = dictionary_len;
    write_raw(dictionary_bytes, dictionary_len);
    pad_to_4k();

    // Write order mirrors the graph: offset table first, then per-point payload.
    _header.per_point_label_offsets_off = cur_offset();
    _header.per_point_label_offsets_len = (_header.npts + 1) * sizeof(uint64_t);
    write_raw(per_point_offsets, _header.per_point_label_offsets_len);
    pad_to_4k();

    _header.per_point_labels_off = cur_offset();
    _header.per_point_labels_len = per_point_data_len;
    write_raw(per_point_data, per_point_data_len);
    pad_to_4k();
}

void UnifiedIndexWriter::finalize()
{
    if (_finalized)
        throw std::logic_error("UnifiedIndexWriter: already finalized");

    // Capture total bytes written so readers can verify on load.
    _header.file_size_bytes = cur_offset();

    // Write offset table back at its reserved spot.
    _out.seekp(static_cast<std::streamoff>(_header.offset_table_off), std::ios::beg);
    write_raw(_node_offsets.data(), _header.offset_table_len);

    // Write final header at byte 0.
    _out.seekp(0, std::ios::beg);
    write_raw(&_header, sizeof(UnifiedIndexHeader));

    _out.flush();
    _out.close();
    _finalized = true;
}

// ---------- Reader ----------

UnifiedIndexReader::UnifiedIndexReader(const std::string &path) : _path(path)
{
    parse_header();
}

void UnifiedIndexReader::parse_header()
{
    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(_path, std::ios::binary);
    in.read(reinterpret_cast<char *>(&_header), sizeof(UnifiedIndexHeader));

    if (_header.magic != UNIFIED_FORMAT_MAGIC)
        throw std::runtime_error("UnifiedIndexReader: bad magic in " + _path);
    if (_header.version > UNIFIED_FORMAT_VERSION)
        throw std::runtime_error("UnifiedIndexReader: unsupported version " + std::to_string(_header.version) +
                                 " in " + _path);

    // Validate file size against the value recorded by the writer. A mismatch
    // typically means truncation, partial-write, or external tampering; the
    // exact size is also useful for disk-quota / resource-planning telemetry.
    in.seekg(0, std::ios::end);
    const uint64_t actual_size = static_cast<uint64_t>(in.tellg());
    if (_header.file_size_bytes != 0 && _header.file_size_bytes != actual_size)
    {
        throw std::runtime_error("UnifiedIndexReader: file_size_bytes mismatch in " + _path + " (header=" +
                                 std::to_string(_header.file_size_bytes) +
                                 ", actual=" + std::to_string(actual_size) + ")");
    }
    diskann::cout << "UnifiedIndexReader: opened " << _path << " (" << actual_size << " bytes)" << std::endl;
}

std::vector<uint64_t> UnifiedIndexReader::load_offset_table()
{
    std::vector<uint64_t> table(_header.npts + 1);
    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(_path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(_header.offset_table_off), std::ios::beg);
    in.read(reinterpret_cast<char *>(table.data()),
            static_cast<std::streamsize>((_header.npts + 1) * sizeof(uint64_t)));
    return table;
}

std::vector<uint8_t> UnifiedIndexReader::load_region(uint64_t off, uint64_t len)
{
    std::vector<uint8_t> buf(len);
    if (len == 0)
        return buf;
    load_region(off, len, buf.data());
    return buf;
}

void UnifiedIndexReader::load_region(uint64_t off, uint64_t len, uint8_t *dst)
{
    if (len == 0)
        return;
    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(_path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    in.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(len));
}

} // namespace diskann
