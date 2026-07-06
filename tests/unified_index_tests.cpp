// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "ann_exception.h"
#include "defaults.h"
#include "disk_utils.h"
#include "filter_match_proxy.h"
#include "index.h"
#include "label_bitmask.h"
#include "parameters.h"
#include "pq.h"
#include "pq_flash_index.h"
#include "unified_index.h"
#include "unified_index_builder.h"
#include "unified_index_format.h"
#include "unified_index_io.h"
#include "unified_label_data.h"
#include "unified_node_store.h"
#include "utils.h"
#include "windows_aligned_file_reader.h"

using namespace diskann;

namespace
{

// Path helper: writes test fixtures under the current working dir.
std::string tmp_path(const char *suffix)
{
    return std::string("unified_index_test_") + suffix + ".bin";
}

// Tiny RAII deleter for temp test files.
struct ScopedFile
{
    std::string path;
    explicit ScopedFile(std::string p) : path(std::move(p))
    {
    }
    ~ScopedFile()
    {
        std::remove(path.c_str());
    }
};

// Test-only subclass exposing init_geometry so the sector-math suite can
// inject a synthetic header + offset table without going through the writer.
template <typename T> class node_store_test_fixture final : public unified_node_store_base<T>
{
  public:
    using unified_node_store_base<T>::init_geometry;
    void get_nodes(const std::vector<uint64_t> & /*ids*/, NodeFetchScratch & /*scratch*/,
                   std::vector<NodeView<T>> & /*out*/) override
    {
        // Not exercised in these tests.
    }
};

// Build a synthetic offset table for `npts` nodes where node i has degree `degrees[i]`,
// laid out as [coords (aligned_dim * sizeof(T)), neighbors (degree * 4)] back-to-back.
std::vector<uint64_t> make_offset_table(uint64_t npts, uint64_t aligned_dim, size_t sizeof_T,
                                        const std::vector<uint32_t> &degrees)
{
    std::vector<uint64_t> off(npts + 1, 0);
    const uint64_t coord_bytes = aligned_dim * sizeof_T;
    for (uint64_t i = 0; i < npts; ++i)
    {
        off[i + 1] = off[i] + coord_bytes + degrees[i] * sizeof(uint32_t);
    }
    return off;
}

// Build a synthetic header for the test fixture.
UnifiedIndexHeader make_header(uint64_t npts, uint64_t dim, uint64_t aligned_dim, uint32_t max_degree,
                               DataTypeTag dt = DataTypeTag::Float)
{
    UnifiedIndexHeader h{};
    h.magic = UNIFIED_FORMAT_MAGIC;
    h.version = UNIFIED_FORMAT_VERSION;
    h.data_type = dt;
    h.metric = MetricTag::L2;
    h.npts = npts;
    h.dim = dim;
    h.aligned_dim = aligned_dim;
    h.max_degree = max_degree;
    h.flags = 0;
    h.start_node = 0;
    h.label_encoding = LabelEncoding::None;
    h.universal_label = 0;
    h.total_labels = 0;
    return h;
}

// Write a minimal unified file with `npts` nodes (all-zero coords, no
// neighbors) so the reader has a valid offset table + graph region. Labels
// regions are filled in by the caller (or skipped) via the encoding-specific
// writer methods.
void write_minimal_unified_file(const std::string &path, uint64_t npts, uint64_t dim, uint64_t aligned_dim,
                                uint32_t max_degree, DataTypeTag dt)
{
    UnifiedIndexWriter w(path);
    MetricTag metric = MetricTag::L2;
    w.begin(npts, dim, aligned_dim, max_degree, dt, metric, /*start_node=*/0);

    w.begin_graph_region();
    std::vector<uint8_t> coord_zero(aligned_dim * /*sizeof float upper bound*/ 4, 0);
    for (uint64_t i = 0; i < npts; ++i)
    {
        // Single neighbor pointing to self to keep neighbor section non-empty
        // (writer doesn't care, but easier to reason about offsets).
        const uint32_t nb = static_cast<uint32_t>(i);
        w.write_node(coord_zero.data(), &nb, /*degree=*/1);
    }
    w.end_graph_region();

    const uint32_t single_medoid = 0;
    w.write_medoids(&single_medoid, 1);
}

// Build a labels-dict byte blob in the canonical wire format:
//   [u32 slen][bytes label_str][u32 label_int][u32 medoid]
std::vector<uint8_t> build_dict_bytes(const std::vector<std::tuple<std::string, uint32_t, uint32_t>> &entries)
{
    std::vector<uint8_t> bytes;
    for (auto &e : entries)
    {
        const std::string &s = std::get<0>(e);
        const uint32_t lid = std::get<1>(e);
        const uint32_t medoid = std::get<2>(e);
        const uint32_t slen = static_cast<uint32_t>(s.size());
        const size_t old = bytes.size();
        bytes.resize(old + sizeof(uint32_t) + slen + 2 * sizeof(uint32_t));
        uint8_t *p = bytes.data() + old;
        std::memcpy(p, &slen, sizeof(uint32_t));
        p += sizeof(uint32_t);
        std::memcpy(p, s.data(), slen);
        p += slen;
        std::memcpy(p, &lid, sizeof(uint32_t));
        p += sizeof(uint32_t);
        std::memcpy(p, &medoid, sizeof(uint32_t));
    }
    return bytes;
}

void write_bitmask_labels_file(const std::string &path, uint64_t npts,
                               const std::vector<std::tuple<std::string, uint32_t, uint32_t>> &dict_entries,
                               const std::vector<std::vector<uint32_t>> &per_point_label_ints,
                               uint64_t total_labels, uint32_t universal_label)
{
    UnifiedIndexWriter w(path);
    const uint64_t dim = 4;
    const uint64_t aligned_dim = 4;
    const uint32_t max_degree = 4;
    w.begin(npts, dim, aligned_dim, max_degree, DataTypeTag::Float, MetricTag::L2, /*start_node=*/0);

    w.begin_graph_region();
    std::vector<float> coord_zero(aligned_dim, 0.0f);
    const uint32_t self_nb = 0;
    for (uint64_t i = 0; i < npts; ++i)
    {
        coord_zero[0] = 0.0f;
        w.write_node(coord_zero.data(), &self_nb, /*degree=*/1);
    }
    w.end_graph_region();

    const uint32_t medoid = 0;
    w.write_medoids(&medoid, 1);

    auto dict_bytes = build_dict_bytes(dict_entries);

    // Build bitmask: one row of `bitmask_size_words` uint64 per point.
    const uint64_t bitmask_words = simple_bitmask::get_bitmask_size(total_labels);
    std::vector<uint64_t> bitmask(npts * bitmask_words, 0);
    for (uint64_t i = 0; i < npts; ++i)
    {
        simple_bitmask bm(bitmask.data() + i * bitmask_words, bitmask_words);
        for (uint32_t lid : per_point_label_ints[i])
        {
            bm.set(lid);
        }
    }
    const uint64_t bitmap_bytes_len = bitmask.size() * sizeof(uint64_t);
    w.write_labels_bitmask(total_labels, universal_label, dict_bytes.data(), dict_bytes.size(), bitmask.data(),
                            bitmap_bytes_len);

    w.finalize();
}

void write_integer_labels_file(const std::string &path, uint64_t npts,
                               const std::vector<std::tuple<std::string, uint32_t, uint32_t>> &dict_entries,
                               const std::vector<std::vector<uint32_t>> &per_point_label_ints,
                               uint64_t total_labels, uint32_t universal_label)
{
    UnifiedIndexWriter w(path);
    const uint64_t dim = 4;
    const uint64_t aligned_dim = 4;
    const uint32_t max_degree = 4;
    w.begin(npts, dim, aligned_dim, max_degree, DataTypeTag::Float, MetricTag::L2, /*start_node=*/0);

    w.begin_graph_region();
    std::vector<float> coord_zero(aligned_dim, 0.0f);
    const uint32_t self_nb = 0;
    for (uint64_t i = 0; i < npts; ++i)
    {
        w.write_node(coord_zero.data(), &self_nb, /*degree=*/1);
    }
    w.end_graph_region();

    const uint32_t medoid = 0;
    w.write_medoids(&medoid, 1);

    auto dict_bytes = build_dict_bytes(dict_entries);

    // Flatten per-point lists + offset table.
    std::vector<uint32_t> flat;
    std::vector<uint64_t> offsets(npts + 1, 0);
    for (uint64_t i = 0; i < npts; ++i)
    {
        offsets[i + 1] = offsets[i] + per_point_label_ints[i].size();
        flat.insert(flat.end(), per_point_label_ints[i].begin(), per_point_label_ints[i].end());
    }
    w.write_labels_integer(total_labels, universal_label, dict_bytes.data(), dict_bytes.size(), flat.data(),
                            flat.size() * sizeof(uint32_t), offsets.data());

    w.finalize();
}

// Deterministic float in [-1, 1] from (seed, i, j) -- no Date::now / random.
inline float det_float(uint64_t seed, uint64_t i, uint64_t j)
{
    uint64_t x = seed * 1103515245ull + i * 12345ull + j * 7919ull + 12345ull;
    x ^= x >> 21;
    x *= 2685821657736338717ull;
    x ^= x >> 31;
    const uint32_t lo = static_cast<uint32_t>(x & 0xFFFFFFFFu);
    return (static_cast<float>(lo) / 2147483648.0f) - 1.0f;
}

// Build a unified float file: npts random points, aligned_dim columns each.
// Graph is a star centered on node 0: node 0 -> [1..npts-1], every other node -> [0].
inline void write_star_graph_unified(const std::string &path, uint64_t npts, uint64_t aligned_dim,
                                     std::vector<std::vector<float>> &points_out)
{
    points_out.assign(npts, std::vector<float>(aligned_dim, 0.0f));
    for (uint64_t i = 0; i < npts; ++i)
        for (uint64_t j = 0; j < aligned_dim; ++j)
            points_out[i][j] = det_float(/*seed=*/42, i, j);

    const uint32_t max_degree = static_cast<uint32_t>(npts);
    UnifiedIndexWriter w(path);
    w.begin(npts, aligned_dim, aligned_dim, max_degree, DataTypeTag::Float, MetricTag::L2, /*start_node=*/0);

    w.begin_graph_region();
    std::vector<uint32_t> hub_neighbors;
    for (uint32_t i = 1; i < npts; ++i)
        hub_neighbors.push_back(i);
    w.write_node(points_out[0].data(), hub_neighbors.data(), static_cast<uint32_t>(hub_neighbors.size()));
    const uint32_t back_to_hub = 0;
    for (uint64_t i = 1; i < npts; ++i)
    {
        w.write_node(points_out[i].data(), &back_to_hub, /*degree=*/1);
    }
    w.end_graph_region();

    const uint32_t medoid = 0;
    w.write_medoids(&medoid, 1);
    w.finalize();
}

inline float l2_sq(const std::vector<float> &a, const float *b)
{
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

inline std::vector<uint64_t> brute_force_topk(const std::vector<std::vector<float>> &points, const float *query,
                                              size_t K)
{
    std::vector<std::pair<float, uint64_t>> pairs;
    pairs.reserve(points.size());
    for (uint64_t i = 0; i < points.size(); ++i)
        pairs.emplace_back(l2_sq(points[i], query), i);
    std::sort(pairs.begin(), pairs.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    if (K > pairs.size())
        K = pairs.size();
    std::vector<uint64_t> out(K);
    for (size_t i = 0; i < K; ++i)
        out[i] = pairs[i].second;
    return out;
}

inline float recall_at_k(const std::vector<uint64_t> &gt, const uint64_t *result, size_t K)
{
    std::unordered_set<uint64_t> gt_set(gt.begin(), gt.begin() + std::min<size_t>(K, gt.size()));
    size_t hits = 0;
    for (size_t i = 0; i < K; ++i)
        if (gt_set.count(result[i]))
            ++hits;
    return static_cast<float>(hits) / static_cast<float>(K);
}

inline std::shared_ptr<::AlignedFileReader> make_reader()
{
    return std::shared_ptr<::AlignedFileReader>(new ::WindowsAlignedFileReader());
}

} // namespace

// ===========================================================================
// Suite 1: unified_node_store_base sector math (pure CPU)
// ===========================================================================

BOOST_AUTO_TEST_SUITE(unified_node_store_tests)

BOOST_AUTO_TEST_CASE(sector_math_basic)
{
    const uint64_t npts = 8;
    const uint64_t dim = 13;
    const uint64_t aligned_dim = 16;
    const uint32_t max_degree = 64;
    const std::vector<uint32_t> degrees = {3, 5, 7, 0, 64, 12, 1, 2};

    auto h = make_header(npts, dim, aligned_dim, max_degree);
    auto offsets = make_offset_table(npts, aligned_dim, sizeof(float), degrees);

    node_store_test_fixture<float> store;
    store.init_geometry(h, offsets);

    BOOST_CHECK_EQUAL(store.num_points(), npts);
    BOOST_CHECK_EQUAL(store.dim(), dim);
    BOOST_CHECK_EQUAL(store.aligned_dim(), aligned_dim);
    BOOST_CHECK_EQUAL(store.max_degree(), max_degree);

    // Offsets should match what make_offset_table computed.
    for (uint64_t i = 0; i < npts; ++i)
    {
        BOOST_CHECK_EQUAL(store.node_byte_offset(i), offsets[i]);
        const uint64_t expected_len = aligned_dim * sizeof(float) + degrees[i] * sizeof(uint32_t);
        BOOST_CHECK_EQUAL(store.node_byte_length(i), expected_len);
        BOOST_CHECK_EQUAL(store.degree(i), degrees[i]);
    }

    // max_node_len = (max_degree+1)*4 + aligned_dim*sizeof(T)
    const uint64_t expected_max = static_cast<uint64_t>(max_degree + 1) * 4 + aligned_dim * sizeof(float);
    BOOST_CHECK_EQUAL(store.max_node_len(), expected_max);

    // num_sectors_per_node = ceil(max_node_len / SECTOR_LEN) + 1
    const uint32_t expected_sectors =
        static_cast<uint32_t>((expected_max + defaults::SECTOR_LEN - 1) / defaults::SECTOR_LEN + 1u);
    BOOST_CHECK_EQUAL(store.num_sectors_per_node(), expected_sectors);
}

BOOST_AUTO_TEST_CASE(sector_math_throws_on_short_node)
{
    // Construct an offset table where node 0's payload is shorter than its
    // coords -- degree() must throw.
    auto h = make_header(/*npts=*/2, /*dim=*/4, /*aligned_dim=*/4, /*max_degree=*/4);
    std::vector<uint64_t> bad_offsets = {0, 4, 4 * sizeof(float) + 16}; // node 0 too short
    node_store_test_fixture<float> store;
    store.init_geometry(h, bad_offsets);
    BOOST_CHECK_THROW(store.degree(0), ANNException);
}

BOOST_AUTO_TEST_SUITE_END()

// ===========================================================================
// Suite 2: unified_label_data (bitmask)
// ===========================================================================

BOOST_AUTO_TEST_SUITE(unified_label_data_bitmask_tests)

BOOST_AUTO_TEST_CASE(bitmask_load_and_match_proxy)
{
    ScopedFile sf(tmp_path("bitmask"));
    const uint64_t npts = 4;
    // Use label ints 10/11/12 (and universal=99) so the bitmask_filter_match
    // ctor's unconditional universal-bit merge doesn't collide with real labels.
    // total_labels MUST be large enough that every label int -- including the
    // universal label -- fits inside the bitmask. The bitmask has
    // get_bitmask_size(total_labels) 64-bit words, so a universal label of 99
    // requires total_labels > 99; otherwise the ctor's universal-bit merge
    // (simple_bitmask_full_val::merge_bitmask_val) indexes word 99/64 == 1 of a
    // single-word query buffer and corrupts the heap.
    const std::vector<std::tuple<std::string, uint32_t, uint32_t>> dict = {
        {"red", 10u, 0u}, {"green", 11u, 1u}, {"blue", 12u, 2u}};
    const std::vector<std::vector<uint32_t>> per_point = {
        {10u},       // point 0 -> red
        {11u, 12u},  // point 1 -> green, blue
        {12u},       // point 2 -> blue
        {10u, 11u}}; // point 3 -> red, green
    write_bitmask_labels_file(sf.path, npts, dict, per_point, /*total_labels=*/100, /*universal_label=*/99);

    UnifiedIndexReader r(sf.path);
    auto labels = make_unified_label_data(r, r.header(), r.header().npts);
    BOOST_REQUIRE(labels != nullptr);
    BOOST_CHECK_EQUAL(static_cast<int>(labels->encoding()), static_cast<int>(LabelEncoding::Bitmask));
    BOOST_CHECK(labels->has_labels());
    BOOST_CHECK_EQUAL(labels->num_labels(), 3u);
    BOOST_CHECK(labels->is_valid_label("red"));
    BOOST_CHECK(labels->is_valid_label("green"));
    BOOST_CHECK(!labels->is_valid_label("yellow"));
    uint32_t out = 99;
    BOOST_CHECK(labels->get_converted_label("blue", out));
    BOOST_CHECK_EQUAL(out, 12u);

    // resolve_filters returns the label int and its medoid in a single lookup.
    std::vector<uint32_t> blue_ints, blue_medoids;
    labels->resolve_filters({"blue"}, blue_ints, blue_medoids);
    BOOST_REQUIRE_EQUAL(blue_ints.size(), 1u);
    BOOST_CHECK_EQUAL(blue_ints[0], 12u);
    BOOST_REQUIRE_EQUAL(blue_medoids.size(), 1u);
    BOOST_CHECK_EQUAL(blue_medoids[0], 2u);

    // collect_label_medoids appends every label's entry-point medoid (one per
    // label). Dict above maps red/green/blue -> medoids 0/1/2. It appends (does
    // not clear), so a pre-existing element must be preserved. Order is
    // unspecified (hash-map iteration), so sort before comparing.
    std::vector<uint32_t> all_medoids = {42u};
    labels->collect_label_medoids(all_medoids);
    BOOST_REQUIRE_EQUAL(all_medoids.size(), 4u);
    std::sort(all_medoids.begin(), all_medoids.end());
    const std::vector<uint32_t> expected_medoids = {0u, 1u, 2u, 42u};
    BOOST_CHECK_EQUAL_COLLECTIONS(all_medoids.begin(), all_medoids.end(), expected_medoids.begin(),
                                  expected_medoids.end());

    // Match-proxy: filter "blue" -> matches points 1, 2, but NOT 0, 3.
    auto proxy = labels->make_match_proxy(blue_ints);
    BOOST_REQUIRE(proxy != nullptr);
    BOOST_CHECK(!proxy->contain_filtered_label(0));
    BOOST_CHECK(proxy->contain_filtered_label(1));
    BOOST_CHECK(proxy->contain_filtered_label(2));
    BOOST_CHECK(!proxy->contain_filtered_label(3));
}

BOOST_AUTO_TEST_CASE(bitmask_unknown_label_string_throws)
{
    ScopedFile sf(tmp_path("bitmask_unknown"));
    write_bitmask_labels_file(sf.path, /*npts=*/2,
                              /*dict=*/{{"a", 0u, 0u}}, /*per_point=*/{{0u}, {}}, /*total_labels=*/1,
                              /*universal_label=*/0);
    UnifiedIndexReader r(sf.path);
    auto labels = make_unified_label_data(r, r.header(), r.header().npts);
    BOOST_REQUIRE(labels != nullptr);
    std::vector<uint32_t> ints, medoids;
    BOOST_CHECK_THROW(labels->resolve_filters({"nonexistent"}, ints, medoids), ANNException);
}

BOOST_AUTO_TEST_SUITE_END()

// ===========================================================================
// Suite 3: unified_label_data (integer)
// ===========================================================================

BOOST_AUTO_TEST_SUITE(unified_label_data_integer_tests)

BOOST_AUTO_TEST_CASE(integer_load_and_match_proxy)
{
    ScopedFile sf(tmp_path("integer"));
    const uint64_t npts = 4;
    const std::vector<std::tuple<std::string, uint32_t, uint32_t>> dict = {
        {"red", 10u, 0u}, {"green", 11u, 1u}, {"blue", 12u, 2u}};
    const std::vector<std::vector<uint32_t>> per_point = {{10u}, {11u, 12u}, {12u}, {10u, 11u}};
    write_integer_labels_file(sf.path, npts, dict, per_point, /*total_labels=*/13, /*universal_label=*/99);

    UnifiedIndexReader r(sf.path);
    auto labels = make_unified_label_data(r, r.header(), r.header().npts);
    BOOST_REQUIRE(labels != nullptr);
    BOOST_CHECK_EQUAL(static_cast<int>(labels->encoding()), static_cast<int>(LabelEncoding::Integer));
    BOOST_CHECK(labels->has_labels());
    BOOST_CHECK_EQUAL(labels->num_labels(), 3u);

    std::vector<uint32_t> blue_ints, blue_medoids;
    labels->resolve_filters({"blue"}, blue_ints, blue_medoids);
    auto proxy = labels->make_match_proxy(blue_ints);
    BOOST_REQUIRE(proxy != nullptr);
    BOOST_CHECK(!proxy->contain_filtered_label(0));
    BOOST_CHECK(proxy->contain_filtered_label(1));
    BOOST_CHECK(proxy->contain_filtered_label(2));
    BOOST_CHECK(!proxy->contain_filtered_label(3));
}

BOOST_AUTO_TEST_CASE(integer_match_proxy_sorts_filter_labels)
{
    // Regression: integer_label_vector::check_label_exists advances its binary-
    // search window between query labels, so make_match_proxy must sort the
    // resolved filter ints. Pass labels in DESCENDING string order ("green",
    // "red" -> ints 11, 10) so an unsorted proxy would search 11 first, advance
    // past index 0, then miss 10 -> false negative for point 0 (labelled red).
    ScopedFile sf(tmp_path("integer_unsorted"));
    const uint64_t npts = 4;
    const std::vector<std::tuple<std::string, uint32_t, uint32_t>> dict = {
        {"red", 10u, 0u}, {"green", 11u, 1u}, {"blue", 12u, 2u}};
    // Per-point label lists are stored ascending (binary-search precondition).
    const std::vector<std::vector<uint32_t>> per_point = {{10u}, {11u, 12u}, {12u}, {10u, 11u}};
    write_integer_labels_file(sf.path, npts, dict, per_point, /*total_labels=*/13, /*universal_label=*/0);

    UnifiedIndexReader r(sf.path);
    auto labels = make_unified_label_data(r, r.header(), r.header().npts);
    BOOST_REQUIRE(labels != nullptr);

    // Filter "green" OR "red" -> matches points 0 (red), 1 (green), 3 (red+green),
    // but NOT 2 (blue only). Strings are intentionally out of int order.
    std::vector<uint32_t> ints, medoids;
    labels->resolve_filters({"green", "red"}, ints, medoids);
    auto proxy = labels->make_match_proxy(ints);
    BOOST_REQUIRE(proxy != nullptr);
    BOOST_CHECK(proxy->contain_filtered_label(0)); // red -- the case the sort fixes
    BOOST_CHECK(proxy->contain_filtered_label(1)); // green
    BOOST_CHECK(!proxy->contain_filtered_label(2)); // blue only
    BOOST_CHECK(proxy->contain_filtered_label(3)); // red + green
}

BOOST_AUTO_TEST_CASE(integer_unknown_label_string_throws)
{
    ScopedFile sf(tmp_path("integer_unknown"));
    write_integer_labels_file(sf.path, /*npts=*/2,
                              /*dict=*/{{"a", 0u, 0u}}, /*per_point=*/{{0u}, {}}, /*total_labels=*/1,
                              /*universal_label=*/0);
    UnifiedIndexReader r(sf.path);
    auto labels = make_unified_label_data(r, r.header(), r.header().npts);
    BOOST_REQUIRE(labels != nullptr);
    std::vector<uint32_t> ints, medoids;
    BOOST_CHECK_THROW(labels->resolve_filters({"nonexistent"}, ints, medoids), ANNException);
}

BOOST_AUTO_TEST_SUITE_END()

// ===========================================================================
// Suite 4: make_unified_label_data factory dispatch
// ===========================================================================

BOOST_AUTO_TEST_SUITE(unified_label_data_factory_tests)

BOOST_AUTO_TEST_CASE(factory_no_labels_returns_null)
{
    ScopedFile sf(tmp_path("no_labels"));
    write_minimal_unified_file(sf.path, /*npts=*/2, /*dim=*/4, /*aligned_dim=*/4, /*max_degree=*/4,
                                DataTypeTag::Float);
    // write_minimal_unified_file doesn't call finalize. Do it here.
    {
        // Re-open as writer? No -- minimal helper writes nothing post-medoids.
        // Use a thin wrapper that *does* finalize.
    }
    // Simpler: use the same helper but call finalize via the writer's API. The
    // helper above doesn't finalize; rewrite the file inline:
    {
        UnifiedIndexWriter w(sf.path);
        w.begin(/*npts=*/2, /*dim=*/4, /*aligned_dim=*/4, /*max_degree=*/4, DataTypeTag::Float, MetricTag::L2, 0);
        w.begin_graph_region();
        std::vector<float> coord(4, 0.0f);
        const uint32_t nb = 0;
        w.write_node(coord.data(), &nb, 1);
        w.write_node(coord.data(), &nb, 1);
        w.end_graph_region();
        const uint32_t medoid = 0;
        w.write_medoids(&medoid, 1);
        w.finalize();
    }

    UnifiedIndexReader r(sf.path);
    BOOST_CHECK_EQUAL(r.header().flags & HAS_LABELS, 0u);
    auto labels = make_unified_label_data(r, r.header(), r.header().npts);
    BOOST_CHECK(labels == nullptr);
}

BOOST_AUTO_TEST_CASE(factory_picks_bitmask_class)
{
    ScopedFile sf(tmp_path("factory_bm"));
    write_bitmask_labels_file(sf.path, /*npts=*/2,
                              /*dict=*/{{"a", 0u, 0u}}, /*per_point=*/{{0u}, {}}, /*total_labels=*/1,
                              /*universal_label=*/0);
    UnifiedIndexReader r(sf.path);
    auto labels = make_unified_label_data(r, r.header(), r.header().npts);
    BOOST_REQUIRE(labels != nullptr);
    BOOST_CHECK_EQUAL(static_cast<int>(labels->encoding()), static_cast<int>(LabelEncoding::Bitmask));
}

BOOST_AUTO_TEST_CASE(factory_picks_integer_class)
{
    ScopedFile sf(tmp_path("factory_int"));
    write_integer_labels_file(sf.path, /*npts=*/2,
                              /*dict=*/{{"a", 0u, 0u}}, /*per_point=*/{{0u}, {}}, /*total_labels=*/1,
                              /*universal_label=*/0);
    UnifiedIndexReader r(sf.path);
    auto labels = make_unified_label_data(r, r.header(), r.header().npts);
    BOOST_REQUIRE(labels != nullptr);
    BOOST_CHECK_EQUAL(static_cast<int>(labels->encoding()), static_cast<int>(LabelEncoding::Integer));
}

BOOST_AUTO_TEST_SUITE_END()

// ===========================================================================
// Suite 5: unified_index factory -- non-templated user-facing surface
// ===========================================================================
//
// Phase A's unified_index_memory<T>::load_storage throws not_implemented after
// constructing the node store, so end-to-end load isn't reachable yet. We
// verify two things: (1) the factory dispatches on data_type correctly by
// confirming the not_implemented throw originates from the right concrete
// type, and (2) UnifiedIndexReader independently confirms the file's
// data_type matches the requested template parameter (the factory's expected
// dispatch).

BOOST_AUTO_TEST_SUITE(unified_index_factory_tests)

BOOST_AUTO_TEST_CASE(factory_dispatches_on_data_type)
{
    struct case_t
    {
        DataTypeTag tag;
        const char *suffix;
    };
    case_t cases[] = {{DataTypeTag::Float, "float"}, {DataTypeTag::Uint8, "u8"}, {DataTypeTag::Int8, "i8"}};

    const uint64_t aligned_dim = 32; // multiple of every type's alignment factor

    for (auto &c : cases)
    {
        ScopedFile sf(tmp_path(c.suffix));
        {
            UnifiedIndexWriter w(sf.path);
            w.begin(/*npts=*/2, /*dim=*/aligned_dim, aligned_dim, /*max_degree=*/4, c.tag, MetricTag::L2, 0);
            w.begin_graph_region();
            // Worst-case sized coords buffer (sizeof(float)) -- writer reads
            // aligned_dim*sizeof(T) bytes depending on the data_type tag.
            std::vector<uint8_t> coord(aligned_dim * sizeof(float), 0);
            const uint32_t nb = 0;
            w.write_node(coord.data(), &nb, 1);
            w.write_node(coord.data(), &nb, 1);
            w.end_graph_region();
            const uint32_t medoid = 0;
            w.write_medoids(&medoid, 1);
            w.finalize();
        }

        // Verify the file says the right data type.
        {
            UnifiedIndexReader r(sf.path);
            BOOST_CHECK_EQUAL(static_cast<int>(r.header().data_type), static_cast<int>(c.tag));
        }

        // Factory should construct the right templated index and load successfully.
        UnifiedLoadContext ctx;
        ctx.path = sf.path;
        auto idx = make_unified_index_memory(ctx);
        BOOST_REQUIRE(idx != nullptr);
        BOOST_CHECK_EQUAL(static_cast<int>(idx->data_type()), static_cast<int>(c.tag));
        BOOST_CHECK_EQUAL(idx->num_points(), 2u);
        BOOST_CHECK_EQUAL(idx->dim(), aligned_dim);
    }
}

BOOST_AUTO_TEST_CASE(factory_ssd_dispatches_on_data_type)
{
    // SSD path requires HAS_PQ. A no-PQ unified file should be rejected
    // cleanly. We verify two things: (1) the factory dispatches to the
    // right templated unified_index_ssd<T> based on data_type, and (2) the
    // SSD store class rejects no-PQ files with a clear error message.
    const uint64_t aligned_dim = 32;
    ScopedFile sf(tmp_path("ssd_disp"));
    {
        UnifiedIndexWriter w(sf.path);
        w.begin(/*npts=*/2, /*dim=*/aligned_dim, aligned_dim, /*max_degree=*/4, DataTypeTag::Float, MetricTag::L2,
                 0);
        w.begin_graph_region();
        std::vector<float> coord(aligned_dim, 0.0f);
        const uint32_t nb = 0;
        w.write_node(coord.data(), &nb, 1);
        w.write_node(coord.data(), &nb, 1);
        w.end_graph_region();
        const uint32_t medoid = 0;
        w.write_medoids(&medoid, 1);
        w.finalize();
    }

    UnifiedLoadContext ctx;
    ctx.path = sf.path;
    auto reader = make_reader();
    bool threw = false;
    try
    {
        auto idx = make_unified_index_ssd(reader, ctx);
    }
    catch (const ANNException &e)
    {
        threw = true;
        std::string msg = e.what();
        // Either the SSD index class flags the missing PQ ("requires HAS_PQ"),
        // or the file fails to open for some other reason. Either way the
        // throw should originate from our SSD code path.
        BOOST_CHECK(msg.find("HAS_PQ") != std::string::npos ||
                    msg.find("unified_index_ssd") != std::string::npos ||
                    msg.find("unified_node_store_ssd") != std::string::npos);
    }
    BOOST_CHECK(threw);
}

BOOST_AUTO_TEST_SUITE_END()

// ===========================================================================
// Suite 6: Phase B -- unified_index_memory<T> end-to-end search
// ===========================================================================
//
// Build a small synthetic unified file with random float points and a star
// graph (every node connected to a hub), load it via make_unified_index_memory,
// then validate that search returns sane top-K against a brute-force baseline.
// Recall@10 should be ≥ 95% on a star graph because every node is reachable
// from the hub in one hop.

BOOST_AUTO_TEST_SUITE(unified_index_memory_tests)

BOOST_AUTO_TEST_CASE(memory_end_to_end_search_recall)
{
    ScopedFile sf(tmp_path("memory_e2e"));
    const uint64_t npts = 64;
    const uint64_t aligned_dim = 16;

    std::vector<std::vector<float>> points;
    write_star_graph_unified(sf.path, npts, aligned_dim, points);

    UnifiedLoadContext ctx;
    ctx.path = sf.path;
    ctx.num_threads = 1;
    ctx.search_l = 64;

    auto idx = make_unified_index_memory(ctx);
    BOOST_REQUIRE(idx != nullptr);
    BOOST_CHECK_EQUAL(idx->num_points(), npts);
    BOOST_CHECK_EQUAL(idx->dim(), aligned_dim);
    BOOST_CHECK_EQUAL(static_cast<int>(idx->data_type()), static_cast<int>(DataTypeTag::Float));

    const size_t K = 10;
    const size_t Q = 50;
    float total_recall = 0.0f;

    for (size_t q = 0; q < Q; ++q)
    {
        std::vector<float> query(aligned_dim, 0.0f);
        for (size_t j = 0; j < aligned_dim; ++j)
            query[j] = det_float(/*seed=*/777, q, j);

        std::vector<uint64_t> out_ids(K, 0);
        std::vector<float> out_dists(K, 0.0f);

        UnifiedSearchContext sctx;
        sctx.query = query.data();
        sctx.K = K;
        sctx.L = 64;
        sctx.indices = out_ids.data();
        sctx.distances = out_dists.data();
        idx->search(sctx);

        auto gt = brute_force_topk(points, query.data(), K);
        total_recall += recall_at_k(gt, out_ids.data(), K);
    }

    const float avg_recall = total_recall / static_cast<float>(Q);
    BOOST_TEST_MESSAGE("memory_end_to_end_search_recall: avg recall@10 = " << avg_recall);
    BOOST_CHECK_GE(avg_recall, 0.95f);
}

BOOST_AUTO_TEST_CASE(memory_search_validates_K_and_L)
{
    ScopedFile sf(tmp_path("memory_validate"));
    std::vector<std::vector<float>> points;
    write_star_graph_unified(sf.path, /*npts=*/8, /*aligned_dim=*/8, points);

    UnifiedLoadContext ctx;
    ctx.path = sf.path;
    auto idx = make_unified_index_memory(ctx);

    std::vector<float> query(8, 0.1f);
    std::vector<uint64_t> out_ids(4, 0);
    std::vector<float> out_dists(4, 0.0f);

    UnifiedSearchContext sctx;
    sctx.query = query.data();
    sctx.K = 4;
    sctx.L = 2; // L < K -> must throw
    sctx.indices = out_ids.data();
    sctx.distances = out_dists.data();
    BOOST_CHECK_THROW(idx->search(sctx), ANNException);

    // filter_labels must be empty on a non-filtered index.
    sctx.L = 8;
    sctx.filter_labels = {"red"};
    BOOST_CHECK_THROW(idx->search(sctx), ANNException);
}

BOOST_AUTO_TEST_SUITE_END()

// ===========================================================================
// Suite 7: Phase C -- unified_node_store_ssd<T> with AlignedFileReader
// ===========================================================================
//
// Build the same star-graph unified file the memory suite uses, but load it
// via the SSD store. Verify (1) get_nodes returns the same coords/neighbors
// the writer wrote, (2) cache priming via cache_bfs_levels populates the
// caches so subsequent get_nodes calls skip IO, (3) recall via a simple
// brute-force-over-cached-nodes baseline matches.

BOOST_AUTO_TEST_SUITE(unified_node_store_ssd_tests)

BOOST_AUTO_TEST_CASE(ssd_get_nodes_uncached_round_trip)
{
    ScopedFile sf(tmp_path("ssd_round_trip"));
    const uint64_t npts = 16;
    const uint64_t aligned_dim = 16;
    std::vector<std::vector<float>> points;
    write_star_graph_unified(sf.path, npts, aligned_dim, points);

    UnifiedIndexReader reader_meta(sf.path);
    auto reader = make_reader();
    unified_node_store_ssd<float> store(reader);
    store.load(reader_meta, reader_meta.header());

    BOOST_CHECK_EQUAL(store.num_points(), npts);
    BOOST_CHECK_EQUAL(store.aligned_dim(), aligned_dim);

    NodeFetchScratch scratch = store.make_fetch_scratch(/*max_batch=*/16);
    std::vector<uint64_t> ids = {0, 1, 5, 10, 15};
    std::vector<NodeView<float>> views;
    store.get_nodes(ids, scratch, views);
    BOOST_REQUIRE_EQUAL(views.size(), ids.size());

    // One batched read should have been issued for all 5 ids.
    BOOST_CHECK_EQUAL(store.io_count(), 1u);

    // Verify coords match what the writer wrote.
    for (size_t i = 0; i < ids.size(); ++i)
    {
        const uint64_t id = ids[i];
        BOOST_REQUIRE(views[i].coords != nullptr);
        for (uint64_t j = 0; j < aligned_dim; ++j)
        {
            BOOST_CHECK_CLOSE(views[i].coords[j], points[id][j], 1e-6f);
        }
    }

    // Star graph: node 0 has neighbors [1..npts-1], every other node has neighbor [0].
    BOOST_CHECK_EQUAL(views[0].degree, npts - 1);
    for (size_t i = 1; i < ids.size(); ++i)
    {
        BOOST_CHECK_EQUAL(views[i].degree, 1u);
        BOOST_CHECK_EQUAL(views[i].neighbors[0], 0u);
    }
}

BOOST_AUTO_TEST_CASE(ssd_cache_hits_skip_io)
{
    ScopedFile sf(tmp_path("ssd_cache"));
    const uint64_t npts = 16;
    const uint64_t aligned_dim = 16;
    std::vector<std::vector<float>> points;
    write_star_graph_unified(sf.path, npts, aligned_dim, points);

    UnifiedIndexReader reader_meta(sf.path);
    auto reader = make_reader();
    unified_node_store_ssd<float> store(reader);
    store.load(reader_meta, reader_meta.header());

    // Prime cache with ids {0, 5, 10}. The prime scratch is sized for a single
    // node (max_batch=1), so load_cache_list batches one node per get_nodes
    // call here -> 3 IOs. (With a larger scratch it would batch all three into
    // a single IO; this test pins the per-node case via the GE assertion.)
    {
        NodeFetchScratch prime_scratch = store.make_fetch_scratch(/*max_batch=*/1);
        store.load_cache_list({0u, 5u, 10u}, prime_scratch);
    }
    const uint64_t io_after_prime = store.io_count();
    BOOST_CHECK_GE(io_after_prime, 3u);

    NodeFetchScratch scratch = store.make_fetch_scratch(/*max_batch=*/16);
    std::vector<NodeView<float>> views;

    // All-cached batch: io_count must stay flat.
    store.get_nodes({0u, 5u, 10u}, scratch, views);
    BOOST_CHECK_EQUAL(store.io_count(), io_after_prime);
    for (size_t i = 0; i < 3; ++i)
    {
        BOOST_REQUIRE(views[i].coords != nullptr);
        BOOST_REQUIRE(views[i].neighbors != nullptr);
    }

    // Mixed batch with one miss (id 7): one IO incurred.
    store.get_nodes({0u, 5u, 7u, 10u}, scratch, views);
    BOOST_CHECK_EQUAL(store.io_count(), io_after_prime + 1u);
    // Verify the miss (id 7) decoded correctly against the writer's data.
    for (uint64_t j = 0; j < aligned_dim; ++j)
    {
        BOOST_CHECK_CLOSE(views[2].coords[j], points[7][j], 1e-6f);
    }
}

BOOST_AUTO_TEST_CASE(ssd_load_cache_list_batches_reads)
{
    // load_cache_list batches its reads: with a scratch large enough to hold
    // the whole node list, all requested nodes are fetched in a single batched
    // IO instead of one IO per node.
    ScopedFile sf(tmp_path("ssd_cache_batch"));
    const uint64_t npts = 16;
    const uint64_t aligned_dim = 16;
    std::vector<std::vector<float>> points;
    write_star_graph_unified(sf.path, npts, aligned_dim, points);

    UnifiedIndexReader reader_meta(sf.path);
    auto reader = make_reader();
    unified_node_store_ssd<float> store(reader);
    store.load(reader_meta, reader_meta.header());

    const std::vector<uint32_t> ids = {0u, 3u, 6u, 9u, 12u, 15u};
    const uint64_t io_before = store.io_count();
    {
        // Scratch capacity 8 >= 6 ids -> exactly one batched read.
        NodeFetchScratch prime_scratch = store.make_fetch_scratch(/*max_batch=*/8);
        store.load_cache_list(ids, prime_scratch);
    }
    BOOST_CHECK_EQUAL(store.io_count() - io_before, 1u);

    // Every primed id now resolves with zero additional IO, and the cached
    // coords match what the writer stored.
    const uint64_t io_after = store.io_count();
    NodeFetchScratch scratch = store.make_fetch_scratch(/*max_batch=*/8);
    std::vector<NodeView<float>> views;
    store.get_nodes({0u, 3u, 6u, 9u, 12u, 15u}, scratch, views);
    BOOST_CHECK_EQUAL(store.io_count(), io_after);
    for (size_t i = 0; i < ids.size(); ++i)
    {
        BOOST_REQUIRE(views[i].coords != nullptr);
        for (uint64_t j = 0; j < aligned_dim; ++j)
            BOOST_CHECK_CLOSE(views[i].coords[j], points[ids[i]][j], 1e-6f);
    }
}

BOOST_AUTO_TEST_CASE(ssd_cache_bfs_levels_seeds_from_medoid)
{
    ScopedFile sf(tmp_path("ssd_bfs"));
    const uint64_t npts = 16;
    const uint64_t aligned_dim = 16;
    std::vector<std::vector<float>> points;
    write_star_graph_unified(sf.path, npts, aligned_dim, points);

    UnifiedIndexReader reader_meta(sf.path);
    auto reader = make_reader();
    unified_node_store_ssd<float> store(reader);
    store.load(reader_meta, reader_meta.header());

    // Star graph from medoid 0 -> all npts nodes reachable in 1 hop.
    std::vector<uint32_t> cached_list;
    {
        NodeFetchScratch prime_scratch = store.make_fetch_scratch(/*max_batch=*/npts);
        store.cache_bfs_levels(/*seeds=*/{0u}, /*num_nodes_to_cache=*/npts, cached_list, prime_scratch);
    }
    BOOST_CHECK_EQUAL(cached_list.size(), npts);

    // After priming, asking for ANY id must not issue IO.
    const uint64_t io_after_prime = store.io_count();
    NodeFetchScratch scratch = store.make_fetch_scratch(/*max_batch=*/16);
    std::vector<NodeView<float>> views;
    store.get_nodes({0u, 7u, 15u}, scratch, views);
    BOOST_CHECK_EQUAL(store.io_count(), io_after_prime);
}

BOOST_AUTO_TEST_SUITE_END()

// ===========================================================================
// Suite 8: Phase E -- unified_index_builder end-to-end
// ===========================================================================
//
// Build a tiny .bin data file, run unified_index_builder, then load the
// produced unified container via the read-side factories and confirm metadata.

BOOST_AUTO_TEST_SUITE(unified_index_builder_tests)

namespace
{
// Write a DiskANN-format .bin file: [int32 npts][int32 dim][float data...].
void write_float_bin(const std::string &path, uint32_t npts, uint32_t dim, uint64_t seed)
{
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(path, std::ios::binary | std::ios::trunc);
    const int32_t n = static_cast<int32_t>(npts);
    const int32_t d = static_cast<int32_t>(dim);
    out.write(reinterpret_cast<const char *>(&n), sizeof(int32_t));
    out.write(reinterpret_cast<const char *>(&d), sizeof(int32_t));
    for (uint32_t i = 0; i < npts; ++i)
    {
        for (uint32_t j = 0; j < dim; ++j)
        {
            const float v = det_float(seed, i, j);
            out.write(reinterpret_cast<const char *>(&v), sizeof(float));
        }
    }
}
} // namespace

BOOST_AUTO_TEST_CASE(builder_no_pq_emits_memory_loadable_file)
{
    ScopedFile data_sf(tmp_path("builder_data_nopq.bin"));
    ScopedFile out_sf(tmp_path("builder_out_nopq.bin"));

    write_float_bin(data_sf.path, /*npts=*/64, /*dim=*/16, /*seed=*/42);

    UnifiedBuildContext ctx;
    ctx.data_file_path = data_sf.path;
    ctx.output_path = out_sf.path;
    ctx.data_type = DataTypeTag::Float;
    ctx.metric = diskann::Metric::L2;
    ctx.R = 16;
    ctx.L = 32;
    ctx.alpha = 1.2f;
    ctx.num_threads = 1;
    ctx.pq_dim = 0; // no PQ

    unified_index_builder builder;
    builder.build(ctx);

    // Verify the produced file: no HAS_PQ flag, metadata matches.
    {
        UnifiedIndexReader r(out_sf.path);
        const auto &h = r.header();
        BOOST_CHECK_EQUAL(static_cast<int>(h.data_type), static_cast<int>(DataTypeTag::Float));
        BOOST_CHECK_EQUAL(h.npts, 64u);
        BOOST_CHECK_EQUAL(h.dim, 16u);
        BOOST_CHECK_EQUAL(h.flags & HAS_PQ, 0u);
    }

    // Confirm it loads via the memory factory and metadata is intact.
    UnifiedLoadContext load_ctx;
    load_ctx.path = out_sf.path;
    auto idx = make_unified_index_memory(load_ctx);
    BOOST_REQUIRE(idx != nullptr);
    BOOST_CHECK_EQUAL(idx->num_points(), 64u);
    BOOST_CHECK_EQUAL(idx->dim(), 16u);
}

BOOST_AUTO_TEST_CASE(builder_with_pq_emits_ssd_loadable_file)
{
    ScopedFile data_sf(tmp_path("builder_data_pq.bin"));
    ScopedFile out_sf(tmp_path("builder_out_pq.bin"));

    // Bigger dataset so PQ training has enough sample points.
    write_float_bin(data_sf.path, /*npts=*/512, /*dim=*/16, /*seed=*/7);

    UnifiedBuildContext ctx;
    ctx.data_file_path = data_sf.path;
    ctx.output_path = out_sf.path;
    ctx.data_type = DataTypeTag::Float;
    ctx.metric = diskann::Metric::L2;
    ctx.R = 16;
    ctx.L = 32;
    ctx.alpha = 1.2f;
    ctx.num_threads = 1;
    ctx.pq_dim = 4; // PQ-compress 16-dim into 4 chunks
    ctx.pq_sampling_rate = 1.0; // train on full data (tiny set)

    unified_index_builder builder;
    builder.build(ctx);

    // Verify HAS_PQ flag is set.
    {
        UnifiedIndexReader r(out_sf.path);
        const auto &h = r.header();
        BOOST_CHECK_EQUAL(static_cast<int>(h.data_type), static_cast<int>(DataTypeTag::Float));
        BOOST_CHECK_EQUAL(h.npts, 512u);
        BOOST_CHECK_NE(h.flags & HAS_PQ, 0u);
        BOOST_CHECK_GT(h.pq_pivots_len, 0u);
        BOOST_CHECK_GT(h.pq_codes_len, 0u);
    }

    // Confirm it loads via the SSD factory end-to-end.
    UnifiedLoadContext load_ctx;
    load_ctx.path = out_sf.path;
    load_ctx.num_threads = 1;
    auto reader = make_reader();
    auto idx = make_unified_index_ssd(reader, load_ctx);
    BOOST_REQUIRE(idx != nullptr);
    BOOST_CHECK_EQUAL(idx->num_points(), 512u);
    BOOST_CHECK_EQUAL(idx->dim(), 16u);
}

BOOST_AUTO_TEST_CASE(builder_pq_dim_equals_dim_still_emits_ssd_loadable_file)
{
    // Regression: pq_dim == dim must still emit PQ so the file is SSD-loadable.
    // Previously the builder skipped PQ when pq_dim >= dim, producing a file the
    // SSD load path (which requires HAS_PQ) would reject.
    ScopedFile data_sf(tmp_path("builder_data_pqfull.bin"));
    ScopedFile out_sf(tmp_path("builder_out_pqfull.bin"));

    const uint32_t dim = 16;
    write_float_bin(data_sf.path, /*npts=*/512, dim, /*seed=*/11);

    UnifiedBuildContext ctx;
    ctx.data_file_path = data_sf.path;
    ctx.output_path = out_sf.path;
    ctx.data_type = DataTypeTag::Float;
    ctx.metric = diskann::Metric::L2;
    ctx.R = 16;
    ctx.L = 32;
    ctx.alpha = 1.2f;
    ctx.num_threads = 1;
    ctx.pq_dim = dim;           // pq_dim == dim -> chunk size 1, full-precision-per-dim PQ
    ctx.pq_sampling_rate = 1.0; // train on full data (tiny set)

    unified_index_builder builder;
    builder.build(ctx);

    // HAS_PQ must be set even though pq_dim == dim.
    {
        UnifiedIndexReader r(out_sf.path);
        const auto &h = r.header();
        BOOST_CHECK_EQUAL(h.npts, 512u);
        BOOST_CHECK_EQUAL(h.dim, static_cast<uint64_t>(dim));
        BOOST_CHECK_NE(h.flags & HAS_PQ, 0u);
        BOOST_CHECK_GT(h.pq_pivots_len, 0u);
        BOOST_CHECK_GT(h.pq_codes_len, 0u);
    }

    // And it loads via the SSD factory end-to-end.
    UnifiedLoadContext load_ctx;
    load_ctx.path = out_sf.path;
    load_ctx.num_threads = 1;
    auto reader = make_reader();
    auto idx = make_unified_index_ssd(reader, load_ctx);
    BOOST_REQUIRE(idx != nullptr);
    BOOST_CHECK_EQUAL(idx->num_points(), 512u);
    BOOST_CHECK_EQUAL(idx->dim(), static_cast<uint64_t>(dim));
}

BOOST_AUTO_TEST_SUITE_END()

// ===========================================================================
// Suite 9: legacy <-> unified parity
// ===========================================================================
//
// Verify the unified format produces the SAME search results as the legacy
// DiskANN format when both are derived from the SAME Vamana graph:
//   - Memory: one Index<float> -> save() (legacy) + save_unified(); load both
//     from disk and search; top-K must match.
//   - SSD: one Index<float> + one shared PQ codebook -> legacy disk index (via
//     a UT-local helper mirroring build_disk_index's internals on an existing
//     Index) + unified SSD file; load PQFlashIndex and unified_index_ssd; top-K
//     must match.

namespace
{

// Write a DiskANN .bin file: [int32 npts][int32 dim][float data...]. (Local to
// the parity suite; the builder suite has its own copy in a nested namespace.)
void write_float_bin_parity(const std::string &path, uint32_t npts, uint32_t dim, uint64_t seed)
{
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(path, std::ios::binary | std::ios::trunc);
    const int32_t n = static_cast<int32_t>(npts);
    const int32_t d = static_cast<int32_t>(dim);
    out.write(reinterpret_cast<const char *>(&n), sizeof(int32_t));
    out.write(reinterpret_cast<const char *>(&d), sizeof(int32_t));
    for (uint32_t i = 0; i < npts; ++i)
        for (uint32_t j = 0; j < dim; ++j)
        {
            const float v = det_float(seed, i, j);
            out.write(reinterpret_cast<const char *>(&v), sizeof(float));
        }
}

// Cleanup helper for the multi-file legacy memory index (save() writes the
// graph file plus a ".data" sidecar, and ".tags" when tags are enabled).
struct ScopedLegacyMemFiles
{
    std::string prefix;
    explicit ScopedLegacyMemFiles(std::string p) : prefix(std::move(p))
    {
    }
    ~ScopedLegacyMemFiles()
    {
        for (const char *suffix : {"", ".data", ".tags"})
            std::remove((prefix + suffix).c_str());
    }
};

// Fraction of `a`'s top-K ids that also appear in `b`'s top-K (set overlap).
template <typename IdA, typename IdB>
double topk_overlap(const std::vector<IdA> &a, const std::vector<IdB> &b)
{
    std::unordered_set<uint64_t> bs;
    for (auto id : b)
        bs.insert(static_cast<uint64_t>(id));
    size_t inter = 0;
    for (auto id : a)
        if (bs.count(static_cast<uint64_t>(id)))
            ++inter;
    return a.empty() ? 1.0 : static_cast<double>(inter) / static_cast<double>(a.size());
}

// Read an entire file into a byte buffer.
std::vector<uint8_t> slurp_all(const std::string &path)
{
    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(path, std::ios::binary | std::ios::ate);
    const std::streamoff sz = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(static_cast<size_t>(sz));
    if (sz > 0)
        in.read(reinterpret_cast<char *>(out.data()), sz);
    return out;
}

// Cleanup helper for the legacy SSD index artifacts (mem graph, PQ files +
// sidecars, and the sector-packed disk index).
struct ScopedLegacySsdFiles
{
    std::string prefix;
    explicit ScopedLegacySsdFiles(std::string p) : prefix(std::move(p))
    {
    }
    ~ScopedLegacySsdFiles()
    {
        for (const char *suffix :
             {"_mem.index", "_mem.index.data", "_mem.index.tags", "_disk.index", "_disk.index_medoids.bin",
              "_disk.index_centroids.bin", "_pq_pivots.bin", "_pq_pivots.bin_centroid.bin",
              "_pq_pivots.bin_chunk_offsets.bin", "_pq_pivots.bin_rearrangement_perm.bin", "_pq_compressed.bin"})
            std::remove((prefix + suffix).c_str());
    }
};

// Build the legacy SSD index files from an EXISTING Index, mirroring the
// relevant internals of diskann::build_disk_index (which we can't reuse
// directly because it constructs its own Index instance). Emits
// <prefix>_disk.index, <prefix>_pq_pivots.bin, <prefix>_pq_compressed.bin.
// Returns the PQ pivot + code bytes so the caller can embed the SAME PQ
// codebook into the unified file -- guaranteeing both indices share graph AND
// PQ, so any search-result difference is purely a format/decoder difference.
void make_legacy_ssd_from_index(Index<float, uint32_t, uint32_t> &idx, const std::string &data_file,
                                const std::string &prefix, uint32_t num_pq_chunks, diskann::Metric metric,
                                std::vector<uint8_t> &pq_pivots_bytes, std::vector<uint8_t> &pq_codes_bytes)
{
    const std::string mem_index = prefix + "_mem.index";
    const std::string pq_pivots = prefix + "_pq_pivots.bin";
    const std::string pq_codes = prefix + "_pq_compressed.bin";
    const std::string disk_index = prefix + "_disk.index";

    // 1) Save the Vamana graph (the same graph that backs the unified file).
    idx.save(mem_index.c_str());

    // 2) Train PQ once. These files feed BOTH the legacy PQFlashIndex and (via
    //    slurp) the unified file, so both sides use byte-identical codes.
    diskann::generate_quantized_data<float>(data_file, pq_pivots, pq_codes, metric, /*p_val=*/1.0,
                                            num_pq_chunks, /*use_opq=*/false, /*codebook_prefix=*/"");

    // 3) Pack coords + adjacency into the sector-aligned legacy disk index.
    diskann::create_disk_layout<float>(data_file, mem_index, disk_index);

    pq_pivots_bytes = slurp_all(pq_pivots);
    pq_codes_bytes = slurp_all(pq_codes);
}

// --- Filtered-index helpers -------------------------------------------------

inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Assign each of `npts` points a small random subset (1..3) of string labels
// drawn from the vocabulary {"0".."vocab-1"}, deterministically from `seed`.
// Every point gets >= 1 label (the filtered build rejects label-less points).
std::vector<std::vector<std::string>> gen_label_sets(uint32_t npts, uint32_t vocab, uint64_t seed)
{
    std::vector<std::vector<std::string>> sets(npts);
    for (uint32_t i = 0; i < npts; ++i)
    {
        uint64_t h = splitmix64(seed ^ (static_cast<uint64_t>(i) * 0x100000001B3ull));
        const uint32_t count = 1u + static_cast<uint32_t>(h % 3u); // 1..3 labels
        std::set<uint32_t> chosen;
        for (uint32_t c = 0; c < count; ++c)
        {
            h = splitmix64(h);
            chosen.insert(static_cast<uint32_t>(h % vocab));
        }
        for (uint32_t lb : chosen)
            sets[i].push_back(std::to_string(lb));
    }
    return sets;
}

// Write a DiskANN string-label file: one comma-separated line of labels per point.
void write_label_file(const std::string &path, const std::vector<std::vector<std::string>> &sets)
{
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(path, std::ios::trunc);
    for (const auto &s : sets)
    {
        for (size_t j = 0; j < s.size(); ++j)
        {
            out << s[j];
            if (j + 1 < s.size())
                out << ",";
        }
        out << "\n";
    }
}

bool point_has_label(const std::vector<std::vector<std::string>> &sets, uint32_t id, const std::string &lbl)
{
    if (id >= sets.size())
        return false;
    const auto &s = sets[id];
    return std::find(s.begin(), s.end(), lbl) != s.end();
}

// Cleanup helper for the many sidecar files a filtered legacy index emits.
struct ScopedFilteredLegacyFiles
{
    std::string prefix;
    explicit ScopedFilteredLegacyFiles(std::string p) : prefix(std::move(p))
    {
    }
    ~ScopedFilteredLegacyFiles()
    {
        for (const char *suffix :
             {"", ".data", ".tags", ".del", "_labels.txt", "_labels_map.txt", "_labels_to_medoids.txt",
              "_universal_label.txt", "_bitmask_labels.bin", "_integer_labels.bin", "_label_formatted.txt"})
            std::remove((prefix + suffix).c_str());
    }
};

// Build the legacy FILTERED SSD index from an existing filtered Index. The mem
// graph is saved to <prefix> (not <prefix>_mem.index) so its label sidecars
// land at <prefix>_labels.txt / _labels_to_medoids.txt / _bitmask_labels.bin,
// which is exactly where PQFlashIndex::load(prefix) looks. The filtered build
// must have used save_path_prefix == prefix so <prefix>_labels_map.txt exists.
void make_legacy_ssd_filtered_from_index(Index<float, uint32_t, uint32_t> &idx, const std::string &data_file,
                                         const std::string &prefix, uint32_t num_pq_chunks, diskann::Metric metric,
                                         std::vector<uint8_t> &pq_pivots_bytes, std::vector<uint8_t> &pq_codes_bytes)
{
    const std::string mem_index = prefix; // graph file; label sidecars co-locate here
    const std::string pq_pivots = prefix + "_pq_pivots.bin";
    const std::string pq_codes = prefix + "_pq_compressed.bin";
    const std::string disk_index = prefix + "_disk.index";

    idx.save(mem_index.c_str());
    diskann::generate_quantized_data<float>(data_file, pq_pivots, pq_codes, metric, /*p_val=*/1.0, num_pq_chunks,
                                            /*use_opq=*/false, /*codebook_prefix=*/"");
    diskann::create_disk_layout<float>(data_file, mem_index, disk_index);

    pq_pivots_bytes = slurp_all(pq_pivots);
    pq_codes_bytes = slurp_all(pq_codes);
}

// Cleanup for the legacy filtered SSD artifacts.
struct ScopedFilteredLegacySsdFiles
{
    std::string prefix;
    explicit ScopedFilteredLegacySsdFiles(std::string p) : prefix(std::move(p))
    {
    }
    ~ScopedFilteredLegacySsdFiles()
    {
        for (const char *suffix :
             {"", ".data", ".tags", ".del", "_labels.txt", "_labels_map.txt", "_labels_to_medoids.txt",
              "_universal_label.txt", "_bitmask_labels.bin", "_integer_labels.bin", "_label_formatted.txt",
              "_disk.index", "_disk.index_medoids.bin", "_disk.index_centroids.bin", "_pq_pivots.bin",
              "_pq_pivots.bin_centroid.bin", "_pq_pivots.bin_chunk_offsets.bin",
              "_pq_pivots.bin_rearrangement_perm.bin", "_pq_compressed.bin"})
            std::remove((prefix + suffix).c_str());
    }
};

} // namespace

BOOST_AUTO_TEST_SUITE(unified_parity_tests)

BOOST_AUTO_TEST_CASE(memory_parity_legacy_vs_unified)
{
    const uint32_t npts = 10000;
    const uint32_t dim = 32;
    const uint32_t nq = 100;
    const uint32_t R = 32, L = 100, K = 10, search_L = 100;

    ScopedFile data_sf(tmp_path("parity_mem_data"));
    write_float_bin_parity(data_sf.path, npts, dim, /*seed=*/123);

    // 1) Build one Vamana Index<float> (single-threaded for determinism).
    auto write_params = std::make_shared<IndexWriteParameters>(
        IndexWriteParametersBuilder(L, R).with_alpha(1.2f).with_num_threads(1).build());
    Index<float, uint32_t, uint32_t> idx(diskann::Metric::L2, dim, npts, write_params, nullptr,
                                         /*num_frozen_pts=*/0, /*dynamic=*/false, /*enable_tags=*/false,
                                         /*concurrent_consolidate=*/false, /*pq_dist_build=*/false,
                                         /*num_pq_chunks=*/0, /*use_opq=*/false, /*filtered_index=*/false);
    idx.build(data_sf.path.c_str(), npts, std::vector<uint32_t>());

    // 2) Emit BOTH formats from the same in-memory graph.
    ScopedLegacyMemFiles legacy(tmp_path("parity_mem_legacy"));
    ScopedFile unified_sf(tmp_path("parity_mem_unified"));
    idx.save(legacy.prefix.c_str());
    idx.save_unified(unified_sf.path.c_str());

    // 3) Load the legacy memory index from disk.
    Index<float, uint32_t, uint32_t> legacy_idx(diskann::Metric::L2, dim, npts, write_params, nullptr,
                                                /*num_frozen_pts=*/0, /*dynamic=*/false, /*enable_tags=*/false,
                                                /*concurrent_consolidate=*/false, /*pq_dist_build=*/false,
                                                /*num_pq_chunks=*/0, /*use_opq=*/false, /*filtered_index=*/false);
    legacy_idx.load(legacy.prefix.c_str(), /*num_threads=*/1, /*search_l=*/search_L);

    // 4) Load the unified memory index from disk.
    UnifiedLoadContext uctx;
    uctx.path = unified_sf.path;
    uctx.num_threads = 1;
    uctx.search_l = search_L;
    auto uidx = make_unified_index_memory(uctx);
    BOOST_REQUIRE(uidx != nullptr);
    BOOST_REQUIRE_EQUAL(uidx->num_points(), npts);

    // 5) Search the same queries on both; top-K must be (near-)identical.
    double total_overlap = 0.0;
    size_t exact = 0;
    for (uint32_t q = 0; q < nq; ++q)
    {
        std::vector<float> query(dim);
        for (uint32_t j = 0; j < dim; ++j)
            query[j] = det_float(/*seed=*/999, q, j);

        std::vector<uint32_t> legacy_ids(K, 0);
        std::vector<float> legacy_dists(K, 0.0f);
        legacy_idx.search<uint32_t>(query.data(), K, search_L, legacy_ids.data(), legacy_dists.data());

        std::vector<uint64_t> uni_ids(K, 0);
        std::vector<float> uni_dists(K, 0.0f);
        UnifiedSearchContext sctx;
        sctx.query = query.data();
        sctx.K = K;
        sctx.L = search_L;
        sctx.indices = uni_ids.data();
        sctx.distances = uni_dists.data();
        uidx->search(sctx);

        const double ov = topk_overlap(legacy_ids, uni_ids);
        total_overlap += ov;
        if (ov >= 1.0)
            ++exact;
    }
    const double avg_overlap = total_overlap / nq;
    BOOST_TEST_MESSAGE("memory parity: avg top-" << K << " overlap = " << avg_overlap << ", exact = " << exact << "/"
                                                 << nq);
    // EXACT parity is expected: both indices are built from the SAME in-memory
    // graph, the memory search has no RNG, runs single-threaded, seeds from the
    // same single medoid, and NeighborPriorityQueue breaks distance ties
    // deterministically by id (see Neighbor::operator< in include/neighbor.h).
    // So every query's top-K must be identical.
    BOOST_CHECK_EQUAL(exact, static_cast<size_t>(nq));
}

BOOST_AUTO_TEST_CASE(parity_unaligned_dim_66)
{
    // Regression for the coord-width mismatch: UnifiedIndexWriter::write_node
    // stores coords as dim*sizeof(T) while the node store decodes them as
    // aligned_dim*sizeof(T). When dim is a multiple of 8 (e.g. 32) the two
    // agree and the bug is hidden; dim=66 -> aligned_dim=72 exposes it. Both
    // the memory and SSD unified paths must still match the legacy results.
    const uint32_t npts = 3000;
    const uint32_t dim = 66; // aligned_dim = 72
    const uint32_t nq = 50;
    const uint32_t R = 32, L = 100, K = 10, search_L = 100, beam = 4, pq_chunks = 22;

    ScopedFile data_sf(tmp_path("parity_u66_data"));
    write_float_bin_parity(data_sf.path, npts, dim, /*seed=*/1234);

    auto write_params = std::make_shared<IndexWriteParameters>(
        IndexWriteParametersBuilder(L, R).with_alpha(1.2f).with_num_threads(1).build());
    Index<float, uint32_t, uint32_t> idx(diskann::Metric::L2, dim, npts, write_params, nullptr,
                                         /*num_frozen_pts=*/0, /*dynamic=*/false, /*enable_tags=*/false,
                                         /*concurrent_consolidate=*/false, /*pq_dist_build=*/false,
                                         /*num_pq_chunks=*/0, /*use_opq=*/false, /*filtered_index=*/false);
    idx.build(data_sf.path.c_str(), npts, std::vector<uint32_t>());

    // ---- Memory parity ----
    ScopedLegacyMemFiles legacy_mem(tmp_path("parity_u66_legacy_mem"));
    ScopedFile unified_mem_sf(tmp_path("parity_u66_unified_mem"));
    idx.save(legacy_mem.prefix.c_str());
    idx.save_unified(unified_mem_sf.path.c_str());

    Index<float, uint32_t, uint32_t> legacy_mem_idx(diskann::Metric::L2, dim, npts, write_params, nullptr, 0, false,
                                                    false, false, false, 0, false, false);
    legacy_mem_idx.load(legacy_mem.prefix.c_str(), /*num_threads=*/1, /*search_l=*/search_L);

    UnifiedLoadContext mctx;
    mctx.path = unified_mem_sf.path;
    mctx.num_threads = 1;
    mctx.search_l = search_L;
    auto umem = make_unified_index_memory(mctx);
    BOOST_REQUIRE(umem != nullptr);
    BOOST_CHECK_EQUAL(umem->dim(), static_cast<uint64_t>(dim));
    BOOST_CHECK_EQUAL(umem->aligned_dim(), 72u);

    // ---- SSD parity (shared PQ) ----
    ScopedLegacySsdFiles legacy_ssd(tmp_path("parity_u66_legacy_ssd"));
    std::vector<uint8_t> pq_pivots_bytes, pq_codes_bytes;
    make_legacy_ssd_from_index(idx, data_sf.path, legacy_ssd.prefix, pq_chunks, diskann::Metric::L2, pq_pivots_bytes,
                               pq_codes_bytes);
    ScopedFile unified_ssd_sf(tmp_path("parity_u66_unified_ssd"));
    idx.save_unified(unified_ssd_sf.path.c_str(), pq_pivots_bytes, pq_codes_bytes);

    auto legacy_reader = make_reader();
    PQFlashIndex<float, uint32_t> pfi(legacy_reader, diskann::Metric::L2);
    BOOST_REQUIRE_EQUAL(pfi.load(/*num_threads=*/1, legacy_ssd.prefix.c_str()), 0);

    UnifiedLoadContext sctx_load;
    sctx_load.path = unified_ssd_sf.path;
    sctx_load.num_threads = 1;
    sctx_load.search_l = search_L;
    auto ureader = make_reader();
    auto ussd = make_unified_index_ssd(ureader, sctx_load);
    BOOST_REQUIRE(ussd != nullptr);

    size_t mem_exact = 0, ssd_exact = 0;
    for (uint32_t q = 0; q < nq; ++q)
    {
        std::vector<float> query(dim);
        for (uint32_t j = 0; j < dim; ++j)
            query[j] = det_float(/*seed=*/4321, q, j);

        std::vector<uint32_t> mem_legacy_ids(K, 0);
        std::vector<float> mem_legacy_dists(K, 0.0f);
        legacy_mem_idx.search<uint32_t>(query.data(), K, search_L, mem_legacy_ids.data(), mem_legacy_dists.data());
        std::vector<uint64_t> mem_uni_ids(K, 0);
        std::vector<float> mem_uni_dists(K, 0.0f);
        UnifiedSearchContext mq;
        mq.query = query.data();
        mq.K = K;
        mq.L = search_L;
        mq.indices = mem_uni_ids.data();
        mq.distances = mem_uni_dists.data();
        umem->search(mq);
        if (topk_overlap(mem_legacy_ids, mem_uni_ids) >= 1.0)
            ++mem_exact;

        std::vector<uint64_t> ssd_legacy_ids(K, 0);
        std::vector<float> ssd_legacy_dists(K, 0.0f);
        pfi.cached_beam_search(query.data(), K, search_L, ssd_legacy_ids.data(), ssd_legacy_dists.data(),
                               static_cast<uint64_t>(beam));
        std::vector<uint64_t> ssd_uni_ids(K, 0);
        std::vector<float> ssd_uni_dists(K, 0.0f);
        UnifiedSearchContext sq;
        sq.query = query.data();
        sq.K = K;
        sq.L = search_L;
        sq.indices = ssd_uni_ids.data();
        sq.distances = ssd_uni_dists.data();
        sq.beam_width = beam;
        ussd->search(sq);
        if (topk_overlap(ssd_legacy_ids, ssd_uni_ids) >= 1.0)
            ++ssd_exact;
    }
    BOOST_TEST_MESSAGE("unaligned dim=66 parity: memory exact = " << mem_exact << "/" << nq
                                                                  << ", ssd exact = " << ssd_exact << "/" << nq);
    BOOST_CHECK_EQUAL(mem_exact, static_cast<size_t>(nq));
    BOOST_CHECK_EQUAL(ssd_exact, static_cast<size_t>(nq));

    ussd.reset();
    ureader->close();
}

BOOST_AUTO_TEST_CASE(ssd_parity_legacy_vs_unified)
{
    const uint32_t npts = 10000;
    const uint32_t dim = 32;
    const uint32_t nq = 100;
    const uint32_t R = 32, L = 100, K = 10, search_L = 100, beam = 4, pq_chunks = 16;

    ScopedFile data_sf(tmp_path("parity_ssd_data"));
    write_float_bin_parity(data_sf.path, npts, dim, /*seed=*/234);

    // 1) Build one Vamana Index<float>.
    auto write_params = std::make_shared<IndexWriteParameters>(
        IndexWriteParametersBuilder(L, R).with_alpha(1.2f).with_num_threads(1).build());
    Index<float, uint32_t, uint32_t> idx(diskann::Metric::L2, dim, npts, write_params, nullptr,
                                         /*num_frozen_pts=*/0, /*dynamic=*/false, /*enable_tags=*/false,
                                         /*concurrent_consolidate=*/false, /*pq_dist_build=*/false,
                                         /*num_pq_chunks=*/0, /*use_opq=*/false, /*filtered_index=*/false);
    idx.build(data_sf.path.c_str(), npts, std::vector<uint32_t>());

    // 2) Legacy SSD index from the Index (also returns the shared PQ bytes).
    ScopedLegacySsdFiles legacy(tmp_path("parity_ssd_legacy"));
    std::vector<uint8_t> pq_pivots_bytes, pq_codes_bytes;
    make_legacy_ssd_from_index(idx, data_sf.path, legacy.prefix, pq_chunks, diskann::Metric::L2, pq_pivots_bytes,
                               pq_codes_bytes);

    // 3) Unified SSD file from the SAME Index + SAME PQ codebook.
    ScopedFile unified_sf(tmp_path("parity_ssd_unified"));
    idx.save_unified(unified_sf.path.c_str(), pq_pivots_bytes, pq_codes_bytes);

    // 4) Load the legacy PQFlashIndex.
    auto legacy_reader = make_reader();
    PQFlashIndex<float, uint32_t> pfi(legacy_reader, diskann::Metric::L2);
    const int rc = pfi.load(/*num_threads=*/1, legacy.prefix.c_str());
    BOOST_REQUIRE_EQUAL(rc, 0);

    // 5) Load the unified SSD index.
    UnifiedLoadContext uctx;
    uctx.path = unified_sf.path;
    uctx.num_threads = 1;
    uctx.search_l = search_L;
    auto ureader = make_reader();
    auto uidx = make_unified_index_ssd(ureader, uctx);
    BOOST_REQUIRE(uidx != nullptr);
    BOOST_REQUIRE_EQUAL(uidx->num_points(), npts);

    // 6) Search the same queries on both; top-K should match.
    double total_overlap = 0.0;
    size_t exact = 0;
    for (uint32_t q = 0; q < nq; ++q)
    {
        std::vector<float> query(dim);
        for (uint32_t j = 0; j < dim; ++j)
            query[j] = det_float(/*seed=*/888, q, j);

        std::vector<uint64_t> legacy_ids(K, 0);
        std::vector<float> legacy_dists(K, 0.0f);
        pfi.cached_beam_search(query.data(), K, search_L, legacy_ids.data(), legacy_dists.data(),
                               static_cast<uint64_t>(beam));

        std::vector<uint64_t> uni_ids(K, 0);
        std::vector<float> uni_dists(K, 0.0f);
        UnifiedSearchContext sctx;
        sctx.query = query.data();
        sctx.K = K;
        sctx.L = search_L;
        sctx.indices = uni_ids.data();
        sctx.distances = uni_dists.data();
        sctx.beam_width = beam;
        uidx->search(sctx);

        const double ov = topk_overlap(legacy_ids, uni_ids);
        total_overlap += ov;
        if (ov >= 1.0)
            ++exact;
    }
    const double avg_overlap = total_overlap / nq;
    BOOST_TEST_MESSAGE("ssd parity: avg top-" << K << " overlap = " << avg_overlap << ", exact = " << exact << "/"
                                              << nq);
    // EXACT parity is expected. Both indices share the SAME graph AND the SAME
    // PQ codebook (the pivot/code bytes are generated once and fed to both), so
    // the beam search is fully deterministic: no RNG on the search path (the
    // cache_bfs_levels shuffle is not triggered -- no cache priming), single
    // thread, one deterministic medoid seed, and NeighborPriorityQueue breaks
    // distance ties by id. Every query's top-K must be identical.
    BOOST_CHECK_EQUAL(exact, static_cast<size_t>(nq));

    // WindowsAlignedFileReader keeps the unified file open (its destructor does
    // not close, unlike PQFlashIndex which calls reader->close()). Release the
    // index and close the reader so ScopedFile can delete the backing file.
    uidx.reset();
    ureader->close();
}

BOOST_AUTO_TEST_CASE(memory_filtered_parity_legacy_vs_unified)
{
    const uint32_t npts = 10000;
    const uint32_t dim = 32;
    const uint32_t nq = 100;
    const uint32_t R = 32, L = 100, K = 10, search_L = 100, vocab = 8;

    ScopedFile data_sf(tmp_path("parity_fmem_data"));
    write_float_bin_parity(data_sf.path, npts, dim, /*seed=*/321);

    // Simulate random per-point labels and write the label file.
    const auto label_sets = gen_label_sets(npts, vocab, /*seed=*/55);
    ScopedFile rawlabels_sf(tmp_path("parity_fmem_rawlabels"));
    write_label_file(rawlabels_sf.path, label_sets);

    // 1) Build ONE filtered Vamana Index<float>. filter_list_size (Lf) MUST be
    // set for a filtered build -- it defaults to 0, which makes the filtered
    // link phase run with an empty search list and crash.
    auto write_params = std::make_shared<IndexWriteParameters>(
        IndexWriteParametersBuilder(L, R).with_alpha(1.2f).with_num_threads(1).with_filter_list_size(L).build());
    Index<float, uint32_t, uint32_t> idx(diskann::Metric::L2, dim, npts, write_params, nullptr,
                                         /*num_frozen_pts=*/0, /*dynamic=*/false, /*enable_tags=*/false,
                                         /*concurrent_consolidate=*/false, /*pq_dist_build=*/false,
                                         /*num_pq_chunks=*/0, /*use_opq=*/false, /*filtered_index=*/true);
    ScopedFilteredLegacyFiles legacy(tmp_path("parity_fmem_legacy"));
    {
        // save_path_prefix == legacy prefix so the labels_map + label files that
        // build/save emit all co-locate for the subsequent legacy load().
        IndexFilterParams fp = IndexFilterParamsBuilder()
                                   .with_label_file(rawlabels_sf.path)
                                   .with_save_path_prefix(legacy.prefix)
                                   .build();
        idx.build(data_sf.path, npts, fp);
    }

    // 2) Emit both formats from the same filtered graph.
    ScopedFile unified_sf(tmp_path("parity_fmem_unified"));
    idx.save(legacy.prefix.c_str());
    idx.save_unified(unified_sf.path.c_str());

    // 3) Load the legacy filtered index.
    Index<float, uint32_t, uint32_t> legacy_idx(diskann::Metric::L2, dim, npts, write_params, nullptr,
                                                /*num_frozen_pts=*/0, /*dynamic=*/false, /*enable_tags=*/false,
                                                /*concurrent_consolidate=*/false, /*pq_dist_build=*/false,
                                                /*num_pq_chunks=*/0, /*use_opq=*/false, /*filtered_index=*/true);
    legacy_idx.load(legacy.prefix.c_str(), /*num_threads=*/1, /*search_l=*/search_L);

    // 4) Load the unified index.
    UnifiedLoadContext uctx;
    uctx.path = unified_sf.path;
    uctx.num_threads = 1;
    uctx.search_l = search_L;
    auto uidx = make_unified_index_memory(uctx);
    BOOST_REQUIRE(uidx != nullptr);

    // get_table_stats() on a real filtered index: node/label cardinality and a
    // non-zero label memory footprint (bitmask storage).
    const TableStats ust = uidx->get_table_stats();
    BOOST_CHECK_EQUAL(ust.node_count, npts);
    BOOST_CHECK_EQUAL(ust.label_count, static_cast<size_t>(vocab));
    BOOST_CHECK_GT(ust.label_mem_usage, 0u);
    BOOST_CHECK_GT(ust.node_mem_usage, 0u);
    double total_overlap = 0.0;
    size_t exact = 0, nonempty = 0, legacy_bad = 0, uni_bad = 0;
    for (uint32_t q = 0; q < nq; ++q)
    {
        std::vector<float> query(dim);
        for (uint32_t j = 0; j < dim; ++j)
            query[j] = det_float(/*seed=*/444, q, j);
        const std::string flabel = std::to_string(q % vocab);

        std::vector<uint32_t> legacy_ids(K, std::numeric_limits<uint32_t>::max());
        std::vector<float> legacy_dists(K, 0.0f);
        std::vector<uint32_t> filter_ints = {legacy_idx.get_converted_label(flabel)};
        legacy_idx.search_with_filters<uint32_t>(query.data(), filter_ints, K, search_L, /*maxLperSeller=*/0,
                                                 legacy_ids.data(), legacy_dists.data());

        std::vector<uint64_t> uni_ids(K, std::numeric_limits<uint64_t>::max());
        std::vector<float> uni_dists(K, 0.0f);
        UnifiedSearchContext sctx;
        sctx.query = query.data();
        sctx.K = K;
        sctx.L = search_L;
        sctx.indices = uni_ids.data();
        sctx.distances = uni_dists.data();
        sctx.filter_labels = {flabel};
        uidx->search(sctx);

        // Correctness: every returned point MUST carry the filter label.
        std::vector<uint32_t> lvalid;
        std::vector<uint64_t> uvalid;
        for (uint32_t id : legacy_ids)
            if (id != std::numeric_limits<uint32_t>::max())
            {
                lvalid.push_back(id);
                if (!point_has_label(label_sets, id, flabel))
                    ++legacy_bad;
            }
        for (uint64_t id : uni_ids)
            if (id != std::numeric_limits<uint64_t>::max())
            {
                uvalid.push_back(id);
                if (!point_has_label(label_sets, static_cast<uint32_t>(id), flabel))
                    ++uni_bad;
            }

        if (!lvalid.empty())
        {
            const double ov = topk_overlap(lvalid, uvalid);
            total_overlap += ov;
            ++nonempty;
            if (ov >= 1.0)
                ++exact;
        }
    }
    const double avg_overlap = nonempty ? total_overlap / nonempty : 1.0;
    BOOST_TEST_MESSAGE("memory filtered parity: avg overlap = " << avg_overlap << ", exact = " << exact << "/"
                                                                << nonempty << ", legacy_bad = " << legacy_bad
                                                                << ", uni_bad = " << uni_bad);
    // Correctness is exact: no result from either index may violate the filter.
    BOOST_CHECK_EQUAL(legacy_bad, 0u);
    BOOST_CHECK_EQUAL(uni_bad, 0u);
    // Legacy seeds init_ids from the global _start medoid PLUS per-label medoids
    // (Index::search_with_filters), while the unified seeds ONLY from per-label
    // medoids (a deliberate recall-oriented choice). So results are highly
    // similar but not required to be bit-identical.
    BOOST_CHECK_GE(avg_overlap, 0.90);
}

BOOST_AUTO_TEST_CASE(ssd_filtered_parity_legacy_vs_unified)
{
    const uint32_t npts = 10000;
    const uint32_t dim = 32;
    const uint32_t nq = 100;
    const uint32_t R = 32, L = 100, K = 10, search_L = 100, beam = 4, pq_chunks = 16, vocab = 8;

    ScopedFile data_sf(tmp_path("parity_fssd_data"));
    write_float_bin_parity(data_sf.path, npts, dim, /*seed=*/876);

    const auto label_sets = gen_label_sets(npts, vocab, /*seed=*/77);
    ScopedFile rawlabels_sf(tmp_path("parity_fssd_rawlabels"));
    write_label_file(rawlabels_sf.path, label_sets);

    // 1) Build ONE filtered Index. save_path_prefix == the SSD prefix so the
    //    labels_map lands where PQFlashIndex::load expects it.
    ScopedFilteredLegacySsdFiles legacy(tmp_path("parity_fssd_legacy"));
    auto write_params = std::make_shared<IndexWriteParameters>(
        IndexWriteParametersBuilder(L, R).with_alpha(1.2f).with_num_threads(1).with_filter_list_size(L).build());
    Index<float, uint32_t, uint32_t> idx(diskann::Metric::L2, dim, npts, write_params, nullptr,
                                         /*num_frozen_pts=*/0, /*dynamic=*/false, /*enable_tags=*/false,
                                         /*concurrent_consolidate=*/false, /*pq_dist_build=*/false,
                                         /*num_pq_chunks=*/0, /*use_opq=*/false, /*filtered_index=*/true);
    {
        IndexFilterParams fp = IndexFilterParamsBuilder()
                                   .with_label_file(rawlabels_sf.path)
                                   .with_save_path_prefix(legacy.prefix)
                                   .build();
        idx.build(data_sf.path, npts, fp);
    }

    // 2) Legacy filtered SSD (+ shared PQ) and unified filtered SSD.
    std::vector<uint8_t> pq_pivots_bytes, pq_codes_bytes;
    make_legacy_ssd_filtered_from_index(idx, data_sf.path, legacy.prefix, pq_chunks, diskann::Metric::L2,
                                        pq_pivots_bytes, pq_codes_bytes);
    ScopedFile unified_sf(tmp_path("parity_fssd_unified"));
    idx.save_unified(unified_sf.path.c_str(), pq_pivots_bytes, pq_codes_bytes);

    // 3) Load the legacy filtered PQFlashIndex.
    auto legacy_reader = make_reader();
    PQFlashIndex<float, uint32_t> pfi(legacy_reader, diskann::Metric::L2);
    const int rc = pfi.load(/*num_threads=*/1, legacy.prefix.c_str());
    BOOST_REQUIRE_EQUAL(rc, 0);

    // 4) Load the unified filtered SSD index.
    UnifiedLoadContext uctx;
    uctx.path = unified_sf.path;
    uctx.num_threads = 1;
    uctx.search_l = search_L;
    auto ureader = make_reader();
    auto uidx = make_unified_index_ssd(ureader, uctx);
    BOOST_REQUIRE(uidx != nullptr);

    // 5) Search each query under a rotating filter label; correctness + parity.
    double total_overlap = 0.0;
    size_t exact = 0, nonempty = 0, legacy_bad = 0, uni_bad = 0;
    for (uint32_t q = 0; q < nq; ++q)
    {
        std::vector<float> query(dim);
        for (uint32_t j = 0; j < dim; ++j)
            query[j] = det_float(/*seed=*/222, q, j);
        const std::string flabel = std::to_string(q % vocab);

        std::vector<uint64_t> legacy_ids(K, std::numeric_limits<uint64_t>::max());
        std::vector<float> legacy_dists(K, 0.0f);
        std::vector<uint32_t> filter_ints = {pfi.get_converted_label(flabel)};
        pfi.cached_beam_search(query.data(), K, search_L, legacy_ids.data(), legacy_dists.data(),
                               static_cast<uint64_t>(beam), /*use_filter=*/true, filter_ints);

        std::vector<uint64_t> uni_ids(K, std::numeric_limits<uint64_t>::max());
        std::vector<float> uni_dists(K, 0.0f);
        UnifiedSearchContext sctx;
        sctx.query = query.data();
        sctx.K = K;
        sctx.L = search_L;
        sctx.indices = uni_ids.data();
        sctx.distances = uni_dists.data();
        sctx.beam_width = beam;
        sctx.filter_labels = {flabel};
        uidx->search(sctx);

        std::vector<uint64_t> lvalid, uvalid;
        for (uint64_t id : legacy_ids)
            if (id != std::numeric_limits<uint64_t>::max())
            {
                lvalid.push_back(id);
                if (!point_has_label(label_sets, static_cast<uint32_t>(id), flabel))
                    ++legacy_bad;
            }
        for (uint64_t id : uni_ids)
            if (id != std::numeric_limits<uint64_t>::max())
            {
                uvalid.push_back(id);
                if (!point_has_label(label_sets, static_cast<uint32_t>(id), flabel))
                    ++uni_bad;
            }

        if (!lvalid.empty())
        {
            const double ov = topk_overlap(lvalid, uvalid);
            total_overlap += ov;
            ++nonempty;
            if (ov >= 1.0)
                ++exact;
        }
    }
    const double avg_overlap = nonempty ? total_overlap / nonempty : 1.0;
    BOOST_TEST_MESSAGE("ssd filtered parity: avg overlap = " << avg_overlap << ", exact = " << exact << "/" << nonempty
                                                             << ", legacy_bad = " << legacy_bad
                                                             << ", uni_bad = " << uni_bad);
    // Correctness: no result may violate the filter.
    BOOST_CHECK_EQUAL(legacy_bad, 0u);
    BOOST_CHECK_EQUAL(uni_bad, 0u);
    // For a single filter label per query, the legacy filtered SSD search seeds
    // ONLY from that label's medoid (no global seed) -- same as the unified --
    // so with the shared graph + PQ the results should be highly similar.
    BOOST_CHECK_GE(avg_overlap, 0.90);

    uidx.reset();
    ureader->close();
}

BOOST_AUTO_TEST_SUITE_END()

// ===========================================================================
// Suite 10: get_table_stats()
// ===========================================================================

BOOST_AUTO_TEST_SUITE(unified_table_stats_tests)

BOOST_AUTO_TEST_CASE(memory_stats_unfiltered)
{
    ScopedFile sf(tmp_path("stats_mem"));
    const uint64_t npts = 64, aligned_dim = 16;
    std::vector<std::vector<float>> points;
    write_star_graph_unified(sf.path, npts, aligned_dim, points);

    UnifiedLoadContext ctx;
    ctx.path = sf.path;
    auto idx = make_unified_index_memory(ctx);
    BOOST_REQUIRE(idx != nullptr);

    const TableStats st = idx->get_table_stats();
    BOOST_CHECK_EQUAL(st.node_count, npts);
    BOOST_CHECK_EQUAL(st.label_count, 0u); // unfiltered
    BOOST_CHECK_EQUAL(st.label_mem_usage, 0u);
    BOOST_CHECK_GT(st.node_mem_usage, 0u);
    // Memory keeps the full graph region resident, so total > 0 and is the sum
    // of the parts.
    BOOST_CHECK_GT(st.total_mem_usage, 0u);
    BOOST_CHECK_EQUAL(st.total_mem_usage, st.node_mem_usage + st.graph_mem_usage + st.label_mem_usage +
                                              st.tag_memory_usage);
}

BOOST_AUTO_TEST_CASE(ssd_stats_pq_codes)
{
    ScopedFile data_sf(tmp_path("stats_ssd_data"));
    const uint32_t npts = 512, dim = 16, pq_dim = 4;
    write_float_bin_parity(data_sf.path, npts, dim, /*seed=*/5);

    ScopedFile out_sf(tmp_path("stats_ssd_out"));
    UnifiedBuildContext bctx;
    bctx.data_file_path = data_sf.path;
    bctx.output_path = out_sf.path;
    bctx.data_type = DataTypeTag::Float;
    bctx.metric = diskann::Metric::L2;
    bctx.R = 16;
    bctx.L = 32;
    bctx.alpha = 1.2f;
    bctx.num_threads = 1;
    bctx.pq_dim = pq_dim;
    bctx.pq_sampling_rate = 1.0;
    unified_index_builder().build(bctx);

    UnifiedLoadContext ctx;
    ctx.path = out_sf.path;
    ctx.num_threads = 1;
    auto reader = make_reader();
    auto idx = make_unified_index_ssd(reader, ctx);
    BOOST_REQUIRE(idx != nullptr);

    const TableStats st = idx->get_table_stats();
    BOOST_CHECK_EQUAL(st.node_count, npts);
    // SSD node_mem_usage == resident PQ codes == npts * n_chunks; graph on disk.
    BOOST_CHECK_EQUAL(st.node_mem_usage, static_cast<size_t>(npts) * pq_dim);
    BOOST_CHECK_EQUAL(st.graph_mem_usage, 0u);
    BOOST_CHECK_EQUAL(st.total_mem_usage, st.node_mem_usage + st.graph_mem_usage + st.label_mem_usage +
                                              st.tag_memory_usage);

    idx.reset();
    reader->close();
}

BOOST_AUTO_TEST_SUITE_END()
