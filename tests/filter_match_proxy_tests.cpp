// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// Unit tests for filter_match_proxy changes introduced by the
// "software prefetch bitmask filter K=8 steps ahead" optimization:
//   1) new virtual prefetch_bitmask(id) on filter_match_proxy and its three
//      subclasses (bitmask_filter_match, integer_label_filter_match,
//      label_filter_match_holder).
//   2) bitmask_filter_match ctor pads query_bitmask_buf to at least 4 uint64
//      words for safe AVX2 256-bit loads.
//   3) contain_filtered_label semantics preserved (regression coverage).

#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <vector>

#include "filter_match_proxy.h"
#include "integer_label_vector.h"
#include "label_bitmask.h"

using diskann::bitmask_filter_match;
using diskann::integer_label_filter_match;
using diskann::label_filter_match_holder;
using diskann::integer_label_vector;
using diskann::simple_bitmask;
using diskann::simple_bitmask_buf;

namespace
{
// Build a simple_bitmask_buf sized for `num_points` points with `total_bits`
// bits per node, and set `set_labels` on point `point_id`.
simple_bitmask_buf make_bitmask_buf(std::uint64_t num_points,
                                    std::uint64_t total_bits,
                                    std::uint32_t point_id,
                                    const std::vector<std::uint32_t>& set_labels)
{
    const std::uint64_t bitmask_size = simple_bitmask::get_bitmask_size(total_bits);
    simple_bitmask_buf buf(num_points * bitmask_size, bitmask_size);
    if (bitmask_size > 0)
    {
        simple_bitmask bm(buf.get_bitmask(point_id), bitmask_size);
        for (auto lbl : set_labels)
            bm.set(lbl);
    }
    return buf;
}
} // namespace

BOOST_AUTO_TEST_SUITE(FilterMatchProxy_tests)

// ---- (2) AVX2 padding of query_bitmask_buf ----------------------------------
//
// build_query_mask sizes the query buffer to:
//     bitmask_size + AVX2_TAIL_PADDING
// (unified with the node-side simple_bitmask_buf padding scheme). This ensures a
// 256-bit (4x uint64) AVX2 load starting at query_bitmask_buf.data() never runs
// past the buffer end for any bitmask_size, and the trailing padding words stay
// zero so they never contribute false bits to the intersection test.

BOOST_AUTO_TEST_CASE(bitmask_ctor_pads_query_buf_when_size_1)
{
    // 1 word covers up to 64 bits; use 64 bits total.
    auto filters = make_bitmask_buf(/*num_points*/ 8, /*total_bits*/ 64,
                                    /*point_id*/ 0, /*set_labels*/ {});
    BOOST_TEST(filters._bitmask_size == (std::uint64_t)1);

    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {3};
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, /*unv*/ 0);

    // size + 4 padding.
    BOOST_TEST(qbuf.size() == (size_t)5);
}

BOOST_AUTO_TEST_CASE(bitmask_ctor_pads_query_buf_when_size_2)
{
    // 128 bits -> bitmask_size == 2.
    auto filters = make_bitmask_buf(4, 128, 0, {});
    BOOST_TEST(filters._bitmask_size == (std::uint64_t)2);

    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {1};
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, 0);

    BOOST_TEST(qbuf.size() == (size_t)6);
}

BOOST_AUTO_TEST_CASE(bitmask_ctor_pads_query_buf_when_size_3)
{
    // 192 bits -> bitmask_size == 3.
    auto filters = make_bitmask_buf(4, 192, 0, {});
    BOOST_TEST(filters._bitmask_size == (std::uint64_t)3);

    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {0};
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, 0);

    BOOST_TEST(qbuf.size() == (size_t)7);
}

BOOST_AUTO_TEST_CASE(bitmask_ctor_pads_query_buf_when_size_4)
{
    // 256 bits -> bitmask_size == 4.
    auto filters = make_bitmask_buf(4, 256, 0, {});
    BOOST_TEST(filters._bitmask_size == (std::uint64_t)4);

    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {0};
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, 0);

    BOOST_TEST(qbuf.size() == (size_t)8);
}

BOOST_AUTO_TEST_CASE(bitmask_ctor_pads_query_buf_when_size_larger_than_4)
{
    // 512 bits -> bitmask_size == 8; padded to 8 + 4.
    auto filters = make_bitmask_buf(4, 512, 0, {});
    BOOST_TEST(filters._bitmask_size == (std::uint64_t)8);

    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {0};
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, 0);

    BOOST_TEST(qbuf.size() == (size_t)12);
}

BOOST_AUTO_TEST_CASE(bitmask_ctor_size_zero_leaves_query_buf_empty)
{
    // bitmask_size == 0 means "no filter set" -- ctor short-circuits and must
    // not resize (nor pad) query_bitmask_buf.
    simple_bitmask_buf filters; // default: _bitmask_size == 0
    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels; // empty; wouldn't be used anyway
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, 0);

    BOOST_TEST(qbuf.size() == (size_t)0);
}

// ---- (1) prefetch_bitmask: bitmask_filter_match ----------------------------
//
// prefetch_bitmask is a CPU hint with no observable effect beyond the side
// effect of loading a cache line. We can't assert cache state from a UT, so
// coverage here is: (a) it does not crash, (b) it does not mutate any state
// observable via contain_filtered_label, (c) it correctly no-ops when
// _bitmask_size == 0.

BOOST_AUTO_TEST_CASE(bitmask_prefetch_does_not_crash_for_valid_ids)
{
    auto filters = make_bitmask_buf(/*num_points*/ 16, /*total_bits*/ 128,
                                    /*point_id*/ 5, /*set_labels*/ {3});
    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {3};
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, 0);

    for (std::uint32_t id = 0; id < 16; ++id)
        m.prefetch_bitmask(id);

    // State unchanged: point 5 still matches on label 3, others do not.
    BOOST_TEST(m.contain_filtered_label(5) == true);
    BOOST_TEST(m.contain_filtered_label(0) == false);
}

BOOST_AUTO_TEST_CASE(bitmask_prefetch_is_noop_when_size_zero)
{
    // With _bitmask_size == 0, prefetch_bitmask must skip the get_bitmask()
    // call (which would otherwise dereference a bogus offset). We just require
    // it not to crash for any id, including out-of-range values.
    simple_bitmask_buf filters;
    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels;
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, 0);

    m.prefetch_bitmask(0);
    m.prefetch_bitmask(1000000);
}

// ---- (1) prefetch_bitmask: integer_label_filter_match ----------------------
//
// Must be a pure no-op. Just verify it does not crash and does not mutate
// contain_filtered_label results.

BOOST_AUTO_TEST_CASE(integer_prefetch_is_noop)
{
    integer_label_vector lv;
    lv.initialize(/*numpoints*/ 4, /*total_labels*/ 8);
    std::vector<std::uint32_t> lbls_p0 = {10, 20};
    std::vector<std::uint32_t> lbls_p1 = {30};
    std::vector<std::uint32_t> lbls_p2 = {}; // point 2 has no labels
    std::vector<std::uint32_t> lbls_p3 = {40};
    lv.add_labels<std::uint32_t>(0, lbls_p0);
    lv.add_labels<std::uint32_t>(1, lbls_p1);
    lv.add_labels<std::uint32_t>(2, lbls_p2);
    lv.add_labels<std::uint32_t>(3, lbls_p3);

    std::vector<std::uint32_t> filter_labels = {20};
    integer_label_filter_match<std::uint32_t> m(lv, filter_labels, /*unv*/ 0);

    // Baseline behavior.
    BOOST_TEST(m.contain_filtered_label(0) == true);   // has 20
    BOOST_TEST(m.contain_filtered_label(1) == false);  // has 30 only
    BOOST_TEST(m.contain_filtered_label(2) == false);  // no labels

    // Prefetch must be a no-op for any id, including out-of-range.
    for (std::uint32_t id = 0; id < 10; ++id)
        m.prefetch_bitmask(id);
    m.prefetch_bitmask(0xFFFFFFFFu);

    // Behavior unchanged.
    BOOST_TEST(m.contain_filtered_label(0) == true);
    BOOST_TEST(m.contain_filtered_label(1) == false);
    BOOST_TEST(m.contain_filtered_label(2) == false);
    BOOST_TEST(m.contain_filtered_label(3) == false);
}

// ---- (1) prefetch_bitmask: label_filter_match_holder -----------------------
//
// Holder dispatches prefetch_bitmask to bitmask_filter_match only when
// !_use_integer_labels; otherwise it does nothing. Coverage: verify both
// paths compile and do not crash, and that contain_filtered_label routes to
// the corresponding backend.

BOOST_AUTO_TEST_CASE(holder_bitmask_path_prefetch_and_contain)
{
    auto filters = make_bitmask_buf(8, 128, /*point_id*/ 2, /*set_labels*/ {5});
    std::vector<std::uint64_t> qbuf;
    integer_label_vector lv; // unused on this path but required by ctor
    std::vector<std::uint32_t> filter_labels = {5};

    label_filter_match_holder<std::uint32_t> holder(
        filters, qbuf, lv, filter_labels, /*unv*/ 0, /*use_integer_labels*/ false);

    for (std::uint32_t id = 0; id < 8; ++id)
        holder.prefetch_bitmask(id);

    BOOST_TEST(holder.contain_filtered_label(2) == true);
    BOOST_TEST(holder.contain_filtered_label(0) == false);
}

BOOST_AUTO_TEST_CASE(holder_integer_path_prefetch_is_noop)
{
    // On the integer_labels path the holder must not touch the bitmask
    // backend at all -- so we can pass a default (size-0) simple_bitmask_buf
    // without crashing prefetch.
    simple_bitmask_buf filters; // _bitmask_size == 0
    std::vector<std::uint64_t> qbuf;

    integer_label_vector lv;
    lv.initialize(3, 4);
    std::vector<std::uint32_t> lbls0 = {7};
    std::vector<std::uint32_t> lbls1 = {8};
    std::vector<std::uint32_t> lbls2 = {9};
    lv.add_labels<std::uint32_t>(0, lbls0);
    lv.add_labels<std::uint32_t>(1, lbls1);
    lv.add_labels<std::uint32_t>(2, lbls2);

    std::vector<std::uint32_t> filter_labels = {8};

    label_filter_match_holder<std::uint32_t> holder(
        filters, qbuf, lv, filter_labels, /*unv*/ 0, /*use_integer_labels*/ true);

    for (std::uint32_t id = 0; id < 3; ++id)
        holder.prefetch_bitmask(id);
    holder.prefetch_bitmask(0xFFFFFFFFu);

    BOOST_TEST(holder.contain_filtered_label(0) == false);
    BOOST_TEST(holder.contain_filtered_label(1) == true);
    BOOST_TEST(holder.contain_filtered_label(2) == false);
}

// ---- (3) contain_filtered_label regression coverage ------------------------

BOOST_AUTO_TEST_CASE(bitmask_contain_matches_on_shared_label)
{
    auto filters = make_bitmask_buf(4, 128, /*point*/ 1, /*labels*/ {2, 40, 90});
    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {40};
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, /*unv*/ 0);

    BOOST_TEST(m.contain_filtered_label(1) == true);  // label 40 set on p1
    BOOST_TEST(m.contain_filtered_label(0) == false); // p0 has no labels
    BOOST_TEST(m.contain_filtered_label(2) == false);
    BOOST_TEST(m.contain_filtered_label(3) == false);
}

BOOST_AUTO_TEST_CASE(bitmask_contain_no_match_when_labels_disjoint)
{
    auto filters = make_bitmask_buf(4, 128, /*point*/ 0, /*labels*/ {1, 2, 3});
    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {10, 20};
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, /*unv*/ 0);

    BOOST_TEST(m.contain_filtered_label(0) == false);
}

BOOST_AUTO_TEST_CASE(bitmask_contain_multi_filter_labels_any_match)
{
    // Query bitmask ORs all filter labels -- any single overlap matches.
    auto filters = make_bitmask_buf(4, 128, /*point*/ 3, /*labels*/ {77});
    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {5, 77, 100};
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, 0);

    BOOST_TEST(m.contain_filtered_label(3) == true);
    BOOST_TEST(m.contain_filtered_label(2) == false);
}

BOOST_AUTO_TEST_CASE(integer_contain_matches_on_filter_or_unv)
{
    integer_label_vector lv;
    lv.initialize(4, 8);
    std::vector<std::uint32_t> p0 = {1, 2};
    std::vector<std::uint32_t> p1 = {3};
    std::vector<std::uint32_t> p2 = {99}; // unv label
    std::vector<std::uint32_t> p3 = {4};
    lv.add_labels<std::uint32_t>(0, p0);
    lv.add_labels<std::uint32_t>(1, p1);
    lv.add_labels<std::uint32_t>(2, p2);
    lv.add_labels<std::uint32_t>(3, p3);

    std::vector<std::uint32_t> filter_labels = {2, 4};
    integer_label_filter_match<std::uint32_t> m(lv, filter_labels, /*unv*/ 99);

    BOOST_TEST(m.contain_filtered_label(0) == true);  // has 2
    BOOST_TEST(m.contain_filtered_label(1) == false); // has 3 only
    BOOST_TEST(m.contain_filtered_label(2) == true);  // has unv 99
    BOOST_TEST(m.contain_filtered_label(3) == true);  // has 4
}

// ---- AVX2 correctness corner cases -----------------------------------------
//
// test_full_mask_val's AVX2 fast path (_bitmask_size <= 4) issues an
// UNCONDITIONAL 256-bit load over both the query and the node bitmask, then a
// single 256-bit AND. This only agrees with the scalar semantics ("intersect
// the first _bitmask_size words") if both operands are zero in the padding
// lanes past _bitmask_size. These tests pin the invariants that make that true;
// each one previously failed when the query buffer was not zero-padded to 4
// words or was reused without re-zeroing.

// A single-word bitmask (size 1) means the AVX2 load reads 3 words of the NEXT
// three points as the query's high lanes. If a match on point p leaked in from
// point p+1/p+2/p+3's labels, this would report a false positive. Only the
// point that actually carries the filter label may match.
BOOST_AUTO_TEST_CASE(bitmask_size1_no_cross_point_leak)
{
    // 64 bits -> size 1. Give ONLY point 5 the filter label; its neighbours
    // 6/7/8 carry other labels that must not leak into point 5's test.
    auto filters = make_bitmask_buf(/*num_points*/ 16, /*total_bits*/ 64,
                                    /*point_id*/ 5, /*set_labels*/ {9});
    // Set unrelated labels on the following points (whose words the AVX2 load
    // over point 5 will also touch as high lanes).
    {
        simple_bitmask bm6(filters.get_bitmask(6), filters._bitmask_size);
        bm6.set(9); // same label as the filter, but on a DIFFERENT point
        simple_bitmask bm7(filters.get_bitmask(7), filters._bitmask_size);
        bm7.set(9);
        simple_bitmask bm8(filters.get_bitmask(8), filters._bitmask_size);
        bm8.set(9);
    }
    BOOST_TEST(filters._bitmask_size == (std::uint64_t)1);

    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {9};
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, /*unv*/ 0);

    BOOST_TEST(m.contain_filtered_label(5) == true);   // really has label 9
    BOOST_TEST(m.contain_filtered_label(4) == false);  // no labels
    // Points 6/7/8 DO carry label 9, so they legitimately match -- the point is
    // that testing point 5 must not have leaked THEIR bits into point 5's result
    // (already asserted by 5==true / 4==false above). Confirm the boundary point
    // just past the filter label owner behaves per its own bits.
    BOOST_TEST(m.contain_filtered_label(6) == true);
}

// Reusing one query buffer across proxies is the exact scenario that broke:
// the second ctor sees a buffer already at size 4 and must still zero every
// word (clear+resize), or stale high/low words from the first proxy produce
// false matches.
BOOST_AUTO_TEST_CASE(bitmask_reused_query_buffer_is_rezeroed)
{
    auto filters = make_bitmask_buf(8, 64, /*point*/ 2, /*labels*/ {5});
    std::vector<std::uint64_t> qbuf;

    // First proxy: filter label 5, and deliberately a label whose bit lands in
    // a high word would require size>1; here size==1 so all bits are word 0.
    {
        std::vector<std::uint32_t> f1 = {5};
        bitmask_filter_match<std::uint32_t> m1(filters, qbuf, f1, /*unv*/ 0);
        BOOST_TEST(m1.contain_filtered_label(2) == true);
        BOOST_TEST(qbuf.size() == (size_t)5); // size 1 + 4 padding
    }
    // qbuf now retains proxy #1's bits (filter label 5 in word 0). A second
    // proxy filtering a DISJOINT label must not inherit label 5.
    {
        std::vector<std::uint32_t> f2 = {7}; // point 2 does NOT have label 7
        bitmask_filter_match<std::uint32_t> m2(filters, qbuf, f2, /*unv*/ 0);
        // If qbuf's word 0 still held label 5's bit, point 2 (which has 5) would
        // falsely match. It must not.
        BOOST_TEST(m2.contain_filtered_label(2) == false);
    }
}

// Labels on 64-bit word boundaries: bit 63 is the last bit of word 0, bit 64
// the first of word 1. With size 2 the AVX2 fast path still applies. A filter
// on bit 64 must match a point carrying bit 64 and not one carrying bit 63.
BOOST_AUTO_TEST_CASE(bitmask_word_boundary_labels_63_and_64)
{
    auto filters = make_bitmask_buf(4, /*total_bits*/ 128, /*point*/ 0, /*labels*/ {63});
    {
        simple_bitmask bm1(filters.get_bitmask(1), filters._bitmask_size);
        bm1.set(64);
    }
    BOOST_TEST(filters._bitmask_size == (std::uint64_t)2);

    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {64};
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, /*unv*/ 0);

    BOOST_TEST(m.contain_filtered_label(1) == true);  // has bit 64
    BOOST_TEST(m.contain_filtered_label(0) == false); // has bit 63, not 64
}

// Universal label participates in the query mask via merge. A point carrying
// only the universal label must match even when it lacks every filter label.
BOOST_AUTO_TEST_CASE(bitmask_universal_label_matches)
{
    auto filters = make_bitmask_buf(4, 64, /*point*/ 0, /*labels*/ {7}); // unv label
    {
        simple_bitmask bm1(filters.get_bitmask(1), filters._bitmask_size);
        bm1.set(3); // a real filter label
    }
    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {3};
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, /*unv*/ 7);

    BOOST_TEST(m.contain_filtered_label(0) == true);  // has universal label 7
    BOOST_TEST(m.contain_filtered_label(1) == true);  // has filter label 3
    BOOST_TEST(m.contain_filtered_label(2) == false); // has neither
}

// A bitmask larger than 4 words exercises the AVX2 blocked loop plus the
// scalar tail (size not a multiple of 4). A match whose only shared bit lives
// in the tail words must still be found.
BOOST_AUTO_TEST_CASE(bitmask_large_size_tail_match)
{
    // 320 bits -> size 5: one 4-word AVX2 block + a 1-word scalar tail.
    auto filters = make_bitmask_buf(4, /*total_bits*/ 320, /*point*/ 2, /*labels*/ {300});
    BOOST_TEST(filters._bitmask_size == (std::uint64_t)5);

    std::vector<std::uint64_t> qbuf;
    std::vector<std::uint32_t> filter_labels = {300}; // bit 300 lives in word 4 (tail)
    bitmask_filter_match<std::uint32_t> m(filters, qbuf, filter_labels, /*unv*/ 0);

    BOOST_TEST(m.contain_filtered_label(2) == true);  // shared bit in the tail
    BOOST_TEST(m.contain_filtered_label(0) == false);
}

BOOST_AUTO_TEST_SUITE_END()
