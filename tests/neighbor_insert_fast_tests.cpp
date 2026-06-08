// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>

#include <vector>

#include "neighbor.h"

using diskann::Neighbor;
using diskann::NeighborVector;

namespace
{
// Drives insert_fast and mirrors what NeighborPriorityQueue does with the
// returned index, so tests can track `size` the same way production code does.
struct Fixture
{
    NeighborVector vec;
    size_t size = 0;
    size_t capacity;

    explicit Fixture(size_t cap) : vec(cap), capacity(cap)
    {
    }

    // Returns the raw index from insert_fast.
    size_t insert(unsigned id, float dist)
    {
        size_t lo = vec.insert_fast(Neighbor(id, dist), size, capacity);
        if (lo > capacity)
            return lo; // rejected, size unchanged
        if (size < capacity)
            size++;
        return lo;
    }

    bool rejected(size_t lo) const
    {
        return lo > capacity;
    }

    // Returns the ordered (dist, id) content of the live portion.
    std::vector<std::pair<float, unsigned>> contents()
    {
        std::vector<std::pair<float, unsigned>> out;
        for (size_t i = 0; i < size; i++)
            out.emplace_back(vec[i].distance, vec[i].id);
        return out;
    }

    // Asserts _dist_arr and _data agree and are sorted ascending by (dist, id).
    void check_invariants()
    {
        float* d = vec.distances();
        for (size_t i = 0; i < size; i++)
        {
            BOOST_TEST(d[i] == vec[i].distance);
        }
        for (size_t i = 1; i < size; i++)
        {
            bool ordered = d[i - 1] < d[i] ||
                           (d[i - 1] == d[i] && vec[i - 1].id <= vec[i].id);
            BOOST_TEST(ordered);
        }
    }
};
} // namespace

BOOST_AUTO_TEST_SUITE(NeighborInsertFast_tests)

// ---- Basic insertion into an empty / non-full queue ----

BOOST_AUTO_TEST_CASE(insert_into_empty_returns_zero)
{
    Fixture f(4);
    size_t lo = f.insert(10, 1.0f);
    BOOST_TEST(lo == (size_t)0);
    BOOST_TEST(f.size == (size_t)1);
    BOOST_TEST(f.vec[0].id == (unsigned)10);
    BOOST_TEST(f.vec[0].distance == 1.0f);
    f.check_invariants();
}

BOOST_AUTO_TEST_CASE(insert_ascending_keeps_order)
{
    Fixture f(4);
    f.insert(1, 1.0f);
    f.insert(2, 2.0f);
    f.insert(3, 3.0f);
    BOOST_TEST(f.size == (size_t)3);
    auto c = f.contents();
    BOOST_TEST(c[0].second == (unsigned)1);
    BOOST_TEST(c[1].second == (unsigned)2);
    BOOST_TEST(c[2].second == (unsigned)3);
    f.check_invariants();
}

BOOST_AUTO_TEST_CASE(insert_descending_reorders)
{
    Fixture f(4);
    f.insert(1, 3.0f);
    f.insert(2, 2.0f);
    f.insert(3, 1.0f);
    auto c = f.contents();
    BOOST_TEST(c[0].second == (unsigned)3); // dist 1.0
    BOOST_TEST(c[1].second == (unsigned)2); // dist 2.0
    BOOST_TEST(c[2].second == (unsigned)1); // dist 3.0
    f.check_invariants();
}

// Insert into the middle of a non-full queue exercises move_count > 0
// with size < capacity (move_count = size - lo).
BOOST_AUTO_TEST_CASE(insert_middle_of_non_full_shifts_right)
{
    Fixture f(8);
    f.insert(1, 1.0f);
    f.insert(2, 3.0f);
    f.insert(3, 5.0f);
    size_t lo = f.insert(4, 2.0f); // goes between 1.0 and 3.0
    BOOST_TEST(lo == (size_t)1);
    auto c = f.contents();
    BOOST_TEST(c[0].second == (unsigned)1);
    BOOST_TEST(c[1].second == (unsigned)4);
    BOOST_TEST(c[2].second == (unsigned)2);
    BOOST_TEST(c[3].second == (unsigned)3);
    BOOST_TEST(f.size == (size_t)4);
    f.check_invariants();
}

// Insert at the tail of a non-full queue: lo == size, move_count == 0.
BOOST_AUTO_TEST_CASE(insert_at_tail_non_full_move_count_zero)
{
    Fixture f(8);
    f.insert(1, 1.0f);
    f.insert(2, 2.0f);
    size_t lo = f.insert(3, 3.0f);
    BOOST_TEST(lo == (size_t)2);
    BOOST_TEST(f.size == (size_t)3);
    f.check_invariants();
}

// ---- Tie-breaking by id within a non-full queue ----

BOOST_AUTO_TEST_CASE(equal_distance_sorts_by_id_ascending)
{
    Fixture f(8);
    f.insert(20, 1.0f);
    f.insert(10, 1.0f); // same dist, smaller id -> before 20
    auto c = f.contents();
    BOOST_TEST(c[0].second == (unsigned)10);
    BOOST_TEST(c[1].second == (unsigned)20);
    f.check_invariants();
}

BOOST_AUTO_TEST_CASE(equal_distance_larger_id_goes_after)
{
    Fixture f(8);
    f.insert(10, 1.0f);
    size_t lo = f.insert(20, 1.0f); // same dist, larger id -> after
    BOOST_TEST(lo == (size_t)1);
    auto c = f.contents();
    BOOST_TEST(c[0].second == (unsigned)10);
    BOOST_TEST(c[1].second == (unsigned)20);
    f.check_invariants();
}

// ---- Full-queue rejection branches ----

// Branch: size >= capacity && dist > _dist_arr[size-1] -> reject (capacity+1).
BOOST_AUTO_TEST_CASE(full_queue_worse_distance_rejected)
{
    Fixture f(3);
    f.insert(1, 1.0f);
    f.insert(2, 2.0f);
    f.insert(3, 3.0f);
    BOOST_TEST(f.size == (size_t)3);
    size_t lo = f.insert(4, 4.0f); // worse than tail 3.0
    BOOST_TEST(f.rejected(lo));
    BOOST_TEST(lo == f.capacity + 1);
    BOOST_TEST(f.size == (size_t)3); // unchanged
    auto c = f.contents();
    BOOST_TEST(c[2].second == (unsigned)3); // tail untouched
    f.check_invariants();
}

// Branch: full, dist == tail dist, nbr.id >= tail id -> reject.
BOOST_AUTO_TEST_CASE(full_queue_tie_distance_worse_id_rejected)
{
    Fixture f(3);
    f.insert(1, 1.0f);
    f.insert(2, 2.0f);
    f.insert(10, 3.0f);
    size_t lo = f.insert(20, 3.0f); // same dist as tail, larger id -> reject
    BOOST_TEST(f.rejected(lo));
    BOOST_TEST(f.size == (size_t)3);
    f.check_invariants();
}

// Branch: full, dist == tail dist, nbr.id == tail id -> reject (<= path).
BOOST_AUTO_TEST_CASE(full_queue_tie_distance_equal_id_rejected)
{
    Fixture f(3);
    f.insert(1, 1.0f);
    f.insert(2, 2.0f);
    f.insert(10, 3.0f);
    size_t lo = f.insert(10, 3.0f); // identical to tail -> reject
    BOOST_TEST(f.rejected(lo));
    BOOST_TEST(f.size == (size_t)3);
    f.check_invariants();
}

// Branch: full, dist == tail dist, nbr.id < tail id -> NOT rejected, evicts tail.
BOOST_AUTO_TEST_CASE(full_queue_tie_distance_better_id_evicts_tail)
{
    Fixture f(3);
    f.insert(1, 1.0f);
    f.insert(2, 2.0f);
    f.insert(20, 3.0f);
    size_t lo = f.insert(10, 3.0f); // same dist, smaller id -> replaces 20
    BOOST_TEST(!f.rejected(lo));
    BOOST_TEST(lo == (size_t)2);
    BOOST_TEST(f.size == (size_t)3);
    auto c = f.contents();
    BOOST_TEST(c[2].second == (unsigned)10); // 20 evicted
    f.check_invariants();
}

// ---- Full-queue insertion that evicts the last element ----

// Insert a better distance into a full queue: evicts tail, shifts, size stays.
BOOST_AUTO_TEST_CASE(full_queue_better_distance_evicts_tail)
{
    Fixture f(3);
    f.insert(1, 2.0f);
    f.insert(2, 4.0f);
    f.insert(3, 6.0f);
    size_t lo = f.insert(4, 1.0f); // best -> front
    BOOST_TEST(!f.rejected(lo));
    BOOST_TEST(lo == (size_t)0);
    BOOST_TEST(f.size == (size_t)3);
    auto c = f.contents();
    BOOST_TEST(c[0].second == (unsigned)4);
    BOOST_TEST(c[1].second == (unsigned)1);
    BOOST_TEST(c[2].second == (unsigned)2); // 6.0 (id 3) evicted
    f.check_invariants();
}

// Full-queue insertion in the middle: move_count uses capacity-1 branch.
BOOST_AUTO_TEST_CASE(full_queue_middle_insertion_uses_capacity_minus_one)
{
    Fixture f(4);
    f.insert(1, 1.0f);
    f.insert(2, 3.0f);
    f.insert(3, 5.0f);
    f.insert(4, 7.0f);
    size_t lo = f.insert(5, 4.0f); // between 3.0 and 5.0 -> index 2
    BOOST_TEST(lo == (size_t)2);
    BOOST_TEST(f.size == (size_t)4);
    auto c = f.contents();
    BOOST_TEST(c[0].second == (unsigned)1); // 1.0
    BOOST_TEST(c[1].second == (unsigned)2); // 3.0
    BOOST_TEST(c[2].second == (unsigned)5); // 4.0 inserted
    BOOST_TEST(c[3].second == (unsigned)3); // 5.0; 7.0 (id 4) evicted
    f.check_invariants();
}

// ---- Capacity == 1 corner cases ----

BOOST_AUTO_TEST_CASE(capacity_one_first_insert)
{
    Fixture f(1);
    size_t lo = f.insert(7, 5.0f);
    BOOST_TEST(lo == (size_t)0);
    BOOST_TEST(f.size == (size_t)1);
    f.check_invariants();
}

BOOST_AUTO_TEST_CASE(capacity_one_worse_rejected)
{
    Fixture f(1);
    f.insert(7, 5.0f);
    size_t lo = f.insert(8, 6.0f);
    BOOST_TEST(f.rejected(lo));
    BOOST_TEST(f.size == (size_t)1);
    BOOST_TEST(f.vec[0].id == (unsigned)7);
    f.check_invariants();
}

BOOST_AUTO_TEST_CASE(capacity_one_better_replaces)
{
    Fixture f(1);
    f.insert(7, 5.0f);
    size_t lo = f.insert(8, 4.0f); // better -> replaces at index 0
    BOOST_TEST(!f.rejected(lo));
    BOOST_TEST(lo == (size_t)0);
    BOOST_TEST(f.size == (size_t)1);
    BOOST_TEST(f.vec[0].id == (unsigned)8);
    f.check_invariants();
}

BOOST_AUTO_TEST_CASE(capacity_one_tie_better_id_replaces)
{
    Fixture f(1);
    f.insert(7, 5.0f);
    size_t lo = f.insert(3, 5.0f); // same dist, smaller id -> replaces
    BOOST_TEST(!f.rejected(lo));
    BOOST_TEST(f.vec[0].id == (unsigned)3);
    f.check_invariants();
}

// ---- Larger queue to exercise the SIMD / binary-search lower_bound path ----

BOOST_AUTO_TEST_CASE(large_queue_lower_bound_paths)
{
    // Capacity > 16 so AVX-512/AVX2 main loops and scalar tails are all hit.
    const size_t cap = 40;
    Fixture f(cap);
    // Fill with even distances 0,2,4,... so odd distances land between them.
    for (unsigned i = 0; i < (unsigned)cap; i++)
        f.insert(i, (float)(i * 2));
    BOOST_TEST(f.size == cap);

    // Insert into a middle slot (better than current tail so not rejected).
    size_t lo = f.insert(1000, 15.0f); // between 14 (idx7) and 16 (idx8)
    BOOST_TEST(lo == (size_t)8);
    BOOST_TEST(f.vec[8].id == (unsigned)1000);
    BOOST_TEST(f.size == cap); // full: tail evicted, size stays
    f.check_invariants();

    // Insert at the very front.
    size_t lo2 = f.insert(2000, -1.0f);
    BOOST_TEST(lo2 == (size_t)0);
    BOOST_TEST(f.vec[0].id == (unsigned)2000);
    f.check_invariants();
}

// Distances straddling exact SIMD lane boundaries (8, 16) to catch tail bugs.
BOOST_AUTO_TEST_CASE(insertion_at_simd_lane_boundaries)
{
    const size_t cap = 32;
    Fixture f(cap);
    for (unsigned i = 0; i < (unsigned)cap; i++)
        f.insert(i, (float)(i * 10));

    // Target exactly matching an existing distance, smaller id -> tie path.
    size_t lo = f.insert(5, 80.0f); // dist 80 == index 8's dist, id 5 < 8
    BOOST_TEST(lo == (size_t)8);
    BOOST_TEST(f.vec[8].id == (unsigned)5);
    BOOST_TEST(f.vec[9].id == (unsigned)8); // shifted right
    f.check_invariants();
}

BOOST_AUTO_TEST_SUITE_END()
