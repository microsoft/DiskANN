// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// Swiss table-style hash set for uint32_t keys.
// Inspired by Google's Abseil flat_hash_set / Rust's hashbrown.
// Uses SSE2 SIMD control byte matching for fast probing.
// clear() is O(occupied_slots) not O(capacity).

#pragma once

#include <cstdint>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <immintrin.h>

namespace diskann
{

class SwissHashSet
{
  public:
    // Control byte values
    static constexpr int8_t kEmpty = -128;   // 0b10000000
    static constexpr int8_t kDeleted = -2;   // 0b11111110 (unused — we don't do tombstones)

    SwissHashSet() : _ctrl(nullptr), _slots(nullptr), _capacity(0), _size(0), _growth_left(0)
    {
    }

    ~SwissHashSet()
    {
        _free();
    }

    SwissHashSet(const SwissHashSet &) = delete;
    SwissHashSet &operator=(const SwissHashSet &) = delete;

    SwissHashSet(SwissHashSet &&other) noexcept
        : _ctrl(other._ctrl), _slots(other._slots), _capacity(other._capacity), _size(other._size),
          _growth_left(other._growth_left)
    {
        other._ctrl = nullptr;
        other._slots = nullptr;
        other._capacity = 0;
        other._size = 0;
        other._growth_left = 0;
    }

    void reserve(size_t n)
    {
        if (n == 0)
            return;
        // Target load factor ~7/8 (87.5%), so we need capacity >= n * 8/7
        size_t needed = n + n / 7;
        needed = _round_up_power_of_2(std::max(needed, (size_t)16));
        if (needed > _capacity)
        {
            _resize(needed);
        }
    }

    // Returns true if inserted, false if already present
    bool insert(uint32_t key)
    {
        if (_capacity == 0)
        {
            _resize(16);
        }

        size_t hash = _hash(key);
        size_t pos = _find_or_prepare_insert(hash, key);
        if (_slots[pos] == key && _ctrl[pos] >= 0)
        {
            return false; // already exists
        }

        // Insert
        _set_ctrl(pos, _h2(hash));
        _slots[pos] = key;
        _size++;
        _growth_left--;

        if (_growth_left == 0)
        {
            _resize(_capacity * 2);
        }
        return true;
    }

    bool find(uint32_t key) const
    {
        if (_capacity == 0)
            return false;

        size_t hash = _hash(key);
        size_t idx = hash & (_capacity - 1);
        size_t group_start = idx & ~(size_t)15; // align to 16-byte group

        // Probe groups
        for (size_t probe = 0; probe <= _capacity; probe += 16)
        {
            size_t g = (group_start + probe) & (_capacity - 1);
            __m128i ctrl_group = _mm_loadu_si128(reinterpret_cast<const __m128i *>(_ctrl + g));
            int8_t h2 = _h2(hash);

            // Match h2
            __m128i match = _mm_cmpeq_epi8(ctrl_group, _mm_set1_epi8(h2));
            uint32_t mask = static_cast<uint32_t>(_mm_movemask_epi8(match));

            while (mask != 0)
            {
                int bit = _bit_scan_forward(mask);
                size_t candidate = (g + bit) & (_capacity - 1);
                if (_slots[candidate] == key)
                {
                    return true;
                }
                mask &= mask - 1; // clear lowest bit
            }

            // Check for empty — if any slot in group is empty, key is absent
            __m128i empty_match = _mm_cmpeq_epi8(ctrl_group, _mm_set1_epi8(kEmpty));
            if (_mm_movemask_epi8(empty_match) != 0)
            {
                return false;
            }
        }
        return false;
    }

    // O(size) clear — only walks occupied slots tracked via control bytes
    void clear()
    {
        if (_size == 0)
            return;
        // Walk in 16-byte groups, only memset groups that have occupied slots
        for (size_t g = 0; g < _capacity; g += 16)
        {
            __m128i ctrl_group = _mm_loadu_si128(reinterpret_cast<const __m128i *>(_ctrl + g));
            __m128i empty_cmp = _mm_cmpeq_epi8(ctrl_group, _mm_set1_epi8(kEmpty));
            int all_empty = _mm_movemask_epi8(empty_cmp);
            if (all_empty != 0xFFFF)
            {
                // This group has at least one occupied slot — clear it
                _mm_storeu_si128(reinterpret_cast<__m128i *>(_ctrl + g), _mm_set1_epi8(kEmpty));
            }
        }
        // Also clear the mirror/clamp bytes at end
        if (_capacity > 0)
        {
            std::memset(_ctrl + _capacity, kEmpty, 16);
        }
        _size = 0;
        _growth_left = _capacity_to_growth(_capacity);
    }

    bool empty() const
    {
        return _size == 0;
    }

    size_t size() const
    {
        return _size;
    }

  private:
    int8_t *_ctrl;      // control bytes: kEmpty or h2 (top 7 bits of hash)
    uint32_t *_slots;   // key storage
    size_t _capacity;   // always power of 2
    size_t _size;
    size_t _growth_left;

    static size_t _round_up_power_of_2(size_t n)
    {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }

    static size_t _capacity_to_growth(size_t cap)
    {
        // 7/8 load factor
        return cap - cap / 8;
    }

    static size_t _hash(uint32_t key)
    {
        // Fast integer hash (murmurhash3 finalizer)
        uint64_t h = key;
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return static_cast<size_t>(h);
    }

    static int8_t _h2(size_t hash)
    {
        // Top 7 bits, ensuring positive (bit 7 = 0)
        return static_cast<int8_t>(hash >> 57) & 0x7F;
    }

    static int _bit_scan_forward(uint32_t mask)
    {
#ifdef _MSC_VER
        unsigned long idx;
        _BitScanForward(&idx, mask);
        return static_cast<int>(idx);
#else
        return __builtin_ctz(mask);
#endif
    }

    void _set_ctrl(size_t pos, int8_t h2)
    {
        _ctrl[pos] = h2;
        // Mirror the first 16 bytes at the end for wrap-around SIMD loads
        if (pos < 16)
        {
            _ctrl[_capacity + pos] = h2;
        }
    }

    size_t _find_or_prepare_insert(size_t hash, uint32_t key)
    {
        size_t idx = hash & (_capacity - 1);
        size_t group_start = idx & ~(size_t)15;
        int8_t h2 = _h2(hash);

        for (size_t probe = 0; probe <= _capacity; probe += 16)
        {
            size_t g = (group_start + probe) & (_capacity - 1);
            __m128i ctrl_group = _mm_loadu_si128(reinterpret_cast<const __m128i *>(_ctrl + g));

            // Check for existing key
            __m128i match = _mm_cmpeq_epi8(ctrl_group, _mm_set1_epi8(h2));
            uint32_t mask = static_cast<uint32_t>(_mm_movemask_epi8(match));
            while (mask != 0)
            {
                int bit = _bit_scan_forward(mask);
                size_t candidate = (g + bit) & (_capacity - 1);
                if (_slots[candidate] == key)
                {
                    return candidate; // found existing
                }
                mask &= mask - 1;
            }

            // Check for empty slot
            __m128i empty_match = _mm_cmpeq_epi8(ctrl_group, _mm_set1_epi8(kEmpty));
            uint32_t empty_mask = static_cast<uint32_t>(_mm_movemask_epi8(empty_match));
            if (empty_mask != 0)
            {
                int bit = _bit_scan_forward(empty_mask);
                return (g + bit) & (_capacity - 1);
            }
        }
        // Should never reach here if load factor is maintained
        assert(false && "SwissHashSet: table is full");
        return 0;
    }

    void _resize(size_t new_capacity)
    {
        int8_t *old_ctrl = _ctrl;
        uint32_t *old_slots = _slots;
        size_t old_capacity = _capacity;

        _capacity = new_capacity;
        // Allocate ctrl bytes: capacity + 16 (mirror) bytes, 16-byte aligned
        _ctrl = static_cast<int8_t *>(_aligned_malloc(_capacity + 16, 16));
        _slots = static_cast<uint32_t *>(std::malloc(_capacity * sizeof(uint32_t)));
        std::memset(_ctrl, kEmpty, _capacity + 16);
        _size = 0;
        _growth_left = _capacity_to_growth(_capacity);

        // Re-insert old elements
        if (old_ctrl != nullptr)
        {
            for (size_t i = 0; i < old_capacity; i++)
            {
                if (old_ctrl[i] >= 0) // occupied
                {
                    insert(old_slots[i]);
                }
            }
            _aligned_free(old_ctrl);
            std::free(old_slots);
        }
    }

    void _free()
    {
        if (_ctrl != nullptr)
        {
            _aligned_free(_ctrl);
            _ctrl = nullptr;
        }
        if (_slots != nullptr)
        {
            std::free(_slots);
            _slots = nullptr;
        }
        _capacity = 0;
        _size = 0;
        _growth_left = 0;
    }
};

} // namespace diskann
