// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include <tsl/robin_map.h>
#include "utils.h"

namespace diskann
{

struct Neighbor
{
    unsigned id;
    float distance;
    bool expanded;

    Neighbor() = default;

    Neighbor(unsigned id, float distance) : id{id}, distance{distance}, expanded(false)
    {
    }

    inline bool operator<(const Neighbor &other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const Neighbor &other) const
    {
        return (id == other.id);
    }
};

// Invariant: after every `insert` and `closest_unexpanded()`, `_cur` points to
//            the first Neighbor which is unexpanded.
class NeighborPriorityQueue
{
  public:
    NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0)
    {
    }

    explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1, Neighbor(std::numeric_limits<uint32_t>::max(), std::numeric_limits<float>::max()))
    {
    }

    void setup(uint32_t capacity) {
        _data.resize(capacity+1,Neighbor(std::numeric_limits<uint32_t>::max(), std::numeric_limits<float>::max()));
        _capacity = capacity;
    }
    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    void insert(const Neighbor &nbr)
    {
        if (_size == _capacity && _data[_size - 1] < nbr)
        {
            return;
        }

        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid])
            {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            }
            else if (_data[mid].id == nbr.id)
            {
                return;
            }
            else
            {
                lo = mid + 1;
                }
        }

        if (lo < _capacity)
        {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
        }
        _data[lo] = {nbr.id, nbr.distance};
        if (_size < _capacity)
        {
            _size++;
        }
        if (lo < _cur)
        {
            _cur = lo;
        }
    }


    // Deletes the item if found.
    void delete_id(const Neighbor &nbr)
    {
        size_t lo = 0, hi = _size;
        size_t loc = std::numeric_limits<uint64_t>::max();
        while ((lo < hi) && loc == std::numeric_limits<uint64_t>::max())
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr.distance < _data[mid].distance)
            {
                hi = mid;
            }
            else if (nbr.distance > _data[mid].distance)
            {
                lo = mid+1;
            }
            else
            {
                uint32_t itr = 0;
                for (;; itr++) {
                    if (mid + itr < hi) {
                    if (_data[mid+itr].id == nbr.id) {
                    loc = mid+itr;
                    break;
                    }
                    }
                    if(mid - itr >= lo) {
                    if (_data[mid-itr].id == nbr.id) {
                    loc = mid-itr;
                    break;
                    }                    
                    }
                }
            }
        }

        if (loc != std::numeric_limits<uint64_t>::max())
        {
            std::memmove(&_data[loc], &_data[loc+1], (_size - loc - 1) * sizeof(Neighbor));
            _size--;
            _cur = 0;
            while (_cur < _size && _data[_cur].expanded) // RK: inefficient!
            {
                _cur++;
            }
        } else {
            std::cout<<"Found a problem! " << lo <<" " << hi <<" " <<nbr.id << "," << nbr.distance << " " <<_size << std::endl;
            uint32_t pos = 0;
            for (auto &x : this->_data) {
                std::cout<<pos<<":" <<x.id<<"," << x.distance <<" " << std::flush;
                pos++;
            }
            std::cout<<std::endl;
        }
    }


    Neighbor closest_unexpanded()
    {
        _data[_cur].expanded = true;
        size_t pre = _cur;
        while (_cur < _size && _data[_cur].expanded)
        {
            _cur++;
        }
        return _data[pre];
    }

    bool has_unexpanded_node() const
    {
        return _cur < _size;
    }

    size_t size() const
    {
        return _size;
    }

    size_t capacity() const
    {
        return _capacity;
    }

    void reserve(size_t capacity)
    {
        if (capacity + 1 > _data.size())
        {
            _data.resize(capacity + 1);
        }
        _capacity = capacity;
    }

    Neighbor &operator[](size_t i)
    {
        return _data[i];
    }

    Neighbor operator[](size_t i) const
    {
        return _data[i];
    }

    void clear()
    {
        _size = 0;
        _cur = 0;
    }

  public:
    size_t _size, _capacity, _cur;
    std::vector<Neighbor> _data;
};


struct bestCandidates {
    NeighborPriorityQueue best_L_nodes;
    
    // For max_K_per_seller == 1 (optimized path): maps seller -> single best Neighbor
    // For max_K_per_seller > 1 (general path): maps seller -> priority queue of neighbors
    tsl::robin_map<uint32_t, Neighbor> seller_to_best;           // Used when _maxLperSeller == 1
    tsl::robin_map<uint32_t, NeighborPriorityQueue> color_to_nodes; // Used when _maxLperSeller > 1
    
    uint32_t _Lsize = 0;
    uint32_t _maxLperSeller = 0;
    std::vector<uint32_t> &_location_to_seller;

    bestCandidates(std::vector<uint32_t> &location_to_seller)
        : _location_to_seller(location_to_seller) {
    }

    bestCandidates(uint32_t Lsize, uint32_t maxLperSeller, std::vector<uint32_t> &location_to_seller)
        : _location_to_seller(location_to_seller) {
        _Lsize = Lsize;
        _maxLperSeller = maxLperSeller;
        best_L_nodes = NeighborPriorityQueue(_Lsize);

        if (_maxLperSeller == 1) {
            seller_to_best.reserve(static_cast<size_t>(_Lsize));
        } else {
            color_to_nodes.reserve(static_cast<size_t>(_Lsize));
        }
    }

    void clear() {
        best_L_nodes.clear();
        seller_to_best.clear();
        color_to_nodes.clear();
    }

    void setup(uint32_t Lsize, uint32_t maxLperSeller) {
        _Lsize = Lsize;
        _maxLperSeller = maxLperSeller;
        best_L_nodes = NeighborPriorityQueue(_Lsize);

        if (_maxLperSeller == 1) {
            seller_to_best.reserve(static_cast<size_t>(_Lsize));
        } else {
            color_to_nodes.reserve(static_cast<size_t>(_Lsize));
        }
    }

    // Optimized insert for max_K_per_seller == 1
    // Key insight: each seller can have at most 1 item in best_L_nodes.
    // We track seller->Neighbor directly, avoiding NeighborPriorityQueue overhead.
    inline void insert_single(uint32_t cur_id, float cur_dist) {
        const uint32_t seller = _location_to_seller[cur_id];
        const uint32_t best_sz = static_cast<uint32_t>(best_L_nodes._size);
        
        auto it = seller_to_best.find(seller);
        
        if (it == seller_to_best.end()) {
            // Seller not seen yet
            if (best_sz < _Lsize) {
                // Global has room - just add
                const Neighbor n(cur_id, cur_dist);
                seller_to_best[seller] = n;
                best_L_nodes.insert(n);
            } else {
                // Global is full - check if we can evict worst
                const Neighbor &worst_global = best_L_nodes._data[_Lsize - 1];
                if (cur_dist < worst_global.distance) {
                    const uint32_t worst_seller = _location_to_seller[worst_global.id];
                    
                    // Remove worst from seller tracking
                    seller_to_best.erase(worst_seller);
                    
                    // Remove from global (it's at the end, so just decrement size)
                    best_L_nodes._size--;
                    
                    // Add new entry
                    const Neighbor n(cur_id, cur_dist);
                    seller_to_best[seller] = n;
                    best_L_nodes.insert(n);
                }
            }
        } else {
            // Seller already has an entry - only replace if better
            Neighbor &existing = it.value();
            if (cur_dist < existing.distance) {
                // Need to remove old entry from global queue
                best_L_nodes.delete_id(existing);
                
                // Update seller's best and insert into global
                const Neighbor n(cur_id, cur_dist);
                existing = n;
                best_L_nodes.insert(n);
            }
        }
    }

    // General insert for max_K_per_seller > 1
    void insert_general(uint32_t cur_id, float cur_dist) {
        const uint32_t seller = _location_to_seller[cur_id];

        // Ensure seller queue exists
        auto it = color_to_nodes.find(seller);
        if (it == color_to_nodes.end()) {
            color_to_nodes[seller] = NeighborPriorityQueue(_maxLperSeller);
            it = color_to_nodes.find(seller);
        }
        NeighborPriorityQueue &cur_list = it.value();

        const uint32_t cur_sz  = static_cast<uint32_t>(cur_list._size);
        const uint32_t best_sz = static_cast<uint32_t>(best_L_nodes._size);

        // Fast path: both have room
        if (cur_sz < _maxLperSeller && best_sz < _Lsize) {
            const Neighbor n(cur_id, cur_dist);
            cur_list.insert(n);
            best_L_nodes.insert(n);
            return;
        }

        // Seller full: only accept if better than seller-worst; evict seller-worst from global.
        if (cur_sz == _maxLperSeller) {
            const Neighbor worst_seller = cur_list._data[_maxLperSeller - 1];
            if (cur_dist < worst_seller.distance) {
                best_L_nodes.delete_id(worst_seller);
                const Neighbor n(cur_id, cur_dist);
                cur_list.insert(n);
                best_L_nodes.insert(n);
            }
            return;
        }

        // Seller has room but global is full.
        if (best_sz == _Lsize) {
            const Neighbor worst_global = best_L_nodes._data[_Lsize - 1];
            if (cur_dist < worst_global.distance) {
                const uint32_t worst_id     = worst_global.id;
                const uint32_t worst_seller = _location_to_seller[worst_id];

                // Delete from the seller's queue
                auto it_w = color_to_nodes.find(worst_seller);
                if (it_w != color_to_nodes.end()) {
                    it_w.value().delete_id(worst_global);
                }

                const Neighbor n(cur_id, cur_dist);
                cur_list.insert(n);
                best_L_nodes.insert(n);
            }
        }
    }

    inline void insert(uint32_t cur_id, float cur_dist) {
        if (_Lsize == 0 || _maxLperSeller == 0) return;
        
        if (_maxLperSeller == 1) {
            insert_single(cur_id, cur_dist);
        } else {
            insert_general(cur_id, cur_dist);
        }
    }
};



} // namespace diskann
