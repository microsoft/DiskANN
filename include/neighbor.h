// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include "utils.h"

namespace diskann {

  struct Neighbor {
    unsigned id;
    float    distance;
    bool     flag;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f)
        : id{id}, distance{distance}, flag(f) {
    }

    inline bool operator<(const Neighbor &other) const {
      return distance < other.distance;
    }
    inline bool operator==(const Neighbor &other) const {
      return (id == other.id);
    }
  };

  typedef std::lock_guard<std::mutex> LockGuard;
  struct nhood {
    std::mutex            lock;
    std::vector<Neighbor> pool;
    unsigned              M;

    std::vector<unsigned> nn_old;
    std::vector<unsigned> nn_new;
    std::vector<unsigned> rnn_old;
    std::vector<unsigned> rnn_new;

    nhood() {
    }
    nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N) {
      M = s;
      nn_new.resize(s * 2);
      GenRandom(rng, &nn_new[0], (unsigned) nn_new.size(), N);
      nn_new.reserve(s * 2);
      pool.reserve(l);
    }

    nhood(const nhood &other) {
      M = other.M;
      std::copy(other.nn_new.begin(), other.nn_new.end(),
                std::back_inserter(nn_new));
      nn_new.reserve(other.nn_new.capacity());
      pool.reserve(other.pool.capacity());
    }
    void insert(unsigned id, float dist) {
      LockGuard guard(lock);
      if (dist > pool.front().distance)
        return;
      for (unsigned i = 0; i < pool.size(); i++) {
        if (id == pool[i].id)
          return;
      }
      if (pool.size() < pool.capacity()) {
        pool.push_back(Neighbor(id, dist, true));
        std::push_heap(pool.begin(), pool.end());
      } else {
        std::pop_heap(pool.begin(), pool.end());
        pool[pool.size() - 1] = Neighbor(id, dist, true);
        std::push_heap(pool.begin(), pool.end());
      }
    }

    template<typename C>
    void join(C callback) const {
      for (unsigned const i : nn_new) {
        for (unsigned const j : nn_new) {
          if (i < j) {
            callback(i, j);
          }
        }
        for (unsigned j : nn_old) {
          callback(i, j);
        }
      }
    }
  };

  struct SimpleNeighbor {
    unsigned id;
    float    distance;

    SimpleNeighbor() = default;
    SimpleNeighbor(unsigned id, float distance) : id(id), distance(distance) {
    }

    inline bool operator<(const SimpleNeighbor &other) const {
      return distance < other.distance;
    }

    inline bool operator==(const SimpleNeighbor &other) const {
      return id == other.id;
    }
  };
  struct SimpleNeighbors {
    std::vector<SimpleNeighbor> pool;
  };

  static inline unsigned InsertIntoPool(Neighbor *addr, unsigned K,
                                        Neighbor nn) {
    // find the location to insert
    unsigned left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
      memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
      addr[left] = nn;
      return left;
    }
    if (addr[right].distance < nn.distance) {
      addr[K] = nn;
      return K;
    }
    while (right > 1 && left < right - 1) {
      unsigned mid = (left + right) / 2;
      if (addr[mid].distance > nn.distance)
        right = mid;
      else
        left = mid;
    }
    // check equal ID

    while (left > 0) {
      if (addr[left].distance < nn.distance)
        break;
      if (addr[left].id == nn.id)
        return K + 1;
      left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
      return K + 1;
    memmove((char *) &addr[right + 1], &addr[right],
            (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
  }
}  // namespace diskann
