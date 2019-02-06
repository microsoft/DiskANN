//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_GRAPH_H
#define EFANNA2E_GRAPH_H

#include <cstddef>
#include <mutex>
#include <vector>
#include "efanna2e/util.h"

namespace efanna2e {

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
      return (id == other.id) && (distance < other.distance) &&
             (flag == other.flag);
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
  };
  struct SimpleNeighbors {
    std::vector<SimpleNeighbor> pool;
  };

  struct SimpleNhood {
    unsigned  nnbrs = 0;
    unsigned  dim = 0;
    unsigned *nbrs = nullptr;
    int8_t *  int8_coords = nullptr;
    void *    buf = nullptr;  // underlying buffer for nbrs, nnbrs int8_t coords
    float *   aligned_fp32_coords = nullptr;  // alloc'ed only when read into,
    // dimension aligned to multiple of 8
    SimpleNhood *redir = nullptr;  // redirection to existing SimpleNhood

    SimpleNhood() {
      assert(buf == nullptr);
      assert(aligned_fp32_coords == nullptr);
      assert(int8_coords == nullptr);
    }

    SimpleNhood(unsigned buf_size, unsigned dim) : dim(dim) {
      assert(buf == nullptr);
      assert(aligned_fp32_coords == nullptr);
      assert(int8_coords == nullptr);
      alloc_aligned(&buf, buf_size, 512);
    }

    void init(unsigned buf_size, unsigned dim) {
      assert(buf == nullptr);
      assert(aligned_fp32_coords == nullptr);
      assert(int8_coords == nullptr);
      this->dim = dim;
      alloc_aligned(&buf, buf_size, 512);
    }

    ~SimpleNhood() {
      // clear up any unused mem
      if (buf != nullptr) {
        free(buf);
      }

      if (aligned_fp32_coords != nullptr) {
        free(aligned_fp32_coords);
      }
    }

    void construct(float scale_factor = 127.0f) {
      assert(buf != nullptr);
      assert(dim > 0);

      // construct basic node info from buf
      nnbrs = *(unsigned *) buf;
      assert(nnbrs > 0);
      nbrs = (unsigned *) buf + 1;
      int8_coords = (int8_t *) buf + (nnbrs + 1) * sizeof(unsigned);

      unsigned aligned_dim = ROUND_UP(dim, 8);
      alloc_aligned((void **) &aligned_fp32_coords,
                    aligned_dim * nnbrs * sizeof(float), 32);
      memset(aligned_fp32_coords, 0, aligned_dim * nnbrs * sizeof(float));
      for (unsigned i = 0; i < nnbrs; i++) {
        for (unsigned j = 0; j < dim; j++) {
          aligned_fp32_coords[i * aligned_dim + j] =
              ((float) int8_coords[i * dim + j]) / scale_factor;
        }
      }
    }
  };

  static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
    // find the location to insert
    int left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
      memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
      addr[left] = nn;
      return left;
    }
    if (addr[right].distance < nn.distance) {
      addr[K] = nn;
      return K;
    }
    while (left < right - 1) {
      int mid = (left + right) / 2;
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
}

#endif  // EFANNA2E_GRAPH_H
