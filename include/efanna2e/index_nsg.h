#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include "index.h"
#include "neighbor.h"
#include "parameters.h"
#include "tsl/robin_set.h"
#include "util.h"

namespace NSG {
  template<typename T>
  class IndexNSG : public Index<T> {
   public:
    explicit IndexNSG(const size_t dimension, const size_t n, Metric m,
                      Index<T> *initializer, const size_t max_points = 0);

    ~IndexNSG();

    void save(const char *filename) override;
    void load(const char *filename) override;

    void init_random_graph(size_t num_points, unsigned k,
                           std::vector<size_t> mapping = std::vector<size_t>());

    void build(const T *data, Parameters &parameters);

    typedef std::vector<SimpleNeighbor> vecNgh;

    int insert_point(const T *point, const Parameters &parameter,
                     std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
                     tsl::robin_set<unsigned> &visited, vecNgh &cut_graph);

    void populate_start_points_bfs(std::vector<unsigned> &start_points);

    std::pair<int, int> beam_search(const T *query, const T *x, const size_t K,
                                    const Parameters &parameters,
                                    unsigned *indices, int beam_width,
                                    std::vector<unsigned> &start_points);

    // Gopal. Added BeamSearch overload that takes L as parameter, so that we
    // can customize L on a per-query basis without tampering with "Parameters"
    std::pair<int, int> beam_search(const T *query, const T *x, const size_t K,
                                    const unsigned int L, unsigned *indices,
                                    int                    beam_width,
                                    std::vector<unsigned> &start_points);

    void save_disk_opt_graph(const char *diskopt_path);

   protected:
    typedef std::vector<std::vector<unsigned>> CompactGraph;

    void reachable_bfs(const unsigned                         start_node,
                       std::vector<tsl::robin_set<unsigned>> &bfs_order,
                       bool *                                 visited);

    CompactGraph _final_graph;

    bool *_is_inner;

    Index<T> *_initializer;

    // version supplied by authors
    // brute-force centroid + all-to-centroid distance computation
    unsigned get_entry_point();

    void iterate_to_fixed_point(const T *query, const Parameters &parameter,
                                std::vector<unsigned> &   init_ids,
                                std::vector<Neighbor> &   retset,
                                std::vector<Neighbor> &   fullset,
                                tsl::robin_set<unsigned> &visited);
    // void compute_distances_batch(const unsigned *ids, float *dists,
    //                             const unsigned n_pts);
    void get_neighbors(const T *query, const Parameters &parameter,
                       std::vector<Neighbor> &retset,
                       std::vector<Neighbor> &fullset);
    void get_neighbors(const T *query, const Parameters &parameter,
                       std::vector<Neighbor> &   retset,
                       std::vector<Neighbor> &   fullset,
                       tsl::robin_set<unsigned> &visited);

    void inter_insert(unsigned n, vecNgh &cut_graph,
                      const Parameters &parameter);

    void sync_prune(unsigned q, std::vector<Neighbor> &pool,
                    const Parameters &        parameter,
                    tsl::robin_set<unsigned> &visited, vecNgh &cut_graph);

    void link(Parameters &parameters);

   private:
    unsigned                _width;
    unsigned                _ep;
    std::vector<std::mutex> _locks;
    char *                  _opt_graph;
    size_t                  _node_size;
    size_t                  _data_len;
    size_t                  _neighbor_len;

    using Index<T>::_dim;
    using Index<T>::_data;
    using Index<T>::_nd;
    using Index<T>::_has_built;
    using Index<T>::_distance;
    using Index<T>::_max_points;
    // KNNGraph                nnd_graph;

    std::mutex _incr_insert_lock;  // Allow only one thread to insert.
  };
}

bool BuildIndex(const char *dataFilePath, const char *indexFilePath,
                const char *indexBuildParameters);
#endif  // EFANNA2E_INDEX_NSG_H
