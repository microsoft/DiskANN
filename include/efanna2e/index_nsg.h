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
  class IndexNSG : public Index {
   public:
    explicit IndexNSG(const size_t dimension, const size_t n, Metric m,
                      Index *initializer, const size_t max_points = 0);

    virtual ~IndexNSG();

    virtual void Save(const char *filename) override;
    virtual void Load(const char *filename) override;

    void Init_rnd_nn_graph(size_t num_points, unsigned k,
                           std::vector<size_t> mapping = std::vector<size_t>());

    void BuildRandomHierarchical(size_t n, const float *data,
                                 Parameters &parameters);
    void populate_start_points_bfs(std::vector<unsigned> &start_points);

    std::pair<int, int> BeamSearch(const float *query, const float *x,
                                   const size_t K, const Parameters &parameters,
                                   unsigned *indices, int beam_width,
                                   std::vector<unsigned> &start_points);

    void prefetch_vector(unsigned id);

   protected:
    typedef std::vector<std::vector<unsigned>> CompactGraph;

    void reachable_bfs(const unsigned                         start_node,
                       std::vector<tsl::robin_set<unsigned>> &bfs_order,
                       bool *                                 visited);

    CompactGraph final_graph_;

    bool *is_inner;

    Index *initializer_;

    // version supplied by authors
    // brute-force centroid + all-to-centroid distance computation
    unsigned get_entry_point();

    void iterate_to_fixed_point(const float *query, const Parameters &parameter,
                                std::vector<unsigned> &   init_ids,
                                std::vector<Neighbor> &   retset,
                                std::vector<Neighbor> &   fullset,
                                tsl::robin_set<unsigned> &visited);
    void get_neighbors(const float *query, const Parameters &parameter,
                       std::vector<Neighbor> &retset,
                       std::vector<Neighbor> &fullset);
    void get_neighbors(const float *query, const Parameters &parameter,
                       std::vector<Neighbor> &   retset,
                       std::vector<Neighbor> &   fullset,
                       tsl::robin_set<unsigned> &visited);

    typedef std::vector<SimpleNeighbor> vecNgh;
    void InterInsertHierarchy(unsigned n, std::vector<std::mutex> &locks,
                              vecNgh &cut_graph_, const Parameters &parameter);

    void sync_prune(unsigned q, std::vector<Neighbor> &pool,
                    const Parameters &        parameter,
                    tsl::robin_set<unsigned> &visited, vecNgh &cut_graph_);

    void LinkHierarchy(Parameters &parameters);

    int insert_point(const float *point, const Parameters &parameter);

   private:
    unsigned                width;
    unsigned                ep_;
    std::vector<std::mutex> locks;
    char *                  opt_graph_;
    size_t                  node_size;
    size_t                  data_len;
    size_t                  neighbor_len;
    // KNNGraph                nnd_graph;

    std::mutex incr_insert_lock;  // Allow only one thread to insert.
  };
}

#endif  // EFANNA2E_INDEX_NSG_H
