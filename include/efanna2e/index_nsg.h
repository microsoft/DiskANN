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
                      Index *initializer);

    virtual ~IndexNSG();

    virtual void Save(const char *filename) override;
    virtual void Load(const char *filename) override;

    void LoadSmallIndex(const char *filename, std::vector<unsigned> &picked);
    void SaveSmallIndex(const char *filename, std::vector<unsigned> &picked);

    void Load_nn_graph(const char *filename);

    virtual void Build(size_t n, const float *data,
                       const Parameters &parameters) override;

    void BuildFromSmall(size_t n, const float *data,
                        const Parameters &parameters, IndexNSG &small_index,
                        const std::vector<unsigned> &picked_pts);

    void BuildFromER(size_t n, size_t nr, const float *data,
                     const Parameters &parameters);

    virtual std::pair<int, int> Search(const float *query, const float *x,
                                       const size_t      K,
                                       const Parameters &parameters,
                                       unsigned *        indices) override;

    void populate_start_points_bfs(std::vector<unsigned> &start_points);

    std::pair<int, int> BeamSearch(const float *query, const float *x,
                                   const size_t K, const Parameters &parameters,
                                   unsigned *indices, int beam_width,
                                   const std::vector<unsigned> &start_points);

    unsigned long long SearchWithOptGraph(const float *query, size_t K,
                                          const Parameters &parameters,
                                          unsigned *        indices);
    void OptimizeGraph(float *data);

    unsigned get_start_node() const;
    void set_start_node(const unsigned s);

    void init_graph_outside(const float *data);

   protected:
    typedef std::vector<std::vector<unsigned>> CompactGraph;
    typedef std::vector<SimpleNeighbors>       LockGraph;
    typedef std::vector<nhood>                 KNNGraph;

    CompactGraph final_graph_;

    Index *initializer_;

    // version supplied by authors
    void init_graph(const Parameters &parameters);
    // brute-force centroid + all-to-centroid distance computation
    void init_graph_bf(const Parameters &parameters);

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
    void get_neighbors(const float *query, const Parameters &parameter,
                       std::vector<Neighbor> &      retset,
                       std::vector<Neighbor> &      fullset,
                       tsl::robin_set<unsigned> &   visited,
                       const std::vector<unsigned> &start_points);

    // void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph&
    // cut_graph_);

    typedef std::vector<SimpleNeighbor> vecNgh;
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks,
                     const Parameters &parameter, vecNgh *cut_graph_);
    void sync_prune(unsigned q, std::vector<Neighbor> &pool,
                    const Parameters &        parameter,
                    tsl::robin_set<unsigned> &visited, vecNgh *cut_graph_);
    void Link(const Parameters &parameters, vecNgh *cut_graph_);

    void LinkFromSmall(const Parameters &parameters, vecNgh *cut_graph_,
                       IndexNSG &                   small_index,
                       const std::vector<unsigned> &picked_pts);

    void tree_grow(const Parameters &parameter);
    void DFS(tsl::robin_set<unsigned> &visited, unsigned root, unsigned &cnt);
    void findroot(tsl::robin_set<unsigned> &visited, unsigned &root,
                  const Parameters &parameter);
    void reachable_bfs(const unsigned                         start_node,
                       std::vector<tsl::robin_set<unsigned>> &bfs_order,
                       bool *                                 visited);

   private:
    unsigned                width;
    unsigned                ep_;
    std::vector<std::mutex> locks;
    char *                  opt_graph_;
    size_t                  node_size;
    size_t                  data_len;
    size_t                  neighbor_len;
    KNNGraph                nnd_graph;
  };
}

#endif  // EFANNA2E_INDEX_NSG_H
