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

typedef int tag_t;
#define NULL_TAG -1

namespace NSG {
  class IndexNSG : public Index {
   public:
    explicit IndexNSG(const size_t dimension, const size_t n, Metric m,
                      Index *initializer, const size_t max_points = 0,
                      const bool enable_tags = false);

    virtual ~IndexNSG();

    virtual void Save(const char *filename) override;
    virtual void Load(const char *filename) override;

    void Init_rnd_nn_graph(size_t num_points, unsigned k,
                           std::vector<size_t> mapping = std::vector<size_t>());

    void BuildRandomHierarchical(
        const float *data, Parameters &parameters,
        const std::vector<tag_t> &tags = std::vector<tag_t>());

    void populate_start_points_bfs(std::vector<unsigned> &start_points);

    std::pair<int, int> BeamSearch(const float *query, const float *x,
                                   const size_t K, const Parameters &parameters,
                                   unsigned *indices, int beam_width,
                                   std::vector<unsigned> &start_points);

    void prefetch_vector(unsigned id);

    /* Methods for inserting and deleting points from the databases*/

    typedef std::vector<SimpleNeighbor> vecNgh;

    int insert_point(const float *point, const Parameters &parameter,
                     std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
                     tsl::robin_set<unsigned> &visited, vecNgh &cut_graph,
                     const tag_t tag = NULL_TAG);

    int enable_delete();
    int disable_delete(const Parameters &parameters,
                       const bool        consolidate = false);

    // Return -1 if tag not found, 0 if OK.
    int delete_point(const tag_t tag);

    /*  Internals of the library */
   protected:
    typedef std::vector<std::vector<unsigned>> CompactGraph;

    CompactGraph final_graph_;

    Index *initializer_;

    void reachable_bfs(const unsigned                         start_node,
                       std::vector<tsl::robin_set<unsigned>> &bfs_order,
                       bool *                                 visited);

    // entry point is centroid based on all-to-centroid distance computation
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

    void InterInsertHierarchy(unsigned n, vecNgh &cut_graph_,
                              const Parameters &parameter);

    void sync_prune(const float *x, unsigned location,
                    std::vector<Neighbor> &pool, const Parameters &parameter,
                    tsl::robin_set<unsigned> &visited, vecNgh &cut_graph);

    void occlude_list(const std::vector<Neighbor> &pool,
                      const unsigned location, const float alpha,
                      const unsigned degree, const unsigned maxc,
                      std::vector<Neighbor> &result);

    void LinkHierarchy(Parameters &parameters);

    // WARNING: Do not call reserve_location() without acquiring change_lock_
    unsigned reserve_location();

    // WARNING: Do not call consolidate_deletes() without acquiring change_lock_
    // Returns number of live points left after consolidation
    size_t consolidate_deletes(const Parameters &parameters);

   private:
    unsigned width;
    unsigned ep_;
    char *   opt_graph_;
    size_t   node_size;
    size_t   data_len;
    size_t   neighbor_len;

    std::vector<std::mutex> locks;  // Per node lock, cardinality=max_points_
    std::mutex change_lock_;        // Allow only 1 thread to insert/delete

    bool can_delete_;
    bool enable_tags_;
    bool consolidated_order_;

    std::unordered_map<tag_t, unsigned> tag_to_point_;
    std::unordered_map<unsigned, tag_t> point_to_tag_;

    tsl::robin_set<unsigned> delete_list_;
    tsl::robin_set<unsigned> empty_slots_;
  };
}

#endif  // EFANNA2E_INDEX_NSG_H
