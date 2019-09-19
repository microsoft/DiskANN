#pragma once

#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include "distance.h"
#include "neighbor.h"
#include "parameters.h"
#include "tsl/robin_set.h"
#include "util.h"

namespace NSG {
  template<typename T, typename TagT = unsigned>
  class IndexNSG {
   public:
    IndexNSG(const size_t dimension, const size_t n, Metric m,
             const size_t max_points = 0, const bool enable_tags = false,
             const bool store_data = false);

    ~IndexNSG();

    void save(const char *filename);
    void load(const char *filename, const bool load_tags = false);
    void gen_fake_point(unsigned fake_points, T *data);
    void init_random_graph(size_t num_points, unsigned k,
                           std::vector<size_t> mapping = std::vector<size_t>());
    void update_in_graph();
    void build(const T *data, Parameters &parameters,
               const std::vector<TagT> &tags = std::vector<TagT>());
    typedef std::vector<SimpleNeighbor> vecNgh;

    void populate_start_points_ep(std::vector<unsigned> &start_points);
    void populate_start_points_bfs(std::vector<unsigned> &start_points);

    // Gopal. Added beam_search overload that takes L as parameter, so that we
    // can customize L on a per-query basis without tampering with "Parameters"
    std::pair<int, int> beam_search(const T *query, const T *x, const size_t K,
                                    const unsigned int L, unsigned *indices,
                                    int                          beam_width,
                                    const std::vector<unsigned> &start_points,
                                    unsigned                     fake_points);

    std::pair<int, int> beam_search(const T *query, const T *x, const size_t K,
                                    const Parameters &parameters,
                                    unsigned *indices, int beam_width,
                                    const std::vector<unsigned> start_points,
                                    unsigned                    fake_points);

    std::pair<int, int> beam_search_tags(
        const T *query, const T *x, const size_t K,
        const Parameters &parameters, TagT *tags, int beam_width,
        const std::vector<unsigned> &start_points, unsigned fake_points,
        unsigned *indices_buffer = NULL);

    void prefetch_vector(unsigned id);

    void save_disk_opt_graph(const char *diskopt_path);

/* Methods for inserting and deleting points from the databases*/
#define NULL_TAG (TagT(-1))

    int insert_point(const T *point, const Parameters &parameter,
                     std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
                     tsl::robin_set<unsigned> &visited, vecNgh &cut_graph,
                     const TagT tag = NULL_TAG);

    int enable_delete();
    int disable_delete(const Parameters &parameters,
                       const bool        consolidate = false);

    // Return -1 if tag not found, 0 if OK.
    int delete_point(const TagT tag);

    int eager_delete(const TagT tag, const Parameters &parameters);
    /*  Internals of the library */
    void set_data(const T *data);

    // print in_degree of points
    void report_in_degree(tsl::robin_set<unsigned> delete_list);

   protected:
    typedef std::vector<std::vector<unsigned>> CompactGraph;
    CompactGraph                               _final_graph;
    CompactGraph                               _in_graph;

    void reachable_bfs(const unsigned                         start_node,
                       std::vector<tsl::robin_set<unsigned>> &bfs_order,
                       bool *                                 visited);

    // entry point is centroid based on all-to-centroid distance computation
    unsigned calculate_entry_point();

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

    // flag = 1 for incremental insertions, flag = 0 for graph build
    void inter_insert(unsigned n, vecNgh &cut_graph,
                      const Parameters &parameter, int flag);

    void sync_prune(const T *x, unsigned location, std::vector<Neighbor> &pool,
                    const Parameters &        parameter,
                    tsl::robin_set<unsigned> &visited, vecNgh &cut_graph);

    void occlude_list(const std::vector<Neighbor> &pool,
                      const unsigned location, const float alpha,
                      const unsigned degree, const unsigned maxc,
                      std::vector<Neighbor> &result);

    void link(Parameters &parameters);

    // WARNING: Do not call reserve_location() without acquiring change_lock_
    unsigned reserve_location();

    // WARNING: Do not call consolidate_deletes() without acquiring change_lock_
    // Returns number of live points left after consolidation

    size_t consolidate_deletes(const Parameters &parameters);

    // Computes the in edges of each node, from the out graph
    void compute_in_degree_stats();

   private:
    const size_t            _dim;
    const T *               _data;
    size_t                  _nd;
    size_t                  _max_points;
    bool                    _has_built;
    Distance<T> *           _distance;
    unsigned                _width;
    unsigned                _ep;
    std::vector<std::mutex> _locks;  // Per node lock, cardinality=max_points_
    char *                  _opt_graph;
    size_t                  _node_size;
    size_t                  _data_len;
    size_t                  _neighbor_len;

    bool _can_delete;
    bool _enable_tags;
    bool _consolidated_order;
    bool _store_data;
    bool _eager_done;  // if eager_done = 1, lazy deletes are not allowed

    std::unordered_map<TagT, unsigned> _tag_to_location;
    std::unordered_map<unsigned, TagT> _location_to_tag;

    tsl::robin_set<unsigned> _delete_set;
    tsl::robin_set<unsigned> _empty_slots;

    std::mutex _change_lock;  // Allow only 1 thread to insert/delete
  };
}
