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
  template<typename T, typename TagT = int>
  class IndexNSG : public Index<T> {
   public:
    explicit IndexNSG(const size_t dimension, const size_t n, Metric m,
                      Index<T> *initializer, const size_t max_points = 0,
                      const bool enable_tags = false);

    ~IndexNSG();

    void save(const char *filename) override;
    void load(const char *filename) override;

    void init_random_graph(size_t num_points, unsigned k,
                           std::vector<size_t> mapping = std::vector<size_t>());

    void build(const T *data, Parameters &parameters,
               const std::vector<TagT> &tags = std::vector<TagT>());

    typedef std::vector<SimpleNeighbor> vecNgh;

    void populate_start_points_bfs(std::vector<unsigned> &start_points);

    std::pair<int, int> beam_search(const T *query, const T *x, const size_t K,
                                    const Parameters &parameters,
                                    unsigned *indices, int beam_width,
                                    std::vector<unsigned> &start_points);

    std::pair<int, int> beam_search_tags(const T *query, const T *x,
                                         const size_t      K,
                                         const Parameters &parameters,
                                         TagT *tags, int beam_width,
                                         std::vector<unsigned> &start_points,
                                         unsigned *indices_buffer = NULL);

    void prefetch_vector(unsigned id);

    // Gopal. Added beam_search overload that takes L as parameter, so that we
    // can customize L on a per-query basis without tampering with "Parameters"
    std::pair<int, int> beam_search(const T *query, const T *x, const size_t K,
                                    const unsigned int L, unsigned *indices,
                                    int                    beam_width,
                                    std::vector<unsigned> &start_points);

    void save_disk_opt_graph(const char *diskopt_path);

    /* Methods for inserting and deleting points from the databases*/

    int insert_point(const T *point, const Parameters &parameter,
                     std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
                     tsl::robin_set<unsigned> &visited, vecNgh &cut_graph,
                     const TagT tag = TagT(-1));

    int enable_delete();
    int disable_delete(const Parameters &parameters,
                       const bool        consolidate = false);

    // Return -1 if tag not found, 0 if OK.
    int delete_point(const TagT tag);

    /*  Internals of the library */
   protected:
    typedef std::vector<std::vector<unsigned>> CompactGraph;

    CompactGraph _final_graph;

    bool *_is_inner;

    Index<T> *_initializer;

    void reachable_bfs(const unsigned                         start_node,
                       std::vector<tsl::robin_set<unsigned>> &bfs_order,
                       bool *                                 visited);

    // entry point is centroid based on all-to-centroid distance computation
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

    std::unordered_map<TagT, unsigned> _tag_to_point;
    std::unordered_map<unsigned, TagT> _point_to_tag;

    tsl::robin_set<unsigned> _delete_set;
    tsl::robin_set<unsigned> _empty_slots;

    std::mutex _change_lock;  // Allow only 1 thread to insert/delete

    using Index<T>::_dim;
    using Index<T>::_data;
    using Index<T>::_nd;
    using Index<T>::_has_built;
    using Index<T>::_distance;
    using Index<T>::_max_points;
  };
}

bool BuildIndex(const char *dataFilePath, const char *indexFilePath,
                const char *indexBuildParameters);
#endif  // EFANNA2E_INDEX_NSG_H
