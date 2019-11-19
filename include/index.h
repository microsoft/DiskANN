#pragma once

#include <cassert>
#include <map>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include "distance.h"
#include "neighbor.h"
#include "parameters.h"
#include "tsl/robin_set.h"
#include "utils.h"

namespace diskann {
  template<typename T, typename TagT = int>
  class Index {
   public:
    Index(Metric m, const char *filename, const size_t max_points = 0,
          const size_t nd = 0, const size_t num_frozen_pts = 0,
          const bool enable_tags = false, const bool store_data = true,
          const bool support_eager_delete = false);
    ~Index();

    // checks if data is consolidated, saves graph, metadata and associated
    // tags.
    void save(const char *filename);
    void load(const char *filename, const bool load_tags = false);
    void init_random_graph(unsigned k);

    void build(Parameters &             parameters,
               const std::vector<TagT> &tags = std::vector<TagT>());

    void populate_start_points_ep(std::vector<unsigned> &start_points);
    void populate_start_points_bfs(std::vector<unsigned> &start_points);

    // Gopal. Added beam_search overload that takes L as parameter, so that we
    // can customize L on a per-query basis without tampering with "Parameters"
    std::pair<int, int> beam_search(const T *query, const size_t K,
                                    const unsigned L, unsigned *indices,
                                    int                   beam_width,
                                    std::vector<unsigned> start_points);

    std::pair<int, int> beam_search_tags(const T *query, const size_t K,
                                         const size_t L, TagT *tags,
                                         int                   beam_width,
                                         std::vector<unsigned> start_points,
                                         unsigned *indices_buffer = NULL);

    void prefetch_vector(unsigned id);

    // repositions frozen points to the end of _data - if they have been moved
    // during deletion
    void readjust_data(unsigned _num_frozen_pts);

    /* insertions possible only when id corresponding to tag does not already
     * exist in the graph */
    int insert_point(const T *point, const Parameters &parameter,
                     std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
                     tsl::robin_set<unsigned> &   visited,
                     std::vector<SimpleNeighbor> &cut_graph, const TagT tag);

    // call before triggering deleteions - sets important flags required for
    // deletion related operations
    int enable_delete();

    // call after all delete requests have been served, checks if deletions were
    // executed correctly, rearranges metadata in case of lazy deletes
    int disable_delete(const Parameters &parameters,
                       const bool        consolidate = false);

    // Record deleted point now and restructure graph later. Return -1 if tag
    // not found, 0 if OK. Do not call if _eager_delete was called earlier and
    // data was not consolidated
    int delete_point(const TagT tag);

    // Delete point from graph and restructure it immediately. Do not call if
    // _lazy_delete was called earlier and data was not consolidated
    int eager_delete(const TagT tag, const Parameters &parameters);

    /*  Internals of the library */
   protected:
    typedef std::vector<SimpleNeighbor>        vecNgh;
    typedef std::vector<std::vector<unsigned>> CompactGraph;
    CompactGraph                               _final_graph;
    CompactGraph                               _in_graph;

    // to get appropriate start points for search
    void reachable_bfs(const unsigned                         start_node,
                       std::vector<tsl::robin_set<unsigned>> &bfs_order,
                       bool *                                 visited);

    // determines navigating node of the graph by calculating medoid of data
    unsigned calculate_entry_point();
    // generates one or more frozen points that will never get deleted from the
    // graph
    int generate_random_frozen_points();
    // called only when _eager_delete is to be supported
    void update_in_graph();

    void iterate_to_fixed_point(const size_t node_id, const unsigned Lindex,
                                std::vector<unsigned> &   init_ids,
                                std::vector<Neighbor> &   retset,
                                std::vector<Neighbor> &   fullset,
                                tsl::robin_set<unsigned> &visited);

    void get_neighbors(const size_t node, const unsigned Lindex,
                       std::vector<Neighbor> &   retset,
                       std::vector<Neighbor> &   fullset,
                       tsl::robin_set<unsigned> &visited);

    void inter_insert(unsigned n, std::vector<unsigned> &pruned_list,
                      const Parameters &parameter, bool update_in_graph);

    void sync_prune(const unsigned location, std::vector<Neighbor> &pool,
                    const Parameters &     parameter,
                    std::vector<unsigned> &pruned_list);

    void occlude_list(std::vector<Neighbor> &pool, const unsigned location,
                      const float alpha, const unsigned degree,
                      const unsigned maxc, std::vector<Neighbor> &result);

    void occlude_list(std::vector<Neighbor> &pool, const unsigned location,
                      const float alpha, const unsigned degree,
                      const unsigned maxc, std::vector<Neighbor> &result,
                      std::vector<float> &occlude_factor);

    void batch_inter_insert(unsigned                     n,
                            const std::vector<unsigned> &pruned_list,
                            const Parameters &           parameter,
                            std::vector<unsigned> &      need_to_sync);

    void link(Parameters &parameters);

    // WARNING: Do not call reserve_location() without acquiring change_lock_
    unsigned reserve_location();

    // get new location corresponding to each undeleted tag after deletions
    std::vector<unsigned> get_new_location(unsigned &active);

    // renumber nodes, update tag and location maps and compact the graph, mode
    // = _consolidated_order in case of lazy deletion and _compacted_order in
    // case of eager deletion
    void compact_data(std::vector<unsigned> new_location, unsigned active,
                      bool &mode);

    // WARNING: Do not call consolidate_deletes without acquiring change_lock_
    // Returns number of live points left after consolidation
    size_t consolidate_deletes(const Parameters &parameters);

   private:
    size_t       _dim;
    size_t       _aligned_dim;
    T *          _data;
    size_t       _nd;  // number of active points i.e. existing in the graph
    size_t       _max_points;  // total number of points in given data set
    size_t       _num_frozen_pts;
    bool         _has_built;
    Distance<T> *_distance;
    unsigned     _width;
    unsigned     _ep;
    std::vector<std::mutex> _locks;  // Per node lock, cardinality=max_points_
    char *                  _opt_graph;
    size_t                  _node_size;
    size_t                  _data_len;
    size_t                  _neighbor_len;

    bool _can_delete;
    bool _eager_done;       // true if eager deletions have been made
    bool _lazy_done;        // true if lazy deletions have been made
    bool _compacted_order;  // true if after eager deletions, data has been
                            // consolidated
    bool _enable_tags;
    bool _consolidated_order;    // true if after lazy deletions, data has been
                                 // consolidated
    bool _support_eager_delete;  //_support_eager_delete = activates extra data
                                 // structures and functions required for eager
    // deletion
    bool _store_data;

    std::unordered_map<TagT, unsigned> _tag_to_location;
    std::unordered_map<unsigned, TagT> _location_to_tag;

    tsl::robin_set<unsigned> _delete_set;
    tsl::robin_set<unsigned> _empty_slots;

    std::mutex _change_lock;  // Allow only 1 thread to insert/delete
  };
}
