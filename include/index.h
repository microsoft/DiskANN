// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <atomic>
#include <cassert>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"

#include "distance.h"
#include "neighbor.h"
#include "parameters.h"
#include "utils.h"
#include "windows_customizations.h"

#include "Neighbor_Tag.h"

#define SLACK_FACTOR 1.3

#define ESTIMATE_RAM_USAGE(size, dim, datasize, degree)                 \
  (SLACK_FACTOR * (((double) size * (double) dim) * (double) datasize + \
                   ((double) size * (double) degree) *                  \
                       (double) sizeof(unsigned) * SLACK_FACTOR))

namespace diskann {
  template<typename T, typename TagT = uint32_t>
  class Index {
   public:
    DISKANN_DLLEXPORT Index(Metric m, const size_t dim, const size_t max_points,
                            const bool dynamic_index,
                            const bool save_index_in_one_file,
                            const bool enable_tags = false,
                            const bool support_eager_delete = false);

    //    DISKANN_DLLEXPORT Index(Index *index);  // deep copy
    DISKANN_DLLEXPORT ~Index();

    // Public Functions for Static Support

    // checks if data is consolidated, saves graph, metadata and associated
    // tags.
    DISKANN_DLLEXPORT void save(const char *filename);

    DISKANN_DLLEXPORT _u64 save_graph(std::string filename, size_t offset = 0);
    DISKANN_DLLEXPORT _u64 save_data(std::string filename, size_t offset = 0);
    DISKANN_DLLEXPORT _u64 save_tags(std::string filename, size_t offset = 0);
    DISKANN_DLLEXPORT _u64 save_delete_list(const std::string &filename,
                                            size_t             offset = 0);

    DISKANN_DLLEXPORT void load(const char *index_file);

    DISKANN_DLLEXPORT size_t load_graph(const std::string filename,
                                        size_t            expected_num_points,
                                        size_t            offset = 0);

    DISKANN_DLLEXPORT size_t load_data(std::string filename, size_t offset = 0);

    DISKANN_DLLEXPORT size_t load_tags(const std::string tag_file_name,
                                       size_t            offset = 0);
    DISKANN_DLLEXPORT size_t load_delete_set(const std::string &filename,
                                             size_t             offset = 0);

    DISKANN_DLLEXPORT size_t get_num_points();

    DISKANN_DLLEXPORT size_t return_max_points();

    DISKANN_DLLEXPORT void build(
        const char *filename, const size_t num_points_to_load,
        Parameters &             parameters,
        const std::vector<TagT> &tags = std::vector<TagT>());

    DISKANN_DLLEXPORT void build(const char * filename,
                                 const size_t num_points_to_load,
                                 Parameters & parameters,
                                 const char * tag_filename);
    // Added search overload that takes L as parameter, so that we
    // can customize L on a per-query basis without tampering with "Parameters"
    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> search(
        const T *query, const size_t K, const unsigned L, unsigned *indices,
        float *distances = nullptr);

    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> search(
        const T *query, const uint64_t K, const unsigned L,
        std::vector<unsigned> init_ids, uint64_t *indices, float *distances);

    DISKANN_DLLEXPORT size_t search_with_tags(const T *query, const uint64_t K,
                                              const unsigned L, TagT *tags,
                                              float *           distances,
                                              std::vector<T *> &res_vectors);

    DISKANN_DLLEXPORT size_t search_with_tags(const T *query, const size_t K,
                                              const unsigned L, TagT *tags,
                                              float *distances);

    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> search(
        const T *query, const size_t K, const unsigned L,
        std::vector<Neighbor_Tag<TagT>> &best_L_tags);

    DISKANN_DLLEXPORT void optimize_graph();

    DISKANN_DLLEXPORT void search_with_opt_graph(const T *query, size_t K,
                                                 size_t L, unsigned *indices);

    DISKANN_DLLEXPORT void clear_index();

    // Public Functions for Incremental Support

    /* insertions possible only when id corresponding to tag does not already
     * exist in the graph */
    DISKANN_DLLEXPORT int insert_point(
        const T *point, const Parameters &parameter,
        const TagT tag);  // only keep point, tag, parameters
    // call before triggering deleteions - sets important flags required for
    // deletion related operations
    DISKANN_DLLEXPORT int enable_delete();

    // Record deleted point now and restructure graph later. Return -1 if tag
    // not found, 0 if OK. Do not call if _eager_delete was called earlier and
    // data was not consolidated
    DISKANN_DLLEXPORT int lazy_delete(const TagT &tag);

    // Record deleted points now and restructure graph later. Add to failed_tags
    // if tag not found. Do not call if _eager_delete was called earlier and
    // data was not consolidated. Return -1 if
    DISKANN_DLLEXPORT int lazy_delete(const tsl::robin_set<TagT> &tags,
                                      std::vector<TagT> &         failed_tags);

    // Delete point from graph and restructure it immediately. Do not call if
    // _lazy_delete was called earlier and data was not consolidated
    DISKANN_DLLEXPORT int eager_delete(const TagT        tag,
                                       const Parameters &parameters,
                                       int               delete_mode = 1);
    // return _data and tag_to_location offset
    DISKANN_DLLEXPORT int extract_data(
        T *ret_data, std::unordered_map<TagT, unsigned> &tag_to_location);

    DISKANN_DLLEXPORT void get_location_to_tag(
        std::unordered_map<unsigned, TagT> &ret_loc_to_tag);

    DISKANN_DLLEXPORT void prune_all_nbrs(const Parameters &parameters);

    DISKANN_DLLEXPORT void compact_data_for_insert();

    DISKANN_DLLEXPORT bool hasIndexBeenSaved();
    // diskv2 API
    void iterate_to_fixed_point(const T *node_coords, const unsigned Lindex,
                                std::vector<Neighbor> &expanded_nodes_info,
                                tsl::robin_map<uint32_t, T *> &coord_map,
                                bool return_frozen_pt = true);
    // convenient access to graph + data (aligned)
    const std::vector<std::vector<unsigned>> *get_graph() const {
      return &this->_final_graph;
    }
    T *                                       get_data();
    const std::unordered_map<unsigned, TagT> *get_tags() const {
      return &this->_location_to_tag;
    };
    // repositions frozen points to the end of _data - if they have been moved
    // during deletion
    DISKANN_DLLEXPORT void reposition_frozen_point_to_end();
    DISKANN_DLLEXPORT void reposition_point(unsigned old_location,
                                            unsigned new_location);

    DISKANN_DLLEXPORT void compact_frozen_point();
    DISKANN_DLLEXPORT void compact_data_for_search();

    DISKANN_DLLEXPORT void consolidate(Parameters &parameters);

    // DISKANN_DLLEXPORT void save_index_as_one_file(bool flag);

    DISKANN_DLLEXPORT void get_active_tags(tsl::robin_set<TagT> &active_tags);

    DISKANN_DLLEXPORT int   get_vector_by_tag(TagT &tag, T *vec);
    DISKANN_DLLEXPORT const T *get_vector_by_tag(const TagT &tag);

    // TODO: Debugging ONLY
    DISKANN_DLLEXPORT void print_status() const;
    DISKANN_DLLEXPORT void are_deleted_points_in_graph() const;
    DISKANN_DLLEXPORT void print_delete_set() const;

    // This variable MUST be updated if the number of entries in the metadata
    // change.
    DISKANN_DLLEXPORT static const int METADATA_ROWS = 5;

    /*  Internals of the library */
   protected:
    std::vector<std::vector<unsigned>> _final_graph;
    std::vector<std::vector<unsigned>> _in_graph;

    // generates one frozen point that will never get deleted from the
    // graph
    int generate_frozen_point();

    // determines navigating node of the graph by calculating medoid of data
    unsigned calculate_entry_point();
    // called only when _eager_delete is to be supported
    void update_in_graph();

    std::pair<uint32_t, uint32_t> iterate_to_fixed_point(
        const T *node_coords, const unsigned Lindex,
        const std::vector<unsigned> &init_ids,
        std::vector<Neighbor> &      expanded_nodes_info,
        tsl::robin_set<unsigned> &   expanded_nodes_ids,
        std::vector<Neighbor> &best_L_nodes, bool ret_frozen = true);

    void get_expanded_nodes(const size_t node, const unsigned Lindex,
                            std::vector<unsigned>     init_ids,
                            std::vector<Neighbor> &   expanded_nodes_info,
                            tsl::robin_set<unsigned> &expanded_nodes_ids);

    void inter_insert(unsigned n, std::vector<unsigned> &pruned_list,
                      const Parameters &parameter, bool update_in_graph);

    void prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                         const Parameters &     parameter,
                         std::vector<unsigned> &pruned_list);

    void occlude_list(std::vector<Neighbor> &pool, const float alpha,
                      const unsigned degree, const unsigned maxc,
                      std::vector<Neighbor> &result);

    void occlude_list(std::vector<Neighbor> &pool, const float alpha,
                      const unsigned degree, const unsigned maxc,
                      std::vector<Neighbor> &result,
                      std::vector<float> &   occlude_factor);

    void batch_inter_insert(unsigned                     n,
                            const std::vector<unsigned> &pruned_list,
                            const Parameters &           parameter,
                            std::vector<unsigned> &      need_to_sync);

    void link(Parameters &parameters);

    // Support for Incremental Indexing
    int  reserve_location();
    void release_location();

    // Support for resizing the index
    // This function must be called ONLY after taking the _change_lock and
    // _update_lock.
    // Anything else in a MT environment will lead to an inconsistent index.
    void resize(uint32_t new_max_points);

    // renumber nodes, update tag and location maps and compact the graph, mode
    // = _compacted_lazy_deletions in case of lazy deletion and
    // _compacted_eager_deletions in
    // case of eager deletion
    void compact_data();

    // WARNING: Do not call consolidate_deletes without acquiring change_lock_
    // Returns number of live points left after consolidation
    size_t consolidate_deletes(const Parameters &parameters);

   private:
    // DEBUG ONLY
    void printTagToLocation();

    std::shared_timed_mutex _tag_lock;  // reader-writer lock on
                                        // _tag_to_location and
    std::mutex _change_lock;  // Lock taken to synchronously modify _nd

    T *_data = nullptr;  // coordinates of all base points
    // T *_pq_data =
    //    nullptr;  // coordinates of pq centroid corresponding to every point
    Distance<T> *   _distance = nullptr;
    diskann::Metric _dist_metric;

    size_t   _dim;
    size_t   _aligned_dim;
    size_t   _nd = 0;  // number of active points i.e. existing in the graph
    size_t   _max_points = 0;  // total number of points in given data set
    size_t   _num_frozen_pts = 0;
    unsigned _width = 0;
    unsigned _ep = 0;
    bool     _has_built = false;
    bool     _saturate_graph = false;
    bool     _save_as_one_file = false;
    bool     _dynamic_index = false;
    bool     _enable_tags = false;

    char * _opt_graph;
    size_t _node_size;
    size_t _data_len;
    size_t _neighbor_len;

    // flags for dynamic indexing
    std::unordered_map<TagT, unsigned> _tag_to_location;
    std::unordered_map<unsigned, TagT> _location_to_tag;

    tsl::robin_set<unsigned> _delete_set;
    tsl::robin_set<unsigned> _empty_slots;

    bool _support_eager_delete =
        false;  //_support_eager_delete = activates extra data
    // bool _can_delete = false;  // only true if deletes can be done (if
    // enabled)
    bool _eager_done = false;     // true if eager deletions have been made
    bool _lazy_done = false;      // true if lazy deletions have been made
    bool _data_compacted = true;  // true if data has been consolidated
    bool _is_saved = false;  // Gopal. Checking if the index is already saved.

    std::vector<std::mutex> _locks;  // Per node lock, cardinality=max_points_
    std::vector<std::mutex> _locks_in;     // Per node lock
    std::shared_timed_mutex _delete_lock;  // Lock on _delete_set and
                                           // _empty_slots when reading and
                                           // writing to them
    // _location_to_tag, has a shared lock
    // and exclusive lock associated with
    // it.
    std::shared_timed_mutex _update_lock;  // coordinate save() and any change
                                           // being done to the graph.

    const float INDEX_GROWTH_FACTOR = 1.5f;
  };
}  // namespace diskann
