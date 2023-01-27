// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "common_includes.h"

#ifdef EXEC_ENV_OLS
#include "aligned_file_reader.h"
#endif

#include "distance.h"
#include "locking.h"
#include "natural_number_map.h"
#include "natural_number_set.h"
#include "neighbor.h"
#include "parameters.h"
#include "utils.h"
#include "windows_customizations.h"
#include "scratch.h"

#define OVERHEAD_FACTOR 1.1
#define EXPAND_IF_FULL 0
#define DEFAULT_MAXC 750

namespace diskann {
  inline double estimate_ram_usage(_u64 size, _u32 dim, _u32 datasize,
                                   _u32 degree) {
    double size_of_data = ((double) size) * ROUND_UP(dim, 8) * datasize;
    double size_of_graph =
        ((double) size) * degree * sizeof(unsigned) * GRAPH_SLACK_FACTOR;
    double size_of_locks = ((double) size) * sizeof(non_recursive_mutex);
    double size_of_outer_vector = ((double) size) * sizeof(ptrdiff_t);

    return OVERHEAD_FACTOR * (size_of_data + size_of_graph + size_of_locks +
                              size_of_outer_vector);
  }

  struct consolidation_report {
    enum status_code {
      SUCCESS = 0,
      FAIL = 1,
      LOCK_FAIL = 2,
      INCONSISTENT_COUNT_ERROR = 3
    };
    status_code _status;
    size_t      _active_points, _max_points, _empty_slots, _slots_released,
        _delete_set_size, _num_calls_to_process_delete;
    double _time;

    consolidation_report(status_code status, size_t active_points,
                         size_t max_points, size_t empty_slots,
                         size_t slots_released, size_t delete_set_size,
                         size_t num_calls_to_process_delete, double time_secs)
        : _status(status), _active_points(active_points),
          _max_points(max_points), _empty_slots(empty_slots),
          _slots_released(slots_released), _delete_set_size(delete_set_size),
          _num_calls_to_process_delete(num_calls_to_process_delete),
          _time(time_secs) {
    }
  };

  template<typename T, typename TagT = uint32_t>
  class Index {
    /**************************************************************************
     *
     * Public functions acquire one or more of _update_lock, _consolidate_lock,
     * _tag_lock, _delete_lock before calling protected functions which DO NOT
     * acquire these locks. They might acquire locks on _locks[i]
     *
     **************************************************************************/

   public:
    // Constructor for Bulk operations and for creating the index object solely
    // for loading a prexisting index.
    DISKANN_DLLEXPORT Index(Metric m, const size_t dim,
                            const size_t max_points = 1,
                            const bool   dynamic_index = false,
                            const bool   enable_tags = false,
                            const bool   concurrent_consolidate = false,
                            const bool   pq_dist_build = false,
                            const size_t num_pq_chunks = 0,
                            const bool   use_opq = false);

    // Constructor for incremental index
    DISKANN_DLLEXPORT Index(Metric m, const size_t dim, const size_t max_points,
                            const bool        dynamic_index,
                            const Parameters &indexParameters,
                            const Parameters &searchParameters,
                            const bool        enable_tags = false,
                            const bool        concurrent_consolidate = false,
                            const bool        pq_dist_build = false,
                            const size_t      num_pq_chunks = 0,
                            const bool        use_opq = false);

    DISKANN_DLLEXPORT ~Index();

    // Saves graph, data, metadata and associated tags.
    DISKANN_DLLEXPORT void save(const char *filename,
                                bool        compact_before_save = false);

    // Load functions
#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT void load(AlignedFileReader &reader, uint32_t num_threads,
                                uint32_t search_l);
#else
    DISKANN_DLLEXPORT void   load(const char *index_file, uint32_t num_threads,
                                  uint32_t search_l);
#endif

    // get some private variables
    DISKANN_DLLEXPORT size_t get_num_points();
    DISKANN_DLLEXPORT size_t get_max_points();

    // Batch build from a file. Optionally pass tags vector.
    DISKANN_DLLEXPORT void build(
        const char *filename, const size_t num_points_to_load,
        Parameters              &parameters,
        const std::vector<TagT> &tags = std::vector<TagT>());

    // Batch build from a file. Optionally pass tags file.
    DISKANN_DLLEXPORT void build(const char  *filename,
                                 const size_t num_points_to_load,
                                 Parameters  &parameters,
                                 const char  *tag_filename);

    // Batch build from a data array, which must pad vectors to aligned_dim
    DISKANN_DLLEXPORT void build(const T *data, const size_t num_points_to_load,
                                 Parameters              &parameters,
                                 const std::vector<TagT> &tags);

    // Set starting point of an index before inserting any points incrementally
    DISKANN_DLLEXPORT void set_start_point(T *data);
    // Set starting point to a random point on a sphere of certain radius
    DISKANN_DLLEXPORT void set_start_point_at_random(T radius);

    // For FastL2 search on a static index, we interleave the data with graph
    DISKANN_DLLEXPORT void optimize_index_layout();

    // For FastL2 search on optimized layout
    DISKANN_DLLEXPORT void search_with_optimized_layout(const T *query,
                                                        size_t K, size_t L,
                                                        unsigned *indices);

    // Added search overload that takes L as parameter, so that we
    // can customize L on a per-query basis without tampering with "Parameters"
    template<typename IDType>
    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> search(
        const T *query, const size_t K, const unsigned L, IDType *indices,
        float *distances = nullptr);

    // Initialize space for res_vectors before calling.
    DISKANN_DLLEXPORT size_t search_with_tags(const T *query, const uint64_t K,
                                              const unsigned L, TagT *tags,
                                              float            *distances,
                                              std::vector<T *> &res_vectors);

    // Will fail if tag already in the index or if tag=0.
    DISKANN_DLLEXPORT int insert_point(const T *point, const TagT tag);

    // call this before issuing deletions to sets relevant flags
    DISKANN_DLLEXPORT int enable_delete();

    // Record deleted point now and restructure graph later. Return -1 if tag
    // not found, 0 if OK.
    DISKANN_DLLEXPORT int lazy_delete(const TagT &tag);

    // Record deleted points now and restructure graph later. Add to failed_tags
    // if tag not found.
    DISKANN_DLLEXPORT void lazy_delete(const std::vector<TagT> &tags,
                                       std::vector<TagT>       &failed_tags);

    // Call after a series of lazy deletions
    // Returns number of live points left after consolidation
    // If _conc_consolidates is set in the ctor, then this call can be invoked
    // alongside inserts and lazy deletes, else it acquires _update_lock
    DISKANN_DLLEXPORT consolidation_report
    consolidate_deletes(const Parameters &parameters);

    DISKANN_DLLEXPORT bool is_index_saved();

    // repositions frozen points to the end of _data - if they have been moved
    // during deletion
    DISKANN_DLLEXPORT void reposition_frozen_point_to_end();
    DISKANN_DLLEXPORT void reposition_point(unsigned old_location,
                                            unsigned new_location);

    // DISKANN_DLLEXPORT void save_index_as_one_file(bool flag);

    DISKANN_DLLEXPORT void get_active_tags(tsl::robin_set<TagT> &active_tags);

    // memory should be allocated for vec before calling this function
    DISKANN_DLLEXPORT int get_vector_by_tag(TagT &tag, T *vec);

    DISKANN_DLLEXPORT void print_status();

    DISKANN_DLLEXPORT void count_nodes_at_bfs_levels();

    // This variable MUST be updated if the number of entries in the metadata
    // change.
    DISKANN_DLLEXPORT static const int METADATA_ROWS = 5;

    // ********************************
    //
    // Internals of the library
    //
    // ********************************

   protected:
    // No copy/assign.
    Index(const Index<T, TagT> &) = delete;
    Index<T, TagT> &operator=(const Index<T, TagT> &) = delete;

    // Use after _data and _nd have been populated
    // Acquire exclusive _update_lock before calling
    void build_with_data_populated(Parameters              &parameters,
                                   const std::vector<TagT> &tags);

    // generates 1 frozen point that will never be deleted from the graph
    // This is not visible to the user
    int generate_frozen_point();

    // determines navigating node of the graph by calculating medoid of datafopt
    unsigned calculate_entry_point();

    std::pair<uint32_t, uint32_t> iterate_to_fixed_point(
        const T *node_coords, const unsigned Lindex,
        const std::vector<unsigned> &init_ids, InMemQueryScratch<T> *scratch,
        bool ret_frozen = true, bool search_invocation = false);

    void search_for_point_and_prune(int location, _u32 Lindex,
                                    std::vector<unsigned> &pruned_list,
                                    InMemQueryScratch<T>  *scratch);

    void prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                         std::vector<unsigned> &pruned_list,
                         InMemQueryScratch<T>  *scratch);

    void prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                         const _u32 range, const _u32 max_candidate_size,
                         const float alpha, std::vector<unsigned> &pruned_list,
                         InMemQueryScratch<T> *scratch);

    // Prunes candidates in @pool to a shorter list @result
    // @pool must be sorted before calling
    void occlude_list(
        const unsigned location, std::vector<Neighbor> &pool, const float alpha,
        const unsigned degree, const unsigned maxc,
        std::vector<unsigned> &result, InMemQueryScratch<T> *scratch,
        const tsl::robin_set<unsigned> *const delete_set_ptr = nullptr);

    // add reverse links from all the visited nodes to node n.
    void inter_insert(unsigned n, std::vector<unsigned> &pruned_list,
                      const _u32 range, InMemQueryScratch<T> *scratch);

    void inter_insert(unsigned n, std::vector<unsigned> &pruned_list,
                      InMemQueryScratch<T> *scratch);

    // Acquire exclusive _update_lock before calling
    void link(Parameters &parameters);

    // Acquire exclusive _tag_lock and _delete_lock before calling
    int reserve_location();

    // Acquire exclusive _tag_lock before calling
    size_t release_location(int location);
    size_t release_locations(const tsl::robin_set<unsigned> &locations);

    // Resize the index when no slots are left for insertion.
    // Acquire exclusive _update_lock and _tag_lock before calling.
    void resize(size_t new_max_points);

    // Acquire unique lock on _update_lock, _consolidate_lock, _tag_lock
    // and _delete_lock before calling these functions.
    // Renumber nodes, update tag and location maps and compact the
    // graph, mode = _consolidated_order in case of lazy deletion and
    // _compacted_order in case of eager deletion
    DISKANN_DLLEXPORT void compact_data();
    DISKANN_DLLEXPORT void compact_frozen_point();

    // Remove deleted nodes from adjacency list of node loc
    // Replace removed neighbors with second order neighbors.
    // Also acquires _locks[i] for i = loc and out-neighbors of loc.
    void process_delete(const tsl::robin_set<unsigned> &old_delete_set,
                        size_t loc, const unsigned range, const unsigned maxc,
                        const float alpha, InMemQueryScratch<T> *scratch);

    void initialize_query_scratch(uint32_t num_threads, uint32_t search_l,
                                  uint32_t indexing_l, uint32_t r,
                                  uint32_t maxc, size_t dim);

    // Do not call without acquiring appropriate locks
    // call public member functions save and load to invoke these.
    DISKANN_DLLEXPORT _u64 save_graph(std::string filename);
    DISKANN_DLLEXPORT _u64 save_data(std::string filename);
    DISKANN_DLLEXPORT _u64 save_tags(std::string filename);
    DISKANN_DLLEXPORT _u64 save_delete_list(const std::string &filename);
#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT size_t load_graph(AlignedFileReader &reader,
                                        size_t             expected_num_points);
    DISKANN_DLLEXPORT size_t load_data(AlignedFileReader &reader);
    DISKANN_DLLEXPORT size_t load_tags(AlignedFileReader &reader);
    DISKANN_DLLEXPORT size_t load_delete_set(AlignedFileReader &reader);
#else
    DISKANN_DLLEXPORT size_t load_graph(const std::string filename,
                                        size_t            expected_num_points);
    DISKANN_DLLEXPORT size_t load_data(std::string filename0);
    DISKANN_DLLEXPORT size_t load_tags(const std::string tag_file_name);
    DISKANN_DLLEXPORT size_t load_delete_set(const std::string &filename);
#endif

   private:
    // Distance functions
    Metric       _dist_metric = diskann::L2;
    Distance<T> *_distance = nullptr;

    // Data
    T    *_data = nullptr;
    char *_opt_graph = nullptr;

    // Graph related data structures
    std::vector<std::vector<unsigned>> _final_graph;

    // Dimensions
    size_t _dim = 0;
    size_t _aligned_dim = 0;
    size_t _nd = 0;  // number of active points i.e. existing in the graph
    size_t _max_points = 0;  // total number of points in given data set
    size_t _num_frozen_pts = 0;
    size_t _max_range_of_loaded_graph = 0;
    size_t _node_size;
    size_t _data_len;
    size_t _neighbor_len;

    unsigned _max_observed_degree = 0;
    unsigned _start = 0;

    bool _has_built = false;
    bool _saturate_graph = false;
    bool _save_as_one_file = false;  // plan to support in next version
    bool _dynamic_index = false;
    bool _enable_tags = false;
    bool _normalize_vecs = false;  // Using normalied L2 for cosine.

    // Indexing parameters
    uint32_t _indexingQueueSize;
    uint32_t _indexingRange;
    uint32_t _indexingMaxC;
    float    _indexingAlpha;
    uint32_t _search_queue_size;

    // Query scratch data structures
    ConcurrentQueue<InMemQueryScratch<T> *> _query_scratch;

    // Flags for PQ based distance calculation
    bool              _pq_dist = false;
    bool              _use_opq = false;
    size_t            _num_pq_chunks = 0;
    _u8              *_pq_data = nullptr;
    bool              _pq_generated = false;
    FixedChunkPQTable _pq_table;

    //
    // Data structures, locks and flags for dynamic indexing and tags
    //

    // lazy_delete removes entry from _location_to_tag and _tag_to_location. If
    // _location_to_tag does not resolve a location, infer that it was deleted.
    tsl::sparse_map<TagT, unsigned>    _tag_to_location;
    natural_number_map<unsigned, TagT> _location_to_tag;

    // _empty_slots has unallocated slots and those freed by consolidate_delete.
    // _delete_set has locations marked deleted by lazy_delete. Will not be
    // immediately available for insert. consolidate_delete will release these
    // slots to _empty_slots.
    natural_number_set<unsigned>              _empty_slots;
    std::unique_ptr<tsl::robin_set<unsigned>> _delete_set;

    bool _data_compacted = true;  // true if data has been compacted
    bool _is_saved = false;  // Gopal. Checking if the index is already saved.
    bool _conc_consolidate = false;  // use _lock while searching

    // Acquire locks in the order below when acquiring multiple locks
    std::shared_timed_mutex  // RW mutex between save/load (exclusive lock) and
        _update_lock;        // search/inserts/deletes/consolidate (shared lock)
    std::shared_timed_mutex  // Ensure only one consolidate or compact_data is
        _consolidate_lock;   // ever active
    std::shared_timed_mutex  // RW lock for _tag_to_location,
        _tag_lock;           // _location_to_tag, _empty_slots, _nd, _max_points
    std::shared_timed_mutex  // RW Lock on _delete_set and _data_compacted
        _delete_lock;        // variable

    // Per node lock, cardinality=_max_points
    std::vector<non_recursive_mutex> _locks;

    static const float INDEX_GROWTH_FACTOR;
  };
}  // namespace diskann
