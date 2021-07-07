#pragma once

#include "v2/graph_delta.h"
#include "v2/fs_allocator.h"
#include "v2/index_merger.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "pq_flash_index.h"
#include "linux_aligned_file_reader.h"
#include "index.h"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <cassert>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>

#include "parameters.h"
#include "Neighbor_Tag.h"
#include "threadpool.h"

#include "windows_customizations.h"

#define MERGE_TH 18000000

namespace diskann {

  template<typename T, typename TagT = uint32_t>
  class MergeInsert {
   public:
    // constructor
    DISKANN_DLLEXPORT MergeInsert(
        Parameters& parameters, size_t dim, const std::string mem_prefix,
        const std::string disk_prefix_in, const std::string disk_prefix_out,
        Distance<T>* dist, diskann::Metric disk_metric, bool single_file_index,
        std::string working_folder);

    DISKANN_DLLEXPORT ~MergeInsert();

    // insertion function - insert into short_term_index
    DISKANN_DLLEXPORT int insert(const T* point, const TagT& tag);

    DISKANN_DLLEXPORT void lazy_delete(const TagT& tag);
    //DISKANN_DLLEXPORT void lazy_delete(tsl::robin_set<TagT>& delete_list);

    // search function - search both short_term_index and long_term_index and
    // return with top L candidate tags of the shard
    DISKANN_DLLEXPORT void search_sync(const T* query, const uint64_t K,
                                       const uint64_t search_L, TagT* tags,
                                       float* distances, QueryStats* stats);
    // void return_active_tags(tsl::robin_set<TagT>& active_tags);

    // continuously runs in background to check if mem index size has exceeded
    // its threshold - triggers index switch and merge
    DISKANN_DLLEXPORT int trigger_merge();

    DISKANN_DLLEXPORT void final_merge();

    DISKANN_DLLEXPORT std::string ret_merge_prefix();

   protected:
    // call constructor to StreamingMerger object
    void construct_index_merger();

    // call StreamingMerger destructor to explicitly de-register threads
    void destruct_index_merger();

    //_active_index flag will be modified only inside this function
    void switch_index();  // function to atomically switch btw indices, makes
                          // older index inactive(read-only), saves it, makes new
                          // index active (r/w)

    // save currently active mem_index and make it inactive
    int save();

    // make a local copy of _deletion_set and save it to a _deleted_tags_file
    void save_del_set();

    // call merge on a StreamingMerger object, only if index switching and
    // saving is successful
    void merge();

   private:
    size_t   _merge_th = 0;
    size_t   _mem_points = 0;  // reflects number of points in active mem index
    size_t   _index_points = 0;
    size_t   _dim;
    _u32     _num_nodes_to_cache;
    _u32     _num_search_threads;
    uint64_t _beamwidth;

    std::unordered_map<unsigned, TagT> curr_location_to_tag;

    std::shared_ptr<Index<T, TagT>>    _mem_index_0 = nullptr;
    std::shared_ptr<Index<T, TagT>>    _mem_index_1 = nullptr;
    std::shared_ptr<AlignedFileReader> reader = nullptr;
    PQFlashIndex<T, TagT>* _disk_index = nullptr;
   StreamingMerger<T, TagT> *  _merger = nullptr;
   std::string TMP_FOLDER;

   diskann::Metric _dist_metric;
    Distance<T>* _dist_comp;

    diskann::Parameters _paras_mem;
    diskann::Parameters _paras_disk;

    tsl::robin_set<TagT> _deletion_set_0;
    tsl::robin_set<TagT> _deletion_set_1;

    std::vector<const std::vector<TagT>*> _deleted_tags_vector;

    int              _active_index = 0;  // reflects value of writable index
    int              _active_delete_set = 0;  // reflects active _deletion_set
    std::atomic_bool _active_0;               // true except when merging
    std::atomic_bool _active_1;               // true except when merging
    std::atomic_bool _active_del_0;           // true except when being saved
    std::atomic_bool _active_del_1;           // true except when being saved
    std::atomic_bool _clearing_index_0;       // don't search mem_index if true
    std::atomic_bool _clearing_index_1;       // don't search mem_index if true
    std::atomic_bool _switching_disk_prefixes =
        false;  // wait if true, search when false
    std::atomic_bool _check_switch_index =
        false;  // true when switch_index acquires _index_lock in writer mode,
                // insert threads wait till it turns back to false
    std::atomic_bool _check_switch_delete =
        false;  // true when switching between _deletion_sets, _delete_lock
    // acquired in write mode, delete thread waits till it turns back to false

    bool _single_file_index = false;

    std::shared_timed_mutex _delete_lock;  // lock to access _deletion_set
    std::shared_timed_mutex _index_lock;  // mutex to switch between mem indices
    std::shared_timed_mutex _change_lock;  // mutex to switch increment _mem_pts
    std::shared_timed_mutex _disk_lock;  // mutex to switch between disk indices
    std::shared_timed_mutex
        _clear_lock_0;  // lock to prevent an index from being cleared when it
                        // is being searched  and vice versa
    std::shared_timed_mutex
        _clear_lock_1;  // lock to prevent an index from being cleared when it
                        // is being searched  and vice versa

    ThreadPool* _search_tpool;

    std::string _mem_index_prefix;
    std::string _disk_index_prefix_in;
    std::string _disk_index_prefix_out;
    std::string _deleted_tags_file;
  };
};  // namespace diskann
