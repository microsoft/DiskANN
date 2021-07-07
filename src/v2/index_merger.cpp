#include "neighbor.h"
#include "timer.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/index_merger.h"
#include <algorithm>
#include <cassert>
#include <csignal>
#include <iterator>
#include <mutex>
#include <thread>
#include <vector>
#include <limits>
#include <omp.h>
#include <future>

#include "tcmalloc/malloc_extension.h"
#include <unistd.h>
#include <sys/syscall.h>
#include "logger.h"
#include "ann_exception.h"

#define SECTORS_PER_MERGE (uint64_t) 65536
// max number of points per mem index being merged -- 32M
#define MAX_PTS_PER_MEM_INDEX (uint64_t)(1 << 25)
#define INDEX_OFFSET (uint64_t)(MAX_PTS_PER_MEM_INDEX * 4)
#define MAX_INSERT_THREADS (uint64_t) 18
#define MAX_N_THREADS (uint64_t) 18
#define NUM_INDEX_LOAD_THREADS (uint64_t) 18
#define PER_THREAD_BUF_SIZE (uint64_t)(65536 * 64 * 4)

#define PQ_FLASH_INDEX_MAX_NODES_TO_CACHE 200000

namespace diskann {
  template<typename T, typename TagT>
  StreamingMerger<T, TagT>::StreamingMerger(
      const uint32_t ndims, Distance<T> *dist, diskann::Metric dist_metric, const uint32_t beam_width,
      const uint32_t range, const uint32_t l_index, const float alpha,
      const uint32_t maxc, bool single_file_index) {
    // book keeping
    this->ndims = ndims;
    this->aligned_ndims = (_u32) ROUND_UP(this->ndims, 8);
    this->range = range;
    this->l_index = l_index;
    this->beam_width = beam_width;
    this->maxc = maxc;
    this->alpha = alpha;
    this->dist_metric = dist_metric;
    this->dist_cmp = dist;
    this->_single_file_index = single_file_index;

    std::cout << "StreamingMerger created with R=" << this->range
              << " L=" << this->l_index << " BW=" << this->beam_width
              << " MaxC=" << this->maxc << " alpha=" << this->alpha
              << " ndims: " << this->ndims << std::endl;
  }

  template<typename T, typename TagT>
  StreamingMerger<T, TagT>::~StreamingMerger() {
    // release scratch alloc memory
    // delete this->fp_alloc;
    // delete this->pq_alloc;

    if (this->disk_index != nullptr)
      delete this->disk_index;

    delete this->disk_delta;

    for (auto &delta : this->mem_deltas) {
      delete delta;
    }
    aligned_free((void *) this->thread_pq_scratch);
    for (auto &data : this->mem_data) {
      //delete[] data;
      aligned_free((void *)data);
    }
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::process_inserts_pq() {
    Timer total_insert_timer;
    this->insert_times.resize(MAX_N_THREADS, 0.0);
    this->delta_times.resize(MAX_N_THREADS, 0.0);
    // iterate through each vector in each mem index
    for (uint32_t i = 0; i < this->mem_data.size(); i++) {
      diskann::cout << "Processing pq of inserts from mem-DiskANN #" << i + 1
                    << "\n";
      const tsl::robin_set<uint32_t> &deleted_set = this->mem_deleted_ids[i];
      const T* coords = this->mem_data[i];
      const uint32_t offset = this->offset_ids[i];
      const uint32_t count = this->mem_npts[i];
// TODO (perf) :: trivially parallelizes ??
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_N_THREADS)
      // iteratively insert each point into full index
      for (int32_t j = 0; j < (int32_t) count; j++) {
        // filter out -- `j` is deleted
        if (deleted_set.find((uint32_t) j) != deleted_set.end()) {
          continue;
        }

        // data for jth point
        const T *j_coords =
            coords + ((uint64_t)(this->aligned_ndims) * (uint64_t) j);
        const uint32_t j_id = offset + (uint32_t) j;

        // get renamed ID
        const uint32_t j_renamed = this->rename(j_id);
        assert(j_renamed != std::numeric_limits<uint32_t>::max());

        // compute PQ coords
        std::vector<uint8_t> j_pq_coords =
            this->disk_index->deflate_vector(j_coords);
        //        std::vector<uint8_t> j_pq_coords(this->pq_nchunks,0);

        // directly copy into PQFlashIndex PQ data
        const uint64_t j_pq_offset =
            (uint64_t) j_renamed * (uint64_t) this->pq_nchunks;
        memcpy(this->pq_data + j_pq_offset, j_pq_coords.data(),
               this->pq_nchunks * sizeof(uint8_t));
      }
    }

    diskann::cout << "Finished deflating all points\n";
    double   e2e_time = ((double) total_insert_timer.elapsed()) / (1000000.0);
    double   insert_time = std::accumulate(this->insert_times.begin(),
                                         this->insert_times.end(), 0.0);
    double   delta_time = std::accumulate(this->delta_times.begin(),
                                        this->delta_times.end(), 0.0);
    uint32_t n_inserts =
        std::accumulate(this->mem_npts.begin(), this->mem_npts.end(), 0);
    diskann::cout << "TIMER:: PQ time per point = " << insert_time / n_inserts
                  << ", Delta = " << delta_time / n_inserts << "\n";
    diskann::cout << " E2E pq time: " << e2e_time << " sec" << std::endl;
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::process_inserts() {
    Timer total_insert_timer;
    this->insert_times.resize(MAX_INSERT_THREADS, 0.0);
    this->delta_times.resize(MAX_INSERT_THREADS, 0.0);
    // iterate through each vector in each mem index
    for (uint32_t i = 0; i < this->mem_data.size(); i++) {
      diskann::cout << "Processing inserts from mem-DiskANN #" << i + 1 << "\n";
      const tsl::robin_set<uint32_t> &deleted_set = this->mem_deleted_ids[i];
      const T* coords = this->mem_data[i];
      const uint32_t offset = this->offset_ids[i];
      const uint32_t count = this->mem_npts[i];

      size_t cur_cache_size = 0;
#ifdef USE_TCMALLOC
      MallocExtension::instance()->GetNumericProperty(
          "tcmalloc.max_total_thread_cache_bytes", &cur_cache_size);
      //diskann::cout << "Current cache size : " << (cur_cache_size >> 10)
      //              << " KiB\n"
      //              << std::endl;
      MallocExtension::instance()->SetNumericProperty(
          "tcmalloc.max_total_thread_cache_bytes", 128 * 1024 * 1024);

#endif
      diskann::Timer timer;
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_INSERT_THREADS)
      // iteratively insert each point into full index
      for (int32_t j = 0; j < (int32_t) count; j++) {
        // filter out -- `j` is deleted
        if (deleted_set.find((uint32_t) j) != deleted_set.end()) {
          continue;
        }

        if (((j % 100000) == 0) && (j > 0)) {
          diskann::cout << "Finished inserting " << j << " points" << std::endl;
          std::cout << "When j = " << j
                    << " elapsed time: " << timer.elapsed() / 1000000 << "s"
                    << std::endl;
        }
        // data for jth point
        const T *j_coords =
            coords + ((uint64_t)(this->aligned_ndims) * (uint64_t) j);
        const uint32_t j_id = offset + (uint32_t) j;

        // insert into index
        this->insert_mem_vec(j_coords, j_id);
      }
    }

    diskann::cout << "Finished inserting all points\n";
    double   e2e_time = ((double) total_insert_timer.elapsed()) / (1000000.0);
    double   insert_time = std::accumulate(this->insert_times.begin(),
                                         this->insert_times.end(), 0.0);
    double   delta_time = std::accumulate(this->delta_times.begin(),
                                        this->delta_times.end(), 0.0);
    uint32_t n_inserts =
        std::accumulate(this->mem_npts.begin(), this->mem_npts.end(), 0);
    diskann::cout << "TIMER:: Insert time per point = "
                  << insert_time / n_inserts
                  << ", Delta = " << delta_time / n_inserts << "\n";
    diskann::cout << " E2E insert time: " << e2e_time << " sec" << std::endl;
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::insert_mem_vec(const T *      mem_vec,
                                                const uint32_t offset_id) {
    Timer timer;
    float insert_time, delta_time;
    // START: mem_vec has no ID, no presence in system
    std::vector<Neighbor>         pool;
    std::vector<Neighbor>         tmp;
    tsl::robin_map<uint32_t, T *> coord_map;

    // std::cout << "TID: " << std::this_thread::get_id()
    //          << " Before offset_iterate_to_Fixed_point()" << std::endl;
    // search on combined graph
    this->offset_iterate_to_fixed_point(mem_vec, this->l_index, pool,
                                        coord_map);
    insert_time = (float) timer.elapsed();

    // prune neighbors using alpha
    std::vector<uint32_t> new_nhood;
    prune_neighbors(coord_map, pool, new_nhood);

    if (new_nhood.size() > range) {
      std::cout << "***ERROR*** After prune, for offset_id: " << offset_id << " found "
                << new_nhood.size() << " neighbors instead of range: " << range
                << std::endl;
    
    }

    //this->disk_delta->insert_vector(offset_id, new_nhood.data(),
    //                                (_u32) new_nhood.size());
    this->disk_delta->inter_insert(offset_id, new_nhood.data(),
                                   (_u32) new_nhood.size());

    // insert into graph
    for (auto &delta : this->mem_deltas) {
      delta->insert_vector(offset_id, new_nhood.data(),
                           (_u32) new_nhood.size());
      //delta->inter_insert(offset_id, new_nhood.data(), (_u32) new_nhood.size());
    }
    delta_time = (float) timer.elapsed();
    // END: mem_vec now connected with new ID
    uint32_t thread_no = omp_get_thread_num();
    this->insert_times[thread_no] += insert_time;
    this->delta_times[thread_no] += delta_time;
    // std::cout << "TID: " << std::this_thread::get_id()
    //          << " Exiting insert_mem_vec() " << std::endl;
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::offset_iterate_to_fixed_point(
      const T *vec, const uint32_t Lsize,
      std::vector<Neighbor> &        expanded_nodes_info,
      tsl::robin_map<uint32_t, T *> &coord_map) {
    std::vector<Neighbor> exp_node_info;
    exp_node_info.reserve(2 * Lsize);
    tsl::robin_map<uint32_t, T *> cmap;
    // first hit PQ iterate to fixed point
    // NOTE :: handling deletes for disk-index inside this call
    // this->disk_iterate_to_fixed_point(vec, this->l_index, exp_node_info,
    // exp_node_id, best_l_nodes, cmap);
    uint32_t       omp_thread_no = omp_get_thread_num();
    if (this->disk_thread_data.size() <= omp_thread_no) {
      throw ANNException(std::string("Found ") + std::to_string(omp_thread_no) +
                             " thread when only " +
                             std::to_string(this->disk_thread_data.size()) + " were expected",
                         -1);
    }
    ThreadData<T> &thread_data = this->disk_thread_data[omp_thread_no];
    //    ThreadData<T> * thread_data = nullptr;

    cmap.reserve(2 * Lsize);
    this->disk_index->disk_iterate_to_fixed_point(
        vec, Lsize, this->beam_width, exp_node_info, &cmap, nullptr,
        &thread_data, &this->disk_deleted_ids);

    // reduce and pick top maxc expanded nodes only
    std::sort(exp_node_info.begin(), exp_node_info.end());
    //    expanded_nodes_info.clear();
    expanded_nodes_info.reserve(this->maxc);
    expanded_nodes_info.insert(expanded_nodes_info.end(), exp_node_info.begin(),
                               exp_node_info.end());

    // insert only relevant coords into coord_map
    for (auto &nbr : expanded_nodes_info) {
      uint32_t id = nbr.id;
      auto     iter = cmap.find(id);
      assert(iter != cmap.end());
      coord_map.insert(std::make_pair(iter->first, iter->second));
    }
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::prune_neighbors(
      const tsl::robin_map<uint32_t, T *> &coord_map,
      std::vector<Neighbor> &pool, std::vector<uint32_t> &pruned_list) {
    if (pool.size() == 0)
      return;

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list(pool, coord_map, result, occlude_factor);

    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      pruned_list.emplace_back(iter.id);
    }

    if (alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if (std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) ==
            pruned_list.end())
          pruned_list.emplace_back(pool[i].id);
      }
    }
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::prune_neighbors_pq(
      std::vector<Neighbor> &pool, std::vector<uint32_t> &pruned_list,
      uint8_t *scratch) {
    if (pool.size() == 0)
      return;

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(this->range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list_pq(pool, result, occlude_factor, scratch);

    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      pruned_list.emplace_back(iter.id);
    }

    if (alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if (std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) ==
            pruned_list.end())
          pruned_list.emplace_back(pool[i].id);
      }
    }
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::occlude_list(
      std::vector<Neighbor> &              pool,
      const tsl::robin_map<uint32_t, T *> &coord_map,
      std::vector<Neighbor> &result, std::vector<float> &occlude_factor) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < range) {
      uint32_t start = 0;
      while (result.size() < range && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result.push_back(p);
        for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
          auto iter_right = coord_map.find(p.id);
          auto iter_left = coord_map.find(pool[t].id);
          // HAS to be in coord_map since it was expanded during
          // iterate_to_fixed_point
          assert(iter_right != coord_map.end());
          assert(iter_left != coord_map.end());
          // WARNING :: correct, but not fast -- NO SIMD version if using MSVC,
          // g++ should auto vectorize
          float djk = this->dist_cmp->compare(iter_left->second,
                                              iter_right->second, this->ndims);
          occlude_factor[t] =
              (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2f;
    }
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::occlude_list_pq(
      std::vector<Neighbor> &pool, std::vector<Neighbor> &result,
      std::vector<float> &occlude_factor, uint8_t *scratch) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < range) {
      uint32_t start = 0;
      while (result.size() < range && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result.push_back(p);
        for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
          // djk = dist(p.id, pool[t.id])
          float djk;
          this->disk_index->compute_pq_dists(p.id, &(pool[t].id), &djk, 1,
                                             scratch);
          occlude_factor[t] =
              (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2f;
    }
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::dump_to_disk(const uint32_t start_id,
                                              const char *   buf,
                                              const uint32_t n_sectors,
                                              std::ofstream & output_writer) {
    assert(start_id % this->nnodes_per_sector == 0);
    uint32_t start_sector = (start_id / this->nnodes_per_sector) + 1;
    uint64_t start_off = start_sector * (uint64_t) SECTOR_LEN;

    // seek fp
    output_writer.seekp(start_off, std::ios::beg);

    // dump
    output_writer.write(buf, (uint64_t) n_sectors * (uint64_t) SECTOR_LEN);

    uint64_t nb_written =
        (uint64_t) output_writer.tellp() - (uint64_t) start_off;
    if (nb_written != (uint64_t) n_sectors * (uint64_t) SECTOR_LEN) {
      std::stringstream sstream;
      sstream << "ERROR!!! Wrote " << nb_written << " bytes to disk instead of "
              << ((uint64_t) n_sectors) * SECTOR_LEN;
      diskann::cerr << sstream.str() << std::endl;
      throw diskann::ANNException(sstream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::compute_deleted_ids() {
    // process disk deleted tags
    for (uint32_t i = 0; i < this->disk_npts; i++) {
      TagT i_tag = this->disk_tags[i];
      if (this->deleted_tags.find(i_tag) != this->deleted_tags.end()) {
        this->disk_deleted_ids.insert(i);
      }
    }
    diskann::cout << "Found " << this->disk_deleted_ids.size()
                  << " tags to delete from SSD-DiskANN\n";

//    this->mem_deleted_ids.resize(this->mem_data.size());
    for (uint32_t i = 0; i < this->mem_data.size(); i++) {
      tsl::robin_set<uint32_t> &deleted_ids = this->mem_deleted_ids[i];
      for (uint32_t id = 0; id < this->mem_npts[i]; id++) {
          if(deleted_ids.find(id) != deleted_ids.end())
              continue;
        const TagT tag = this->mem_tags[i][id];
        //if (this->deleted_tags.find(tag) != this->deleted_tags.end()) {
        //  deleted_ids.insert(id);
        //}
        if (this->latter_deleted_tags[i].find(tag) !=
            this->latter_deleted_tags[i].end()) {
          deleted_ids.insert(id);
        }
      }
      diskann::cout << "Found " << deleted_ids.size()
                    << " tags to delete from mem-DiskANN #" << i + 1 << "\n";
    }
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::process_deletes() {
    // buf to hold data being read
    char *buf = nullptr;
    alloc_aligned((void **) &buf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);

    // open output file for writing
    // Was: this->disk_index_out_path + "_disk.index";
    
    diskann::cout << "Writing delete consolidated graph to "
                  << this->temp_disk_index_path << std::endl;
    std::ofstream output_writer(this->temp_disk_index_path, 
                                std::ios::out | std::ios::binary);
    assert(output_writer.is_open());
    // skip writing header for now
    //    this->output_writer.seekp(SECTOR_LEN, std::ios::beg);
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
    output_writer.write(sector_buf.get(), SECTOR_LEN);

    Timer delete_timer;
    // batch consolidate deletes
    std::vector<DiskNode<T>>                                disk_nodes;
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> id_nhoods;
    uint32_t start_id = 0, new_start_id;
    diskann::cout << "Consolidating deletes\n";

    while (start_id < this->disk_npts) {
      new_start_id = this->disk_index->merge_read(disk_nodes, start_id,
                                                  SECTORS_PER_MERGE, buf);
#pragma omp parallel for schedule(dynamic, 128) num_threads(MAX_N_THREADS)
      for (int64_t i = 0; i < (int64_t) disk_nodes.size(); i++) {
        // get thread-specific scratch
        int      omp_thread_no = omp_get_thread_num();
        uint8_t *pq_coord_scratch = this->thread_bufs[omp_thread_no];
        assert(pq_coord_scratch != nullptr);
        DiskNode<T> &disk_node = disk_nodes[i];
        this->consolidate_deletes(disk_node, pq_coord_scratch);
      }
      for (auto &disk_node : disk_nodes) {
        if (this->is_deleted(disk_node)) {
          this->free_ids.push_back(disk_node.id);
        }
      }

      uint64_t prev_pos = output_writer.tellp();
      this->dump_to_disk(start_id, buf, SECTORS_PER_MERGE, output_writer);
      output_writer.flush();
      uint64_t cur_pos = output_writer.tellp();
      if (!(cur_pos - prev_pos == (SECTORS_PER_MERGE * SECTOR_LEN)))
        diskann::cout << "ERROR! During dumping to disk" << std::endl;

      // advance to next block
      disk_nodes.clear();
      id_nhoods.clear();
      diskann::cout << new_start_id << " / " << this->disk_npts
                    << " nodes processed.\n";
      start_id = new_start_id;

    }
    double e2e_time = ((double) delete_timer.elapsed()) / (1000000.0);
    diskann::cout << "Processed Deletes in " << e2e_time << " s." << std::endl;
    diskann::cout << "Writing header.\n";

    // write header
    output_writer.seekp(0, std::ios::beg);
    // HEADER --> [_u32 #metadata items][_u32 1][_u64 nnodes][_u64 ndims][_u64
    // medoid ID]
    // [_u64 max_node_len][_u64 nnodes_per_sector][_u64 #frozen points in vamana
    // index][_u64 frozen point location][_u64 file size]
    uint64_t file_size =
        SECTOR_LEN + (ROUND_UP(ROUND_UP(this->disk_npts, nnodes_per_sector) /
                                   nnodes_per_sector,
                               SECTORS_PER_MERGE)) *
                         (uint64_t) SECTOR_LEN;
    std::vector<uint64_t> output_metadata;
    output_metadata.push_back(this->disk_npts);
    output_metadata.push_back((uint64_t) this->ndims);
    // determine medoid
    uint64_t medoid = this->init_ids[0];
    // TODO (correct?, misc) :: better way of selecting new medoid
    while (this->disk_deleted_ids.find((_u32) medoid) !=
           this->disk_deleted_ids.end()) {
      diskann::cout << "Medoid deleted. Choosing another start node.\n";
      auto iter = this->disk_deleted_nhoods.find((_u32) medoid);
      assert(iter != this->disk_deleted_nhoods.end());
      medoid = iter->second[0];
    }
    output_metadata.push_back((uint64_t) medoid);
    uint64_t max_node_len = (this->ndims * sizeof(T)) + sizeof(uint32_t) +
                            (this->range * sizeof(uint32_t));
    uint64_t nnodes_per_sector = SECTOR_LEN / max_node_len;
    output_metadata.push_back(max_node_len);
    output_metadata.push_back(nnodes_per_sector);
    output_metadata.push_back(this->disk_index_num_frozen);
    output_metadata.push_back(this->disk_index_frozen_loc);
    output_metadata.push_back(file_size);

    // close index
    output_writer.close();
    diskann::save_bin<_u64>(this->temp_disk_index_path, output_metadata.data(),
                            output_metadata.size(), 1, 0);
    // free buf
    aligned_free((void *) buf);

    // free backing buf for deletes
    aligned_free((void *) this->delete_backing_buf);
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::populate_deleted_nhoods() {
    // buf for scratch
    char *buf = nullptr;
    alloc_aligned((void **) &buf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);

    // scan deleted nodes and get
    std::vector<DiskNode<T>> deleted_nodes;
    uint64_t backing_buf_size = (uint64_t) this->disk_deleted_ids.size() *
                                ROUND_UP(this->max_node_len, 32);
    backing_buf_size = ROUND_UP(backing_buf_size, 256);
    diskann::cout << "ALLOC: " << (backing_buf_size << 10)
                  << "KiB aligned buffer for deletes.\n";
    alloc_aligned((void **) &this->delete_backing_buf, backing_buf_size, 256);
    memset(this->delete_backing_buf, 0, backing_buf_size);
    this->disk_index->scan_deleted_nodes(this->disk_deleted_ids, deleted_nodes,
                                         buf, this->delete_backing_buf,
                                         SECTORS_PER_MERGE);

    // insert into deleted_nhoods
    this->disk_deleted_nhoods.clear();
    this->disk_deleted_nhoods.reserve(deleted_nodes.size());
    for (auto &nhood : deleted_nodes) {
      // WARNING :: ASSUMING DISK GRAPH DEGREE NEVER GOES OVER 512
      assert(nhood.nnbrs < 512);
      std::vector<uint32_t> non_deleted_nbrs;
      for (uint32_t i = 0; i < nhood.nnbrs; i++) {
        uint32_t id = nhood.nbrs[i];
        auto     iter = this->disk_deleted_ids.find(id);
        if (iter == this->disk_deleted_ids.end()) {
          non_deleted_nbrs.push_back(id);
        }
      }
      this->disk_deleted_nhoods.insert(
          std::make_pair(nhood.id, non_deleted_nbrs));
    }

    // free buf
    aligned_free((void *) buf);
    assert(deleted_nodes.size() == this->disk_deleted_ids.size());
    assert(this->disk_deleted_nhoods.size() == this->disk_deleted_ids.size());
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::consolidate_deletes(DiskNode<T> &disk_node,
                                                     uint8_t *    scratch) {
    // if node is deleted
    if (this->is_deleted(disk_node)) {
      disk_node.nnbrs = 0;
      *(disk_node.nbrs - 1) = 0;
      return;
    }

    const uint32_t id = disk_node.id;

    assert(disk_node.nnbrs < 512);


    std::vector<uint32_t> id_nhood(disk_node.nbrs,
                                   disk_node.nbrs + disk_node.nnbrs);

    tsl::robin_set<uint32_t> new_edges;

    bool change = false;
    for (auto &nbr : id_nhood) {
      auto iter = this->disk_deleted_nhoods.find(nbr);
      if (iter != this->disk_deleted_nhoods.end()) {
        change = true;
        new_edges.insert(iter->second.begin(), iter->second.end());
      } else {
        new_edges.insert(nbr);
      }
    }
    // no refs to deleted nodes --> move to next node
    if (!change) {
      return;
    }

    // refs to deleted nodes
    id_nhood.clear();
    id_nhood.reserve(new_edges.size());
    for (auto &nbr : new_edges) {
      // 2nd order deleted edge
      auto iter = this->disk_deleted_ids.find(nbr);
      if (iter != this->disk_deleted_ids.end()) {
        continue;
      } else {
        id_nhood.push_back(nbr);
      }
    }

    // TODO (corner case) :: id_nhood might be empty in adversarial cases
    if (id_nhood.empty()) {
      diskann::cout << "Adversarial case -- all neighbors of node's neighbors "
                       "deleted -- ID : "
                    << id << "; exiting\n";
      exit(-1);
    }

    // compute PQ dists and shrink
    std::vector<float> id_nhood_dists(id_nhood.size(), 0.0f);
    assert(scratch != nullptr);
    this->disk_index->compute_pq_dists(id, id_nhood.data(),
                                       id_nhood_dists.data(),
                                       (_u32) id_nhood.size(), scratch);

    // prune neighbor list using PQ distances
    std::vector<Neighbor> cand_nbrs(id_nhood.size());
    for (uint32_t i = 0; i < id_nhood.size(); i++) {
      cand_nbrs[i].id = id_nhood[i];
      //      auto iter = this->disk_deleted_ids.find(id_nhood[i]);
      //      assert(iter == this->disk_deleted_ids.end());
      cand_nbrs[i].distance = id_nhood_dists[i];
    }
    // sort and keep only maxc neighbors
    std::sort(cand_nbrs.begin(), cand_nbrs.end());
    if (cand_nbrs.size() > this->maxc) {
      cand_nbrs.resize(this->maxc);
    }
    std::vector<Neighbor> pruned_nbrs;
    std::vector<float>    occlude_factor(cand_nbrs.size(), 0.0f);
    pruned_nbrs.reserve(this->range);
    this->occlude_list_pq(cand_nbrs, pruned_nbrs, occlude_factor, scratch);

    // copy back final nbrs
    disk_node.nnbrs = (_u32) pruned_nbrs.size();
    *(disk_node.nbrs - 1) = disk_node.nnbrs;
    for (uint32_t i = 0; i < (_u32) pruned_nbrs.size(); i++) {
      disk_node.nbrs[i] = pruned_nbrs[i].id;
      //     auto iter = this->disk_deleted_ids.find(disk_node.nbrs[i]);
      //      assert(iter == this->disk_deleted_ids.end());
    }
  }

  template<typename T, typename TagT>
  bool StreamingMerger<T, TagT>::is_deleted(const DiskNode<T> &disk_node) {
    // short circuit when disk_node is a `hole` on disk
    if (this->disk_tags[disk_node.id] == std::numeric_limits<uint32_t>::max()) {
      if (disk_node.nnbrs != 0) {
        throw diskann::ANNException(std::string("Found node with id: ") +
                                        std::to_string(disk_node.id) +
                                        " that has non-zero degree.",
                                    -1, __FUNCSIG__, __FILE__, __LINE__);
        diskann::cerr << "Node with id " << disk_node.id
                      << " is a hole but has non-zero degree "
                      << disk_node.nnbrs << std::endl;
      } else {
        return true;
      }
    }
    return (this->disk_deleted_ids.find(disk_node.id) !=
            this->disk_deleted_ids.end());
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::compute_rename_map() {
    uint32_t needed = 0;
    for (auto &mem_npt : this->mem_npts) {
      needed += mem_npt;
    }
    for (auto &del_set : this->mem_deleted_ids) {
      needed -= (_u32) del_set.size();
    }
    diskann::cout << "RENAME: Need " << needed
                  << ", free: " << this->free_ids.size() << "\n";

    uint32_t last_id = this->disk_npts;
    if (needed > this->free_ids.size()) {
      this->free_ids.reserve(needed);
    }
    while (this->free_ids.size() < needed) {
      this->free_ids.push_back(last_id);
      last_id++;
    }

    // assign free IDs to all new IDs
    diskann::cout << "RENAME: Assigning IDs.\n";
    uint32_t next_free_index = 0;
    this->rename_map.reserve(needed);
    this->inverse_map.reserve(needed);
    std::vector<std::pair<uint32_t, uint32_t>> rename_pairs(needed);
    std::vector<std::pair<uint32_t, uint32_t>> inverse_pairs(needed);
    for (uint32_t mem_id = 0; mem_id < this->mem_data.size(); mem_id++) {
      diskann::cout << "Processing Mem-DiskANN #" << mem_id + 1 << "\n";
      uint32_t                        offset = this->offset_ids[mem_id];
      const tsl::robin_set<uint32_t> &del_set = this->mem_deleted_ids[mem_id];
      std::vector<bool>               deleted(this->mem_npts[mem_id], false);
      for (auto &id : del_set) {
        deleted[id] = true;
      }
      for (uint32_t j = 0; j < this->mem_npts[mem_id]; j++) {
        // ignore any deleted points
        if (deleted[j]) {
          continue;
        }
        const uint32_t new_id = this->free_ids[next_free_index];
        assert(new_id < last_id);
        rename_pairs[next_free_index].first = offset + j;
        rename_pairs[next_free_index].second = new_id;
        inverse_pairs[next_free_index].first = new_id;
        inverse_pairs[next_free_index].second = offset + j;
        next_free_index++;
      }
    }
    diskann::cout << "RENAME: Storing mappings for " << next_free_index
                  << " points.\n";
    this->rename_list.clear();
    this->rename_list.reserve(next_free_index);
    this->rename_list.insert(this->rename_list.end(), rename_pairs.begin(),
                             rename_pairs.end());
    this->inverse_list.clear();
    this->inverse_list.reserve(next_free_index);
    this->inverse_list.insert(this->inverse_list.end(), inverse_pairs.begin(),
                              inverse_pairs.end());
  }

  template<typename T, typename TagT>
  uint32_t StreamingMerger<T, TagT>::rename(uint32_t id) const {
    auto iter = std::lower_bound(
        this->rename_list.begin(), this->rename_list.end(),
        std::make_pair(id, std::numeric_limits<uint32_t>::max()),
        [](const auto &left, const auto &right) {
          return left.first < right.first;
        });
    if (iter == this->rename_list.end()) {
      return std::numeric_limits<uint32_t>::max();
    } else {
      uint32_t idx = (_u32) std::distance(this->rename_list.begin(), iter);
      const std::pair<uint32_t, uint32_t> &p = this->rename_list[idx];
      if (p.first == id)
        return p.second;
      else
        return std::numeric_limits<uint32_t>::max();
    }
  }

  template<typename T, typename TagT>
  uint32_t StreamingMerger<T, TagT>::rename_inverse(uint32_t renamed_id) const {
    auto iter = std::lower_bound(
        this->inverse_list.begin(), this->inverse_list.end(),
        std::make_pair(renamed_id, std::numeric_limits<uint32_t>::max()),
        [](const auto &left, const auto &right) {
          return left.first < right.first;
        });
    if (iter == this->inverse_list.end()) {
      return std::numeric_limits<uint32_t>::max();
    } else {
      uint32_t idx = (_u32) std::distance(this->inverse_list.begin(), iter);
      const std::pair<uint32_t, uint32_t> &p = this->inverse_list[idx];
      if (p.first == renamed_id)
        return p.second;
      else
        return std::numeric_limits<uint32_t>::max();
    }
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::rename(DiskNode<T> &node) const {
    uint32_t renamed_id = this->rename(node.id);
    if (renamed_id != std::numeric_limits<uint32_t>::max()) {
      node.id = renamed_id;
    }
    uint32_t nnbrs = node.nnbrs;
    for (uint32_t i = 0; i < nnbrs; i++) {
      uint32_t renamed_nbr_i = this->rename(node.nbrs[i]);
      if (renamed_nbr_i != std::numeric_limits<uint32_t>::max()) {
        node.nbrs[i] = renamed_nbr_i;
      }
    }
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::rename(std::vector<uint32_t> &ids) const {
    for (uint32_t i = 0; i < ids.size(); i++) {
      uint32_t renamed_id = this->rename(ids[i]);
      if (renamed_id != std::numeric_limits<uint32_t>::max()) {
        ids[i] = renamed_id;
      }
    }
  }

  template<typename T, typename TagT>
  uint32_t StreamingMerger<T, TagT>::get_index_id(
      const uint32_t offset_id) const {
    if (offset_id < this->offset_ids[0]) {
      return std::numeric_limits<uint32_t>::max();
    }
    // should not happen unless some buffer is corrupted
    if (offset_id > this->offset_ids.back() + INDEX_OFFSET) {
      diskann::cout << "Seen: " << offset_id << ", min: " << offset_ids[0]
                    << ", max: " << offset_ids.back() << "\n";
    }
    assert(offset_id < this->offset_ids.back() + INDEX_OFFSET);
    uint32_t index_no =
        (uint32_t)((offset_id - this->offset_ids[0]) / INDEX_OFFSET);
    assert(index_no < this->offset_ids.size());
    return index_no;
  }

  template<typename T, typename TagT>
  std::vector<uint32_t> StreamingMerger<T, TagT>::get_edge_list(
      const uint32_t offset_id) {
    const uint32_t index_no = this->get_index_id(offset_id);
    if (index_no == std::numeric_limits<uint32_t>::max()) {
      assert(offset_id < this->offset_ids[0]);
      return this->disk_delta->get_nhood(offset_id);
    }
    //    uint32_t local_id = offset_id - this->offset_ids[index_no];
    //    assert(local_id < this->mem_npts[index_no]);
    std::vector<uint32_t> ret =
        this->mem_deltas[index_no]->get_nhood(offset_id);
    // this->rename(ret);
    return ret;
  }

  template<typename T, typename TagT>
  const T *StreamingMerger<T, TagT>::get_mem_data(const uint32_t offset_id) {
    const uint32_t index_no = this->get_index_id(offset_id);
    if (index_no == std::numeric_limits<uint32_t>::max()) {
      assert(offset_id < this->offset_ids[0]);
      return nullptr;
    }
    uint32_t local_id = offset_id - this->offset_ids[index_no];
    assert(local_id < this->mem_npts[index_no]);
    return this->mem_data[index_no] +
           ((uint64_t) local_id * (uint64_t) this->aligned_ndims);
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::write_tag_file(
      const std::string &tag_out_filename, const uint32_t npts) {
    diskann::Timer timer;
    diskann::cout << "Writing new tags to " << tag_out_filename << "\n";

    TagT *cur_tags;

    size_t allocSize = npts * sizeof(TagT);
    alloc_aligned(((void **) &cur_tags), allocSize, 8 * sizeof(TagT));

    //TODO: We must detect holes in a better way. Currently, it is possible
    //that one of the tags will be uint32_t::max() and will fail.
    for (uint32_t i = 0; i < npts; i++) {
      TagT cur_tag;
      // check if `i` is in inverse map
      const uint32_t offset_id = this->rename_inverse(i);
      if (offset_id == std::numeric_limits<uint32_t>::max()) {
        cur_tag = this->disk_tags[i];
        if (this->deleted_tags.find(cur_tag) != this->deleted_tags.end()) {
          *(cur_tags + i) = std::numeric_limits<uint32_t>::max();
        } else
          *(cur_tags + i) = cur_tag;
      } else {
        const uint32_t index_no = this->get_index_id(offset_id);
        const uint32_t index_local_id = offset_id - this->offset_ids[index_no];
        cur_tag = this->mem_tags[index_no][index_local_id];
        if (this->latter_deleted_tags[index_no].find(cur_tag) !=
            this->latter_deleted_tags[index_no].end()) {
          *(cur_tags + i) = std::numeric_limits<uint32_t>::max();
        } else
          *(cur_tags + i) = cur_tag;
      }
    }
    diskann::save_bin<TagT>(tag_out_filename, cur_tags, npts, 1);

    diskann::cout << "Tags written to  " << tag_out_filename << " in " << timer.elapsed()
                  << " microsec" << std::endl;

    //Should not mix delete with alloc aligned
    //TODO: This will work because we are dealing with uint64 at the moment. 
    //If we ever have string tags, this'll fail spectacularly.
    //delete[] cur_tags;
    aligned_free(cur_tags);
    // release all tags -- automatically deleted since using `unique_ptr`
    this->mem_tags.clear();
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::process_merges() {
    // buf to hold data being read
    char *buf = nullptr;
    alloc_aligned((void **) &buf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);

    Timer                   merge_timer;
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);

    std::ofstream output_writer(this->final_index_file,
                                std::ios::binary | std::ios::ate);

    output_writer.write(sector_buf.get(), SECTOR_LEN);
    // start at sector=1 in output file for reading + writing
    uint64_t cur_offset = SECTOR_LEN;

    // batch consolidate deletes
    std::vector<DiskNode<T>>                                disk_nodes;
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> id_nhoods;
    uint32_t start_id = 0, new_start_id;
    diskann::cout << "Merging inserts into SSD-DiskANN.\n";
    uint64_t              delta_avg = 0, delta_max = 0, delta_count = 0;
    std::atomic<uint64_t> counts;
    counts.store(0);

    // aux_writer << "buf_min = " << (uint64_t) buf << ", buf_max = " <<
    // (uint64_t) (buf + SECTORS_PER_MERGE * SECTOR_LEN) << "\n";
    while (start_id < this->disk_npts) {
      // zero buf for better consistency
      memset(buf, 0, SECTORS_PER_MERGE * SECTOR_LEN);
      new_start_id = this->disk_index->merge_read(disk_nodes, start_id,
                                                  SECTORS_PER_MERGE, buf);
#pragma omp parallel for schedule(dynamic, 128) num_threads(MAX_N_THREADS)
      for (int64_t idx = 0; idx < (int64_t) disk_nodes.size(); idx++) {
        // get thread-specific scratch
        int      omp_thread_no = omp_get_thread_num();
        uint8_t *thread_scratch = this->thread_bufs[omp_thread_no];

        DiskNode<T> &disk_node = disk_nodes[idx];
        uint32_t     id = disk_node.id;

        std::vector<uint32_t> nhood;
        std::vector<uint32_t> deltas;
        uint32_t              offset_id = this->rename_inverse(id);
        // replaced by new vector, copy coords and proceed as normal
        if (offset_id != std::numeric_limits<uint32_t>::max()) {
          // copy coords
          const T *vec = this->get_mem_data(offset_id);
          assert(vec != nullptr);
          memcpy(disk_node.coords, vec, this->ndims * sizeof(T));
          disk_node.nnbrs = 0;
          *(disk_node.nbrs - 1) = 0;  // also set on buffer
          deltas = this->get_edge_list(offset_id);
          // delta_count++;
          // delta_avg += deltas.size();
          // delta_max = std::max(delta_max, (uint64_t) deltas.size());
        } else {
          // not replaced
          deltas = this->get_edge_list(id);
          delta_count++;
          delta_avg += deltas.size();
          delta_max = std::max(delta_max, (uint64_t) deltas.size());
        }

        // if no edges to add, continue
        if (deltas.empty()) {
          continue;
        }

        uint32_t nnbrs = disk_node.nnbrs;
        nhood.insert(nhood.end(), disk_node.nbrs, disk_node.nbrs + nnbrs);
        nhood.insert(nhood.end(), deltas.begin(), deltas.end());
        // rename nbrs in nhood to use PQ dist comparisons
        // this->rename(nhood); // skipping since get_edge_list() renames delta
        // edges

        // prune neighbor list ONLY if exceeding graph max out-degree (`range`)
        if (nhood.size() > this->range) {
          std::vector<float>    dists(nhood.size(), 0.0f);
          std::vector<Neighbor> pool(nhood.size());
          this->disk_index->compute_pq_dists(id, nhood.data(), dists.data(),
                                             (_u32) nhood.size(),
                                             thread_scratch);
          for (uint32_t k = 0; k < nhood.size(); k++) {
            pool[k].id = nhood[k];
            pool[k].distance = dists[k];
            // diskann::cout << k << ": id=" << nhood[k] << ", dist=" <<
            // dists[k] << "\n";
          }
          nhood.clear();
          // prune pool
          std::sort(pool.begin(), pool.end());
          this->prune_neighbors_pq(pool, nhood, thread_scratch);
        }
        // copy edges from nhood to disk node
        disk_node.nnbrs = (_u32) nhood.size();
        // *(disk_node.nbrs - 1) = nhood.size(); // write to buf
        *(disk_node.nbrs - 1) = (_u32) nhood.size();  // write to buf
        for (uint32_t i = 0; i < disk_node.nnbrs; i++) {
          disk_node.nbrs[i] = nhood[i];
        }
        memcpy(disk_node.nbrs, nhood.data(),
               disk_node.nnbrs * sizeof(uint32_t));
        // 	aux_writer << "disk_node.nbrs =  " << (uint64_t) disk_node.nbrs <<
        // ", id = ";
        uint32_t lcounts = 0;
        for (auto &nbr : nhood) {
          if (nbr >= 980000)
            lcounts++;
        }
        counts += lcounts;
      }

      cur_offset += SECTORS_PER_MERGE * SECTOR_LEN;
      output_writer.write(buf, SECTORS_PER_MERGE * SECTOR_LEN);
      diskann::cout << new_start_id << " / " << this->disk_npts
                    << " nodes processed.\n";
      start_id = new_start_id;
    }
    diskann::cout << "Delta statistics:\nMax: " << delta_max
                  << ", Avg: " << (float) delta_avg / (float) delta_count
                  << "\n";
    diskann::cout << "Old -> new edges: " << counts.load() << "\n";

    // close index
    output_writer.close();

    // [_u64 file size][_u64 nnodes][_u64 medoid ID][_u64 max_node_len][_u64
    // nnodes_per_sector]
    uint64_t file_size =
        SECTOR_LEN + (ROUND_UP(ROUND_UP(this->disk_npts, nnodes_per_sector) /
                                   nnodes_per_sector,
                               SECTORS_PER_MERGE)) *
                         (uint64_t) SECTOR_LEN;
    memset(buf, 0, SECTORS_PER_MERGE * SECTOR_LEN);
    std::vector<uint64_t> output_metadata;
    output_metadata.push_back((uint64_t) this->disk_npts);
    output_metadata.push_back((uint64_t) this->ndims);
    // determine medoid
    uint64_t medoid = this->init_ids[0];
    output_metadata.push_back((uint64_t) medoid);
    uint64_t max_node_len = this->ndims * sizeof(T) + sizeof(uint32_t) +
                            this->range * sizeof(uint32_t);
    uint64_t nnodes_per_sector = SECTOR_LEN / max_node_len;
    output_metadata.push_back(max_node_len);
    output_metadata.push_back(nnodes_per_sector);
    output_metadata.push_back(this->disk_index_num_frozen);
    output_metadata.push_back(this->disk_index_frozen_loc);
    output_metadata.push_back(file_size);

    diskann::save_bin<_u64>(final_index_file, output_metadata.data(),
                            output_metadata.size(), 1, 0);
    // free buf
    aligned_free((void *) buf);
    double e2e_time = ((double) merge_timer.elapsed()) / (1000000.0);
    diskann::cout << "Time to merge the inserts to disk: " << e2e_time << "s."
                  << std::endl;
  }


  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::merge(const char *                    disk_in,
                                       const std::vector<std::string> &mem_in,
                                       const char *                    disk_out,
                                       std::vector<const std::vector<TagT>*> &deleted_tags_vectors,
                                       std::string  &working_folder) {
    // load disk index
    this->disk_index_out_path = disk_out;
    this->disk_index_in_path = disk_in;
    this->TMP_FOLDER = working_folder;
    std::cout << "Working folder : " << working_folder << std::endl;
    this->temp_disk_index_path = getTempFilePath(working_folder, "temp_disk_index");
    this->temp_pq_coords_path = getTempFilePath(working_folder, "temp_pq_compressed");
    this->temp_tags_path = getTempFilePath(working_folder, "temp_tags");
    std::cout << this->temp_disk_index_path << " , "  << this->temp_pq_coords_path << "  ,  " << this->temp_tags_path << std::endl;
    this->final_index_file =
        this->_single_file_index ? this->disk_index_out_path
                                 : this->disk_index_out_path + "_disk.index";
    this->final_pq_coords_file =
        this->_single_file_index
            ? this->disk_index_out_path
            : this->disk_index_out_path + "_pq_compressed.bin";
    this->final_tags_file =
        this->_single_file_index
            ? this->disk_index_out_path
            : this->disk_index_out_path + "_disk.index.tags";


#ifndef _WINDOWS
    std::shared_ptr<AlignedFileReader> reader =
        std::make_shared<LinuxAlignedFileReader>();
#else
    std::shared_ptr<AlignedFileReader> reader =
        std::make_shared<BingAlignedFileReader>();
#endif

    //    std::shared_ptr<AlignedFileReader> reader =
    //    std::make_shared<MemAlignedFileReader>();
    this->disk_index = new PQFlashIndex<T, TagT>(
        this->dist_metric, reader, this->_single_file_index, true);
    diskann::cout << "Created PQFlashIndex inside index_merger " << std::endl;

    diskann::cout << "Loading PQFlashIndex from file: " << disk_in
                  << " into object: " << std::hex << (_u64) &
        (this->disk_index) << std::dec << std::endl;
    this->disk_index->load(disk_in, NUM_INDEX_LOAD_THREADS);

    uint32_t node_cache_count = 1 + (uint32_t) round(this->disk_index->return_nd() * 0.01);
    node_cache_count = node_cache_count > PQ_FLASH_INDEX_MAX_NODES_TO_CACHE
                           ? PQ_FLASH_INDEX_MAX_NODES_TO_CACHE
                           : node_cache_count;
    std::vector<uint32_t> cache_node_list;
    this->disk_index->cache_bfs_levels(node_cache_count,
                                       cache_node_list);
    this->disk_index->load_cache_list(cache_node_list);
    this->disk_tags = this->disk_index->get_tags();
    this->init_ids = this->disk_index->get_init_ids();
    this->disk_npts = (_u32) this->disk_index->return_nd();
    this->disk_thread_data = this->disk_index->get_thread_data();
    auto res_pq = this->disk_index->get_pq_config();
    this->pq_data = res_pq.first;
    this->pq_nchunks = res_pq.second;
    this->nnodes_per_sector = (_u32) this->disk_index->nnodes_per_sector;
    this->max_node_len = (_u32) this->disk_index->max_node_len;
    _u32 max_degree =
        (max_node_len - (sizeof(T) * this->ndims)) / sizeof(uint32_t) - 1;
    this->range = max_degree; 
    diskann::cout << "Setting range to: " << this->range << std::endl;
    this->disk_index_num_frozen = this->disk_index->get_num_frozen_points();
    this->disk_index_frozen_loc = this->disk_index->get_frozen_loc(); 

    // create deltas
    this->disk_delta = new GraphDelta(0, this->disk_npts);
    uint64_t base_offset = ROUND_UP(this->disk_npts, INDEX_OFFSET);

    // load mem-indices
    for (auto &mem_index_path : mem_in) {
      uint32_t npts;
      if (!(this->_single_file_index)) {
        std::string   ind_path = mem_index_path;
        std::string   data_path = mem_index_path + ".data";
        std::ifstream bin_reader(data_path, std::ios::binary);
        uint32_t      bin_npts, bin_ndims;
        bin_reader.read((char *) &bin_npts, sizeof(uint32_t));
        bin_reader.read((char *) &bin_ndims, sizeof(uint32_t));
        bin_reader.close();
        diskann::cout << "Index Path: " << ind_path << "\n";
        diskann::cout << "Data Path: " << data_path << "\n";
        diskann::cout << "Detected # pts = " << bin_npts
                      << ", # dims = " << bin_ndims << "\n";

        auto mem_index = std::make_unique<Index<T, TagT>>(
            this->dist_metric, bin_ndims, bin_npts + 100, true,
            this->_single_file_index, true, false);
        _u64 n1, n2, n3;
        T *  data_load;
        diskann::load_aligned_bin<T>(data_path, data_load, n1, n2, n3);
        npts = (_u32)(n1 - 1);
        assert(npts < MAX_PTS_PER_MEM_INDEX);
        this->mem_npts.push_back(npts);
        this->mem_data.push_back(data_load);
        uint32_t index_offset = (_u32) base_offset;
        base_offset += INDEX_OFFSET;
        this->offset_ids.push_back(index_offset);
        this->mem_deltas.push_back(new GraphDelta(index_offset, npts));
        tsl::robin_set<uint32_t> temp_del_set;
        if(file_exists(mem_index_path + ".del"))
        {
            mem_index->load_delete_set(mem_index_path + ".del");
            mem_index->get_delete_set(temp_del_set);
        }
        this->mem_deleted_ids.push_back(temp_del_set);
        mem_index->load_tags(mem_index_path + ".tags");
        // manage tags
        std::unique_ptr<TagT[]> index_tags;
        index_tags.reset(new TagT[npts]);

        const std::unordered_map<uint32_t, TagT> &loc_tag_map =
            *mem_index->get_tags();
        for (uint32_t k = 0; k < npts; k++) {
          auto iter = loc_tag_map.find(k);
          if (iter == loc_tag_map.end()) {
            index_tags[k] = (TagT) 0;
          } else {
            index_tags[k] = iter->second;
          }
        }
        this->mem_tags.push_back(std::move(index_tags));
      } else {
        // read metadata from single index file for npts and ndims
        _u64                    nr, nc;
        std::unique_ptr<_u64[]> file_offset_data;
        diskann::load_bin<_u64>(mem_index_path, file_offset_data, nr, nc, 0);

        size_t data_dim, data_num_points;
        diskann::get_bin_metadata(mem_index_path, data_num_points, data_dim,
                                  file_offset_data[1]);
        diskann::cout << "Detected # pts = " << data_num_points
                      << ", # dims = " << data_dim << "\n";
        diskann::cout << "Since vamana index is dynamic, it will have one "
                         "frozen point, hence #pts = "
                      << data_num_points - 1 << std::endl;

        // load mem_index_data with appropriate offset
        _u64 n1, n2, n3;
        T *  data_load;
        diskann::load_aligned_bin<T>(mem_index_path, data_load, n1, n2, n3,
                                     file_offset_data[1]);
        npts = (_u32)(n1 - 1);
        assert(npts < MAX_PTS_PER_MEM_INDEX);
        this->mem_npts.push_back(npts);
        this->mem_data.push_back(data_load);

        // call mem_index constructor with dynamic index and single index file
        // set to true
        auto mem_index = std::make_unique<Index<T, TagT>>(
            this->dist_metric, data_dim, data_num_points + 100, true,
            this->_single_file_index, true, false);

        // load tags with appropriate offset
        uint32_t index_offset = (_u32) base_offset;
        base_offset += INDEX_OFFSET;
        this->offset_ids.push_back(index_offset);
        this->mem_deltas.push_back(new GraphDelta(index_offset, npts));
        mem_index->load_tags(mem_index_path, file_offset_data[2]);
        // manage tags
        std::unique_ptr<TagT[]> index_tags;
        index_tags.reset(new TagT[npts]);

        const std::unordered_map<uint32_t, TagT> &loc_tag_map =
            *mem_index->get_tags();
        for (uint32_t k = 0; k < npts; k++) {
          auto iter = loc_tag_map.find(k);
          if (iter == loc_tag_map.end()) {
            diskann::cout << "Index # " << this->mem_data.size()
                          << " : missing tag for node #" << k << "\n";
            exit(-1);
            index_tags[k] = (TagT) k;
          } else {
            index_tags[k] = iter->second;
          }
        }
        this->mem_tags.push_back(std::move(index_tags));
      }
    }

#ifdef USE_TCMALLOC
    MallocExtension::instance()->ReleaseFreeMemory();
#endif

    for (size_t j = 0; j < deleted_tags_vectors.size(); j++) {
      this->latter_deleted_tags.push_back(tsl::robin_set<TagT>());
      for (size_t i = j+1; i < deleted_tags_vectors.size(); i++) {
        for (size_t k = 0; k < deleted_tags_vectors[i]->size(); k++) {
          this->latter_deleted_tags[j].insert((*deleted_tags_vectors[i])[k]);
        }
      }
    }

    //TODO: See if this can be included in the previous loop
    for (auto &deleted_tags_vector : deleted_tags_vectors) {
      for (size_t i = 0; i < deleted_tags_vector->size(); i++) {
        this->deleted_tags.insert((*deleted_tags_vector)[i]);
      }
    }

    diskann::cout << "Allocating thread scratch space -- "
                  << PER_THREAD_BUF_SIZE / (1 << 20) << " MB / thread.\n";
    alloc_aligned((void **) &this->thread_pq_scratch,
                  MAX_N_THREADS * PER_THREAD_BUF_SIZE, SECTOR_LEN);
    this->thread_bufs.resize(MAX_N_THREADS);
    for (uint32_t i = 0; i < thread_bufs.size(); i++) {
      this->thread_bufs[i] = this->thread_pq_scratch + i * PER_THREAD_BUF_SIZE;
    }

    mergeImpl();
  }

  template<typename T, typename TagT>
  void StreamingMerger<T, TagT>::mergeImpl() {
    // populate deleted IDs
    this->compute_deleted_ids();
    // BEGIN -- graph on disk has deleted references, maybe some holes
    // populate deleted nodes
    this->populate_deleted_nhoods();
    // process all deletes
    this->process_deletes();
    // END -- graph on disk has NO deleted references, maybe some holes

    diskann::cout << "Computing rename-map.\n";
    // compute rename map
    this->compute_rename_map();

    // get max ID + 1 in rename-map as new max pts
    uint32_t new_max_pts = this->disk_npts - 1;
    // alternative using list
    new_max_pts = std::max(this->inverse_list.back().first, new_max_pts);
    new_max_pts = new_max_pts + 1;

    // TODO (correct) :: figure out naming scheme
    //std::string new_disk_out(this->disk_index_out_path + "_disk.index");
    //diskann::cout << "RELOAD: Creating new disk graph at " << new_disk_out
    //              << "\n";
    //std::string new_pq_prefix(this->disk_index_out_path + "_pq");
    //std::string new_pq_coords(new_pq_prefix + "_compressed.bin");
    diskann::cout << "RELOAD: Creating new PQ coords file "
                  << this->temp_pq_coords_path << std::endl;

   
    // TODO (correct) :: write to the right file
    //std::string tmp_file = TMP_FOLDER + "/index_ravi";

#ifdef USE_TCMALLOC
    MallocExtension::instance()->ReleaseFreeMemory();
#endif

    //this->output_writer.open(tmp_file, std::ios::out | std::ios::binary);
    //assert(this->output_writer.is_open());

    // BEGIN -- PQ data on disk not consistent, not in right order
    // write outdated PQ data into pq writer with intentionally wrong header -
    // all these updates are made to a separate file, to be merged later into
    // thw index file if instructed
    std::fstream pq_writer(this->temp_pq_coords_path,
                           std::ios::out | std::ios::binary | std::ios::trunc);
    assert(pq_writer.is_open());
    uint64_t pq_file_size =
        ((uint64_t) new_max_pts * (uint64_t) this->pq_nchunks) +
        (2 * sizeof(uint32_t));

    // inflate file size to accommodate new points
    uint64_t dummy = 0;
    pq_writer.seekp(pq_file_size - sizeof(uint64_t), std::ios::beg);
    pq_writer.write((char *) (&dummy), sizeof(uint64_t));

    // write PQ compressed coords bin and close file
    pq_writer.seekp(0, std::ios::beg);
    uint32_t npts_u32 = new_max_pts, ndims_u32 = this->pq_nchunks;
    pq_writer.write((char *) &npts_u32, sizeof(uint32_t));
    pq_writer.write((char *) &ndims_u32, sizeof(uint32_t));
    pq_writer.write((char *) this->pq_data,
                    (uint64_t) this->disk_npts * (uint64_t) ndims_u32);
    pq_writer.close();

    // write out tags
    //const std::string tag_file = new_disk_out + ".tags";
    this->write_tag_file(this->temp_tags_path, new_max_pts);

    // switch index to read-only mode
    this->disk_index->reload_index(this->temp_disk_index_path,
                                   this->temp_pq_coords_path,
                                   this->temp_tags_path);
#ifdef USE_TCMALLOC
    MallocExtension::instance()->ReleaseFreeMemory();
#endif

    // re-acquire pointers
    auto res = this->disk_index->get_pq_config();
    this->pq_nchunks = res.second;
    this->pq_data = res.first;
    this->disk_npts = (_u32) this->disk_index->return_nd();
    this->init_ids.clear();
    this->init_ids = this->disk_index->get_init_ids();
    assert(this->disk_npts == new_max_pts);

    std::cout << "AFTER RELOAD: PQ_NChunks: " << res.second
              << " Disk points: " << this->disk_npts
              << " Frozen point id: " << this->init_ids[0] << std::endl;

    // call inserts
    this->process_inserts();
#ifdef USE_TCMALLOC
    MallocExtension::instance()->ReleaseFreeMemory();
#endif

    this->process_inserts_pq();
#ifdef USE_TCMALLOC
    MallocExtension::instance()->ReleaseFreeMemory();
#endif

    diskann::cout << "Dumping full compressed PQ vectors from memory.\n";
    // re-open PQ writer
    pq_writer.open(this->temp_pq_coords_path,
                   std::ios::in | std::ios::out | std::ios::binary);
    pq_writer.seekp(2 * sizeof(uint32_t), std::ios::beg);
    // write all (old + new) PQ data to disk; no need to modify header
    pq_writer.write((char *) this->pq_data,
                    ((uint64_t) new_max_pts * (uint64_t) this->pq_nchunks));
    pq_writer.close();
    // END -- PQ data on disk consistent and in correct order

    // batch rename all inserted edges in each delta
    diskann::cout << "Renaming edges for easier access during merge.\n";
    // const std::function<uint32_t(uint32_t)> rename_func =
    // std::bind(&StreamingMerger<T, TagT>::rename, this);
    const std::function<uint32_t(uint32_t)> rename_func = [this](uint32_t id) {
      return this->rename(id);
    };
    this->disk_delta->rename_edges(rename_func);
    for (auto &delta : this->mem_deltas) {
      delta->rename_edges(rename_func);
    }

    // start merging
    // BEGIN -- graph on disk has NO deleted references, NO newly inserted
    // points
    
    
    this->process_merges();
    // END -- graph on disk has NO deleted references, has newly inserted points

    /* copy output from temp_file -> new_disk_out */
    // reset temp_file ptr
    //this->output_writer.close();

    auto copy_file = [](const std::string &src, const std::string &dest) {
      diskann::cout << "COPY :: " << src << " --> " << dest << "\n";
      std::ofstream dest_writer(dest, std::ios::binary);
      std::ifstream src_reader(src, std::ios::binary);
      dest_writer << src_reader.rdbuf();
      dest_writer.close();
      src_reader.close();
    };
    // copy index
    //copy_file(tmp_file, this->disk_index_out_path);

    // merge files if needed
    if (this->_single_file_index) {
      // update metadata with pq_pivots_file_size, pq_vector_file_size
      size_t                nr, nc;
      std::vector<uint64_t> output_metadata;
      uint64_t *            out_metadata;

      diskann::load_bin<uint64_t>(this->final_index_file, out_metadata, nr,
                                  nc);
      for (size_t i = 0; i < nr; i++)
        output_metadata.push_back(out_metadata[i]);

      delete[] out_metadata;

      // tags
      TagT *   tags;
      uint64_t tag_num, tag_dim;
      diskann::load_bin(this->temp_tags_path, tags, tag_num, tag_dim);
      size_t tag_bytes_written =
          diskann::save_bin(this->final_index_file, tags, tag_num, tag_dim,
                            output_metadata[output_metadata.size() - 1]);
      delete[] tags;

      output_metadata.push_back(output_metadata[output_metadata.size() - 1] +
                                tag_bytes_written);

      size_t      nr_in, nc_in;
      uint64_t *  in_metadata;
      std::string disk_in = this->_single_file_index
                                ? this->disk_index_in_path
                                : this->disk_index_in_path + "_disk.index";
      diskann::load_bin<uint64_t>(disk_in, in_metadata, nr_in, nc_in);

      uint64_t *pq_metadata_in;
      size_t    nr_pq_in, nc_pq_in;
      diskann::load_bin<uint64_t>(disk_in, pq_metadata_in, nr_pq_in, nc_pq_in,
                                  in_metadata[8]);
      diskann::save_bin<uint64_t>(this->final_index_file, pq_metadata_in,
                                  nr_pq_in,
                                  nc_pq_in,
                                  output_metadata[output_metadata.size() - 1]);

      size_t pq_pivots_total_bytes_written = 0;
      // pq_pivots
      float *  pq_pivots_data;
      uint64_t pq_pts, pq_dims;
      diskann::load_bin<float>(disk_in, pq_pivots_data, pq_pts, pq_dims,
                               in_metadata[8] + pq_metadata_in[0]);
      size_t pq_pivots_bytes = diskann::save_bin<float>(
          this->final_index_file, pq_pivots_data, pq_pts, pq_dims,
          output_metadata[output_metadata.size() - 1] + pq_metadata_in[0]);
      delete[] pq_pivots_data;
      diskann::cout << "Written pivots to single index file" << std::endl;

      // pq centroids
      float *  pq_centroid_data;
      uint64_t centroid_num, centroid_dim;
      diskann::load_bin<float>(disk_in, pq_centroid_data, centroid_num,
                               centroid_dim,
                               in_metadata[8] + pq_metadata_in[1]);
      size_t pq_centroid_bytes = diskann::save_bin<float>(
          this->final_index_file, pq_centroid_data, centroid_num,
          centroid_dim,
          output_metadata[output_metadata.size() - 1] + pq_metadata_in[1]);
      delete[] pq_centroid_data;
      diskann::cout << "Written centroids to single index file" << std::endl;

      // pq_rearrangment_perm
      uint32_t *pq_rearrange_data;
      uint64_t  rearrange_num, rearrange_dim;
      diskann::load_bin<uint32_t>(disk_in, pq_rearrange_data, rearrange_num,
                                  rearrange_dim,
                                  in_metadata[8] + pq_metadata_in[2]);
      size_t pq_rearrange_bytes = diskann::save_bin<uint32_t>(
          this->final_index_file, pq_rearrange_data, rearrange_num,
          rearrange_dim,
          output_metadata[output_metadata.size() - 1] + pq_metadata_in[2]);
      delete[] pq_rearrange_data;
      diskann::cout << "Written rearrangement data to single index file"
                    << std::endl;

      // pq_chunk_offsets
      uint32_t *pq_offset_data;
      uint64_t  chunk_offset_num, chunk_offset_dim;
      diskann::load_bin<uint32_t>(disk_in, pq_offset_data, chunk_offset_num,
                                  chunk_offset_dim,
                                  in_metadata[8] + pq_metadata_in[3]);
      size_t pq_offset_bytes = diskann::save_bin<uint32_t>(
          this->final_index_file, pq_offset_data, chunk_offset_num,
          chunk_offset_dim,
          output_metadata[output_metadata.size() - 1] + pq_metadata_in[3]);
      delete[] pq_offset_data;
      diskann::cout << "Written offsets to single index file" << std::endl;

      pq_pivots_total_bytes_written = pq_pivots_bytes + pq_centroid_bytes +
                                      pq_rearrange_bytes + pq_offset_bytes;
      output_metadata.push_back(output_metadata[output_metadata.size() - 1] +
                                pq_pivots_total_bytes_written +
                                pq_metadata_in[0]);

      // pq vectors
      size_t pq_vector_bytes = diskann::save_bin<uint8_t>(
          this->final_index_file, this->pq_data, (uint64_t) this->disk_npts,
          (uint64_t) ndims_u32, output_metadata[output_metadata.size() - 1]);

      output_metadata.push_back(output_metadata[output_metadata.size() - 1] +
                                pq_vector_bytes);

      delete[] pq_metadata_in;

      diskann::save_bin<uint64_t>(this->final_index_file,
                                  output_metadata.data(),
                                  output_metadata.size(), 1);
    } else {
      // update pq table related data into new files
      /* copy PQ tables */
      std::string prefix_pq_in = this->disk_index_in_path + "_pq";
      std::string prefix_pq_out = this->disk_index_out_path + "_pq";
      // PQ pivots

      copy_file(prefix_pq_in + "_pivots.bin", prefix_pq_out + "_pivots.bin");
      diskann::save_bin<uint8_t>(
          this->final_pq_coords_file, this->pq_data, (uint64_t) this->disk_npts,
          (uint64_t) ndims_u32);
      
      copy_file(this->temp_tags_path, this->final_tags_file);

    }

    // destruct PQFlashIndex
    delete this->disk_index;
    this->disk_index = nullptr;
    diskann::cout << "Destroyed PQ Flash Index\n";

  }

  // template class instantiations
  template class StreamingMerger<float, uint32_t>;
  template class StreamingMerger<uint8_t, uint32_t>;
  template class StreamingMerger<int8_t, uint32_t>;
  template class StreamingMerger<float, int64_t>;
  template class StreamingMerger<uint8_t, int64_t>;
  template class StreamingMerger<int8_t, int64_t>;
  template class StreamingMerger<float, uint64_t>;
  template class StreamingMerger<uint8_t, uint64_t>;
  template class StreamingMerger<int8_t, uint64_t>;
}  // namespace diskann
