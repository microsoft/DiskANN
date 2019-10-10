#include "pq_flash_index_nsg.h"
#include <malloc.h>
#include "percentile_stats.h"

#include <omp.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iterator>
#include <thread>
#include "distance.h"
#include "exceptions.h"
#include "parameters.h"
#include "timer.h"
#include "utils.h"

#include "tsl/robin_set.h"

#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#endif

#define SECTOR_LEN 4096

#define READ_U64(stream, val) stream.read((char *) &val, sizeof(_u64))
#define READ_UNSIGNED(stream, val) stream.read((char *) &val, sizeof(unsigned))

// sector # on disk where node_id is present
#define NODE_SECTOR_NO(node_id) ((node_id / nnodes_per_sector) + 1)

// obtains region of sector containing node
#define OFFSET_TO_NODE(sector_buf, node_id) \
  ((char *) sector_buf + (node_id % nnodes_per_sector) * max_node_len)

// offset into sector where node_id's nhood starts
#define NODE_SECTOR_OFFSET(sector_buf, node_id) \
  ((char *) sector_buf + ((node_id % nnodes_per_sector) * max_node_len))

// returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
#define OFFSET_TO_NODE_NHOOD(node_buf) \
  (unsigned *) ((char *) node_buf + data_dim * sizeof(T))

// returns region of `node_buf` containing [COORD(T)]
#define OFFSET_TO_NODE_COORDS(node_buf) (T *) (node_buf)

namespace {
  void aggregate_coords(const unsigned *ids, const _u64 n_ids,
                        const _u8 *all_coords, const _u64 ndims, _u8 *out) {
    for (_u64 i = 0; i < n_ids; i++) {
      memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(_u8));
    }
  }

  void pq_dist_lookup(const _u8 *pq_ids, const _u64 n_pts,
                      const _u64 pq_nchunks, const float *pq_dists,
                      float *dists_out) {
    _mm_prefetch((char *) dists_out, _MM_HINT_T0);
    _mm_prefetch((char *) pq_ids, _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 64), _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 128), _MM_HINT_T0);
    memset(dists_out, 0, n_pts * sizeof(float));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
      const float *chunk_dists = pq_dists + 256 * chunk;
      if (chunk < pq_nchunks - 1) {
        _mm_prefetch((char *) (chunk_dists + 256), _MM_HINT_T0);
      }
      for (_u64 idx = 0; idx < n_pts; idx++) {
        _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
        dists_out[idx] += chunk_dists[pq_centerid];
      }
    }
  }
}  // namespace

namespace NSG {
  template<>
  PQFlashNSG<_u8>::PQFlashNSG() {
    this->dist_cmp = new DistanceL2UInt8();
    //    medoid_nhood.second = nullptr;
  }

  template<>
  PQFlashNSG<_s8>::PQFlashNSG() {
    this->dist_cmp = new DistanceL2Int8();
    //    medoid_nhood.second = nullptr;
  }

  template<>
  PQFlashNSG<float>::PQFlashNSG() {
    this->dist_cmp = new DistanceL2();
    //    medoid_nhood.second = nullptr;
  }

  template<typename T>
  PQFlashNSG<T>::~PQFlashNSG() {
    if (data != nullptr) {
      delete[] data;
    }

    // delete backing bufs for nhood and coord cache
    if (nhood_cache_buf != nullptr) {
      delete[] nhood_cache_buf;
      NSG::aligned_free(coord_cache_buf);
    }
    for (auto m : medoid_nhoods)
      delete[] m.second;

    delete this->dist_cmp;
    if (load_flag) {
      this->destroy_thread_data();
      reader->close();
      delete reader;
    }
  }

  template<typename T>
  void PQFlashNSG<T>::setup_thread_data(_u64 nthreads) {
    std::cout << "Setting up thread-specific contexts for nthreads: "
              << nthreads << "\n";
// omp parallel for to generate unique thread IDs
#pragma omp parallel for
    for (_s64 thread = 0; thread < (_s64) nthreads; thread++) {
#pragma omp critical
      {
        this->reader->register_thread();
        IOContext ctx = this->reader->get_ctx();
        // std::cout << "ctx: " << ctx << "\n";
        QueryScratch<T> scratch;
        _u64 coord_alloc_size = ROUND_UP(MAX_N_CMPS * this->aligned_dim, 256);
        NSG::alloc_aligned((void **) &scratch.coord_scratch, coord_alloc_size,
                           256);
        // scratch.coord_scratch = new T[MAX_N_CMPS * this->aligned_dim];
        // //Gopal. Commenting out the reallocation!
        NSG::alloc_aligned((void **) &scratch.sector_scratch,
                           MAX_N_SECTOR_READS * SECTOR_LEN, SECTOR_LEN);
        NSG::alloc_aligned((void **) &scratch.aligned_scratch,
                           256 * sizeof(float), 256);
        NSG::alloc_aligned((void **) &scratch.aligned_pq_coord_scratch,
                           16384 * sizeof(_u8), 256);
        NSG::alloc_aligned((void **) &scratch.aligned_pqtable_dist_scratch,
                           16384 * sizeof(float), 256);
        NSG::alloc_aligned((void **) &scratch.aligned_dist_scratch,
                           512 * sizeof(float), 256);
        memset(scratch.aligned_scratch, 0, 256 * sizeof(float));
        memset(scratch.coord_scratch, 0, MAX_N_CMPS * this->aligned_dim);
        ThreadData<T> data;
        data.ctx = ctx;
        data.scratch = scratch;
        this->thread_data.push(data);
      }
    }
    load_flag = true;
  }

  template<typename T>
  void PQFlashNSG<T>::destroy_thread_data() {
    std::cerr << "Clearing scratch" << std::endl;
    assert(this->thread_data.size() == this->max_nthreads);
    while (this->thread_data.size() > 0) {
      ThreadData<T> data = this->thread_data.pop();
      while (data.scratch.sector_scratch == nullptr) {
        this->thread_data.wait_for_push_notify();
        data = this->thread_data.pop();
      }
      auto &scratch = data.scratch;
      NSG::aligned_free((void *) scratch.coord_scratch);
      NSG::aligned_free((void *) scratch.sector_scratch);
      NSG::aligned_free((void *) scratch.aligned_scratch);
      NSG::aligned_free((void *) scratch.aligned_pq_coord_scratch);
      NSG::aligned_free((void *) scratch.aligned_pqtable_dist_scratch);
      NSG::aligned_free((void *) scratch.aligned_dist_scratch);
    }
  }

  template<typename T>
  void PQFlashNSG<T>::set_cache_create_flag() {
    this->create_visit_cache = true;
  }

  /*  This function loads the nhood_cache and coord_cache with cached nodes
   * present in node_list..
   *  The num_nodes parameter tells how many nodes to cache. */
  template<typename T>
  void PQFlashNSG<T>::load_cache_from_file(std::string cache_bin) {
    _u64  num_cached_nodes;
    _u64  dummy_ones;
    _u64 *node_list;
    NSG::load_bin<_u64>(cache_bin, node_list, num_cached_nodes, dummy_ones);
    std::cout << "Caching " << num_cached_nodes
              << "nodes full-precision vectors and graph neighborhood "
                 "information in memory..."
              << std::flush;

    // borrow thread data
    ThreadData<T> this_thread_data = this->thread_data.pop();
    while (this_thread_data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      this_thread_data = this->thread_data.pop();
    }

    IOContext ctx = this_thread_data.ctx;

    nhood_cache_buf = new unsigned[num_cached_nodes * (max_degree + 1)];
    memset(nhood_cache_buf, 0, num_cached_nodes * (max_degree + 1));

    _u64 coord_cache_buf_len = num_cached_nodes * aligned_dim;
    NSG::alloc_aligned((void **) &coord_cache_buf,
                       coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    size_t BLOCK_SIZE = 1000;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);

    for (_u64 block = 0; block < num_blocks; block++) {
      _u64 start_idx = block * BLOCK_SIZE;
      _u64 end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);
      for (_u64 node_idx = start_idx; node_idx < end_idx; node_idx++) {
        std::vector<AlignedRead> read_reqs;
        std::vector<std::pair<_u64, char *>> nhoods;
        AlignedRead read;
        char *      buf = nullptr;
        alloc_aligned((void **) &buf, SECTOR_LEN, SECTOR_LEN);
        nhoods.push_back(std::make_pair(node_list[node_idx], buf));
        read.len = SECTOR_LEN;
        read.buf = buf;
        read.offset = NODE_SECTOR_NO(node_list[node_idx]) * SECTOR_LEN;
        read_reqs.push_back(read);

        reader->read(read_reqs, ctx);

        for (auto &nhood : nhoods) {
          char *node_buf = OFFSET_TO_NODE(nhood.second, nhood.first);
          T *   node_coords = OFFSET_TO_NODE_COORDS(node_buf);
          T *   cached_coords = coord_cache_buf + node_idx * aligned_dim;
          memcpy(cached_coords, node_coords, data_dim * sizeof(T));
          coord_cache.insert(std::make_pair(nhood.first, cached_coords));

          // insert node nhood into nhood_cache
          unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);
          _u64      nnbrs = (_u64) *node_nhood;
          unsigned *nbrs = node_nhood + 1;
          // std::cerr << "CACHE: nnbrs = " << nnbrs << "\n";
          std::pair<_u64, unsigned *> cnhood;
          cnhood.first = nnbrs;
          cnhood.second = nhood_cache_buf + node_idx * (max_degree + 1);
          memcpy(cnhood.second, nbrs, nnbrs * sizeof(unsigned));
          nhood_cache.insert(std::make_pair(nhood.first, cnhood));
          aligned_free(nhood.second);
        }
      }
    }
    // return thread data
    this->thread_data.push(this_thread_data);
    std::cout << "..done" << std::endl;
  }

  template<typename T>
  void PQFlashNSG<T>::cache_bfs_levels(_u64 nlevels) {
    assert(nlevels > 1);

    // borrow thread data
    ThreadData<T> this_thread_data = this->thread_data.pop();
    while (this_thread_data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      this_thread_data = this->thread_data.pop();
    }

    IOContext ctx = this_thread_data.ctx;

    tsl::robin_set<unsigned> *cur_level, *prev_level;
    cur_level = new tsl::robin_set<unsigned>();
    prev_level = new tsl::robin_set<unsigned>();

    // add medoid nhood to cur_level
    for (_u64 miter = 0; miter < medoid_nhoods.size(); miter++) {
      for (_u64 idx = 0; idx < medoid_nhoods[miter].first; idx++) {
        unsigned nbr_id = medoid_nhoods[miter].second[idx];
        cur_level->insert(nbr_id);
      }
    }

    for (_u64 lvl = 1; lvl < nlevels; lvl++) {
      // swap prev_level and cur_level
      std::swap(prev_level, cur_level);
      // clear cur_level
      cur_level->clear();

      // read in all pre_level nhoods
      std::vector<AlignedRead> read_reqs;
      std::vector<std::pair<_u64, char *>> nhoods;

      for (const unsigned &id : *prev_level) {
        // skip node if already read into
        if (nhood_cache.find(id) != nhood_cache.end()) {
          continue;
        }
        char *buf = nullptr;
        alloc_aligned((void **) &buf, SECTOR_LEN, SECTOR_LEN);
        nhoods.push_back(std::make_pair(id, buf));
        AlignedRead read;
        read.len = SECTOR_LEN;
        read.buf = buf;
        read.offset = NODE_SECTOR_NO(id) * SECTOR_LEN;
        read_reqs.push_back(read);
      }

      // issue read requests
      reader->read(read_reqs, ctx);

      // process each nhood buf
      // TODO:: cache all nhoods in each sector instead of just one
      for (auto &nhood : nhoods) {
        // insert node coord into coord_cache
        char *node_buf = OFFSET_TO_NODE(nhood.second, nhood.first);
        T *   node_coords = OFFSET_TO_NODE_COORDS(node_buf);
        T *   cached_coords = new T[data_dim];
        memcpy(cached_coords, node_coords, data_dim * sizeof(T));
        coord_cache.insert(std::make_pair(nhood.first, cached_coords));

        // insert node nhood into nhood_cache
        unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);
        _u64      nnbrs = (_u64) *node_nhood;
        unsigned *nbrs = node_nhood + 1;
        // std::cerr << "CACHE: nnbrs = " << nnbrs << "\n";
        std::pair<_u64, unsigned *> cnhood;
        cnhood.first = nnbrs;
        cnhood.second = new unsigned[nnbrs];
        memcpy(cnhood.second, nbrs, nnbrs * sizeof(unsigned));
        nhood_cache.insert(std::make_pair(nhood.first, cnhood));

        // explore next level
        for (_u64 j = 0; j < nnbrs; j++) {
          cur_level->insert(nbrs[j]);
        }
        aligned_free(nhood.second);
      }
      std::cout << "Level: " << lvl << ", #nodes: " << nhoods.size()
                << std::endl;
    }

    // return thread data
    this->thread_data.push(this_thread_data);

    delete cur_level;
    delete prev_level;
#ifdef DEBUG
    // verify non-null
    for (auto &k_v : nhood_cache) {
      unsigned *nbrs = k_v.second.second;
      _u64      nnbrs = k_v.second.first;
#ifndef _WINDOWS
      assert(malloc_usable_size(nbrs) >= nnbrs * sizeof(unsigned));
#else
      assert(_msize(nbrs) >= nnbrs * sizeof(unsigned));
#endif
    }
#endif

    std::cerr << "Consolidating nhood_cache: # cached nhoods = "
              << nhood_cache.size() << "\n";
    // consolidate nhood_cache down to single buf
    _u64 nhood_cache_buf_len = 0;
    for (auto &k_v : nhood_cache) {
      nhood_cache_buf_len += k_v.second.first;
    }
    nhood_cache_buf = new unsigned[nhood_cache_buf_len];
    memset(nhood_cache_buf, 0, nhood_cache_buf_len);
    _u64 cur_off = 0;
    for (auto &k_v : nhood_cache) {
      std::pair<_u64, unsigned *> &val = nhood_cache[k_v.first];
      unsigned *&ptr = val.second;
      _u64       nnbrs = val.first;
      memcpy(nhood_cache_buf + cur_off, ptr, nnbrs * sizeof(unsigned));
      delete[] ptr;
      ptr = nhood_cache_buf + cur_off;
      cur_off += nnbrs;
    }

    std::cerr << "Consolidating coord_cache: # cached coords = "
              << coord_cache.size() << "\n";
    // consolidate coord_cache down to single buf
    _u64 coord_cache_buf_len = coord_cache.size() * aligned_dim;
    NSG::alloc_aligned((void **) &coord_cache_buf,
                       coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));
    cur_off = 0;
    for (auto &k_v : coord_cache) {
      T *&val = coord_cache[k_v.first];
      memcpy(coord_cache_buf + cur_off, val, data_dim * sizeof(T));
      delete[] val;
      val = coord_cache_buf + cur_off;
      cur_off += aligned_dim;
    }
  }

  template<typename T>
  void PQFlashNSG<T>::save_cached_nodes(_u64        num_nodes,
                                        std::string cache_file_path) {
    if (this->create_visit_cache) {
      std::sort(this->node_visit_counter.begin(), node_visit_counter.end(),
                [](std::pair<_u64, _u32> &left, std::pair<_u64, _u32> &right) {
                  return left.second > right.second;
                });

      std::vector<_u64> node_ids;
      for (_u64 i = 0; i < num_nodes; i++) {
        node_ids.push_back(this->node_visit_counter[i].first);
      }

      save_bin<_u64>(cache_file_path.c_str(), node_ids.data(), num_nodes, 1);
    }
  }

  template<typename T>
  int PQFlashNSG<T>::load(uint32_t num_threads, const char *pq_centroids_bin,
                          const char *compressed_data_bin,
                          const char *disk_index_file,
                          const char *medoids_file) {
    size_t pq_file_dim, pq_file_num_centroids;
    get_bin_metadata(pq_centroids_bin, pq_file_num_centroids, pq_file_dim);

    if (pq_file_num_centroids != 256) {
      std::cout << "Error. Number of PQ centroids is not 256. Exitting"
                << std::endl;
      return -1;
    }

    this->data_dim = pq_file_dim;
    this->aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
    NSG::load_bin<_u8>(compressed_data_bin, data, npts_u64, nchunks_u64);

    this->num_points = npts_u64;
    this->n_chunks = nchunks_u64;
    this->chunk_size = DIV_ROUND_UP(this->data_dim, nchunks_u64);

    pq_table.load_pq_centroid_bin(pq_centroids_bin, n_chunks, chunk_size);

    std::cout
        << "Loaded PQ centroids and in-memory compressed vectors. #points: "
        << num_points << " #dim: " << data_dim
        << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks
        << std::endl;

    if (this->create_visit_cache) {
      this->node_visit_counter.resize(npts_u64);
      for (_u64 i = 0; i < node_visit_counter.size(); i++) {
        this->node_visit_counter[i].first = i;
        this->node_visit_counter[i].second = 0;
      }
    }

    // read nsg metadata
    size_t        actual_index_size = get_file_size(disk_index_file);
    std::ifstream nsg_meta(disk_index_file, std::ios::binary);
    size_t        expected_file_size;
    READ_U64(nsg_meta, expected_file_size);
    if (actual_index_size != expected_file_size) {
      std::cout << "File size mismatch for " << disk_index_file
                << " with meta-data size" << expected_file_size << std::endl;
      return -1;
    }
    _u64 disk_nnodes;
    READ_U64(nsg_meta, disk_nnodes);
    if (disk_nnodes != num_points) {
      std::cout
          << "Mismatch in #points for compressed data file and disk index file."
          << std::endl;
      return -1;
    }

    size_t medoid_id_on_file;

    std::ifstream file_exists(medoids_file);
    if (!file_exists) {
      file_exists.close();
      num_medoids = 1;
      // medoids file does not exist. loading the medoid on disk index as only
      // medoid for starting search
      medoids = new uint32_t[1];
      READ_U64(nsg_meta, medoid_id_on_file);
      medoids[0] = (_u32)(medoid_id_on_file);
    } else {
      file_exists.close();
      size_t tmp_dim;
      // given medoids file exists. using that data, and reading from disk_index
      // just as dummy. query search starts from nearest medoid
      load_bin<uint32_t>(medoids_file, medoids, num_medoids, tmp_dim);
      READ_U64(nsg_meta, medoid_id_on_file);
    }

    // caching full-precision data and neighborhood information of all medoids
    medoid_nhoods = std::vector<std::pair<_u64, unsigned *>>(num_medoids);
    medoid_full_precs = new T[num_medoids * aligned_dim];
    memset((void *) medoid_full_precs, 0,
           num_medoids * aligned_dim * sizeof(T));

    READ_U64(nsg_meta, max_node_len);
    READ_U64(nsg_meta, nnodes_per_sector);
    max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;

    std::cout << "Disk-Index File Meta-data: ";
    std::cout << "# nodes per sector: " << nnodes_per_sector;
    std::cout << ", max node len (bytes): " << max_node_len << std::endl;
    std::cout << ", max node degree: " << max_degree << std::endl;
    nsg_meta.close();

    // open AlignedFileReader handle to nsg_file
    std::string nsg_fname(disk_index_file);
#ifdef _WINDOWS
    reader = new WindowsAlignedFileReader();
#else
    reader = new LinuxAlignedFileReader();
#endif
    reader->open(nsg_fname);

    this->setup_thread_data(num_threads);
    this->max_nthreads = num_threads;

    // borrow ctx
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }
    IOContext ctx = data.ctx;
    std::cout << "Loading neighborhood info and full-precision vectors of "
              << num_medoids << " medoid(s) to memory" << std::endl;
    coord_cache.clear();
    for (uint64_t cur_m = 0; cur_m < num_medoids; cur_m++) {
      _u64 medoid = (_u64) medoids[cur_m];
      // read medoid nhood
      char *medoid_buf = nullptr;
      alloc_aligned((void **) &medoid_buf, SECTOR_LEN, SECTOR_LEN);
      std::vector<AlignedRead> medoid_read(1);
      medoid_read[0].len = SECTOR_LEN;
      medoid_read[0].buf = medoid_buf;
      medoid_read[0].offset = NODE_SECTOR_NO(medoid) * SECTOR_LEN;
      reader->read(medoid_read, ctx);

      // all data about medoid
      char *medoid_node_buf = OFFSET_TO_NODE(medoid_buf, medoid);

      // add medoid coords to `coord_cache`
      T *medoid_coords = new T[data_dim];
      T *medoid_disk_coords = OFFSET_TO_NODE_COORDS(medoid_node_buf);
      memcpy(medoid_coords, medoid_disk_coords, data_dim * sizeof(T));
      memcpy(medoid_full_precs + cur_m * aligned_dim, medoid_coords,
             data_dim * sizeof(T));

      coord_cache.insert(std::make_pair(medoid, medoid_coords));

      // add medoid nhood to nhood_cache
      unsigned *medoid_nhood_buf = OFFSET_TO_NODE_NHOOD(medoid_node_buf);
      medoid_nhoods[cur_m].first = *(unsigned *) (medoid_nhood_buf);
      medoid_nhoods[cur_m].second = new unsigned[medoid_nhoods[cur_m].first];
      memcpy(medoid_nhoods[cur_m].second, (medoid_nhood_buf + 1),
             medoid_nhoods[cur_m].first * sizeof(unsigned));
      aligned_free(medoid_buf);

      // make a copy and insert into nhood_cache
      unsigned *medoid_nhood_copy = new unsigned[medoid_nhoods[cur_m].first];
      memcpy(medoid_nhood_copy, medoid_nhoods[cur_m].second,
             medoid_nhoods[cur_m].first * sizeof(unsigned));
      nhood_cache.insert(std::make_pair(
          medoid,
          std::make_pair(medoid_nhoods[cur_m].first, medoid_nhood_copy)));
    }

    // return ctx
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
    return 0;
  }

  template<typename T>
  void PQFlashNSG<T>::create_disk_layout(const std::string base_file,
                                         const std::string mem_index_file,
                                         const std::string output_file) {
    unsigned npts, ndims;

    // amount to read or write in one shot
    _u64            read_blk_size = 64 * 1024 * 1024;
    _u64            write_blk_size = read_blk_size;
    cached_ifstream base_reader(base_file, read_blk_size);
    base_reader.read((char *) &npts, sizeof(uint32_t));
    base_reader.read((char *) &ndims, sizeof(uint32_t));

    size_t npts_64, ndims_64;
    npts_64 = npts;
    ndims_64 = ndims;

    // create cached reader + writer
    size_t          actual_file_size = get_file_size(mem_index_file);
    cached_ifstream nsg_reader(mem_index_file, read_blk_size);
    cached_ofstream nsg_writer(output_file, write_blk_size);

    // metadata: width, medoid
    unsigned width_u32, medoid_u32;
    size_t   index_file_size;

    nsg_reader.read((char *) &index_file_size, sizeof(uint64_t));
    if (index_file_size != actual_file_size) {
      std::cout << "Rand-NSG Index file size does not match expected size per "
                   "meta-data."
                << std::endl;
      exit(-1);
    }
    nsg_reader.read((char *) &width_u32, sizeof(unsigned));
    nsg_reader.read((char *) &medoid_u32, sizeof(unsigned));

    // compute
    _u64 medoid, max_node_len, nnodes_per_sector;
    npts_64 = (_u64) npts;
    medoid = (_u64) medoid_u32;
    max_node_len =
        (((_u64) width_u32 + 1) * sizeof(unsigned)) + (ndims_64 * sizeof(T));
    nnodes_per_sector = SECTOR_LEN / max_node_len;

    std::cout << "medoid: " << medoid << "B\n";
    std::cout << "max_node_len: " << max_node_len << "B\n";
    std::cout << "nnodes_per_sector: " << nnodes_per_sector << "B\n";

    // SECTOR_LEN buffer for each sector
    char *    sector_buf = new char[SECTOR_LEN];
    char *    node_buf = new char[max_node_len];
    unsigned &nnbrs = *(unsigned *) (node_buf + ndims_64 * sizeof(T));
    unsigned *nhood_buf =
        (unsigned *) (node_buf + (ndims_64 * sizeof(T)) + sizeof(unsigned));

    // number of sectors (1 for meta data)
    _u64 n_sectors = ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector;
    _u64 disk_index_file_size = (n_sectors + 1) * SECTOR_LEN;
    // write first sector with metadata
    *(_u64 *) (sector_buf + 0 * sizeof(_u64)) = disk_index_file_size;
    *(_u64 *) (sector_buf + 1 * sizeof(_u64)) = npts_64;
    *(_u64 *) (sector_buf + 2 * sizeof(_u64)) = medoid;
    *(_u64 *) (sector_buf + 3 * sizeof(_u64)) = max_node_len;
    *(_u64 *) (sector_buf + 4 * sizeof(_u64)) = nnodes_per_sector;
    nsg_writer.write(sector_buf, SECTOR_LEN);

    T *cur_node_coords = new T[ndims_64];
    std::cout << "# sectors: " << n_sectors << "\n";
    _u64 cur_node_id = 0;
    for (_u64 sector = 0; sector < n_sectors; sector++) {
      if (sector % 100000 == 0) {
        std::cout << "Sector #" << sector << "written\n";
      }
      memset(sector_buf, 0, SECTOR_LEN);
      for (_u64 sector_node_id = 0;
           sector_node_id < nnodes_per_sector && cur_node_id < npts_64;
           sector_node_id++) {
        memset(node_buf, 0, max_node_len);
        // read cur node's nnbrs
        nsg_reader.read((char *) &nnbrs, sizeof(unsigned));

        // sanity checks on nnbrs
        assert(nnbrs > 0);
        assert(nnbrs <= width_u32);

        // read node's nhood
        nsg_reader.read((char *) nhood_buf, nnbrs * sizeof(unsigned));

        // write coords of node first
        //  T *node_coords = data + ((_u64) ndims_64 * cur_node_id);
        base_reader.read((char *) cur_node_coords, sizeof(T) * ndims_64);
        memcpy(node_buf, cur_node_coords, ndims_64 * sizeof(T));

        // write nnbrs
        *(unsigned *) (node_buf + ndims_64 * sizeof(T)) = nnbrs;

        // write nhood next
        memcpy(node_buf + ndims_64 * sizeof(T) + sizeof(unsigned), nhood_buf,
               nnbrs * sizeof(unsigned));

        // get offset into sector_buf
        char *sector_node_buf = sector_buf + (sector_node_id * max_node_len);

        // copy node buf into sector_node_buf
        memcpy(sector_node_buf, node_buf, max_node_len);
        cur_node_id++;
      }
      // flush sector to disk
      nsg_writer.write(sector_buf, SECTOR_LEN);
    }
    delete[] sector_buf;
    delete[] node_buf;
    delete[] cur_node_coords;
    std::cout << "Output file written\n";
  }

  template<typename T>
  void PQFlashNSG<T>::cached_beam_search(const T *query, const _u64 k_search,
                                         const _u64 l_search, _u64 *indices,
                                         float *      distances,
                                         const _u64   beam_width,
                                         QueryStats * stats,
                                         Distance<T> *output_dist_func) {
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    IOContext ctx = data.ctx;
    auto      query_scratch = &(data.scratch);

    // reset query
    query_scratch->reset();

    // scratch space to compute distances between FP32 Query and INT8 data
    float *scratch = query_scratch->aligned_scratch;
    _mm_prefetch((char *) scratch, _MM_HINT_T0);

    // pointers to buffers for data
    T *   data_buf = query_scratch->coord_scratch;
    _u64 &data_buf_idx = query_scratch->coord_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

#ifdef USE_ACCELERATED_PQ
    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](
        const unsigned *ids, const _u64 n_ids, float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };
#endif
    Timer                 query_timer, io_timer;
    std::vector<Neighbor> retset(l_search + 1);
    std::vector<_u64>     init_ids(l_search);
    tsl::robin_set<_u64>  visited(4096);

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);
    tsl::robin_map<_u64, T *> fp_coords;

    _u32  best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();
    for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
      float cur_expanded_dist = dist_cmp->compare(
          query, medoid_full_precs + aligned_dim * cur_m, aligned_dim);
      if (cur_expanded_dist < best_dist) {
        best_medoid = cur_m;
        best_dist = cur_expanded_dist;
      }
    }

// compute medoid nhood <-> query distances
#ifdef USE_ACCELERATED_PQ
    compute_dists(medoid_nhoods[best_medoid].second,
                  medoid_nhoods[best_medoid].first, dist_scratch);
#endif

    _u64 tmp_l = 0;
    // add each neighbor of medoid
    for (; tmp_l < l_search && tmp_l < medoid_nhoods[best_medoid].first;
         tmp_l++) {
      _u64 id = medoid_nhoods[best_medoid].second[tmp_l];
      init_ids[tmp_l] = id;
      visited.insert(id);
#ifdef USE_ACCELERATED_PQ
      float dist = dist_scratch[tmp_l];
#else
      pq_table.convert(data + id * n_chunks, scratch);
      float dist = dist_cmp->compare(scratch, query, aligned_dim);
#endif
      // std::cout << "cmp: " << id << ", dist: " << dist << std::endl;
      // std::cerr << "dist: " << dist << std::endl;
      retset[tmp_l] = Neighbor(id, dist, true);
      if (stats != nullptr) {
        stats->n_cmps++;
      }
    }

    // TODO:: create dummy ids
    for (; tmp_l < l_search; tmp_l++) {
      _u64 id = (std::numeric_limits<unsigned>::max)() - tmp_l;
      init_ids[tmp_l] = id;
      float dist = (std::numeric_limits<float>::max)();
      retset[tmp_l] = Neighbor(id, dist, false);
    }

    std::sort(retset.begin(), retset.begin() + l_search);

    _u64 hops = 0;
    _u64 cmps = 0;
    _u64 k = 0;

    // cleared every iteration
    std::vector<_u64> frontier;
    std::vector<std::pair<_u64, char *>> frontier_nhoods;
    std::vector<AlignedRead> frontier_read_reqs;
    std::vector<std::pair<_u64, std::pair<_u64, unsigned *>>> cached_nhoods;

    while (k < l_search) {
      _u64 nk = l_search;

      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;

      // find new beam
      _u64 marker = k - 1;
      while (++marker < l_search && frontier.size() < beam_width) {
        if (retset[marker].flag) {
          auto iter = nhood_cache.find(retset[marker].id);
          if (iter != nhood_cache.end()) {
            cached_nhoods.push_back(
                std::make_pair(retset[marker].id, iter->second));
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
          } else {
            frontier.push_back(retset[marker].id);
          }
          retset[marker].flag = false;
          if (this->create_visit_cache) {
            reinterpret_cast<std::atomic<_u32> &>(
                this->node_visit_counter[retset[marker].id].second)
                .fetch_add(1);
          }
        }
      }

      // read nhoods of frontier ids
      if (!frontier.empty()) {
        hops++;
        for (_u64 i = 0; i < frontier.size(); i++) {
          unsigned id = frontier[i];
          std::pair<_u64, char *> fnhood;
          fnhood.first = id;
          fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(NODE_SECTOR_NO(id) * SECTOR_LEN,
                                          SECTOR_LEN, fnhood.second);
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
        }
        io_timer.reset();
        reader->read(frontier_read_reqs, ctx);
        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();
        }

        // process each frontier nhood - compute distances to unvisited nodes
        for (auto &frontier_nhood : frontier_nhoods) {
          // if (retset[k].flag) {
          //   retset[k].flag = false;
          //   unsigned n = retset[k].id;
          // }
          //

          char *node_disk_buf =
              OFFSET_TO_NODE(frontier_nhood.second, frontier_nhood.first);
          unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
          _u64      nnbrs = (_u64)(*node_buf);
          T *       node_fp_coords = OFFSET_TO_NODE_COORDS(node_disk_buf);
          assert(data_buf_idx < MAX_N_CMPS);

          T *node_fp_coords_copy = data_buf + (data_buf_idx * aligned_dim);
          data_buf_idx++;
          memcpy(node_fp_coords_copy, node_fp_coords, data_dim * sizeof(T));

          float cur_expanded_dist =
              dist_cmp->compare(query, node_fp_coords_copy, aligned_dim);
          full_retset.push_back(
              Neighbor(frontier_nhood.first, cur_expanded_dist, true));

          //          fp_coords.insert(
          //             std::make_pair(frontier_nhood.first,
          //             node_fp_coords_copy));

          unsigned *node_nbrs = (node_buf + 1);
#ifdef USE_ACCELERATED_PQ
          // compute node_nbrs <-> query dist in PQ space
          compute_dists(node_nbrs, nnbrs, dist_scratch);
#else
          // issue prefetches
          for (_u64 m = 0; m < nnbrs; ++m) {
            unsigned next_id = node_nbrs[m];
            _mm_prefetch((char *) data + next_id * n_chunks, _MM_HINT_T1);
          }
#endif
          // process prefetch-ed nhood
          for (_u64 m = 0; m < nnbrs; ++m) {
            unsigned id = node_nbrs[m];
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              visited.insert(id);
              cmps++;
#ifdef USE_ACCELERATED_PQ
              float dist = dist_scratch[m];
#else
              pq_table.convert(data + id * n_chunks, scratch);
              float dist = dist_cmp->compare(scratch, query, aligned_dim);
#endif
              // std::cout << "cmp: " << id << ", dist: " << dist << std::endl;
              // std::cerr << "dist: " << dist << std::endl;
              if (stats != nullptr) {
                stats->n_cmps++;
              }
              if (dist >= retset[l_search - 1].distance)
                continue;
              Neighbor nn(id, dist, true);
              _u64     r = InsertIntoPool(
                  retset.data(), l_search,
                  nn);  // Return position in sorted list where nn inserted.
              if (r < nk)
                nk = r;  // nk logs the best position in the retset that was
                         // updated
                         // due to neighbors of n.
            }
          }
        }
      }

      // process cached nhoods
      for (auto &cached_nhood : cached_nhoods) {
        auto  global_cache_iter = coord_cache.find(cached_nhood.first);
        T *   node_fp_coords_copy = global_cache_iter->second;
        float cur_expanded_dist =
            dist_cmp->compare(query, node_fp_coords_copy, aligned_dim);
        full_retset.push_back(
            Neighbor(cached_nhood.first, cur_expanded_dist, true));

        _u64      nnbrs = cached_nhood.second.first;
        unsigned *node_nbrs = cached_nhood.second.second;

#ifdef USE_ACCELERATED_PQ
        // compute node_nbrs <-> query dists in PQ space
        compute_dists(node_nbrs, nnbrs, dist_scratch);
#else
        // issue prefetches
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned next_id = node_nbrs[m];
          _mm_prefetch((char *) data + next_id * n_chunks, _MM_HINT_T1);
        }
#endif
        // process prefetched nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
            cmps++;
#ifdef USE_ACCELERATED_PQ
            float dist = dist_scratch[m];
#else
            pq_table.convert(data + id * n_chunks, scratch);
            float dist = dist_cmp->compare(scratch, query, aligned_dim);
#endif
            // std::cout << "cmp: " << id << ", dist: " << dist << std::endl;
            // std::cerr << "dist: " << dist << std::endl;
            if (stats != nullptr) {
              stats->n_cmps++;
            }
            if (dist >= retset[l_search - 1].distance)
              continue;
            Neighbor nn(id, dist, true);
            _u64     r = InsertIntoPool(
                retset.data(), l_search,
                nn);  // Return position in sorted list where nn inserted.
            if (r < nk)
              nk = r;  // nk logs the best position in the retset that was
            // updated
            // due to neighbors of n.
          }
        }
      }
      // update best inserted position
      //

      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;
    }

    /*
        // prefetch coords backing buf
        _mm_prefetch((char *) data_buf, _MM_HINT_T1);
        // RE-RANKING STEP
        for (_u64 i = 0; i < l_search; i++) {
          _u64 idx = retset[i].id;
          T *  node_coords;
          auto global_cache_iter = coord_cache.find(idx);
          if (global_cache_iter == coord_cache.end()) {
            auto local_cache_iter = fp_coords.find(idx);
            assert(local_cache_iter != fp_coords.end());
            node_coords = local_cache_iter->second;
          } else {
            node_coords = global_cache_iter->second;
          }

          // std::cout << "left: " << query << ", right: " << node_coords <<
       "\n";

          // USE different distance function for final ranking -- like L2 for
       search
          // and cosine for ranking
          if (output_dist_func != nullptr) {
            retset[i].distance =
                output_dist_func->compare(query, node_coords, aligned_dim);
          } else {
            retset[i].distance = dist_cmp->compare(query, node_coords,
       aligned_dim);
          }

          if (stats != nullptr) {
            stats->n_cmps++;
          }
        }
        // std::cout << "end\n";
    */
    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    // copy k_search values
    for (_u64 i = 0; i < k_search; i++) {
      indices[i] = full_retset[i].id;
      if (distances != nullptr) {
        distances[i] = full_retset[i].distance;
      }
    }

    this->thread_data.push(data);
    this->thread_data.push_notify_all();

    if (stats != nullptr) {
      stats->total_us = query_timer.elapsed();
    }
  }

  // instantiations
  template class PQFlashNSG<_u8>;
  template class PQFlashNSG<_s8>;
  template class PQFlashNSG<float>;
}  // namespace NSG
