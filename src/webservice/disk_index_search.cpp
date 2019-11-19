#include <cosine_similarity.h>
#include <webservice/disk_index_search.h>
#include <ctime>
#include <iomanip>
#include "utils.h"

namespace diskann {
  const unsigned int DEFAULT_BEAM_WIDTH = 8;
  // const unsigned int L_MULTIPLIER = 10;
  // const unsigned int MAX_L = 300;
  const unsigned int DEFAULT_L = 704;

  DiskIndexSearch::DiskIndexSearch(const char* indexFilePrefix, const char* idsFile,
                               const _u64 cache_nlevels, const _u64 nthreads) {
    this->cosine_distance = new DistanceCosine<float>();

    const std::string index_prefix_path(indexFilePrefix);

    // convert strs into params
    std::string data_bin = index_prefix_path + "_compressed_uint32.bin";
    std::string pq_tables_bin = index_prefix_path + "_pq_pivots.bin";

    // determine nchunks
    std::string params_path = index_prefix_path + "_params.bin";
    uint32_t*   params;
    size_t      nargs, one;
    diskann::load_bin<uint32_t>(params_path.c_str(), params, nargs, one);

    // infer chunk_size
    _u64 m_dimension = (_u64) params[3];

    _u64 n_chunks = (_u64) params[4];
    _u64 chunk_size = (_u64)(m_dimension / n_chunks);

    std::string nsg_disk_opt = index_prefix_path + "_diskopt.rnsg";

    std::string stars(40, '*');
    std::cout << stars << "\nPQ -- n_chunks: " << n_chunks
              << ", chunk_size: " << chunk_size << ", data_dim: " << m_dimension
              << "\n";
    std::cout << "Search meta-params -- cache_nlevels: " << cache_nlevels
              << ", nthreads: " << nthreads << "\n"
              << stars << "\n";

    // create object
    _pFlashIndex.reset(new PQFlashIndex<float>());

    // load index
    _pFlashIndex->load(data_bin.c_str(), nsg_disk_opt.c_str(),
                       pq_tables_bin.c_str(), chunk_size, n_chunks, m_dimension,
                       nthreads);

    // cache bfs levels
    _pFlashIndex->cache_bfs_levels(cache_nlevels);

    // obtain # dims and # points
    this->_dimensions = m_dimension;
    this->_numPoints = _pFlashIndex->n_base;

    // load point IDs from file
    _ids = load_ids(idsFile);
  }

  NSGSearchResult DiskIndexSearch::search(const float*       query,
                                        const unsigned int dimensions,
                                        const unsigned int K) {
    std::vector<unsigned int> start_points;

    std::vector<_u64>  indices(K);
    std::vector<float> distances(K);

    auto startTime = std::chrono::high_resolution_clock::now();

    _pFlashIndex->cached_beam_search(
        query, K,
        /* (std::min)(K * L_MULTIPLIER, MAX_L)*/ DEFAULT_L, indices.data(), distances.data(),
        DEFAULT_BEAM_WIDTH, nullptr, cosine_distance);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - startTime)
                        .count();

    // indices has the indexes of the results. Select the results from the
    // ids_vector.
    NSGSearchResult searchResult(K, (unsigned int) duration);
    std::for_each(indices.begin(), indices.begin() + K, [&](const unsigned int& index) {
      searchResult.addResult(_ids[index]);
      searchResult.finalResultIndices.push_back(
          index);  // TEMPORARY FOR IDENTIFYING RECALL
    });

    searchResult.distances = distances;

    return searchResult;
  }

  DiskIndexSearch::~DiskIndexSearch() {
    delete this->cosine_distance;
  }

  std::vector<std::wstring> DiskIndexSearch::load_ids(const char* idsFile) {
    std::wifstream            in(idsFile);
    std::vector<std::wstring> ids;

    if (!in.is_open()) {
      std::cerr << "Could not open " << idsFile << std::endl;
    }

    std::wstring id;
    while (!in.eof()) {
      in >> id;
      ids.push_back(id);
    }

    std::cout << "Loaded " << ids.size() << " from " << idsFile << std::endl;
    return ids;
  }

}  // namespace diskann
