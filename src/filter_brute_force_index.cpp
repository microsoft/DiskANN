// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "filter_brute_force_index.h"
#include "pq_flash_index.h"

namespace diskann
{
void dump_index(inverted_index_t &bf_index, std::ostream &out)
{
    size_t max_count = 0;
    std::string max_label = "";
    for (auto &label_and_points : bf_index)
    {
        if (label_and_points.second.size() > max_count)
        {
            max_count = label_and_points.second.size();
            max_label = label_and_points.first;
        }
        if (max_count > 10000)
        {
            out << "WARNING: max count changed to > 10k at label: " << label_and_points.first
                << " count is: " << label_and_points.second.size() << std::endl;
            break;
        }
        //out << label_and_points.first << ", " << label_and_points.second.size() << std::endl;
    }
    out << "Found maximum of " << max_count << " points for label: " << max_label << std::endl;
    out << "Size of brute force filter index: " << bf_index.size() << std::endl;
}

template <typename T, typename LabelT> FilterBruteForceIndex<T, LabelT>::FilterBruteForceIndex(const std::string &disk_index_file, 
                                        std::shared_ptr<PQFlashIndex<T, LabelT>> pq_flash_index)
    : _pq_flash_index(pq_flash_index)
{
    _disk_index_file = disk_index_file;
    _filter_bf_data_file = _disk_index_file + "_brute_force.txt";
    _is_loaded = false;
    _data_dim = _pq_flash_index->get_data_dim();
}

template <typename T, typename LabelT> void FilterBruteForceIndex<T, LabelT>::setup_thread_data(uint32_t num_threads)
{
    uint64_t max_count = 0;
    for (auto& label_and_fast_set : _bf_filter_index)
    {
        if (label_and_fast_set.second.size() > max_count)
        {
            max_count = label_and_fast_set.second.size();
        }
    }
    for (uint32_t i = 0; i < num_threads; i++)
    {
        auto bfscratch = new BruteForceScratch(max_count, MAX_PQ_CHUNKS);
        _scratch.push(bfscratch);
    }
}

template<typename T, typename LabelT> FilterBruteForceIndex<T, LabelT>::~FilterBruteForceIndex()
{
    while (! _scratch.empty())
    {
        auto p_bf_scratch = _scratch.pop();
        delete p_bf_scratch;
    }
}


template <typename T, typename LabelT> 
bool FilterBruteForceIndex<T, LabelT>::index_available() const
{
    return _is_loaded;
}

template <typename T, typename LabelT>
bool FilterBruteForceIndex<T, LabelT>::brute_forceable_filter(const std::string &filter) const
{
    return _bf_filter_index.find(filter) != _bf_filter_index.end();
}

template <typename T, typename LabelT> int FilterBruteForceIndex<T, LabelT>::load(uint32_t num_threads)
{
    if (false == file_exists(_filter_bf_data_file))
    {
        diskann::cerr << "Index does not have brute force support." << std::endl;
        return 1;
    }
    std::ifstream bf_in(_filter_bf_data_file);
    if (!bf_in.is_open())
    {
        std::stringstream ss;
        ss << "Could not open " << _filter_bf_data_file << " for reading. " << std::endl;
        diskann::cerr << ss.str() << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }

    std::string line;
    std::vector<std::string> label_and_points;
    label_and_points.reserve(2);
    diskann::fast_set points;

    size_t line_num = 0;
    while (getline(bf_in, line))
    {
        split_string(line, '\t', label_and_points);
        if (label_and_points.size() == 2)
        {
            std::istringstream iss(label_and_points[1]);
            std::string pt_str;
            while (getline(iss, pt_str, ','))
            {
                points.insert((location_t)stoull(pt_str));
            }
            assert(points.size() > 0);
            _bf_filter_index.insert(std::pair<std::string,fast_set>(label_and_points[0], points));
            points.clear();
            label_and_points.clear();
        }
        else
        {
            std::stringstream ss;
            ss << "Error reading brute force data at line: " << line_num << " found " << label_and_points.size()
               << " tab separated entries instead of 2" << std::endl;
            diskann::cerr << ss.str();
            throw diskann::ANNException(ss.str(), -1);
        }
        line_num++;
    }
    dump_index(_bf_filter_index, std::cout);
    setup_thread_data(num_threads);
    _is_loaded = true;
    return 0;
}

template <typename T, typename LabelT>
int FilterBruteForceIndex<T, LabelT>::search(const T *query, const std::string &filter, uint32_t k, uint64_t *res_ids,
                                         float *res_dists, QueryStats* stats)
{
    if (this->brute_forceable_filter(filter))
    {
       auto& candidates = _bf_filter_index[filter]; //we know the filter exists.

       ScratchStoreManager<BruteForceScratch> manager(_scratch);
       auto p_bf_scratch = manager.scratch_space();

       return _pq_flash_index->pq_search(query, k, (const location_t *)candidates.data().data(), candidates.size(),
                                         k * NUM_OF_PQ_RESULTS_MULTIPLIER, p_bf_scratch->pq_coords_scratch(),
                                         p_bf_scratch->pq_dists_scratch(), res_ids, res_dists, stats);
    }
    return -1;
}


template DISKANN_DLLEXPORT bool FilterBruteForceIndex<int8_t, uint32_t>::index_available() const;
template DISKANN_DLLEXPORT bool FilterBruteForceIndex<float, uint32_t>::index_available() const;
template DISKANN_DLLEXPORT bool FilterBruteForceIndex<uint8_t, uint32_t>::index_available() const;
template DISKANN_DLLEXPORT bool FilterBruteForceIndex<int8_t, uint16_t>::index_available() const;
template DISKANN_DLLEXPORT bool FilterBruteForceIndex<float, uint16_t>::index_available() const;
template DISKANN_DLLEXPORT bool FilterBruteForceIndex<uint8_t, uint16_t>::index_available() const;

template DISKANN_DLLEXPORT  FilterBruteForceIndex<int8_t, uint32_t>::FilterBruteForceIndex(
    const std::string &disk_index_file, std::shared_ptr<PQFlashIndex<int8_t, uint32_t>> pq_flash_index);
template DISKANN_DLLEXPORT  FilterBruteForceIndex<uint8_t, uint32_t>::FilterBruteForceIndex(
    const std::string &disk_index_file, std::shared_ptr<PQFlashIndex<uint8_t, uint32_t>> pq_flash_index);
template DISKANN_DLLEXPORT  FilterBruteForceIndex<float, uint32_t>::FilterBruteForceIndex(
    const std::string &disk_index_file, std::shared_ptr<PQFlashIndex<float, uint32_t>> pq_flash_index);
template DISKANN_DLLEXPORT  FilterBruteForceIndex<int8_t, uint16_t>::FilterBruteForceIndex(
    const std::string &disk_index_file, std::shared_ptr<PQFlashIndex<int8_t, uint16_t>> pq_flash_index);
template DISKANN_DLLEXPORT  FilterBruteForceIndex<uint8_t, uint16_t>::FilterBruteForceIndex(
    const std::string &disk_index_file, std::shared_ptr<PQFlashIndex<uint8_t, uint16_t>> pq_flash_index);
template DISKANN_DLLEXPORT  FilterBruteForceIndex<float, uint16_t>::FilterBruteForceIndex(
    const std::string &disk_index_file, std::shared_ptr<PQFlashIndex<float, uint16_t>> pq_flash_index);


template DISKANN_DLLEXPORT FilterBruteForceIndex<int8_t, uint32_t>::~FilterBruteForceIndex();
template DISKANN_DLLEXPORT FilterBruteForceIndex<uint8_t, uint32_t>::~FilterBruteForceIndex();
template DISKANN_DLLEXPORT FilterBruteForceIndex<float, uint32_t>::~FilterBruteForceIndex();
template DISKANN_DLLEXPORT FilterBruteForceIndex<int8_t, uint16_t>::~FilterBruteForceIndex();
template DISKANN_DLLEXPORT FilterBruteForceIndex<uint8_t, uint16_t>::~FilterBruteForceIndex();
template DISKANN_DLLEXPORT FilterBruteForceIndex<float, uint16_t>::~FilterBruteForceIndex();


template DISKANN_DLLEXPORT int FilterBruteForceIndex<int8_t, uint32_t>::load(uint32_t num_threads);
template DISKANN_DLLEXPORT int FilterBruteForceIndex<uint8_t, uint32_t>::load(uint32_t num_threads);
template DISKANN_DLLEXPORT int FilterBruteForceIndex<float, uint32_t>::load(uint32_t num_threads);
template DISKANN_DLLEXPORT int FilterBruteForceIndex<int8_t, uint16_t>::load(uint32_t num_threads);
template DISKANN_DLLEXPORT int FilterBruteForceIndex<uint8_t, uint16_t>::load(uint32_t num_threads);
template DISKANN_DLLEXPORT int FilterBruteForceIndex<float, uint16_t>::load(uint32_t num_threads);

template DISKANN_DLLEXPORT int FilterBruteForceIndex<int8_t, uint32_t>::search(const int8_t *query,
                                                                               const std::string &filter, uint32_t k,
                                                                               uint64_t *res_ids, float *res_dists,
                                                                               QueryStats *stats);
template DISKANN_DLLEXPORT int FilterBruteForceIndex<uint8_t, uint32_t>::search(const uint8_t *query,
                                                                               const std::string &filter, uint32_t k,
                                                                                uint64_t *res_ids, float *res_dists,
                                                                                QueryStats *stats);
template DISKANN_DLLEXPORT int FilterBruteForceIndex<float, uint32_t>::search(const float *query,
                                                                               const std::string &filter, uint32_t k,
                                                                              uint64_t *res_ids, float *res_dists,
                                                                              QueryStats *stats);
template DISKANN_DLLEXPORT int FilterBruteForceIndex<int8_t, uint16_t>::search(const int8_t *query,
                                                                               const std::string &filter, uint32_t k,
                                                                               uint64_t *res_ids, float *res_dists,
                                                                               QueryStats *stats);
template DISKANN_DLLEXPORT int FilterBruteForceIndex<uint8_t, uint16_t>::search(const uint8_t *query,
                                                                                const std::string &filter, uint32_t k,
                                                                                uint64_t *res_ids, float *res_dists,
                                                                                QueryStats *stats);
template DISKANN_DLLEXPORT int FilterBruteForceIndex<float, uint16_t>::search(const float *query,
                                                                              const std::string &filter, uint32_t k,
                                                                              uint64_t *res_ids, float *res_dists,
                                                                              QueryStats *stats);



} // namespace diskann
  