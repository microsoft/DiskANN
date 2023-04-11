#include <numeric>
#include "utils.h"
#include "index.h"
#include "index_factory.h"

namespace diskann
{
template <typename T, typename TagT, typename LabelT>
MemoryIndex<T, TagT, LabelT>::MemoryIndex(IndexConfig &config) : _config(config)
{
}

/*Initialize Index class with provided Dimenrion and max points*/
template <typename T, typename TagT, typename LabelT>
void MemoryIndex<T, TagT, LabelT>::initialize_index(size_t dimension, size_t max_points, size_t frozen_points)
{

    _index = std::make_unique<Index<T, TagT, LabelT>>(
        _config.metric, dimension, max_points, _config.dynamic_index, _config.enable_tags,
        _config.concurrent_consolidate, _config.pq_dist_build, _config.num_pq_chunks, _config.use_opq, frozen_points);
}

template <typename T, typename TagT, typename LabelT>
void MemoryIndex<T, TagT, LabelT>::build(const std::string &data_file, BuildParams &build_params,
                                         const std::string &save_path)
{
    // Initialize index
    size_t data_num, data_dim;
    diskann::get_bin_metadata(data_file, data_num, data_dim);
    if (data_dim == 0 || data_num == 0)
    {
        throw ANNException("ERROR: Data Dimenrion mismatch", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    initialize_index(data_dim, data_num);

    // Build index
    auto s = std::chrono::high_resolution_clock::now();
    if (_config.filtered_build)
    {
        build_filtered_index(data_file, build_params, save_path);
    }
    else
    {
        build_unfiltered_index(data_file, build_params);
    }
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
    std::cout << "Indexing time: " << diff.count() << "\n";
    // Save index
    _index->save(save_path.c_str());
}

template <typename T, typename TagT, typename LabelT>
void MemoryIndex<T, TagT, LabelT>::search(const std::string &query_file, SearchParams &search_params,
                                          const std::vector<std::string> &query_filters)
{
    // NEED to implament code...
}

template <typename T, typename TagT, typename LabelT>
int MemoryIndex<T, TagT, LabelT>::search_prebuilt_index(const std::string &index_file, const std::string &query_file,
                                                        SearchParams &search_params,
                                                        std::vector<std::string> &query_filters,
                                                        const std::string &result_path_prefix)
{
    std::string truthset_file = search_params.gt_file;
    uint32_t recall_at = search_params.K, num_threads = search_params.num_threads;
    bool show_qps_per_thread = search_params.show_qps_per_thread, print_all_recalls = search_params.print_all_recalls,
         fail_if_recall_below = search_params.fail_if_recall_below;
    std::vector<uint32_t> Lvec = search_params.Lvec;

    // Load the query file
    T *query = nullptr;
    size_t query_num, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    // Load Truthset
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t gt_num, gt_dim;
    // Check for ground truth
    bool calc_recall_flag = false;
    if (truthset_file != std::string("null") && file_exists(truthset_file))
    {
        diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }
    else
    {
        diskann::cout << " Truthset file " << truthset_file << " not found. Not computing recall." << std::endl;
    }

    // is this a filtered search
    bool filtered_search = false;
    if (!query_filters.empty())
    {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and size of query filters file" << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    // init index class
    const size_t num_frozen_pts = diskann::Index<T, TagT, LabelT>::get_graph_num_frozen_points(index_file);
    _index.release();
    initialize_index(query_dim, 0, num_frozen_pts);

    // load index
    _index->load(index_file.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));
    std::cout << "Index loaded" << std::endl;
    if (_config.metric == diskann::FAST_L2)
        _index->optimize_index_layout();

    std::cout << "Using " << num_threads << " threads to search" << std::endl;

    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
    if (_config.enable_tags)
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(20) << "Mean Latency (mus)"
                  << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 20 + 15;
    }
    else
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
                  << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 18 + 20 + 15;
    }
    uint32_t recalls_to_print = 0;
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;
    if (calc_recall_flag)
    {
        for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
        {
            std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
        }
        recalls_to_print = recall_at + 1 - first_recall;
        table_width += recalls_to_print * 12;
    }
    std::cout << std::endl;
    std::cout << std::string(table_width, '=') << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats;
    if (not _config.enable_tags)
    {
        cmp_stats = std::vector<uint32_t>(query_num, 0);
    }

    std::vector<TagT> query_result_tags;
    if (_config.enable_tags)
    {
        query_result_tags.resize(recall_at * query_num);
    }

    float best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint64_t L = Lvec[test_id];
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<T *> res = std::vector<T *>();

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            auto qs = std::chrono::high_resolution_clock::now();
            if (filtered_search)
            {
                LabelT filter_label_as_num;
                if (query_filters.size() == 1)
                {
                    filter_label_as_num = _index->get_converted_label(query_filters[0]);
                }
                else
                {
                    filter_label_as_num = _index->get_converted_label(query_filters[i]);
                }
                auto retval = _index->search_with_filters(query + i * query_aligned_dim, filter_label_as_num, recall_at,
                                                          L, query_result_ids[test_id].data() + i * recall_at,
                                                          query_result_dists[test_id].data() + i * recall_at);
                cmp_stats[i] = retval.second;
            }
            else if (_config.metric == diskann::FAST_L2)
            {
                _index->search_with_optimized_layout(query + i * query_aligned_dim, recall_at, L,
                                                     query_result_ids[test_id].data() + i * recall_at);
            }
            else if (_config.enable_tags)
            {
                _index->search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                         query_result_tags.data() + i * recall_at, nullptr, res);
                for (int64_t r = 0; r < (int64_t)recall_at; r++)
                {
                    query_result_ids[test_id][recall_at * i + r] = query_result_tags[recall_at * i + r];
                }
            }
            else
            {
                cmp_stats[i] = _index
                                   ->search(query + i * query_aligned_dim, recall_at, L,
                                            query_result_ids[test_id].data() + i * recall_at)
                                   .second;
            }
            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = diff.count() * 1000000;
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

        float displayed_qps = static_cast<float>(query_num) / diff.count();

        if (show_qps_per_thread)
            displayed_qps /= num_threads;

        std::vector<float> recalls;
        if (calc_recall_flag)
        {
            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall));
            }
        }

        std::sort(latency_stats.begin(), latency_stats.end());
        float mean_latency =
            std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

        float avg_cmps = (float)std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float)query_num;

        if (_config.enable_tags)
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(20) << (float)mean_latency
                      << std::setw(15) << (float)latency_stats[(uint64_t)(0.999 * query_num)];
        }
        else
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << avg_cmps
                      << std::setw(20) << (float)mean_latency << std::setw(15)
                      << (float)latency_stats[(uint64_t)(0.999 * query_num)];
        }
        for (float recall : recalls)
        {
            std::cout << std::setw(12) << recall;
            best_recall = std::max(recall, best_recall);
        }
        std::cout << std::endl;
    }

    std::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }
        std::string cur_result_path = "res_" + std::to_string(L) + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);
        test_id++;
    }

    diskann::aligned_free(query);

    return best_recall >= fail_if_recall_below ? 0 : -1;
}

template <typename T, typename TagT, typename LabelT>
void MemoryIndex<T, TagT, LabelT>::build_filtered_index(const std::string &data_file, BuildParams &build_params,
                                                        const std::string &save_path)
{

    if (build_params.label_file != "" && !file_exists(build_params.label_file))
    {
        diskann::cout << "Error: for filtered_build you need to provide path to label_file in params." << std::endl;
        exit(-1);
    }

    std::string labels_file_to_use = save_path + "_label_formatted.txt";
    std::string mem_labels_int_map_file = save_path + "_labels_map.txt";
    convert_labels_string_to_int(build_params.label_file.c_str(), labels_file_to_use, mem_labels_int_map_file,
                                 build_params.universal_label);
    if (build_params.universal_label != "")
    {
        LabelT unv_label_as_num = 0;
        _index->set_universal_label(unv_label_as_num);
    }
    _index->build_filtered_index(data_file.c_str(), labels_file_to_use, _index->get_max_points(),
                                 build_params.index_write_params);

    if (build_params.label_file != "")
    {
        clean_up_artifacts({labels_file_to_use}, {});
    }
}

template <typename T, typename TagT, typename LabelT>
void MemoryIndex<T, TagT, LabelT>::build_unfiltered_index(const std::string &data_file, BuildParams &build_params)
{
    _index->build(data_file.c_str(), _index->get_max_points(), build_params.index_write_params);
}

template DISKANN_DLLEXPORT class MemoryIndex<float, uint32_t, uint16_t>;
template DISKANN_DLLEXPORT class MemoryIndex<uint8_t, uint32_t, uint16_t>;
template DISKANN_DLLEXPORT class MemoryIndex<int8_t, uint32_t, uint16_t>;

template DISKANN_DLLEXPORT class MemoryIndex<float, uint32_t, uint32_t>;
template DISKANN_DLLEXPORT class MemoryIndex<uint8_t, uint32_t, uint32_t>;
template DISKANN_DLLEXPORT class MemoryIndex<int8_t, uint32_t, uint32_t>;

} // namespace diskann
