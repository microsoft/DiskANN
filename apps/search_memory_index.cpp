// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>
#include <random>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "index.h"
#include "roaring.h"
#include "memory_mapper.h"
#include "utils.h"
#include "program_options_utils.hpp"
#include "index_factory.h"

namespace po = boost::program_options;

template <typename T, typename LabelT = uint32_t>
int search_memory_index(diskann::Metric &metric, const std::string &index_path, const std::string &result_path_prefix,
                        const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                        const uint32_t recall_at, const bool print_all_recalls, const std::vector<uint32_t> &Lvec,
                        const bool dynamic, const bool tags, const bool show_qps_per_thread,
                        const std::vector<std::vector<std::vector<std::string>>> &query_filters,
                        const uint32_t filter_penalty_threshold, const uint32_t bruteforce_threshold,
                        uint32_t L_for_print, const float fail_if_recall_below,
                        uint32_t maxN = 10000000, float p1 = 0.1, float p2 = 0.1)
{
    using TagT = uint32_t;
    // Load the query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    std::vector<double> filter_match_time(query_num);
    std::vector<double> dist_cmp_time(query_num);

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

    bool filtered_search = false;
    if (!query_filters.empty())
    {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and size of query "
                         "filters file"
                      << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);

    std::cout << filter_penalty_threshold << " is value of filter_penalty_threshold at driver file" << std::endl;
    auto search_params =
        diskann::IndexSearchParams(*(std::max_element(Lvec.begin(), Lvec.end())), num_threads, filter_penalty_threshold,
                                   bruteforce_threshold);
    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(query_dim)
                      .with_max_points(0)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type(diskann_type_to_name<T>())
                      .with_label_type(diskann_type_to_name<LabelT>())
                      .with_tag_type(diskann_type_to_name<TagT>())
                      .is_dynamic_index(dynamic)
                      .is_enable_tags(tags)
                      .is_concurrent_consolidate(false)
                      .is_pq_dist_build(false)
                      .is_use_opq(false)
                      .with_num_pq_chunks(0)
                      .with_num_frozen_pts(num_frozen_pts)
                      .with_index_search_params(search_params)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));
    std::cout << "Index loaded" << std::endl;

    if (metric == diskann::FAST_L2)
        index->optimize_index_layout();

    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
    if (tags)
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(20) << "Mean Latency (mus)"
                  << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 20 + 15;
    }
    else
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
                  << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "Recall" 
                  #ifdef INSTRUMENT
                  << std::setw(20) << "Brute Latency (mus)" << std::setw(20) << "Brute Recall" <<  std::setw(20) << "Graph Latency (mus)" << std::setw(20)
                  << "Graph Recall" 
                  #else
                  << std::endl;
                  #endif
        table_width += 4 + 12 + 18 + 20 + 15 
        #ifdef INSTRUMENT
        + 20 + 20 + 20 + 20 + 20 + 20;
        #else
        ;
        #endif
    }
    /*    uint32_t recalls_to_print = 0;
        const uint32_t first_recall = print_all_recalls ? 1 : recall_at;
        if (calc_recall_flag)
        {
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
            }
            recalls_to_print = recall_at + 1 - first_recall;
            table_width += recalls_to_print * 12;
        } */
    std::cout << std::endl;
    std::cout << std::string(table_width, '=') << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<std::vector<uint32_t>> query_result_class(Lvec.size());
    std::vector<float> brute_recalls(Lvec.size(), 0);
    std::vector<float> graph_recalls(Lvec.size(), 0);
    std::vector<float> brute_lat(Lvec.size(), 0);
    std::vector<float> graph_lat(Lvec.size(), 0);
    for (auto &x : query_result_class)
        x.resize(query_num, 0);
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats;
    if (not tags || filtered_search)
    {
        cmp_stats = std::vector<uint32_t>(query_num, 0);
    }

    std::vector<TagT> query_result_tags;
    if (tags)
    {
        query_result_tags.resize(recall_at * query_num);
    }
     //query_num = 4;
    //query_num = 1;
    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        query_result_class[test_id].resize(query_num, 0);
        time_to_get_valid = 0;
        time_to_intersect = 0;
        time_to_filter_check_and_compare = 0;
        time_to_detect_penalty = 0;
        num_brutes = 0;
        num_graphs = 0;
        uint32_t L = Lvec[test_id];
        /*        if (L < recall_at)
                {
                    diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at <<
           std::endl; continue;
                }*/

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<T *> res = std::vector<T *>();
        int method_used = 0;
        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            curr_query = i;
/*            std::cout<<"\n\nQuery #" <<i <<"\n*******************\n";
            for (uint32_t rnr =  0; rnr < L; rnr++) {
                std::cout<<std::setw(10)<< gt_ids[i*gt_dim+rnr]<<":" << gt_dists[i*gt_dim+rnr]<<"\t";
            }
            std::cout<<std::endl;*/
            if (L_for_print == L)
                print_qstats = true;
            else
                print_qstats = false;

            auto qs = std::chrono::high_resolution_clock::now();
            if (filtered_search && !tags)
            {
                uint32_t old_b, old_g, old_c;
                old_b = num_brutes;
                old_g = num_graphs;
                method_used = 0;
                std::vector<std::vector<std::string>> raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

                auto retval = index->search_with_filters(query + i * query_aligned_dim, raw_filter, recall_at, L,
                                                         query_result_ids[test_id].data() + i * recall_at,
                                                         query_result_dists[test_id].data() + i * recall_at);
                if (num_graphs > old_g)
                    method_used = 1;
                else
                    method_used = 0;
                cmp_stats[i] = retval.second;
//                                filter_match_time[i] = time_to_get_valid*1000000;
                //                dist_cmp_time[i] = time_to_compare*1000000;
            }
            else if (metric == diskann::FAST_L2)
            {
                index->search_with_optimized_layout(query + i * query_aligned_dim, recall_at, L,
                                                    query_result_ids[test_id].data() + i * recall_at);
            }
            else if (tags)
            {
                if (!filtered_search)
                {
                    index->search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res);
                }
                else
                {
                    std::vector<std::string> raw_filter =
                        query_filters.size() == 1 ? query_filters[0][0] : query_filters[i][0];

                    index->search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res, true,
                                            raw_filter[0]);
                }

                for (int64_t r = 0; r < (int64_t)recall_at; r++)
                {
                    query_result_ids[test_id][recall_at * i + r] = query_result_tags[recall_at * i + r];
                }
            }
            else
            {
                cmp_stats[i] = index
                                   ->search(query + i * query_aligned_dim, recall_at, L,
                                            query_result_ids[test_id].data() + i * recall_at)
                                   .second;
            }
            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float)(diff.count() * 1000000);
            switch (method_used)
            {
            case 0:
                query_result_class[test_id][i] = 0;
                brute_lat[test_id] += latency_stats[i];
                break;
            case 1:
                query_result_class[test_id][i] = 1;
                graph_lat[test_id] += latency_stats[i];
                break;
            }
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

        double displayed_qps = query_num / diff.count();

        if (show_qps_per_thread)
            displayed_qps /= num_threads;

        std::vector<double> recalls;
        if (calc_recall_flag)
        {
            if (L == L_for_print)
            {
                std::ofstream query_stats_file;
                query_stats_file.open(result_path_prefix + "_query_stats.txt");
                query_stats_file << "cmps\tnum correct\tfilt time\tcmp time\tlatency" << std::endl;
                for (size_t i = 0; i < query_num; i++)
                {
                    std::set<uint32_t> gt, res;
                    uint32_t *gt_vec = gt_ids + gt_dim * i;
                    uint32_t *res_vec = query_result_ids[test_id].data() + recall_at * i;
                    size_t tie_breaker = recall_at;
                    if (gt_dists != nullptr)
                    {
                        tie_breaker = recall_at - 1;
                        float *gt_dist_vec = gt_dists + gt_dim * i;
                        while (tie_breaker < gt_dim && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
                            tie_breaker++;
                    }

                    gt.insert(gt_vec, gt_vec + tie_breaker);
                    res.insert(res_vec,
                               res_vec + recall_at); // change to recall_at for recall k@k
                                                     // or dim_or for k@dim_or
                    uint32_t cur_recall = 0;
                    for (auto &v : gt)
                    {
                        if (res.find(v) != res.end())
                        {
                            cur_recall++;
                        }
                    }
                    query_stats_file << cmp_stats[i] << "\t" << cur_recall << "\t" << filter_match_time[i] << "\t"
                                     << dist_cmp_time[i] << "\t" << latency_stats[i] << "\t";
                    for (auto const &r : res)
                        query_stats_file << r << " ";
                    query_stats_file << std::endl;
                }
                query_stats_file.close();
            }

            recalls.reserve(1);

            for (size_t i = 0; i < query_num; i++)
            {
                std::set<uint32_t> gt, res;
                uint32_t *gt_vec = gt_ids + gt_dim * i;
                uint32_t *res_vec = query_result_ids[test_id].data() + recall_at * i;
                size_t tie_breaker = recall_at;
                if (gt_dists != nullptr)
                {
                    tie_breaker = recall_at - 1;
                    float *gt_dist_vec = gt_dists + gt_dim * i;
                    while (tie_breaker < gt_dim && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
                        tie_breaker++;
                }

                gt.insert(gt_vec, gt_vec + tie_breaker);
                res.insert(res_vec,
                           res_vec + recall_at); // change to recall_at for recall k@k
                                                 // or dim_or for k@dim_or
                uint32_t cur_recall = 0;
                for (auto &v : gt)
                {
                    if (res.find(v) != res.end())
                    {
                        cur_recall++;
                    }
                }
                switch (query_result_class[test_id][i])
                {
                case 0:
                    brute_recalls[test_id] += cur_recall;
                    break;
                case 1:
                    graph_recalls[test_id] += cur_recall;
                    break;
                }
            }

            for (uint32_t curr_recall = recall_at; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall));
            }
        }

        std::sort(latency_stats.begin(), latency_stats.end());
        double mean_latency =
            std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

        float avg_cmps = (float)std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float)query_num;

        if (tags && !filtered_search)
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(20) << (float)mean_latency
                      << std::setw(15) << (float)latency_stats[(uint64_t)(0.999 * query_num)];
        }
        else
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << avg_cmps
                      << std::setw(20) << (float)mean_latency << std::setw(15) << (float)recalls[0] 
                      #ifdef INSTRUMENT
                      << std::setw(20) << (float)(brute_lat[test_id] * 1.0) / (num_brutes * 1.0) << std::setw(20)
                      << (float)(brute_recalls[test_id] * 100.0) / (num_brutes * recall_at * 1.0) << std::setw(20)
                      << (float)(graph_lat[test_id] * 1.0) / (num_graphs * 1.0) << std::setw(20)
                      << (float)(graph_recalls[test_id] * 100.0) / (num_graphs * recall_at * 1.0) << " " << (1000000*time_to_detect_penalty) / query_num << "\t" << (1000000*time_to_get_valid) / query_num 
                      //                      << std::setw(20) << (float)(brute_lat[test_id]*1.0) << std::setw(20) <<
                      //                      (float)(brute_recalls[test_id]*100.0)
                      //                     << std::setw(20) << (float)(graph_lat[test_id]*1.0) << std::setw(20) <<
                      //                     (float)(graph_recalls[test_id]*100.0)
                      << std::endl;
                      #else
                      << std::endl;
                      #endif
        }
    }
    std::cout << "num_graphs " << num_graphs << std::endl;
    std::cout << "num_brutes " << num_brutes << std::endl;

    std::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }
        std::string cur_result_path_prefix = result_path_prefix + "_" + std::to_string(L);

        std::string cur_result_path = cur_result_path_prefix + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = cur_result_path_prefix + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id].data(), query_num, recall_at);

        test_id++;
    }

    diskann::aligned_free(query);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, result_path, query_file, gt_file, filter_label, label_type,
        query_filters_file;
    uint32_t num_threads, K, filter_penalty_threshold, bruteforce_threshold, L_for_print, num_local;
    std::vector<uint32_t> Lvec;
    bool print_all_recalls, dynamic, tags, show_qps_per_thread, global_start;
    float fail_if_recall_below = 0.0f;

    uint32_t maxN;
    float p1, p2;

    po::options_description desc{
        program_options_utils::make_program_description("search_memory_index", "Searches in-memory DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print this information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("filter_label",
                                       po::value<std::string>(&filter_label)->default_value(std::string("")),
                                       program_options_utils::FILTER_LABEL_DESCRIPTION);
        optional_configs.add_options()("query_filters_file",
                                       po::value<std::string>(&query_filters_file)->default_value(std::string("")),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        optional_configs.add_options()("filter_penalty_threshold",
                                       po::value<uint32_t>(&filter_penalty_threshold)->default_value(0),
                                       "What penalty threshold to tolerate for multiple filter search");
        optional_configs.add_options()("bruteforce_threshold",
                                       po::value<uint32_t>(&bruteforce_threshold)->default_value(0),
                                       "Threshold under which we bruteforce the filtered search");
        optional_configs.add_options()("use_global_start",
                                       po::value<bool>(&global_start)->default_value(false),
                                       "Whether or not to use global start or predicate-aware starting point in graph search");
        optional_configs.add_options()("expand_two_hops",
                                       po::value<bool>(&expand_two_hops)->default_value(false),
                                       "Whether or not to use ACORN-like idea of two hops at search");                                       
        optional_configs.add_options()("num_local_start",
                                       po::value<uint32_t>(&num_local)->default_value(0),
                                       "How many local start points to use");


        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()(
            "L_to_print", po::value<uint32_t>(&L_for_print)->default_value(0),
            "Which of the given L's to provide query statistics for (written to index_path + \"_query_stats.txt\")");
        optional_configs.add_options()(
            "dynamic", po::value<bool>(&dynamic)->default_value(false),
            "Whether the index is dynamic. Dynamic indices must have associated tags.  Default false.");
        optional_configs.add_options()("tags", po::value<bool>(&tags)->default_value(false),
                                       "Whether to search with external identifiers (tags). Default false.");
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);

        optional_configs.add_options()("maxN", po::value<uint32_t>(&maxN)->default_value(10000000), "maxN");
        optional_configs.add_options()("p1", po::value<float>(&p1)->default_value(0.1), "p1");
        optional_configs.add_options()("p2", po::value<float>(&p2)->default_value(0.1), "p2");

        // Output controls
        po::options_description output_controls("Output controls");
        output_controls.add_options()("print_all_recalls", po::bool_switch(&print_all_recalls),
                                      "Print recalls at all positions, from 1 up to specified "
                                      "recall_at value");
        output_controls.add_options()("print_qps_per_thread", po::bool_switch(&show_qps_per_thread),
                                      "Print overall QPS divided by the number of threads in "
                                      "the output table");

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs).add(output_controls);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if ((dist_fn == std::string("mips")) && (data_type == std::string("float")))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else if ((dist_fn == std::string("fast_l2")) && (data_type == std::string("float")))
    {
        metric = diskann::Metric::FAST_L2;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                     "supported in general, and mips/fast_l2 only for floating "
                     "point data."
                  << std::endl;
        return -1;
    }

    if (dynamic && not tags)
    {
        std::cerr << "Tags must be enabled while searching dynamically built indices" << std::endl;
        return -1;
    }

    if (fail_if_recall_below < 0.0 || fail_if_recall_below >= 100.0)
    {
        std::cerr << "fail_if_recall_below parameter must be between 0 and 100%" << std::endl;
        return -1;
    }

    if (filter_label != "" && query_filters_file != "")
    {
        std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
        return -1;
    }

    std::vector<std::vector<std::vector<std::string>>> query_filters;
    if (filter_label != "")
    {
        std::vector<std::vector<std::string>> single_filter;
        std::vector<std::string> tmp;
        tmp.push_back(filter_label);
        single_filter.push_back(tmp);
        query_filters.push_back(single_filter);
    }
    else if (query_filters_file != "")
    {
        query_filters = read_file_to_vector_of_vector_of_strings(query_filters_file);
        for (auto &x : query_filters[0]) {
            std::cout<<"(";
            for (auto &y : x) {
                std::cout<<y<<"|";
            }
            std::cout<<")&";
        }
    }

    use_global_start = global_start;
    num_start_points = num_local;

    std::cout<<"Num local start points: " << num_start_points << std::endl;

    try
    {
        if (!query_filters.empty() && label_type == "ushort")
        {
            if (data_type == std::string("int8"))
            {
                return search_memory_index<int8_t, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters, filter_penalty_threshold,
                    bruteforce_threshold, L_for_print, fail_if_recall_below);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters, filter_penalty_threshold,
                    bruteforce_threshold, L_for_print, fail_if_recall_below);
            }
            else if (data_type == std::string("float"))
            {
                return search_memory_index<float, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters, filter_penalty_threshold,
                    bruteforce_threshold, L_for_print, fail_if_recall_below);
            }
            else
            {
                std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("int8"))
            {
                return search_memory_index<int8_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters, filter_penalty_threshold,
                    bruteforce_threshold, L_for_print, fail_if_recall_below, maxN, p1, p2);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters, filter_penalty_threshold,
                    bruteforce_threshold, L_for_print, fail_if_recall_below);
            }
            else if (data_type == std::string("float"))
            {
                return search_memory_index<float>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters, filter_penalty_threshold,
                    bruteforce_threshold, L_for_print, fail_if_recall_below);
            }
            else
            {
                std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                return -1;
            }
        }
    }
    catch (std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}
