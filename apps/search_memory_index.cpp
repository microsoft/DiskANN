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

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "index.h"
#include "memory_mapper.h"
#include "utils.h"
#include "program_options_utils.hpp"
#include "index_factory.h"

namespace po = boost::program_options;


void parse_seller_file(const std::string &label_file, size_t &num_points, std::vector<uint32_t> &location_to_seller)
{
    // Format of Label txt file: filters with comma separators

    std::ifstream infile(label_file);
    if (infile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + label_file, -1);
    }

    std::string line, token;
    uint32_t line_cnt = 0;
    std::set<uint32_t> sellers;
    while (std::getline(infile, line))
    {
        line_cnt++;
    }
    location_to_seller.resize(line_cnt);

    infile.clear();
    infile.seekg(0, std::ios::beg);
    line_cnt = 0;

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        std::istringstream new_iss(token);
        uint32_t seller;
        while (getline(new_iss, token, ','))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            uint32_t token_as_num = (uint32_t)std::stoul(token);
            seller = token_as_num;
            sellers.insert(seller);
        }

        location_to_seller[line_cnt] = seller;
        line_cnt++;
    }
    num_points = (size_t)line_cnt;
    diskann::cout << " Search code: Identified " << sellers.size() << " distinct seller(s) across " << num_points <<" points." << std::endl;
}



template <typename T, typename LabelT = uint32_t>
int search_memory_index(diskann::Metric &metric, const std::string &index_path, const std::string &result_path_prefix,
                        const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                        const uint32_t recall_at, const bool print_all_recalls, const std::vector<uint32_t> &Lvec,
                        const bool dynamic, const bool tags, const bool show_qps_per_thread,
                        const std::vector<std::string> &query_filters, const float fail_if_recall_below, const uint32_t max_K_per_seller = std::numeric_limits<uint32_t>::max(), const bool diverse_search = false, const bool scale_seller_limits = false, const bool post_process = false)
{
    std::cout<<max_K_per_seller <<" " << diverse_search <<" " << scale_seller_limits << " " << post_process << std::endl;
    std::vector<uint32_t> location_to_sellers;
    std::string seller_file = index_path +"_sellers.txt";
    if (file_exists(seller_file)) {
        std::cout<<"Here" << std::endl;
        uint64_t num_pts_seller_file;
        parse_seller_file(seller_file, num_pts_seller_file, location_to_sellers);
    }
    using TagT = uint32_t;
    // Load the query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

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

    //query_num = 2;
    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);

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
    if (not tags || filtered_search)
    {
        cmp_stats = std::vector<uint32_t>(query_num, 0);
    }

    std::vector<TagT> query_result_tags;
    if (tags)
    {
        query_result_tags.resize(recall_at * query_num);
    }

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num, std::numeric_limits<uint32_t>::max());
        query_result_dists[test_id].resize(recall_at * query_num, std::numeric_limits<float>::max());
        std::vector<T *> res = std::vector<T *>();

        //uint32_t maxLperSeller = (max_L_per_seller > 0) ? max_L_per_seller : L;

        //maxLperSeller = (maxLperSeller == 0)? 1 : maxLperSeller;
        uint32_t maxLperSeller = max_K_per_seller;
        if (diverse_search && scale_seller_limits) {
            maxLperSeller = (1.0*L* max_K_per_seller)/(1.0*recall_at);
   //         std::cout<<maxLperSeller<<std::endl;
        }

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            auto qs = std::chrono::high_resolution_clock::now();
            if (filtered_search && !tags)
            {
                std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

                auto retval = index->search_with_filters(query + i * query_aligned_dim, raw_filter, recall_at, L,
                                                         query_result_ids[test_id].data() + i * recall_at,
                                                         query_result_dists[test_id].data() + i * recall_at);
                cmp_stats[i] = retval.second;
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
                    std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

                    index->search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res, true, raw_filter);
                }

                for (int64_t r = 0; r < (int64_t)recall_at; r++)
                {
                    query_result_ids[test_id][recall_at * i + r] = query_result_tags[recall_at * i + r];
                }
            }
            else
            {
                std::vector<uint32_t> results(L,std::numeric_limits<uint32_t>::max());
                std::vector<float> dists(L,std::numeric_limits<float>::max());
                uint32_t K_to_use = (post_process == true) ? L : recall_at;

                if (diverse_search) {
                    
                cmp_stats[i] = index
                                   ->diverse_search(query + i * query_aligned_dim, K_to_use, L, maxLperSeller,
                                            results.data(), dists.data())
                                   .second;
               } else {
//                {
                cmp_stats[i] = index
                                   ->search(query + i * query_aligned_dim, K_to_use, L, 
                                            results.data(), dists.data())
                                   .second;
            }
            if (post_process) {
                    diskann::bestCandidates final_results(recall_at, max_K_per_seller, location_to_sellers);
                    for (uint32_t rr = 0; rr < L; rr++) {
                        final_results.insert(results[rr], dists[rr]);
                    }
                                        
                    for (uint32_t ctr = 0; ctr < std::min(final_results.best_L_nodes.size(), (uint64_t)recall_at); ctr++) {
                        query_result_ids[test_id][recall_at * i + ctr] = final_results.best_L_nodes._data[ctr].id;                        
                        query_result_dists[test_id][recall_at * i + ctr] = final_results.best_L_nodes._data[ctr].distance;                        
                    }
            } else {
                    for (uint32_t ctr = 0; ctr < std::min(results.size(),(uint64_t)recall_at); ctr++) {
                        query_result_ids[test_id][recall_at * i + ctr] = results[ctr];
                        query_result_dists[test_id][recall_at * i + ctr] = dists[ctr];
                    }
            }
            }
            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float)(diff.count() * 1000000);
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

        double displayed_qps = query_num / diff.count();

        if (show_qps_per_thread)
            displayed_qps /= num_threads;

        std::vector<double> recalls;
        if (calc_recall_flag)
        {
            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall, query_result_dists[test_id].data()));
//                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
//                                                            query_result_ids[test_id].data(), recall_at, curr_recall));

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
                      << std::setw(20) << (float)mean_latency << std::setw(15)
                      << (float)latency_stats[(uint64_t)(0.999 * query_num)];
        }
        for (double recall : recalls)
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
    uint32_t num_threads, K, max_L_per_seller;
    std::vector<uint32_t> Lvec;
    bool print_all_recalls, dynamic, tags, show_qps_per_thread, post_process, diverse_search, scale_seller_limits;
    float fail_if_recall_below = 0.0f;

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
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_K_per_seller",
                                       po::value<uint32_t>(&max_L_per_seller)->default_value(0),
                                       "How many results per seller we want search results to contain");
        optional_configs.add_options()("diverse_search",
                                       po::value<bool>(&diverse_search)->default_value(false),
                                       "Whether to run diverse search or baseline search");
        optional_configs.add_options()("scale_seller_limits",
                                       po::value<bool>(&scale_seller_limits)->default_value(false),
                                       "Whether to run scale the max_L_per_seller based on the L value");
        optional_configs.add_options()("post_process",
                                       po::value<bool>(&post_process)->default_value(false),
                                       "Whether to post-processing to ensure correct diversity");


        optional_configs.add_options()(
            "dynamic", po::value<bool>(&dynamic)->default_value(false),
            "Whether the index is dynamic. Dynamic indices must have associated tags.  Default false.");
        optional_configs.add_options()("tags", po::value<bool>(&tags)->default_value(false),
                                       "Whether to search with external identifiers (tags). Default false.");
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);

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

    std::vector<std::string> query_filters;
    if (filter_label != "")
    {
        query_filters.push_back(filter_label);
    }
    else if (query_filters_file != "")
    {
        query_filters = read_file_to_vector_of_strings(query_filters_file);
    }

    try
    {
        if (!query_filters.empty() && label_type == "ushort")
        {
            if (data_type == std::string("int8"))
            {
                return search_memory_index<int8_t, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters, fail_if_recall_below, max_L_per_seller, diverse_search, scale_seller_limits, post_process);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters, fail_if_recall_below, max_L_per_seller, diverse_search, scale_seller_limits, post_process);
            }
            else if (data_type == std::string("float"))
            {
                return search_memory_index<float, uint16_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                            num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                            show_qps_per_thread, query_filters, fail_if_recall_below, max_L_per_seller, diverse_search, scale_seller_limits, post_process);
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
                return search_memory_index<int8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                   num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                   show_qps_per_thread, query_filters, fail_if_recall_below, max_L_per_seller, diverse_search, scale_seller_limits, post_process);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                    num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                    show_qps_per_thread, query_filters, fail_if_recall_below, max_L_per_seller, diverse_search, scale_seller_limits, post_process);
            }
            else if (data_type == std::string("float"))
            {
                return search_memory_index<float>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                  num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                  show_qps_per_thread, query_filters, fail_if_recall_below, max_L_per_seller, diverse_search, scale_seller_limits, post_process);
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
