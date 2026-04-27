// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <thread>
#include <vector>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"
#include "index_factory.h"

namespace po = boost::program_options;

template <typename T>
uint32_t brute_force_knn(const T *base, size_t npts, size_t dim, const T *query, size_t K, uint32_t *result_ids,
                         float *result_dists)
{
    // Priority queue: max-heap of (dist, id)
    std::vector<std::pair<float, uint32_t>> heap;
    heap.reserve(K + 1);

    for (size_t i = 0; i < npts; i++)
    {
        const T *vec = base + i * dim;
        float dist = 0;
        for (size_t d = 0; d < dim; d++)
        {
            float diff = (float)query[d] - (float)vec[d];
            dist += diff * diff;
        }

        if (heap.size() < K)
        {
            heap.push_back({dist, (uint32_t)i});
            std::push_heap(heap.begin(), heap.end());
        }
        else if (dist < heap.front().first)
        {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {dist, (uint32_t)i};
            std::push_heap(heap.begin(), heap.end());
        }
    }

    std::sort_heap(heap.begin(), heap.end());
    for (size_t i = 0; i < heap.size(); i++)
    {
        result_ids[i] = heap[i].second;
        result_dists[i] = heap[i].first;
    }
    return (uint32_t)heap.size();
}

template <typename T>
void compute_ground_truth(const std::string &data_file, size_t num_queries, size_t K, uint32_t num_threads,
                          const std::string &gt_output_file)
{
    T *data = nullptr;
    size_t npts, dim, aligned_dim;
    diskann::load_aligned_bin<T>(data_file, data, npts, dim, aligned_dim);
    std::cout << "Loaded " << npts << " points, dim=" << dim << " for ground truth computation" << std::endl;

    if (num_queries > npts)
        num_queries = npts;

    std::vector<uint32_t> gt_ids(num_queries * K);
    std::vector<float> gt_dists(num_queries * K);

    std::atomic<uint32_t> progress{0};
    auto start = std::chrono::high_resolution_clock::now();

    // Parallel brute-force using std::thread
    std::vector<std::thread> threads;
    size_t queries_per_thread = (num_queries + num_threads - 1) / num_threads;

    for (uint32_t t = 0; t < num_threads; t++)
    {
        size_t q_start = t * queries_per_thread;
        size_t q_end = std::min(q_start + queries_per_thread, num_queries);
        if (q_start >= q_end)
            break;

        threads.emplace_back([&, q_start, q_end]() {
            for (size_t q = q_start; q < q_end; q++)
            {
                const T *query = data + q * aligned_dim;
                brute_force_knn<T>(data, npts, dim, query, K, gt_ids.data() + q * K, gt_dists.data() + q * K);

                uint32_t done = progress.fetch_add(1) + 1;
                if (done % 100 == 0)
                {
                    std::cout << "\rGround truth progress: " << done << "/" << num_queries << std::flush;
                }
            }
        });
    }

    for (auto &th : threads)
        th.join();

    auto elapsed =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "\nGround truth computed in " << elapsed << "s" << std::endl;

    // Save: ids then dists (truthset format with distances)
    diskann::save_bin<uint32_t>(gt_output_file, gt_ids.data(), num_queries, K);

    // Append distances after the ids file
    {
        std::ofstream writer(gt_output_file, std::ios::binary | std::ios::app);
        writer.write((char *)gt_dists.data(), num_queries * K * sizeof(float));
        writer.close();
    }
    std::cout << "Ground truth saved to " << gt_output_file << std::endl;

    diskann::aligned_free(data);
}

template <typename T>
int run_benchmark(diskann::Metric &metric, const std::string &index_path, const std::string &gt_file,
                  uint32_t num_threads, uint32_t K, const std::vector<uint32_t> &Lvec, size_t num_queries,
                  const std::string &graph_compress)
{
    // Load query vectors from the data file (first num_queries vectors)
    T *query = nullptr;
    size_t query_num, query_dim, query_aligned_dim;
    std::string data_file = index_path + ".data";
    diskann::load_aligned_bin<T>(data_file, query, query_num, query_dim, query_aligned_dim);
    std::cout << "Loaded data file: " << query_num << " points, dim=" << query_dim << std::endl;

    if (num_queries > query_num)
        num_queries = query_num;
    std::cout << "Using first " << num_queries << " vectors as queries" << std::endl;

    // Load ground truth if available
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t gt_num = 0, gt_dim = 0;
    bool calc_recall = false;
    if (!gt_file.empty() && gt_file != "null" && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num >= num_queries)
        {
            calc_recall = true;
            std::cout << "Ground truth loaded: " << gt_num << " queries, K=" << gt_dim << std::endl;
        }
        else
        {
            std::cout << "Warning: ground truth has fewer queries than requested" << std::endl;
        }
    }

    // Select graph store strategy
    diskann::GraphStoreStrategy graph_strategy = diskann::GraphStoreStrategy::REFORMAT_STATICMEMORY;
    if (graph_compress == "delta_varint")
    {
        graph_strategy = diskann::GraphStoreStrategy::COMPRESSED_REFORMAT_STATICMEMORY;
        std::cout << "Using compressed graph: delta + varint" << std::endl;
    }
    else if (graph_compress == "delta_bitpack")
    {
        graph_strategy = diskann::GraphStoreStrategy::BITPACK_REFORMAT_STATICMEMORY;
        std::cout << "Using compressed graph: delta + bitpack" << std::endl;
    }
    else if (graph_compress == "reorder_varint")
    {
        graph_strategy = diskann::GraphStoreStrategy::REORDER_COMPRESSED_REFORMAT_STATICMEMORY;
        std::cout << "Using compressed graph: BFS reorder + delta + varint" << std::endl;
    }
    else if (graph_compress == "streamvbyte")
    {
        graph_strategy = diskann::GraphStoreStrategy::STREAMVBYTE_REFORMAT_STATICMEMORY;
        std::cout << "Using compressed graph: StreamVByte SIMD delta" << std::endl;
    }
    else if (graph_compress == "reorder_streamvbyte")
    {
        graph_strategy = diskann::GraphStoreStrategy::REORDER_STREAMVBYTE_REFORMAT_STATICMEMORY;
        std::cout << "Using compressed graph: BFS reorder + StreamVByte SIMD delta" << std::endl;
    }
    else if (graph_compress == "maskedvbyte")
    {
        graph_strategy = diskann::GraphStoreStrategy::MASKEDVBYTE_REFORMAT_STATICMEMORY;
        std::cout << "Using compressed graph: MaskedVByte BMI2 delta" << std::endl;
    }
    else if (graph_compress == "lz4")
    {
        graph_strategy = diskann::GraphStoreStrategy::LZ4_REFORMAT_STATICMEMORY;
        std::cout << "Using compressed graph: LZ4 per-node delta" << std::endl;
    }
    else if (graph_compress.empty() || graph_compress == "none")
    {
        std::cout << "Using uncompressed graph (baseline)" << std::endl;
    }
    else
    {
        std::cerr << "Unknown graph_compress method: " << graph_compress << std::endl;
        return -1;
    }

    // Load index
    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);
    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(query_dim)
                      .with_max_points(0)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(graph_strategy)
                      .with_data_type(diskann_type_to_name<T>())
                      .is_dynamic_index(false)
                      .is_enable_tags(false)
                      .with_num_frozen_pts(num_frozen_pts)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();

    auto load_start = std::chrono::high_resolution_clock::now();
    index->load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));
    auto load_elapsed =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - load_start).count();
    std::cout << "Index loaded in " << load_elapsed << "s" << std::endl;

    // Report memory usage
    auto stats = index->get_table_stats();
    std::cout << "\n=== Memory Usage ===" << std::endl;
    std::cout << "  Graph memory:  " << stats.graph_mem_usage / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Node memory:   " << stats.node_mem_usage / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Total memory:  " << stats.total_mem_usage / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Node count:    " << stats.node_count << std::endl;
    std::cout << std::endl;

    // Print header
    std::cout << std::setw(6) << "L" << std::setw(12) << "QPS" << std::setw(18) << "Mean Lat (us)" << std::setw(18)
              << "P99.9 Lat (us)";
    if (calc_recall)
        std::cout << std::setw(12) << "Recall@" + std::to_string(K);
    std::cout << std::endl;
    std::cout << std::string(66 + (calc_recall ? 12 : 0), '=') << std::endl;

    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);

    for (uint32_t L : Lvec)
    {
        if (L < K)
        {
            std::cout << "Skipping L=" << L << " < K=" << K << std::endl;
            continue;
        }

        std::vector<uint32_t> result_ids(num_queries * K);
        std::vector<float> latency_stats(num_queries, 0);

        auto wall_start = std::chrono::high_resolution_clock::now();

        // Parallel search using std::thread
        std::vector<std::thread> threads;
        size_t queries_per_thread = (num_queries + num_threads - 1) / num_threads;

        for (uint32_t t = 0; t < num_threads; t++)
        {
            size_t q_start = t * queries_per_thread;
            size_t q_end = std::min(q_start + queries_per_thread, num_queries);
            if (q_start >= q_end)
                break;

            threads.emplace_back([&, q_start, q_end]() {
                for (size_t i = q_start; i < q_end; i++)
                {
                    auto qs = std::chrono::high_resolution_clock::now();
                    index->search(query + i * query_aligned_dim, K, L, result_ids.data() + i * K);
                    auto qe = std::chrono::high_resolution_clock::now();
                    latency_stats[i] = (float)(std::chrono::duration<double, std::micro>(qe - qs).count());
                }
            });
        }

        for (auto &th : threads)
            th.join();

        double wall_elapsed =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wall_start).count();
        double qps = num_queries / wall_elapsed;

        std::sort(latency_stats.begin(), latency_stats.end());
        double mean_latency =
            std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / (double)num_queries;
        float p999_latency = latency_stats[(uint64_t)(0.999 * num_queries)];

        std::cout << std::setw(6) << L << std::setw(12) << qps << std::setw(18) << mean_latency << std::setw(18)
                  << p999_latency;

        if (calc_recall)
        {
            double recall = diskann::calculate_recall((uint32_t)num_queries, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                      result_ids.data(), K, K);
            std::cout << std::setw(12) << recall;
        }
        std::cout << std::endl;
    }

    diskann::aligned_free(query);
    if (gt_ids)
        delete[] gt_ids;
    if (gt_dists)
        delete[] gt_dists;

    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path, gt_file, graph_compress;
    uint32_t num_threads, K;
    size_t num_queries;
    std::vector<uint32_t> Lvec;
    bool compute_gt;

    po::options_description desc{"bench_graph_compress: Benchmark tool for graph compression experiments"};
    try
    {
        desc.add_options()("help,h", "Print help message");

        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       "Data type: float, int8, uint8");
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       "Distance function: l2, mips, cosine");
        required_configs.add_options()("index_path", po::value<std::string>(&index_path)->required(),
                                       "Index path prefix (without .bin.data suffix)");
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(), "Number of results (K)");
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       "Search list sizes (L values)");

        po::options_description optional_configs("Optional");
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(""),
                                       "Ground truth file path");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(std::thread::hardware_concurrency()),
                                       "Number of search threads");
        optional_configs.add_options()("num_queries",
                                       po::value<size_t>(&num_queries)->default_value(10000),
                                       "Number of queries (taken from start of data file)");
        optional_configs.add_options()("compute_gt", po::bool_switch(&compute_gt)->default_value(false),
                                       "Compute ground truth and save to gt_file, then exit");
        optional_configs.add_options()("graph_compress",
                                       po::value<std::string>(&graph_compress)->default_value(""),
                                       "Graph compression method: none, delta_varint, delta_bitpack, reorder_varint, streamvbyte, reorder_streamvbyte, maskedvbyte, lz4");

        desc.add(required_configs).add(optional_configs);

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
    if (dist_fn == "l2")
        metric = diskann::Metric::L2;
    else if (dist_fn == "mips" && data_type == "float")
        metric = diskann::Metric::INNER_PRODUCT;
    else if (dist_fn == "cosine")
        metric = diskann::Metric::COSINE;
    else
    {
        std::cerr << "Unsupported distance function: " << dist_fn << std::endl;
        return -1;
    }

    // Compute ground truth mode
    if (compute_gt)
    {
        if (gt_file.empty())
        {
            std::cerr << "Must specify --gt_file when using --compute_gt" << std::endl;
            return -1;
        }
        std::string data_file = index_path + ".data";
        if (data_type == "uint8")
            compute_ground_truth<uint8_t>(data_file, num_queries, K, num_threads, gt_file);
        else if (data_type == "int8")
            compute_ground_truth<int8_t>(data_file, num_queries, K, num_threads, gt_file);
        else if (data_type == "float")
            compute_ground_truth<float>(data_file, num_queries, K, num_threads, gt_file);
        else
        {
            std::cerr << "Unsupported data type: " << data_type << std::endl;
            return -1;
        }
        return 0;
    }

    // Benchmark mode
    try
    {
        if (data_type == "uint8")
            return run_benchmark<uint8_t>(metric, index_path, gt_file, num_threads, K, Lvec, num_queries, graph_compress);
        else if (data_type == "int8")
            return run_benchmark<int8_t>(metric, index_path, gt_file, num_threads, K, Lvec, num_queries, graph_compress);
        else if (data_type == "float")
            return run_benchmark<float>(metric, index_path, gt_file, num_threads, K, Lvec, num_queries, graph_compress);
        else
        {
            std::cerr << "Unsupported data type: " << data_type << std::endl;
            return -1;
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
