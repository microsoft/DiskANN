// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"
#include "index_factory.h"

namespace po = boost::program_options;

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
int build_in_memory_index(const diskann::Metric &metric, const std::string &data_path, const uint32_t R,
                          const uint32_t L, const float alpha, const std::string &save_path, const uint32_t num_threads,
                          const bool use_pq_build, const size_t num_pq_bytes, const bool use_opq,
                          const std::string &label_file, const std::string &universal_label, const uint32_t Lf)
{
    diskann::IndexWriteParameters paras = diskann::IndexWriteParametersBuilder(L, R)
                                              .with_filter_list_size(Lf)
                                              .with_alpha(alpha)
                                              .with_saturate_graph(false)
                                              .with_num_threads(num_threads)
                                              .build();
    std::string labels_file_to_use = save_path + "_label_formatted.txt";
    std::string mem_labels_int_map_file = save_path + "_labels_map.txt";

    size_t data_num, data_dim;
    diskann::get_bin_metadata(data_path, data_num, data_dim);

    diskann::Index<T, TagT, LabelT> index(metric, data_dim, data_num, false, false, false, use_pq_build, num_pq_bytes,
                                          use_opq);
    auto s = std::chrono::high_resolution_clock::now();
    if (label_file == "")
    {
        index.build(data_path.c_str(), data_num, paras);
    }
    else
    {
        convert_labels_string_to_int(label_file, labels_file_to_use, mem_labels_int_map_file, universal_label);
        if (universal_label != "")
        {
            LabelT unv_label_as_num = 0;
            index.set_universal_label(unv_label_as_num);
        }
        index.build_filtered_index(data_path.c_str(), labels_file_to_use, data_num, paras);
    }
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    std::cout << "Indexing time: " << diff.count() << "\n";
    index.save(save_path.c_str());
    if (label_file != "")
        std::remove(labels_file_to_use.c_str());
    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, data_path, index_path_prefix, label_file, universal_label, label_type;
    uint32_t num_threads, R, L, Lf, build_PQ_bytes;
    float alpha;
    bool use_pq_build, use_opq;

    po::options_description desc{"Arguments"};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/cosine>");
        desc.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                           "Input data file in bin format");
        desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                           "Path prefix for saving index file components");
        desc.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64), "Maximum graph degree");
        desc.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                           "Build complexity, higher value results in better graphs");
        desc.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                           "alpha controls density and diameter of graph, set "
                           "1 for sparse graph, "
                           "1.2 or 1.4 for denser graphs with lower diameter");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads used for building index (defaults to "
                           "omp_get_num_procs())");
        desc.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ_bytes)->default_value(0),
                           "Number of PQ bytes to build the index; 0 for full precision "
                           "build");
        desc.add_options()("use_opq", po::bool_switch()->default_value(false),
                           "Set true for OPQ compression while using PQ "
                           "distance comparisons for "
                           "building the index, and false for PQ compression");
        desc.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
                           "Input label file in txt format for Filtered Index search. "
                           "The file should contain comma separated filters for each node "
                           "with each line corresponding to a graph node");
        desc.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                           "Universal label, if using it, only in conjunction with "
                           "labels_file");
        desc.add_options()("FilteredLbuild", po::value<uint32_t>(&Lf)->default_value(0),
                           "Build complexity for filtered points, higher value "
                           "results in better graphs");
        desc.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                           "Storage type of Labels <uint/ushort>, default value is uint which "
                           "will consume memory 4 bytes per filter");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        use_pq_build = (build_PQ_bytes > 0);
        use_opq = vm["use_opq"].as<bool>();
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
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
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    try
    {
        diskann::cout << "Starting index build with R: " << R << "  Lbuild: " << L << "  alpha: " << alpha
                      << "  #threads: " << num_threads << std::endl;

        size_t data_num, data_dim;
        diskann::get_bin_metadata(data_path, data_num, data_dim);

        auto config = diskann::IndexConfigBuilder()
                          .with_metric(metric)
                          .with_dimension(data_dim)
                          .with_max_points(data_num)
                          .with_data_load_store_strategy(diskann::MEMORY)
                          .with_data_type(data_type)
                          .with_label_type(label_type)
                          .is_dynamic_index(false)
                          .is_enable_tags(false)
                          .is_use_opq(use_opq)
                          .is_pq_dist_build(use_pq_build)
                          .with_num_pq_chunks(build_PQ_bytes)
                          .build();

        auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                      .with_filter_list_size(Lf)
                                      .with_alpha(alpha)
                                      .with_saturate_graph(false)
                                      .with_num_threads(num_threads)
                                      .build();

        auto build_params = diskann::IndexBuildParamsBuilder(index_build_params)
                                .with_universal_label(universal_label)
                                .with_label_file(label_file)
                                .with_save_path_prefix(index_path_prefix)
                                .build();
        auto index_factory = diskann::IndexFactory(config);
        auto index = index_factory.create_instance();
        index->build(data_path, data_num, build_params);
        index->save(index_path_prefix.c_str());
        index.reset();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
