// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"
#include "../src/program_options_utils.hpp"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"

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

    po::options_description desc{
            program_options_utils::make_program_description("build_memory_index", "Build a memory-based DiskANN index.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64), program_options_utils::MAX_BUILD_DEGREE);
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);
        optional_configs.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                           program_options_utils::GRAPH_BUILD_ALPHA);
        optional_configs.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ_bytes)->default_value(0),
                                       program_options_utils::BUIlD_GRAPH_PQ_BYTES);
        optional_configs.add_options()("use_opq", po::bool_switch()->default_value(false),program_options_utils::USE_OPQ);
        optional_configs.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
                                       program_options_utils::LABEL_FILE);
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL);

        optional_configs.add_options()("FilteredLbuild,Lf", po::value<uint32_t>(&Lf)->default_value(0),
                                       program_options_utils::FILTERED_LBUILD);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

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
        if (label_file != "" && label_type == "ushort")
        {
            if (data_type == std::string("int8"))
                return build_in_memory_index<int8_t, uint32_t, uint16_t>(
                    metric, data_path, R, L, alpha, index_path_prefix, num_threads, use_pq_build, build_PQ_bytes,
                    use_opq, label_file, universal_label, Lf);
            else if (data_type == std::string("uint8"))
                return build_in_memory_index<uint8_t, uint32_t, uint16_t>(
                    metric, data_path, R, L, alpha, index_path_prefix, num_threads, use_pq_build, build_PQ_bytes,
                    use_opq, label_file, universal_label, Lf);
            else if (data_type == std::string("float"))
                return build_in_memory_index<float, uint32_t, uint16_t>(
                    metric, data_path, R, L, alpha, index_path_prefix, num_threads, use_pq_build, build_PQ_bytes,
                    use_opq, label_file, universal_label, Lf);
            else
            {
                std::cout << "Unsupported type. Use one of int8, uint8 or float." << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("int8"))
                return build_in_memory_index<int8_t>(metric, data_path, R, L, alpha, index_path_prefix, num_threads,
                                                     use_pq_build, build_PQ_bytes, use_opq, label_file, universal_label,
                                                     Lf);
            else if (data_type == std::string("uint8"))
                return build_in_memory_index<uint8_t>(metric, data_path, R, L, alpha, index_path_prefix, num_threads,
                                                      use_pq_build, build_PQ_bytes, use_opq, label_file,
                                                      universal_label, Lf);
            else if (data_type == std::string("float"))
                return build_in_memory_index<float>(metric, data_path, R, L, alpha, index_path_prefix, num_threads,
                                                    use_pq_build, build_PQ_bytes, use_opq, label_file, universal_label,
                                                    Lf);
            else
            {
                std::cout << "Unsupported type. Use one of int8, uint8 or float." << std::endl;
                return -1;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
