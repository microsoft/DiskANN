// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <boost/program_options.hpp>

#include "utils.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "index.h"
#include "partition.h"
#include "program_options_utils.hpp"
#include "bfloat16.h"

namespace po = boost::program_options;

static int convert_bf16_bin_to_f32_bin(const std::string &bf16_path, const std::string &f32_path)
{
    std::ifstream reader(bf16_path, std::ios::binary);
    if (!reader)
    {
        diskann::cerr << "Error: could not open input file " << bf16_path << std::endl;
        return -1;
    }
    std::ofstream writer(f32_path, std::ios::binary);
    if (!writer)
    {
        diskann::cerr << "Error: could not open output file " << f32_path << std::endl;
        return -1;
    }

    uint32_t npts = 0, dim = 0;
    reader.read(reinterpret_cast<char *>(&npts), sizeof(uint32_t));
    reader.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
    if (!reader)
    {
        diskann::cerr << "Error: failed to read header from " << bf16_path << std::endl;
        return -1;
    }
    writer.write(reinterpret_cast<const char *>(&npts), sizeof(uint32_t));
    writer.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));

    constexpr size_t kBlockElems = 1u << 20; // 1M elements (~2MB bf16, ~4MB float)
    std::vector<diskann::bfloat16> in_buf;
    std::vector<float> out_buf;
    in_buf.resize(kBlockElems);
    out_buf.resize(kBlockElems);

    const uint64_t total_elems = static_cast<uint64_t>(npts) * static_cast<uint64_t>(dim);
    uint64_t done = 0;
    while (done < total_elems)
    {
        const size_t this_block = static_cast<size_t>(std::min<uint64_t>(kBlockElems, total_elems - done));
        reader.read(reinterpret_cast<char *>(in_buf.data()), this_block * sizeof(diskann::bfloat16));
        if (!reader)
        {
            diskann::cerr << "Error: failed reading bf16 payload from " << bf16_path << std::endl;
            return -1;
        }
        for (size_t i = 0; i < this_block; i++)
        {
            out_buf[i] = static_cast<float>(in_buf[i]);
        }
        writer.write(reinterpret_cast<const char *>(out_buf.data()), this_block * sizeof(float));
        done += this_block;
    }
    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, data_path, index_path_prefix, codebook_prefix, label_file, universal_label,
        label_type;
    uint32_t num_threads, R, L, disk_PQ, build_PQ, QD, Lf, filter_threshold;
    float B, M;
    bool append_reorder_data = false;
    bool use_opq = false;

    po::options_description desc{
        program_options_utils::make_program_description("build_disk_index", "Build a disk-based index.")};
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
        required_configs.add_options()("search_DRAM_budget,B", po::value<float>(&B)->required(),
                                       "DRAM budget in GB for searching the index to set the "
                                       "compressed level for data while search happens");
        required_configs.add_options()("build_DRAM_budget,M", po::value<float>(&M)->required(),
                                       "DRAM budget in GB for building the index");

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE);
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);
        optional_configs.add_options()("QD", po::value<uint32_t>(&QD)->default_value(0),
                                       " Quantized Dimension for compression");
        optional_configs.add_options()("codebook_prefix", po::value<std::string>(&codebook_prefix)->default_value(""),
                                       "Path prefix for pre-trained codebook");
        optional_configs.add_options()("PQ_disk_bytes", po::value<uint32_t>(&disk_PQ)->default_value(0),
                                       "Number of bytes to which vectors should be compressed "
                                       "on SSD; 0 for no compression");
        optional_configs.add_options()("append_reorder_data", po::bool_switch()->default_value(false),
                                       "Include full precision data in the index. Use only in "
                                       "conjuction with compressed data on SSD.");
        optional_configs.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ)->default_value(0),
                                       program_options_utils::BUIlD_GRAPH_PQ_BYTES);
        optional_configs.add_options()("use_opq", po::bool_switch()->default_value(false),
                                       program_options_utils::USE_OPQ);
        optional_configs.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
                                       program_options_utils::LABEL_FILE);
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL);
        optional_configs.add_options()("FilteredLbuild", po::value<uint32_t>(&Lf)->default_value(0),
                                       program_options_utils::FILTERED_LBUILD);
        optional_configs.add_options()("filter_threshold,F", po::value<uint32_t>(&filter_threshold)->default_value(0),
                                       "Threshold to break up the existing nodes to generate new graph "
                                       "internally where each node has a maximum F labels.");
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
        if (vm["append_reorder_data"].as<bool>())
            append_reorder_data = true;
        if (vm["use_opq"].as<bool>())
            use_opq = true;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    bool use_filters = (label_file != "") ? true : false;
    diskann::Metric metric;
    if (dist_fn == std::string("l2"))
        metric = diskann::Metric::L2;
    else if (dist_fn == std::string("mips"))
        metric = diskann::Metric::INNER_PRODUCT;
    else if (dist_fn == std::string("cosine"))
        metric = diskann::Metric::COSINE;
    else
    {
        std::cout << "Error. Only l2 and mips distance functions are supported" << std::endl;
        return -1;
    }

    if (append_reorder_data)
    {
        if (disk_PQ == 0)
        {
            std::cout << "Error: It is not necessary to append data for reordering "
                         "when vectors are not compressed on disk."
                      << std::endl;
            return -1;
        }
        if (data_type != std::string("float") && data_type != std::string("bf16") &&
            data_type != std::string("bfloat16"))
        {
            std::cout << "Error: Appending data for reordering currently only "
                         "supported for float/bf16 data type."
                      << std::endl;
            return -1;
        }
    }

    std::string params = std::string(std::to_string(R)) + " " + std::string(std::to_string(L)) + " " +
                         std::string(std::to_string(B)) + " " + std::string(std::to_string(M)) + " " +
                         std::string(std::to_string(num_threads)) + " " + std::string(std::to_string(disk_PQ)) + " " +
                         std::string(std::to_string(append_reorder_data)) + " " +
                         std::string(std::to_string(build_PQ)) + " " + std::string(std::to_string(QD));

    try
    {
        if (label_file != "" && label_type == "ushort")
        {
            if (data_type == std::string("int8"))
                return diskann::build_disk_index<int8_t>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                         metric, use_opq, codebook_prefix, use_filters, label_file,
                                                         universal_label, filter_threshold, Lf);
            else if (data_type == std::string("uint8"))
                return diskann::build_disk_index<uint8_t, uint16_t>(
                    data_path.c_str(), index_path_prefix.c_str(), params.c_str(), metric, use_opq, codebook_prefix,
                    use_filters, label_file, universal_label, filter_threshold, Lf);
            else if (data_type == std::string("float"))
                return diskann::build_disk_index<float, uint16_t>(
                    data_path.c_str(), index_path_prefix.c_str(), params.c_str(), metric, use_opq, codebook_prefix,
                    use_filters, label_file, universal_label, filter_threshold, Lf);
            else if (data_type == std::string("bf16") || data_type == std::string("bfloat16"))
            {
                return diskann::build_disk_index<diskann::bfloat16, uint16_t>(
                    data_path.c_str(), index_path_prefix.c_str(), params.c_str(), metric, use_opq, codebook_prefix,
                    use_filters, label_file, universal_label, filter_threshold, Lf);
            }
            else
            {
                diskann::cerr << "Error. Unsupported data type" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("int8"))
                return diskann::build_disk_index<int8_t>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                         metric, use_opq, codebook_prefix, use_filters, label_file,
                                                         universal_label, filter_threshold, Lf);
            else if (data_type == std::string("uint8"))
                return diskann::build_disk_index<uint8_t>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                          metric, use_opq, codebook_prefix, use_filters, label_file,
                                                          universal_label, filter_threshold, Lf);
            else if (data_type == std::string("float"))
                return diskann::build_disk_index<float>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                        metric, use_opq, codebook_prefix, use_filters, label_file,
                                                        universal_label, filter_threshold, Lf);
            else if (data_type == std::string("bf16") || data_type == std::string("bfloat16"))
            {
                return diskann::build_disk_index<diskann::bfloat16>(data_path.c_str(), index_path_prefix.c_str(),
                                                                    params.c_str(), metric, use_opq, codebook_prefix,
                                                                    use_filters, label_file, universal_label,
                                                                    filter_threshold, Lf);
            }
            else
            {
                diskann::cerr << "Error. Unsupported data type" << std::endl;
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
