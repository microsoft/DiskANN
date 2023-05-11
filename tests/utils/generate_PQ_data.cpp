// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <boost/program_options.hpp>

#include "utils.h"
#include "disk_utils.h"
#include "pq.h"

namespace po = boost::program_options;

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, data_path, codebook_prefix;
    uint32_t QD;
    float sample_rate;
    bool use_opq = false;

    po::options_description desc{"Arguments"};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(), "distance function <l2/mips>");
        desc.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                           "Input data file in bin format");
        desc.add_options()("QD", po::value<uint32_t>(&QD)->default_value(0), " Quantized Dimension for compression");
        desc.add_options()("codebook_prefix", po::value<std::string>(&codebook_prefix)->default_value(""),
                           "Path prefix for pre-trained codebook");
        desc.add_options()("use_opq", po::bool_switch()->default_value(false),
                           "Use Optimized Product Quantization (OPQ).");
        desc.add_options()("sample_rate", po::value<float>(&sample_rate)->default_value(1),
                           "Fraction of data to be sampled for computing PQ codebook. Use value between 0.01 and 1.");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (vm["use_opq"].as<bool>())
            use_opq = true;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("l2"))
        metric = diskann::Metric::L2;
    else if (dist_fn == std::string("mips"))
        metric = diskann::Metric::INNER_PRODUCT;
    else
    {
        std::cout << "Error. Only l2 and mips distance functions are supported" << std::endl;
        return -1;
    }

    if (sample_rate > 1 || sample_rate < 0.01)
    {
        std::cout << "Error. Use a value of sample rate between 0.01 and 1." << std::endl;
        return -1;
    }

    std::string pq_compressed_vectors_path = data_path;
    pq_compressed_vectors_path += use_opq ? "_OPQ" : "_PQ";
    pq_compressed_vectors_path += std::to_string(QD) + "_compressed.bin";

    std::string pq_pivots_path_base = codebook_prefix;
    std::string pq_pivots_path =
        file_exists(pq_pivots_path_base) ? pq_pivots_path_base + "_pq_pivots.bin" : data_path + "_pq_pivots.bin";

    try
    {
        if (data_type == std::string("int8"))
            diskann::generate_quantized_data<int8_t>(data_path, pq_pivots_path, pq_compressed_vectors_path, metric,
                                                     sample_rate, QD, use_opq, codebook_prefix);
        else if (data_type == std::string("uint8"))
            diskann::generate_quantized_data<uint8_t>(data_path, pq_pivots_path, pq_compressed_vectors_path, metric,
                                                      sample_rate, QD, use_opq, codebook_prefix);
        else if (data_type == std::string("float"))
            diskann::generate_quantized_data<float>(data_path, pq_pivots_path, pq_compressed_vectors_path, metric,
                                                    sample_rate, QD, use_opq, codebook_prefix);
        else
        {
            diskann::cerr << "Error. Unsupported data type" << std::endl;
            return -1;
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
