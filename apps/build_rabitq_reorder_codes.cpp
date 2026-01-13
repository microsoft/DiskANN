#include <boost/program_options.hpp>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "rabitq.h"
#include "utils.h"

namespace po = boost::program_options;

namespace
{
#pragma pack(push, 1)
struct RaBitQReorderHeader
{
    char magic[8];
    uint32_t version;
    uint32_t metric;
    uint32_t nb_bits;
    uint32_t dim;
    uint64_t num_points;
    uint64_t code_size;
};
#pragma pack(pop)

void write_header(std::ofstream &out, uint32_t metric, uint32_t nb_bits, uint32_t dim, uint64_t num_points,
                  uint64_t code_size)
{
    RaBitQReorderHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    // "DARBQ1\0" for easy identification.
    hdr.magic[0] = 'D';
    hdr.magic[1] = 'A';
    hdr.magic[2] = 'R';
    hdr.magic[3] = 'B';
    hdr.magic[4] = 'Q';
    hdr.magic[5] = '1';
    hdr.magic[6] = '\0';
    hdr.magic[7] = '\0';
    hdr.version = 1;
    hdr.metric = metric;
    hdr.nb_bits = nb_bits;
    hdr.dim = dim;
    hdr.num_points = num_points;
    hdr.code_size = code_size;

    out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
}

} // namespace

int main(int argc, char **argv)
{
    std::string data_file;
    std::string output_file;
    uint32_t nb_bits = 4;
    std::string metric_str = "ip";

    po::options_description desc("build_rabitq_reorder_codes options");
    desc.add_options()("help,h", "Show help")

        ("data_file", po::value<std::string>(&data_file)->required(),
         "Input base vectors in DiskANN .bin format (float32, n x d)")

        ("output_file", po::value<std::string>(&output_file)->required(),
         "Output RaBitQ reorder code file (to be loaded by PQFlashIndex)")

        ("nb_bits", po::value<uint32_t>(&nb_bits)->default_value(4), "Bits per dimension (1..9)")

        ("metric", po::value<std::string>(&metric_str)->default_value("ip"), "Metric: ip or l2");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    if (nb_bits < 1 || nb_bits > 9)
    {
        std::cerr << "nb_bits must be in [1,9]" << std::endl;
        return 1;
    }

    diskann::rabitq::Metric metric = diskann::rabitq::Metric::INNER_PRODUCT;
    if (metric_str == "ip" || metric_str == "inner_product")
        metric = diskann::rabitq::Metric::INNER_PRODUCT;
    else if (metric_str == "l2")
        metric = diskann::rabitq::Metric::L2;
    else
    {
        std::cerr << "Unknown metric: " << metric_str << " (expected ip or l2)" << std::endl;
        return 1;
    }

    uint64_t n = 0, d = 0;
    std::unique_ptr<float[]> x;
    diskann::load_bin<float>(data_file, x, n, d);

    const uint64_t code_size = diskann::rabitq::compute_code_size(static_cast<size_t>(d), static_cast<size_t>(nb_bits));

    std::vector<uint8_t> codes;
    codes.resize(static_cast<size_t>(n * code_size));

    for (uint64_t i = 0; i < n; ++i)
    {
        const float *row = x.get() + i * d;
        uint8_t *out = codes.data() + i * code_size;
        diskann::rabitq::encode_vector(row, static_cast<size_t>(d), metric, static_cast<size_t>(nb_bits), out);
    }

    std::ofstream out(output_file, std::ios::binary);
    if (!out.is_open())
    {
        std::cerr << "Failed to open output_file: " << output_file << std::endl;
        return 1;
    }

    write_header(out, static_cast<uint32_t>(metric), nb_bits, static_cast<uint32_t>(d), n, code_size);
    out.write(reinterpret_cast<const char *>(codes.data()), static_cast<std::streamsize>(codes.size()));

    std::cout << "Wrote RaBitQ reorder codes: n=" << n << " d=" << d << " nb_bits=" << nb_bits
              << " code_size=" << code_size << " -> " << output_file << std::endl;

    return 0;
}
