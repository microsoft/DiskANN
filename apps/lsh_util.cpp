#include <string>
#include <limits>
#include <float.h>
#include <random>
#include <chrono>
#include <vector>
#include <boost/program_options.hpp>
#include <program_options_utils.hpp>

#include "distance.h"
#include "utils.h"
#include "LSH.h"
#include "lsh_util.h"

namespace po = boost::program_options;

const int NUM_THREADS = 32;
const double MAX_MEM_IN_GB = 8;
const std::string HASH_SUFFIX_TXT = "_hashes.txt";
const std::string AXES_SUFFIX_TXT = "_axes.txt";
const std::string HASH_SUFFIX_BIN = "_hashes.bin";
const std::string AXES_SUFFIX_BIN = "_axes.bin";

//Signatures
template<typename HashT>
void write_to_text(const std::string &output_prefix, const uint64_t &num_pts,
                   std::unique_ptr<HashT[]> &lsh_values,
                   std::unique_ptr<diskann::LSH<HashT>> &lsh);

template<typename HashT>
void write_to_bin(const std::string &output_prefix, const uint64_t &num_pts,
                  std::unique_ptr<HashT[]> &lsh_values,
                  std::unique_ptr<diskann::LSH<HashT>> &lsh);


template<typename HashT>
std::unique_ptr<HashT[]> compute_lsh(uint64_t num_pts, uint64_t num_dims, const std::string &input_file,
                                       const std::vector<uint32_t> pt_chunks, diskann::LSH<HashT> &lsh)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_ptr<HashT[]> lsh_values = std::make_unique<HashT[]>(num_pts);
    memset((void*)lsh_values.get(), 0, num_pts);
    auto lsh_num_axes = sizeof(HashT) * 8;

    std::ifstream file(input_file, std::ios::binary);
    if (file.good())
    {
        for (auto i = 0; i < pt_chunks.size() - 1; i++)
        {
            uint64_t num_pts_in_chunk = pt_chunks[i + 1] - pt_chunks[i];
            auto read_size = num_pts_in_chunk * num_dims * sizeof(float);

            std::unique_ptr<float[]> data = std::make_unique<float[]>(read_size);
           
            file.seekg(8 + pt_chunks[i] * num_dims * sizeof(float));
            file.read((char*)data.get(), read_size);

            std::cout << "Loaded data for chunk " << i << " from: " << pt_chunks[i] << " to: " << pt_chunks[i + 1]
                      << " size: " << read_size << std::endl;

//#pragma omp parallel num_threads(NUM_THREADS)
            {
                std::unique_ptr<float[]> dot_product = std::make_unique<float[]>(lsh_num_axes);
//#pragma omp for schedule(dynamic, 64)
                for (int64_t j = pt_chunks[i]; j < pt_chunks[i+1]; j++)
                {
                    //if (j % 1000000 == 0)
                    //{
                    //    std::cout << "Thread: " << std::this_thread::get_id()
                    //              << " dot_product: " << (uint64_t)dot_product.get() << std::endl;
                    //}
                    HashT lsh_value = lsh.get_hash(data.get() + (num_dims * (j - pt_chunks[i])),
                                                            const_cast<float *>(dot_product.get()));
                    lsh_values[j] = lsh_value;
                }
            }
            std::cout << "Finished batch of " << num_pts_in_chunk << " out of " << num_pts << std::endl;
        }
    }
    auto time_elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start) ;

    std::cout << "Generated " << num_pts << " hashes in " << time_elapsed.count() << "s." << std::endl;

    return lsh_values;
}

uint64_t ceiling_fn(uint64_t numerator, uint64_t denominator)
{
    if (numerator < denominator)
    {
        return 1;
    }
    return (numerator % denominator) == 0 ? numerator / denominator : (numerator / denominator) + 1;
    
}

void calculate_point_chunks(uint64_t num_pts, uint64_t num_dims, std::vector<uint32_t>& point_chunks)
{
    uint64_t available_mem_bytes = (uint64_t) (MAX_MEM_IN_GB * 1024 * 1024 * 1024);
    uint64_t size_of_data = num_pts * num_dims * sizeof(float);
    uint64_t num_chunks = ceiling_fn(size_of_data, available_mem_bytes);
    uint32_t points_per_chunk = (uint32_t) (num_pts / num_chunks); 

    std::cout << "Available memory: " << available_mem_bytes << ", size_of_data: " << size_of_data
              << ", num_chunks: " << num_chunks << ", points_per_chunk:" << points_per_chunk << std::endl;

    uint32_t chunk_start_point = 0;

    for (auto i = 0; i < num_chunks; i++)
    {
        point_chunks.push_back(chunk_start_point);
        chunk_start_point += points_per_chunk;
    }
    point_chunks.push_back((uint32_t)num_pts); //as a terminating condition later in the code.
}

float *get_centroid(const std::string &input_file, uint64_t num_pts, uint64_t num_dims)
{
    float *centroid = new float[num_dims];
    memset(centroid, 0, sizeof(float) * num_dims);


    const uint32_t BATCH_SIZE = 1000000;
    size_t chunk_size = num_pts < BATCH_SIZE ? num_pts : BATCH_SIZE;
    float *data_chunk = new float[chunk_size * num_dims];

    std::ifstream in(input_file, std::ios::binary);
    in.seekg(8);
    size_t count = 0; 
    while (in.good() && count < num_pts)
    {
        in.read((char *)data_chunk, sizeof(float) * chunk_size * num_dims);
#pragma omp parallel for 
        for (int j = 0; j < (int)num_dims; j++)
        {
            for (size_t i = 0; i < chunk_size; i++)
            {
                centroid[j] += *(data_chunk + j + i * num_dims);
            }
        }
        count += chunk_size;
        chunk_size = (num_pts - count) < BATCH_SIZE ? (num_pts - count) : BATCH_SIZE;
        std::cout << "After processing " << count << " points out of " << num_pts << " chunk_size: " << chunk_size
                  << std::endl;
    }
    std::cout << "Centroid: [";
    for (int i = 0; i < (int)num_dims; i++)
    {
        centroid[i] /= num_pts;
        std::cout << centroid[i] << ",";
    }
    std::cout << "]" << std::endl;
  
    delete[] data_chunk;
    return centroid;
}

template<typename HashT>
void do_lsh(const std::string &input_file, const std::string &output_prefix, const std::string &axes_file = "")
{
    float *data = nullptr;
    uint64_t num_pts, num_dims;
    diskann::get_bin_metadata(input_file, num_pts, num_dims);
    auto lsh_num_axes = sizeof(HashT) * 8;

    std::unique_ptr<diskann::LSH<HashT>> lsh = nullptr;
    if (axes_file != "")
    {
        float *axes = new float[lsh_num_axes * num_dims];
        std::ifstream in(axes_file);
        //TODO: Sanity check on the file contents to ensure that it is indeed NUM_AXES * num_dims.
        for (int i = 0; i < lsh_num_axes * num_dims; i++)
        {
            in >> axes[i];
        }
        lsh = std::make_unique<diskann::LSH<HashT>>((uint32_t)num_dims, (uint32_t) lsh_num_axes,
                                                             axes); // ownership of axes is transfered to lsh.
    }
    else
    {
        float *centroid = get_centroid(input_file, num_pts, num_dims);
        lsh = std::make_unique<diskann::LSH<HashT>>((uint32_t)num_dims, (uint32_t)lsh_num_axes);
        lsh->with_centroid(centroid);
    }
    
    std::vector<uint32_t> point_chunks;
    calculate_point_chunks(num_pts, num_dims, point_chunks);

    auto lsh_values = compute_lsh<HashT>(num_pts, num_dims, input_file, point_chunks, *lsh);

    write_to_text(output_prefix, num_pts, lsh_values, lsh);
    write_to_bin(output_prefix, num_pts, lsh_values, lsh);
}

template<typename HashT>
void write_to_text(const std::string &output_prefix, const uint64_t &num_pts,
                   std::unique_ptr<HashT[]> &lsh_values,
                 std::unique_ptr<diskann::LSH<HashT>> &lsh)
{
    std::string lsh_file = output_prefix + HASH_SUFFIX_TXT;
    std::ofstream out(lsh_file);
    for (auto i = 0; i < num_pts; i++)
    {
        out << (uint32_t)lsh_values[i] << std::endl;
    }
    out.close();

    std::string axes_file_name = output_prefix + AXES_SUFFIX_TXT;
    std::ofstream axes_out(axes_file_name);
    lsh->dump_axes_to_text_file(axes_out);
    axes_out.close();
}

template<typename HashT>
void write_to_bin(const std::string &output_prefix, const uint64_t &num_pts,
                  std::unique_ptr<HashT[]> &lsh_values,
                  std::unique_ptr<diskann::LSH<HashT>> &lsh)
{
    std::string lsh_file = output_prefix + HASH_SUFFIX_BIN;
    std::ofstream out(lsh_file, std::ios::binary);
    
    uint32_t row_count = (uint32_t) num_pts;
    uint32_t col_count = 1;
    out.write((const char *)&row_count, sizeof(uint32_t));
    out.write((const char *)&col_count, sizeof(uint32_t));
    out.write((const char *)lsh_values.get(), num_pts * sizeof(uint8_t));

    std::string axes_file = output_prefix + AXES_SUFFIX_BIN;
    std::ofstream axes_out(axes_file, std::ios::binary);
    lsh->dump_axes_to_bin_file(axes_out);
}

template<typename HashT>
void cluster(const std::string &hash_file_name, const std::string &clusters_file)
{
    std::vector<HashT> lsh_hashes;
    std::ifstream hash_file(hash_file_name);

    uint64_t value;
    while (hash_file.good())
    {
        hash_file >> value;
        lsh_hashes.push_back((HashT)value);
    }

    std::cout << "Read " << lsh_hashes.size() << " hashes" << std::endl;

    std::vector<bool> visited(lsh_hashes.size());
    std::unordered_map<HashT, std::vector<uint32_t>> final_clusters; 

    for (auto i = 0; i < lsh_hashes.size(); i++)
    {
        visited[i] = false;
    }

    uint32_t running_count = 0;
    for (auto i = 0; i < lsh_hashes.size(); i++)
    {
        if (visited[i])
        {
            continue;
        }
        std::vector<uint32_t> running_cluster;
        running_cluster.push_back(i);
        HashT current_hash = lsh_hashes[i];

        visited[i] = true;
        for (auto j = i + 1; j < lsh_hashes.size(); j++)
        {
            if (lsh_hashes[i]  == lsh_hashes[j])
            {
                running_cluster.push_back(j);
                visited[j] = true;
            }
        }
        running_count += (uint32_t) running_cluster.size();
        if (running_count > lsh_hashes.size())
        {
            std::cerr << "ERROR! Cluster sum " << running_count
                      << " exceeds total number of points " << lsh_hashes.size() << std::endl;
            break;
        }
        std::cout << "Computed cluster with " << running_cluster.size() << " points starting with "
                  << " node: " << running_cluster[0] << " hash value: " << current_hash << std::endl;
        final_clusters[current_hash] = running_cluster;
    }

    std::cout << "Computed " << final_clusters.size() << " clusters. " << std::endl;

    std::ofstream cluster_out(clusters_file);
    uint32_t count = 0;
    for (auto &cluster : final_clusters)
    {
        cluster_out << cluster.first << "\t" << cluster.second.size() << "\t[";
        auto i = 0;
        for (; i < cluster.second.size() - 1; i++)
        {
            cluster_out << cluster.second[i] << ",";
        }
        count += (uint32_t) cluster.second.size();
        cluster_out << cluster.second[i] << "]" << std::endl;
    }
    cluster_out.close();

    if (count != lsh_hashes.size())
    {
        std::cerr << "Mismatch in total cluster member count: " << count
                  << " and number of points: " << lsh_hashes.size() << std::endl;
    
    }
    else
    {
        std::cout << "Wrote: " << final_clusters.size() << " clusters to file: " << clusters_file << std::endl;
    }
}

int generate_hashes(uint32_t hash_size_bits, const std::string &input_file, const std::string &output_prefix,
                     const std::string &axes_file = "")
{
    switch (hash_size_bits)
    {
    case 8:
        do_lsh<uint8_t>(input_file, output_prefix, axes_file);
        return 0;
    case 16:
        do_lsh<uint16_t>(input_file, output_prefix, axes_file);
        return 0;
    case 32:
        do_lsh<uint32_t>(input_file, output_prefix, axes_file);
        return 0;
    case 64:
        do_lsh<uint64_t>(input_file, output_prefix, axes_file);
        return 0;
    default:
        std::cerr << "Cannot process hashes of size: " << hash_size_bits << ". Only 8,16,32 and 64 are supported"
                  << std::endl;
        return -1;
    }
}

int do_clustering(uint32_t hash_size_bits, const std::string &input_file, const std::string &output_file)
{
    switch (hash_size_bits)
    {
    case 8:
        cluster<uint8_t>(input_file, output_file);
        return 0;
    case 16:
        cluster<uint16_t>(input_file, output_file);
        return 0;
    case 32:
        cluster<uint32_t>(input_file, output_file);
        return 0;
    case 64:
        cluster<uint64_t>(input_file, output_file);
        return 0;
    default:
        std::cerr << "Cannot process hashes of size: " << hash_size_bits << ". Only 8,16,32 and 64 are supported"
                  << std::endl;
        return -1;
    }
}


int main(int argc, char **argv)
{
    std::string data_type, operation, input_file, output_prefix_or_file, axes_file;
    uint32_t hash_size_bits, ls_bits_for_clustering;

    try
    {
        po::options_description desc{
            program_options_utils::make_program_description("lsh_util", "Performs LSH operations on vector data")};
        desc.add_options()("help,h", "Print information on arguments");

        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       "data type of the data");
        required_configs.add_options()("operation", po::value<std::string>(&operation)->required(),
                                       "operation to perform. generate will generate hashes from the vector data, "
                                       "cluster will cluster existing hashes.");
        required_configs.add_options()(
            "hash_size", po::value<std::uint32_t>(&hash_size_bits)->required(),
            "Size of the hashes in bits. Only 8, 16, 32, 64. For 'generate' this will be the size of the hashes "
            "generated. For 'cluster' it is the expected size of the hashes in the input file.");
        required_configs.add_options()("input_file", po::value<std::string>(&input_file)->required(),
                                       "Input file. For 'generate' option, this is the vector bin file. For the "
                                       "'cluster' option, this is the file containing hashes.");
        required_configs.add_options()(
            "output_prefix_or_file", po::value<std::string>(&output_prefix_or_file)->required(),
            "Output file prefix. For 'generate' option, this is the prefix of the hashes and axes file. For the "
            "'cluster' option, this is the file to which the clusters will be written.");

        po::options_description optional_configs("Optional");
        optional_configs.add_options()("axes_file", po::value<std::string>(&axes_file)->default_value(std::string("")),
                                       "Axes file to be used for generation. Typically used for generating query "
                                       "hashes once data hashes have been generated");
        optional_configs.add_options()("ls_bits_for_clustering",
                                       po::value<std::uint32_t>(&ls_bits_for_clustering)->default_value(0),
                                       "Set this if you want to use fewer bits for clustering than the hash_size");

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
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
    }

    if (data_type != "float")
    {
        std::cerr << "Only floating data is supported" << std::endl;
        return -1;
    }

    if (operation == "generate")
    {
        return generate_hashes(hash_size_bits, input_file, output_prefix_or_file, axes_file);
    }
    else if (operation == "cluster")
    {
        return do_clustering(hash_size_bits, input_file, output_prefix_or_file);
    }
    else
    {
        std::cerr << "Option " << operation << " is not supported. Only generate|cluster are supported." << std::endl;
    }


    //if (argc != 5 && argc != 6)
    //{
    //    std::cout << "Usage: " << argv[0] << " "
    //              << "<data_type> <operation> <input_bin_file> <output_prefix> [axes_file]"
    //              << "Only float is supported at the moment."
    //              << "<operation> can be 'generate' to generate lsh hashes or 'cluster' to cluster hashes"
    //              << "Both ops require two files. For 'generate' the input file is the binary file and the "
    //                 "output_prefix is used to write the hashes and axes."
    //              << "The generate option can also take an optional axes_file parameter which defines the axes that "
    //                 "must be used to generate hashes."
    //              << "For the 'cluster' option, the input file is the hash file prefix and the output file is the "
    //                 "cluster assignment of points"
    //              << std::endl;
    //    return -1;
    //}
    //std::string data_type = argv[1];
    //if (data_type != "float")
    //{
    //    std::cerr << "Only floating point data types are supported." << std::endl;
    //    return -1;
    //}
    //std::string op(argv[2]);
    //if (op == "generate")
    //{
    //    if (argc == 5)
    //    {
    //        do_lsh(argv[3], argv[4]);
    //    }
    //    else
    //    {
    //        do_lsh(argv[3], argv[4], argv[5]);
    //    }
    //}
    //else if (op == "cluster")
    //{
    //    cluster(argv[3], argv[4]);
    //}
    //else
    //{
    //    std::cerr << "Unknown operation " << op << ". Only 'generate' and 'cluster' are supported." << std::endl;
    //}
}
        


