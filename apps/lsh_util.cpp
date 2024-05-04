#include <string>
#include <limits>
#include <float.h>
#include <random>
#include <chrono>
#include <vector>

#include "distance.h"
#include "utils.h"
#include "LSH.h"


const int NUM_THREADS = 32;
const float MAX_MEM_IN_GB = 16;
const std::string HASH_SUFFIX = "_hashes.txt";
const std::string AXES_SUFFIX = "_axes.txt";


std::unique_ptr<uint8_t[]> compute_lsh(uint64_t num_pts, uint64_t num_dims, const std::string& input_file, const std::vector<uint32_t> pt_chunks, diskann::LSH& lsh)
{
    std::unique_ptr<uint8_t[]> lsh_values = std::make_unique<uint8_t[]>(num_pts);
    memset((void*)lsh_values.get(), 0, num_pts);

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
                std::unique_ptr<float[]> dot_product = std::make_unique<float[]>(LSH_NUM_AXES);
//#pragma omp for schedule(dynamic, 64)
                for (int64_t j = pt_chunks[i]; j < pt_chunks[i+1]; j++)
                {
                    //if (j % 1000000 == 0)
                    //{
                    //    std::cout << "Thread: " << std::this_thread::get_id()
                    //              << " dot_product: " << (uint64_t)dot_product.get() << std::endl;
                    //}
                    uint8_t lsh_value = lsh.get_hash(data.get() + (num_dims * (j - pt_chunks[i])), const_cast<float *>(dot_product.get()));
                    lsh_values[j] = lsh_value;
                }
            }
            std::cout << "Finished batch of " << num_pts_in_chunk << " out of " << num_pts << std::endl;
        }
    }
    std::cout << "Generated all " << num_pts << " hashes." << std::endl;

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

void do_lsh(const std::string &input_file, const std::string &output_prefix, const std::string &axes_file = "")
{
    float *data = nullptr;
    uint64_t num_pts, num_dims;
    diskann::get_bin_metadata(input_file, num_pts, num_dims);

    std::unique_ptr<diskann::LSH> lsh = nullptr;
    if (axes_file != "")
    {
        float *axes = new float[LSH_NUM_AXES * num_dims];
        std::ifstream in(axes_file);
        //TODO: Sanity check on the file contents to ensure that it is indeed NUM_AXES * num_dims.
        for (int i = 0; i < LSH_NUM_AXES * num_dims; i++)
        {
            in >> axes[i];
        }
        lsh = std::make_unique<diskann::LSH>(num_dims, LSH_NUM_AXES, axes);//ownership of axes is transfered to lsh.
    }
    else
    {
        lsh = std::make_unique<diskann::LSH>(num_dims, LSH_NUM_AXES);
    }
    
    std::vector<uint32_t> point_chunks;
    calculate_point_chunks(num_pts, num_dims, point_chunks);

    auto lsh_values = compute_lsh(num_pts, num_dims, input_file, point_chunks, *lsh);

    std::string lsh_file = output_prefix + HASH_SUFFIX;
    std::ofstream out(lsh_file);
    for (auto i = 0; i < num_pts; i++)
    {
        out << (uint32_t) lsh_values[i] << std::endl;
    }
    out.close();

    std::string axes_file_name = output_prefix + AXES_SUFFIX;
    std::ofstream axes_out(axes_file_name);
    lsh->dump_axes_to_text_file(axes_out);
    axes_out.close();
}

void cluster(const std::string &lsh_prefix, const std::string &clusters_file)
{
    std::vector<std::uint8_t> lsh_hashes;
    std::ifstream hash_file(lsh_prefix + HASH_SUFFIX);

    uint32_t value;
    while (hash_file.good())
    {
        hash_file >> value;
        lsh_hashes.push_back((uint8_t)value);
    }

    std::cout << "Read " << lsh_hashes.size() << " hashes" << std::endl;

    std::vector<bool> visited(lsh_hashes.size());
    std::vector<std::vector<uint32_t>> final_clusters; 

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
        visited[i] = true;
        for (auto j = i + 1; j < lsh_hashes.size(); j++)
        {
            if (lsh_hashes[i]  == lsh_hashes[j])
            {
                running_cluster.push_back(j);
                visited[j] = true;
            }
        }
        running_count += running_cluster.size();
        if (running_count > lsh_hashes.size())
        {
            std::cerr << "ERROR! Cluster sum " << running_count
                      << " exceeds total number of points " << lsh_hashes.size() << std::endl;
            break;
        }
        std::cout << "Computed cluster with " << running_cluster.size() << " points starting with "
                  << " node: " << running_cluster[0] << std::endl;
        final_clusters.push_back(running_cluster);
    }

    std::cout << "Computed " << final_clusters.size() << " clusters. " << std::endl;

    std::ofstream cluster_out(clusters_file);
    uint32_t count = 0;
    for (auto &cluster : final_clusters)
    {
        cluster_out << "[";
        auto i = 0;
        for (; i < cluster.size() - 1; i++)
        {
            cluster_out << cluster[i] << ",";
        }
        count += (uint32_t) cluster.size();
        cluster_out << cluster[i] << "]" << std::endl;
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


int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6)
    {
        std::cout << "Usage: " << argv[0] << " "
                  << "<data_type> <operation> <input_bin_file> <output_prefix> [axes_file]"
                  << "Only float is supported at the moment."
                  << "<operation> can be 'generate' to generate lsh hashes or 'cluster' to cluster hashes"
                  << "Both ops require two files. For 'generate' the input file is the binary file and the "
                     "output_prefix is used to write the hashes and axes."
                  << "The generate option can also take an optional axes_file parameter which defines the axes that "
                     "must be used to generate hashes."
                  << "For the 'cluster' option, the input file is the hash file prefix and the output file is the "
                     "cluster assignment of points"
                  << std::endl;
        return -1;
    }

    std::string data_type = argv[1];
    if (data_type != "float")
    {
        std::cerr << "Only floating point data types are supported." << std::endl;
        return -1;
    }
    std::string op(argv[2]);
    if (op == "generate")
    {
        if (argc == 5)
        {
            do_lsh(argv[3], argv[4]);
        }
        else
        {
            do_lsh(argv[3], argv[4], argv[5]);
        }
    }
    else if (op == "cluster")
    {
        cluster(argv[3], argv[4]);
    }
    else
    {
        std::cerr << "Unknown operation " << op << ". Only 'generate' and 'cluster' are supported." << std::endl;
    }
}
        


