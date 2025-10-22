#include <iostream>
#include "index.h"
#include "utils.h"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <data_file>" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];
    diskann::Metric metric = diskann::Metric::L2;
    size_t num_points, dim;
    diskann::get_bin_metadata(data_path, num_points, dim);

    // Build parameters
    unsigned R = 24;
    unsigned L = 100;
    float alpha = 1.2f;
    unsigned num_threads = 4;

    // Create index
    auto write_params =
        diskann::IndexWriteParametersBuilder(L, R).with_num_threads(num_threads).with_alpha(alpha).build();
    auto search_params = std::make_shared<diskann::IndexSearchParams>(L, num_threads);
    diskann::Index<float, uint32_t> index(metric, dim, num_points,
                                          std::make_shared<diskann::IndexWriteParameters>(write_params), search_params);
    index.build(data_path.c_str(), num_points);

    // Search parameters
    unsigned Lsearch = 100;
    unsigned K = 10; // Number of neighbors to search for
    unsigned num_queries = 1;
    std::vector<uint32_t> query_result_ids(num_queries * K);
    std::vector<float> query_result_dists(num_queries * K);

    // Load query data (using the first point from the dataset as a query)
    float *query_data = nullptr;
    diskann::alloc_aligned((void **)&query_data, dim * sizeof(float), 8 * sizeof(float));
    std::ifstream reader(data_path, std::ios::binary);
    reader.seekg(2 * sizeof(int)); // Skip npts and dim
    reader.read((char *)query_data, dim * sizeof(float));
    reader.close();

    // Add a pause to allow the debugger to attach properly
    // std::cout << "\nPress Enter to start the search..." << std::endl;
    // std::cin.get();

    // Perform search
    index.search(query_data, K, Lsearch, query_result_ids.data(), query_result_dists.data());

    std::cout << "Search complete. Found " << K << " neighbors." << std::endl;
    std::cout << "Nearest neighbor ID: " << query_result_ids[0] << " with distance " << query_result_dists[0]
              << std::endl;

    diskann::aligned_free(query_data);

    return 0;
}
