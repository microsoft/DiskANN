// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <ctime>
#include <functional>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <codecvt>
#include <boost/program_options.hpp>

#include <restapi/server.h>

using namespace diskann;
namespace po = boost::program_options;

std::unique_ptr<Server> g_httpServer(nullptr);
std::vector<std::unique_ptr<diskann::BaseSearch>> g_inMemorySearch;

void setup(const utility::string_t &address, const std::string &typestring)
{
    web::http::uri_builder uriBldr(address);
    auto uri = uriBldr.to_uri();

    std::cout << "Attempting to start server on " << uri.to_string() << std::endl;

    g_httpServer = std::unique_ptr<Server>(new Server(uri, g_inMemorySearch, typestring));
    std::cout << "Created a server object" << std::endl;

    g_httpServer->open().wait();
    ucout << U"Listening for requests on: " << address << std::endl;
}

void teardown(const utility::string_t &address)
{
    g_httpServer->close().wait();
}

int main(int argc, char *argv[])
{
    std::string data_type, index_file, data_file, address, dist_fn, tags_file;
    uint32_t num_threads;
    uint32_t l_search;

    po::options_description desc{"Arguments"};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("address", po::value<std::string>(&address)->required(), "Web server address");
        desc.add_options()("data_file", po::value<std::string>(&data_file)->required(),
                           "File containing the data found in the index");
        desc.add_options()("index_path_prefix", po::value<std::string>(&index_file)->required(),
                           "Path prefix for saving index file components");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->required(),
                           "Number of threads used for building index");
        desc.add_options()("l_search", po::value<uint32_t>(&l_search)->required(), "Value of L");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->default_value("l2"),
                           "distance function <l2/mips>");
        desc.add_options()("tags_file", po::value<std::string>(&tags_file)->default_value(std::string()),
                           "Tags file location");
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
        std::cerr << ex.what() << std::endl;
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

    if (data_type == std::string("float"))
    {
        auto searcher = std::unique_ptr<diskann::BaseSearch>(
            new diskann::InMemorySearch<float>(data_file, index_file, tags_file, metric, num_threads, l_search));
        g_inMemorySearch.push_back(std::move(searcher));
    }
    else if (data_type == std::string("int8"))
    {
        auto searcher = std::unique_ptr<diskann::BaseSearch>(
            new diskann::InMemorySearch<int8_t>(data_file, index_file, tags_file, metric, num_threads, l_search));
        g_inMemorySearch.push_back(std::move(searcher));
    }
    else if (data_type == std::string("uint8"))
    {
        auto searcher = std::unique_ptr<diskann::BaseSearch>(
            new diskann::InMemorySearch<uint8_t>(data_file, index_file, tags_file, metric, num_threads, l_search));
        g_inMemorySearch.push_back(std::move(searcher));
    }
    else
    {
        std::cerr << "Unsupported data type " << argv[2] << std::endl;
    }

    while (1)
    {
        try
        {
            setup(address, data_type);
            std::cout << "Type 'exit' (case-sensitive) to exit" << std::endl;
            std::string line;
            std::getline(std::cin, line);
            if (line == "exit")
            {
                teardown(address);
                g_httpServer->close().wait();
                exit(0);
            }
        }
        catch (const std::exception &ex)
        {
            std::cerr << "Exception occurred: " << ex.what() << std::endl;
            std::cerr << "Restarting HTTP server";
            teardown(address);
        }
        catch (...)
        {
            std::cerr << "Unknown exception occurreed" << std::endl;
            std::cerr << "Restarting HTTP server";
            teardown(address);
        }
    }
}
