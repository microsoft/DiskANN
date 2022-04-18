// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <ctime>
#include <functional>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <codecvt>

#include <restapi/server.h>

using namespace diskann;

std::unique_ptr<Server>                           g_httpServer(nullptr);
std::vector<std::unique_ptr<diskann::BaseSearch>> g_ssdSearch;

void setup(const utility::string_t& address, const std::string& typestring) {
  web::http::uri_builder uriBldr(address);
  auto                   uri = uriBldr.to_uri();

  std::cout << "Attempting to start server on " << uri.to_string() << std::endl;

  g_httpServer =
      std::unique_ptr<Server>(new Server(uri, g_ssdSearch, typestring));
  std::cout << "Created a server object" << std::endl;

  g_httpServer->open().wait();
  ucout << U"Listening for requests on: " << address << std::endl;
}

void teardown(const utility::string_t& address) {
  g_httpServer->close().wait();
}

int main(int argc, char* argv[]) {
  if (argc != 6 && argc != 7) {
    std::cout << "Usage: ssd_server ip_addr:port data_type<float/int8/uint8> "
                 "index_file_prefix num_nodes_to_cache num_threads [tags_file]"
              << std::endl;
    exit(1);
  }

  // auto address = getHostingAddress(argv[1]);
  std::string address(argv[1]);
  auto        index_prefix = argv[3];
  unsigned    num_nodes_to_cache = atoi(argv[4]);
  unsigned    num_threads = atoi(argv[5]);
  const char* tags_file = argc == 7 ? argv[6] : nullptr;

  const std::string typestring(argv[2]);
  if (typestring == std::string("float")) {
    auto searcher = std::unique_ptr<diskann::BaseSearch>(
        new diskann::PQFlashSearch<float>(index_prefix, num_nodes_to_cache,
                                          num_threads, tags_file, diskann::L2));
    g_ssdSearch.push_back(std::move(searcher));
  } else if (typestring == std::string("int8")) {
    auto searcher =
        std::unique_ptr<diskann::BaseSearch>(new diskann::PQFlashSearch<int8_t>(
            index_prefix, num_nodes_to_cache, num_threads, tags_file,
            diskann::L2));
    g_ssdSearch.push_back(std::move(searcher));
  } else if (typestring == std::string("uint8")) {
    auto searcher = std::unique_ptr<diskann::BaseSearch>(
        new diskann::PQFlashSearch<uint8_t>(index_prefix, num_nodes_to_cache,
                                            num_threads, tags_file,
                                            diskann::L2));
    g_ssdSearch.push_back(std::move(searcher));
  } else {
    std::cerr << "Unsupported data type " << argv[2] << std::endl;
  }

  while (1) {
    try {
      setup(address, typestring);
      std::cout << "Type 'exit' (case-sensitive) to exit" << std::endl;
      std::string line;
      std::getline(std::cin, line);
      if (line == "exit") {
        teardown(address);
        g_httpServer->close().wait();
        exit(0);
      }
    } catch (const std::exception& ex) {
      std::cerr << "Exception occurred: " << ex.what() << std::endl;
      std::cerr << "Restarting HTTP server";
      teardown(address);
    } catch (...) {
      std::cerr << "Unknown exception occurreed" << std::endl;
      std::cerr << "Restarting HTTP server";
      teardown(address);
    }
  }
}
