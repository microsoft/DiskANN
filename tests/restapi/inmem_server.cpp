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

std::unique_ptr<Server>              g_httpServer(nullptr);
std::unique_ptr<diskann::BaseSearch> g_inMemorySearch(nullptr);

void setup(const utility::string_t& address, const std::string& typestring) {
  web::http::uri_builder uriBldr(address);
  auto                   uri = uriBldr.to_uri();

  std::cout << "Attempting to start server on " << uri.to_string() << std::endl;

  g_httpServer =
      std::unique_ptr<Server>(new Server(uri, g_inMemorySearch, typestring));
  std::cout << "Created a server object" << std::endl;

  g_httpServer->open().wait();
  ucout << U"Listening for requests on: " << address << std::endl;
}

void teardown(const utility::string_t& address) {
  g_httpServer->close().wait();
}

int main(int argc, char* argv[]) {
  if (argc != 6 && argc != 5) {
    std::cout << "Usage: server ip_addr:port <float/int8/uint8> data_file "
                 "index_file <tags_file>"
              << std::endl;
    exit(1);
  }

  std::string address(argv[1]);

  const std::string typestring(argv[2]);
  if (typestring == std::string("float")) {
    auto searcher = new diskann::InMemorySearch<float>(
        argv[3], argv[4], argc == 6 ? argv[5] : nullptr, diskann::L2);
    g_inMemorySearch =
        std::unique_ptr<diskann::InMemorySearch<float>>(searcher);
  } else if (typestring == std::string("int8")) {
    auto searcher = new diskann::InMemorySearch<int8_t>(
        argv[3], argv[4], argc == 6 ? argv[5] : nullptr, diskann::L2);
    g_inMemorySearch =
        std::unique_ptr<diskann::InMemorySearch<int8_t>>(searcher);
  } else if (typestring == std::string("uint8")) {
    auto searcher = new diskann::InMemorySearch<uint8_t>(
        argv[3], argv[4], argc == 6 ? argv[5] : nullptr, diskann::L2);
    g_inMemorySearch =
        std::unique_ptr<diskann::InMemorySearch<uint8_t>>(searcher);
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
