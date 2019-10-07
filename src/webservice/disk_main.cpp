// nsg_server.cpp : REST interface for diskann search.
//
#include <utils.h>
#include <webservice/disk_nsg_search.h>
#include <webservice/disk_server.h>
#include <codecvt>
#include <iostream>

std::unique_ptr<DiskServer>             g_httpServer(nullptr);
std::unique_ptr<diskann::DiskNSGSearch> g_diskNSGSearch(nullptr);

void setup(const utility::string_t& address) {
  web::http::uri_builder uriBldr(address);
  auto                   uri = uriBldr.to_uri();

  std::wcout << L"Attempting to start server on " << uri.to_string()
             << std::endl;

  g_httpServer = std::unique_ptr<DiskServer>(new DiskServer(uri, g_diskNSGSearch));
  g_httpServer->open().wait();

  ucout << U"Listening for requests on: " << address << std::endl;
}

void teardown(const utility::string_t& address) {
  g_httpServer->close().wait();
}


void loadIndex(const char* indexFilePrefix, const char* idsFile,
               const _u64 cache_nlevels, const _u64 nthreads) {
  auto nsgSearch =
      new diskann::DiskNSGSearch(indexFilePrefix, idsFile, cache_nlevels, nthreads);
  g_diskNSGSearch = std::unique_ptr<diskann::DiskNSGSearch>(nsgSearch);
}

std::wstring getHostingAddress(const char* hostNameAndPort) {
  wchar_t buffer[4096];
  mbstowcs_s(nullptr, buffer, sizeof(buffer) / sizeof(buffer[0]),
             hostNameAndPort, sizeof(buffer) / sizeof(buffer[0]));
  return std::wstring(buffer);
}


int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cout << "Usage: nsg_server <ip_addr_and_port> <prefix> "
                 "<ids_file> <cache_nlevels> <nthreads>"
              << std::endl;
    exit(1);
  }

  auto       address = getHostingAddress(argv[1]);
  const _u64 cache_nlevels = (_u64) std::atoi(argv[4]);
  const _u64 nthreads = (_u64) std::atoi(argv[5]);
  loadIndex(argv[2], argv[3], cache_nlevels, nthreads);
  while (1) {
    try {
      setup(address);
      std::cout << "Type 'exit' (case-sensitive) to exit" << std::endl;
      std::string line;
      std::getline(std::cin, line);
      if (line == "exit") {
        teardown(address);
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