// nsg_server.cpp : REST interface for NSG search.
//

#include <webservice/server.h>
#include <webservice/in_memory_nsg_search.h>
#include <codecvt>
#include <iostream>

std::unique_ptr<Server>                 g_httpServer(nullptr);
std::unique_ptr<NSG::InMemoryNSGSearch> g_inMemoryNSGSearch(nullptr);

void setup(const utility::string_t& address) {
  web::http::uri_builder uriBldr(address);
  auto                   uri = uriBldr.to_uri();

  g_httpServer = std::unique_ptr<Server>(new Server(uri, g_inMemoryNSGSearch));
  g_httpServer->open().wait();

  ucout << U"Listening for requests on: " << address << std::endl;
}

void teardown(const utility::string_t& address) {
  g_httpServer->close().wait();
}

void loadIndex(const char* indexFile, const char* baseFile,
               const char* idsFile) {
  auto nsgSearch =
      new NSG::InMemoryNSGSearch(baseFile, indexFile, idsFile, NSG::L2);
  g_inMemoryNSGSearch = std::unique_ptr<NSG::InMemoryNSGSearch>(nsgSearch);
}

std::wstring getHostingAddress(const char* portNum) {
  wchar_t buffer[16];
  mbstowcs_s(nullptr, buffer, sizeof(buffer), portNum, sizeof(buffer));

  utility::string_t port(buffer);
  utility::string_t address = U("http://127.0.0.1:");

  address.append(port);

  return address;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cout << "Usage: nsg_server <port> <index_file> <base_file> <ids_file> "
              << std::endl;
    exit(1);
  }

  auto address = getHostingAddress(argv[1]);
  loadIndex(argv[2], argv[3], argv[4]);

  setup(address);
  std::cout << "Press ENTER to exit" << std::endl;

  std::string line;
  std::getline(std::cin, line);

  teardown(address);
}
