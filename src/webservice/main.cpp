// nsg_server.cpp : REST interface for NSG search.
//

#include <codecvt>
#include <iostream>
#include "server.h"

std::unique_ptr<Server> g_httpServer;

void setup(const utility::string_t& address) {
  web::http::uri_builder uriBldr(address);
  auto                   uri = uriBldr.to_uri();

  g_httpServer = std::unique_ptr<Server>(new Server(uri));
  g_httpServer->open().wait();

  ucout << U"Listening for requests on: " << address << std::endl;
}

void teardown(const utility::string_t& address) {
  g_httpServer->close().wait();
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: nsg_server <port>" << std::endl;
    exit(1);
  }

  wchar_t buffer[16];
  mbstowcs_s(nullptr, buffer, sizeof(buffer), argv[1], sizeof(buffer));

  utility::string_t port(buffer);
  utility::string_t     address = U("http://127.0.0.1:");

  address.append(port);

  setup(address);
  std::cout << "Press ENTER to exit" << std::endl;

  std::string line;
  std::getline(std::cin, line);

  teardown(address);
}
