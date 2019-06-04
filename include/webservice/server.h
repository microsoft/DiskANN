#pragma once

#include <cpprest/base_uri.h>
#include <cpprest/http_listener.h>
#include <webservice/in_memory_nsg_search.h>

class Server {
 public:
  Server(web::uri& url, std::unique_ptr<NSG::InMemoryNSGSearch>& searcher);
  virtual ~Server();

  pplx::task<void> open();
  pplx::task<void> close();

 protected:
  void handle_post(web::http::http_request message);

 private:
  std::unique_ptr<web::http::experimental::listener::http_listener> _listener;
  std::unique_ptr<NSG::InMemoryNSGSearch>&                          _searcher;
};
