#pragma once

#include <cpprest/base_uri.h>
#include <cpprest/http_listener.h>
#include <webservice/disk_nsg_search.h>

class DiskServer {
 public:
  DiskServer(web::uri& url, std::unique_ptr<NSG::DiskNSGSearch>& searcher);
  virtual ~DiskServer();

  pplx::task<void> open();
  pplx::task<void> close();

 protected:
  void handle_post(web::http::http_request message);

 private:
  bool                                                              _isDebug;
  std::unique_ptr<web::http::experimental::listener::http_listener> _listener;
  std::unique_ptr<NSG::DiskNSGSearch>&                          _searcher;
};
