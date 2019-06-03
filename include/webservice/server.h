#pragma once

#include <cpprest/http_listener.h>
#include <cpprest/base_uri.h>

class Server {
 public:
  
  Server(web::uri& url);
  virtual ~Server();

  pplx::task<void> open();
  pplx::task<void> close();

  protected:
  void handle_post(web::http::http_request message);

  private:
  web::http::experimental::listener::http_listener* _listener;
};
