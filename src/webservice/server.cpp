#include "server.h"
#include <cpprest/base_uri.h>
#include <string>

const std::wstring VECTOR_KEY = L"query", K_KEY = L"k", FIRST_KEY = L"first",
                   START_TIME_KEY = L"start_time", END_TIME_KEY = L"end_time",
					TIME_TAKEN_KEY=L"time_taken_in_us";


Server::Server(web::uri& uri) {
  _listener = new web::http::experimental::listener::http_listener(uri);
  _listener->support(
      web::http::methods::POST,
      std::bind(&Server::handle_post, this, std::placeholders::_1));
}

Server::~Server() {
  delete _listener;
}

pplx::task<void> Server::open() {
  return _listener->open();
}
pplx::task<void> Server::close() {
  return _listener->close();
}

void Server::handle_post(web::http::http_request message) {
  auto         startTime = std::chrono::steady_clock::now();
  std::wstring body = message.extract_string(true).get();

  web::json::value val = web::json::value::parse(body);
  web::json::array query = val.at(VECTOR_KEY).as_array();
  int              k = val.at(K_KEY).as_integer();


  web::json::value response = web::json::value::object();
  response[K_KEY] = k;
  response[FIRST_KEY] = query[0];
  //response[START_TIME_KEY] = startTime.time_since_epoch().count();
  //response[END_TIME_KEY] =
  //    std::chrono::system_clock::now().time_since_epoch().count();
  response[TIME_TAKEN_KEY] =
      std::chrono::duration_cast<std::chrono::microseconds>
          (std::chrono::steady_clock::now() - startTime).count();

  // std::wstringstream responseStream;
  // responseStream << L"{ \"k\": " << k << ", \"first\":" << query[0] << L"}";

  // web::json::value response = web::json::value::parse(responseStream.str());
  message.reply(web::http::status_codes::OK, response);
}
