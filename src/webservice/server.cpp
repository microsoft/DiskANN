#include <string>
#include <cpprest/base_uri.h>
#include <webservice/server.h>
#include <webservice/in_memory_nsg_search.h>



// Utility function declarations
static web::json::value rsltIdsToJsonArray(const NSG::NSGSearchResult& srchRslt);

const std::wstring VECTOR_KEY = L"query", K_KEY = L"k",
                   RESULTS_KEY = L"results",
                   TIME_TAKEN_KEY = L"time_taken_in_us";

Server::Server(web::uri& uri, std::unique_ptr<NSG::InMemoryNSGSearch>& searcher)
    : _searcher(searcher) {
  _listener = std::unique_ptr<web::http::experimental::listener::http_listener>(
      new web::http::experimental::listener::http_listener(uri));
  _listener->support(
      web::http::methods::POST,
      std::bind(&Server::handle_post, this, std::placeholders::_1));
}

Server::~Server() {
}

pplx::task<void> Server::open() {
  return _listener->open();
}
pplx::task<void> Server::close() {
  return _listener->close();
}

void Server::handle_post(web::http::http_request message) {
  auto startTime = std::chrono::high_resolution_clock::now();
  auto bodyTask = message.extract_string(true);

  bodyTask.then([=](utility::string_t body) {
    web::json::value val = web::json::value::parse(body);
    web::json::array queryArr = val.at(VECTOR_KEY).as_array();
    int              k = val.at(K_KEY).as_integer();

    assert(k > 0);

    float* query = new float[queryArr.size()];
    for (int i = 0; i < queryArr.size(); i++) {
      query[i] = (float) queryArr[i].as_double();
    }

    NSG::NSGSearchResult srchRslt = _searcher->search(query, (unsigned int) k);

    web::json::value ids = rsltIdsToJsonArray(srchRslt);

    web::json::value response = web::json::value::object();
    response[K_KEY] = k;
    response[RESULTS_KEY] = ids;
    response[TIME_TAKEN_KEY] =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - startTime)
            .count();
    message.reply(web::http::status_codes::OK, response);
  });
}

// Utility functions
web::json::value rsltIdsToJsonArray(const NSG::NSGSearchResult& srchRslt) {
  web::json::value rslts = web::json::value::array();
  for (int i = 0; i < srchRslt.finalResults.size(); i++) {
    auto idVal = web::json::value::string(srchRslt.finalResults[i]);
    rslts[i] = idVal;
  }
  return rslts;
}
