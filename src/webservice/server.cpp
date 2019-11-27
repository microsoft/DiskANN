#include <ctime>
#include <functional>
#include <iomanip>
#include <string>

#include <cpprest/base_uri.h>
#include <webservice/server.h>
#include <webservice/in_memory_nsg_search.h>


// Utility function declarations
static std::wstring     to_wstring(const char* str);
template<typename T>
web::json::value toJsonArray(
    const std::vector<T>&                     v,
    std::function<web::json::value(const T&)> valConverter);

static web::json::value rsltIdsToJsonArray(
    const diskann::NSGSearchResult& srchRslt);
static web::json::value scoresToJsonArray(const diskann::NSGSearchResult& srchRslt);
static void parseJson(const utility::string_t& body, int& k, long long& queryId,
                      float*& queryVector, unsigned int& dimensions);
static web::json::value prepareResponse(const long long& queryId, const int k);

// Constants
const std::wstring VECTOR_KEY = L"query", K_KEY = L"k",
                   RESULTS_KEY = L"results", SCORES_KEY = L"scores",
                   INDICES_KEY = L"indices",
                   QUERY_ID_KEY = L"query_id", ERROR_MESSAGE_KEY = L"error",
                   TIME_TAKEN_KEY = L"time_taken_in_us";

// class Server
Server::Server(web::uri& uri, std::unique_ptr<diskann::InMemoryNSGSearch>& searcher)
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
  message.extract_string(true).then([=](utility::string_t body) {
    long long queryId = -1;
    int       k = 0;
    try {
      float*       queryVector = nullptr;
      unsigned int dimensions = 0;
      parseJson(body, k, queryId, queryVector, dimensions);

      diskann::NSGSearchResult srchRslt =
          _searcher->search(queryVector, dimensions, (unsigned int) k);
      diskann::aligned_free(queryVector);

      web::json::value response = prepareResponse(queryId, k);
      web::json::value ids = rsltIdsToJsonArray(srchRslt);
      response[RESULTS_KEY] = ids;
      web::json::value indices = toJsonArray<unsigned int>(srchRslt.finalResultIndices, [](const unsigned int& i){
        return web::json::value::number(i);
	  });
      response[INDICES_KEY] = indices;

      web::json::value scores = scoresToJsonArray(srchRslt);
      response[SCORES_KEY] = scores;
      response[TIME_TAKEN_KEY] =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now() - startTime)
              .count();

      std::wcout << L"Responding to: " << queryId << std::endl;
      message.reply(web::http::status_codes::OK, response).wait();
    } catch (const std::exception& ex) {
      std::cerr << "Exception while processing request: " << queryId << ":"
                << ex.what() <<  std::endl;
      web::json::value response = prepareResponse(queryId, k);
      response[ERROR_MESSAGE_KEY] =
          web::json::value::string(to_wstring(ex.what()));
      message.reply(web::http::status_codes::InternalError, response).wait();
    }
  });
}

static web::json::value prepareResponse(const long long& queryId, const int k) {
  web::json::value response = web::json::value::object();
  response[QUERY_ID_KEY] = queryId;
  response[K_KEY] = k;

  return response;
}

static void parseJson(const utility::string_t& body, int& k, long long& queryId,
                      float*& queryVector, unsigned int& dimensions) {
  web::json::value val = web::json::value::parse(body);
  web::json::array queryArr = val.at(VECTOR_KEY).as_array();
  queryId = val.has_field(QUERY_ID_KEY)
                ? val.at(QUERY_ID_KEY).as_number().to_int64()
                : -1;
  k = val.at(K_KEY).as_integer();

  if (k <= 0) {
    throw new std::exception(
        "Num of expected NN (k) must be greater than zero.");
  }
  if (queryArr.size() == 0) {
    throw new std::exception("Query vector has zero elements.");
  }

  dimensions = static_cast<unsigned int>(queryArr.size());
  unsigned new_dim = ROUND_UP(dimensions, 8);
  diskann::alloc_aligned((void**) &queryVector, new_dim * sizeof(float), 256);
  memset(queryVector, 0, new_dim * sizeof(float));
  for (int i = 0; i < queryArr.size(); i++) {
    queryVector[i] = (float) queryArr[i].as_double();
  }
}

// Utility functions

template<typename T>
web::json::value toJsonArray(
    const std::vector<T>&                     v,
    std::function<web::json::value(const T&)> valConverter) {
  web::json::value rslts = web::json::value::array();
  for (int i = 0; i < v.size(); i++) {
    auto jsonVal = valConverter(v[i]);
    rslts[i] = jsonVal;
  }
  return rslts;
}

web::json::value rsltIdsToJsonArray(const diskann::NSGSearchResult& srchRslt) {
  web::json::value rslts = web::json::value::array();
  for (int i = 0; i < srchRslt.finalResults.size(); i++) {
    auto idVal = web::json::value::string(srchRslt.finalResults[i]);
    rslts[i] = idVal;
  }
  return rslts;
}

web::json::value scoresToJsonArray(const diskann::NSGSearchResult& srchRslt) {
  web::json::value scores = web::json::value::array();
  for (int i = 0; i < srchRslt.distances.size(); i++) {
    scores[i] = web::json::value::number(srchRslt.distances[i]);
  }
  return scores;
}

static std::wstring to_wstring(const char* str) {
  wchar_t buffer[4096];
  mbstowcs_s(nullptr, buffer, sizeof(buffer) / sizeof(buffer[0]), str,
             sizeof(buffer) / sizeof(buffer[0]));
  return std::wstring(buffer);
}
