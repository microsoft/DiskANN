// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <ctime>
#include <functional>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <codecvt>

#include <restapi/server.h>

namespace diskann {
  const unsigned int DEFAULT_L = 100;

  Server::Server(web::uri& uri, std::unique_ptr<diskann::BaseSearch>& searcher,
                 const std::string& typestring)
      : _searcher(searcher) {
    _listener =
        std::unique_ptr<web::http::experimental::listener::http_listener>(
            new web::http::experimental::listener::http_listener(uri));
    if (typestring == std::string("float")) {
      _listener->support(
          std::bind(&Server::handle_post<float>, this, std::placeholders::_1));
    } else if (typestring == std::string("int8_t")) {
      _listener->support(
          web::http::methods::POST,
          std::bind(&Server::handle_post<int8_t>, this, std::placeholders::_1));
    } else if (typestring == std::string("uint8_t")) {
      _listener->support(web::http::methods::POST,
                         std::bind(&Server::handle_post<uint8_t>, this,
                                   std::placeholders::_1));
    } else {
      throw "Unsupported type in server constuctor";
    }
  }

  Server::~Server() {
  }

  pplx::task<void> Server::open() {
    return _listener->open();
  }
  pplx::task<void> Server::close() {
    return _listener->close();
  }

  template<class T>
  void Server::handle_post(web::http::http_request message) {
    message.extract_string(true).then([=](utility::string_t body) {
      int64_t queryId = -1;
      int     k = 0;
      try {
        T*           queryVector = nullptr;
        unsigned int dimensions = 0;
        unsigned int Ls;
        parseJson(body, k, queryId, queryVector, dimensions, Ls);

        auto startTime = std::chrono::high_resolution_clock::now();
        diskann::SearchResult result =
            _searcher->search(queryVector, dimensions, (unsigned int) k, Ls);
        diskann::aligned_free(queryVector);

        web::json::value response = prepareResponse(queryId, k);
        response[INDICES_KEY] = idsToJsonArray(result);
        response[DISTANCES_KEY] = distancesToJsonArray(result);
        if (result.tags_enabled())
          response[TAGS_KEY] = tagsToJsonArray(result);

        response[TIME_TAKEN_KEY] =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - startTime)
                .count();

        std::cout << "Responding to: " << queryId << std::endl;
        message.reply(web::http::status_codes::OK, response).wait();
      } catch (const std::exception& ex) {
        std::cerr << "Exception while processing request: " << queryId << ":"
                  << ex.what() << std::endl;
        web::json::value response = prepareResponse(queryId, k);
        response[ERROR_MESSAGE_KEY] = web::json::value::string(ex.what());
        // web::json::value::string(to_wstring(ex.what()));
        message.reply(web::http::status_codes::InternalError, response).wait();
      }
    });
  }

  web::json::value Server::prepareResponse(const int64_t& queryId,
                                           const int      k) {
    web::json::value response = web::json::value::object();
    response[QUERY_ID_KEY] = queryId;
    response[K_KEY] = k;

    return response;
  }

  template<class T>
  void Server::parseJson(const utility::string_t& body, int& k,
                         int64_t& queryId, T*& queryVector,
                         unsigned int& dimensions, unsigned& Ls) {
    std::cout << body << std::endl;
    web::json::value val = web::json::value::parse(body);
    web::json::array queryArr = val.at(VECTOR_KEY).as_array();
    queryId = val.has_field(QUERY_ID_KEY)
                  ? val.at(QUERY_ID_KEY).as_number().to_int64()
                  : -1;
    Ls = val.has_field(L_KEY) ? val.at(L_KEY).as_number().to_uint32()
        : DEFAULT_L;
    k = val.at(K_KEY).as_integer();

    if (k <= 0) {
      throw new std::invalid_argument(
          "Num of expected NN (k) must be greater than zero.");
    }
    if (queryArr.size() == 0) {
      throw new std::invalid_argument("Query vector has zero elements.");
    }

    dimensions = static_cast<unsigned int>(queryArr.size());
    unsigned new_dim = ROUND_UP(dimensions, 8);
    diskann::alloc_aligned((void**) &queryVector, new_dim * sizeof(T),
                           8 * sizeof(T));
    memset(queryVector, 0, new_dim * sizeof(float));
    for (size_t i = 0; i < queryArr.size(); i++) {
      queryVector[i] = (float) queryArr[i].as_double();
    }
  }

  template<typename T>
  web::json::value Server::toJsonArray(
      const std::vector<T>&                     v,
      std::function<web::json::value(const T&)> valConverter) {
    web::json::value rslts = web::json::value::array();
    for (size_t i = 0; i < v.size(); i++) {
      auto jsonVal = valConverter(v[i]);
      rslts[i] = jsonVal;
    }
    return rslts;
  }

  web::json::value Server::idsToJsonArray(const diskann::SearchResult& result) {
    web::json::value idArray = web::json::value::array();
    auto             ids = result.get_indices();
    for (size_t i = 0; i < ids.size(); i++) {
      auto idVal = web::json::value::number(ids[i]);
      idArray[i] = idVal;
    }
    std::cout << "Vector size: " << ids.size() << std::endl;
    return idArray;
  }

  web::json::value Server::distancesToJsonArray(
      const diskann::SearchResult& result) {
    web::json::value distArray = web::json::value::array();
    auto             distances = result.get_distances();
    for (size_t i = 0; i < distances.size(); i++) {
      distArray[i] = web::json::value::number(distances[i]);
    }
    return distArray;
  }

  web::json::value Server::tagsToJsonArray(
      const diskann::SearchResult& result) {
    web::json::value tagArray = web::json::value::array();
    auto             tags = result.get_tags();
    for (size_t i = 0; i < tags.size(); i++) {
      tagArray[i] = web::json::value::string(tags[i]);
    }
    return tagArray;
  }
};  // namespace diskann