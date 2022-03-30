// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <restapi/common.h>
#include <cpprest/http_listener.h>

namespace diskann {
  class Server {
   public:
    Server(web::uri& url, std::unique_ptr<diskann::BaseSearch>& searcher,
           const std::string& typestring);
    virtual ~Server();

    pplx::task<void> open();
    pplx::task<void> close();

   protected:
    template<class T>
    void handle_post(web::http::http_request message);

    template<typename T>
    web::json::value toJsonArray(
        const std::vector<T>&                     v,
        std::function<web::json::value(const T&)> valConverter);
    web::json::value prepareResponse(const int64_t& queryId, const int k);

    template<class T>
    void parseJson(const utility::string_t& body, int& k, int64_t& queryId,
                   T*& queryVector, unsigned int& dimensions, unsigned &Ls);

    web::json::value idsToJsonArray(const diskann::SearchResult& result);
    web::json::value distancesToJsonArray(const diskann::SearchResult& result);
    web::json::value tagsToJsonArray(const diskann::SearchResult& result);

   private:
    bool                                                              _isDebug;
    std::unique_ptr<web::http::experimental::listener::http_listener> _listener;
    std::unique_ptr<diskann::BaseSearch>&                             _searcher;
  };
}  // namespace diskann
