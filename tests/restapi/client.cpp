// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <ctime>
#include <functional>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <codecvt>

#include <cpprest/http_client.h>
#include <restapi/common.h>

using namespace web;
using namespace web::http;
using namespace web::http::client;

using namespace diskann;

template<typename T>
void query_loop(const std::string& ip_addr_port, const std::string& query_file,
                const unsigned nq, const unsigned Ls) {
  web::http::client::http_client client(U(ip_addr_port));

  T*     data;
  size_t npts = 1, ndims = 128, rounded_dim = 128;
  diskann::load_aligned_bin<T>(query_file, data, npts, ndims, rounded_dim);

  for (unsigned i = 0; i < nq; ++i) {
    T*                      vec = data + i * rounded_dim;
    web::http::http_request http_query(methods::POST);
    web::json::value        queryJson = web::json::value::object();
    queryJson[QUERY_ID_KEY] = i;
    queryJson[K_KEY] = 10;
    queryJson[L_KEY] = Ls;
    for (size_t i = 0; i < ndims; ++i) {
      queryJson[VECTOR_KEY][i] = web::json::value::number(vec[i]);
    }
    http_query.set_body(queryJson);

    client.request(http_query)
        .then([](web::http::http_response response)
                  -> pplx::task<utility::string_t> {
          if (response.status_code() == status_codes::OK) {
            return response.extract_string();
          }
          std::cerr << "Query failed" << std::endl;
          return pplx::task_from_result(utility::string_t());
        })
        .then([](pplx::task<utility::string_t> previousTask) {
          try {
            std::cout << previousTask.get() << std::endl;
          } catch (http_exception const& e) {
            std::wcout << e.what() << std::endl;
          }
        })
        .wait();
  }
}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cout << "Usage: client ip_addr:port <float/int8/uint8> query_file "
                 "num_queries Ls"
              << std::endl;
    exit(1);
  }

  const std::string ip_addr_port(argv[1]);
  const std::string typestring(argv[2]);
  const std::string query_file(argv[3]);
  unsigned          nq = atoi(argv[4]);
  unsigned          Ls = atoi(argv[5]);

  if (typestring == std::string("float")) {
    query_loop<float>(ip_addr_port, query_file, nq, Ls);
  } else if (typestring == std::string("int8")) {
    query_loop<int8_t>(ip_addr_port, query_file, nq, Ls);
  } else if (typestring == std::string("uint8")) {
    query_loop<uint8_t>(ip_addr_port, query_file, nq, Ls);
  } else {
    std::cerr << "Unsupported type " << argv[2] << std::endl;
    return -1;
  }

  return 0;
}