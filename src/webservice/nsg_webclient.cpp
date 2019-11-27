// nsg_webclient.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <cpprest/base_uri.h>
#include <cpprest/http_client.h>
#include <webservice/in_memory_nsg_search.h>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <set>


const std::wstring VECTOR_KEY = L"query", K_KEY = L"k",
                   RESULTS_KEY = L"results", INDICES_KEY = L"indices",
                   QUERY_ID_KEY = L"query_id",
                   TIME_TAKEN_KEY = L"time_taken_in_us";

class LessOperator {
 public:
  constexpr bool operator()(
      const std::tuple<int, unsigned int*>& first,
      const std::tuple<int, unsigned int*>& second) const {
    return std::get<0>(first) < std::get<0>(second);
  }
};

std::mutex                                             setLock;
std::set<std::tuple<int, unsigned int*>, LessOperator> resultsSet;

void addResult(int queryId, unsigned int* indices) {
  auto tuple = std::make_tuple(queryId, indices);
  setLock.lock();
  resultsSet.emplace(tuple);
  setLock.unlock();
}

void save_result(std::ostream& out, unsigned* results, unsigned nd,
                 unsigned nr) {
  for (unsigned i = 0; i < nd; i++) {
    out.write((char*) &nr, sizeof(unsigned));
    out.write((char*) (results + i * nr), nr * sizeof(unsigned));
  }
  out.flush();
}
// TEMPORARY FOR IDENTIFYING RECALL ENDS

std::wstring getHostAddress(const char* hostName) {
  size_t  numBytes;
  wchar_t buffer[4096];  // cannot be more than this!
  mbstowcs_s(&numBytes, buffer, 4096, hostName, 4096);
  return std::wstring(buffer);
}

web::json::value preparePostBody(int numResults, float* queryVec,
                                 int dimensions, int queryId) {
  web::json::value queryArr = web::json::value::array();
  for (int i = 0; i < dimensions; i++) {
    queryArr[i] = web::json::value::number(queryVec[i]);
  }

  web::json::value k = web::json::value::number(numResults);
  web::json::value body = web::json::value::object();
  body[VECTOR_KEY] = queryArr;
  body[K_KEY] = k;
  body[QUERY_ID_KEY] = queryId;

  return body;
}

std::vector<float> getQueryVector(
    const std::string& line,
    char
        delimiter /* , std::function< T(const std::string&) >& conversionFn*/) {
  std::vector<float> extractedValues;

  size_t findOffset = 0, extractStart = 0;
  do {
    findOffset = line.find(delimiter, findOffset);

    std::string substr;
    if (findOffset != std::string::npos) {
      substr = line.substr(extractStart, findOffset - extractStart);
      findOffset++;
      extractStart = findOffset;
    } else {
      substr = line.substr(extractStart, std::string::npos);
    }
    extractedValues.push_back(static_cast<float>(atof(substr.c_str())));

  } while (findOffset != std::string::npos);

  return extractedValues;
}

void loadTextData(const std::string& fileName, float*& queries,
                  unsigned int& numPoints, unsigned int& dimensions) {
  std::ifstream fin(fileName);
  if (!fin.is_open()) {
    throw std::exception(
        (std::string("Could not open file ") + fileName).c_str());
  }

  std::string        line;
  std::vector<float> allValues;
  while (!fin.eof()) {
    std::getline(fin, line);
    line.erase(
        std::remove_if(line.begin(), line.end(),
                       [](char c) { return c == '[' || c == ']' || c == '"'; }),
        line.end());

    auto queryVector = getQueryVector(line, '|');
    if (dimensions != 0 && queryVector.size() != dimensions) {
      std::stringstream message;
      message << "Found mismatched dimensions in points. " << dimensions
              << "v/s" << queryVector.size();
      throw std::exception(message.str().c_str());
    }
    dimensions = static_cast<unsigned int>(queryVector.size());
    allValues.insert(allValues.end(), queryVector.begin(), queryVector.end());
    numPoints++;
  }

  queries = new float[allValues.size()];
  std::copy(allValues.begin(), allValues.end(), queries);
}

void loadData(const std::string& fileName, float*& queries,
              unsigned int& numPoints, unsigned int& dimensions) {
  if (fileName.rfind(".txt") != std::string::npos) {
    loadTextData(fileName, queries, numPoints, dimensions);
  } else if (fileName.rfind(".fvecs") != std::string::npos) {
    diskann::InMemoryNSGSearch::load_data(fileName.c_str(), queries, numPoints,
                                      dimensions);
  } else {
    std::string message = std::string("Unknown file extension in filename ") +
                          fileName +
                          " supported extensions are .txt and .fvecs";
    throw std::exception(message.c_str());
  }
}

void processResponse(const std::wstring& jsonStr, std::ostream& ivecStream) {
  web::json::value parsedResponse = web::json::value::parse(jsonStr);
  if (parsedResponse[INDICES_KEY].is_array()) {
    const web::json::array indicesArr = parsedResponse[INDICES_KEY].as_array();
    unsigned int*          indices = new unsigned int[indicesArr.size()];
    for (int i = 0; i < indicesArr.size(); i++) {
      indices[i] = indicesArr.at(i).as_integer();
    }

    auto queryId = parsedResponse[QUERY_ID_KEY].as_integer();
    addResult(queryId, indices);

    std::cout << "Processed response for query " << queryId << std::endl;
  } else {
    std::cout << "ParsedResponse[\"indices\"] is not an array! but "
              << parsedResponse[INDICES_KEY].type() << std::endl;
  }
}

pplx::task<void> runQuery(float* queries, unsigned int dimensions,
                          unsigned int i, unsigned int numResults,
                          const web::uri& serviceUri,
                          std::ostream&   ivecStream) {
  float* query = new float[dimensions];
  // each vector is float[0] -> float[dimensions]
  std::copy(queries + i * dimensions, (queries + i * dimensions) + dimensions,
            query);

  std::cout << "******************" << std::endl
            << "Firing query: " << i << std::endl;
  // std::for_each(query, query + dimensions, [](const float value) {
  //  std::cout << value << ",";
  //});
  // std::cout << std::endl;

  web::json::value body = preparePostBody(numResults, query, dimensions, i);

  web::http::client::http_client_config config;
  config.set_timeout<std::chrono::seconds>(std::chrono::seconds(10));
  web::http::client::http_client client(serviceUri, config);

  auto responseTask =
      client.request(web::http::methods::POST, L"", body)
          .then([&](web::http::http_response response) {
            try {
              auto jsonStr = response.extract_string(true).get();
              processResponse(jsonStr, ivecStream);
              // std::wcout << jsonStr << std::endl;
            } catch (const web::http::http_exception& hex) {
              std::cout << "HTTP exception: " << hex.what() << std::endl;
              throw;
            } catch (const std::exception& ex) {
              std::cout << "STD exception " << ex.what() << std::endl;
              throw;
            }
          });

  delete[] query;
  return responseTask;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: nsg_webclient <service_addr> <query_file> "
                 "<num_nns_per_query> <ivecs_file> [<output_file>]"
              << std::endl;
    exit(1);
  }

  web::uri     serviceUri = web::uri_builder(getHostAddress(argv[1])).to_uri();
  float*       queries = nullptr;
  unsigned int dimensions = 0, numPoints = 0;
  loadData(std::string(argv[2]), queries, numPoints, dimensions);
  int           numResults = atoi(argv[3]);
  std::ofstream ivecStream(argv[4], std::ios_base::binary | std::ios_base::out);
  std::wostream& outputStream = (argc >= 6)
                                    ? (std::wostream&) std::wcout
                                    : (std::wostream&) std::wofstream(argv[5]);

  std::cout << "Connecting to service: " << argv[1]
            << " to run queries from file: " << argv[2]
            << " num expected results: " << argv[3] << " writing output to "
            << ((argc >= 5) ? argv[4] : "cout") << " and writing ivecs to "
            << ((argc >= 6) ? argv[5] : "none") << std::endl;

  auto startTime = std::chrono::system_clock::now();
  std::vector<Concurrency::task<void>> runningTasks;

  try {
    for (unsigned int i = 0; i < numPoints; i++) {
      auto task =
          runQuery(queries, dimensions, i, numResults, serviceUri, ivecStream);
      runningTasks.push_back(task);

      if (runningTasks.size() % 100 == 0) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
	  }
    }

    // wait for all tasks to complete.
    std::for_each(runningTasks.begin(), runningTasks.end(),
                  [](Concurrency::task<void> rtask) { rtask.get(); });

    int lastId = -1;
    std::for_each(resultsSet.begin(), resultsSet.end(),
                  [&](const std::tuple<int, unsigned int*> entry) {
                    if (lastId >= std::get<0>(entry)) {
                      throw std::exception("Set is not ordered.");
                    } else {
                      lastId = std::get<0>(entry);
                    }
                    save_result(ivecStream, std::get<1>(entry), 1, numResults);
                  });

    outputStream.flush();
    ivecStream.flush();
  } catch (const web::http::http_exception& hex) {
    std::cout << "Outer block HTTP Exception: " << hex.what() << std::endl;
    throw;
  } catch (const std::exception& ex) {
    std::cout << "Outer block STD Exception: " << ex.what() << std::endl;
    throw;
  }

  std::cout << "Completed " << numPoints << " queries in "
            << std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now() - startTime)
                   .count()
            << "s" << std::endl;
}
