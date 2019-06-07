// nsg_webclient.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <cpprest/base_uri.h>
#include <cpprest/http_client.h>
#include <webservice/in_memory_nsg_search.h>
#include <iostream>

const std::wstring VECTOR_KEY = L"query", K_KEY = L"k",
                   RESULTS_KEY = L"results", QUERY_ID_KEY = L"query_id",
                   TIME_TAKEN_KEY = L"time_taken_in_us";

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
        delimiter /* , std::function< T(const std::string&) >& conversionFn*/) 
{
  std::vector<float> extractedValues;

  int findOffset = 0, extractStart = 0;
  do {
    findOffset = line.find(delimiter, findOffset);

    std::string substr;
    if (findOffset != std::string::npos) {
      substr = line.substr(extractStart, findOffset - extractStart);
      findOffset++;
      extractStart++;
    } else {
      substr = line.substr(extractStart, std::string::npos);
    }
    extractedValues.push_back(atof(substr.c_str()));

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
    dimensions = queryVector.size();
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
    NSG::InMemoryNSGSearch::load_data(fileName.c_str(), queries, numPoints,
                                      dimensions);
  } else {
    std::string message = std::string("Unknown file extension in filename ") +
                          fileName +
                          " supported extensions are .txt and .fvecs";
    throw std::exception(message.c_str());
  }
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: nsg_webclient <service_addr> <query_file> "
                 "<num_nns_per_query> [<output_file>]"
              << std::endl;
    exit(1);
  }

  web::uri_builder bldr(getHostAddress(argv[1]));

  float*       queries = nullptr;
  unsigned int dimensions = 0, numPoints = 0;
  loadData(std::string(argv[2]), queries, numPoints, dimensions);
  int              numResults = atoi(argv[3]);
  
  std::wostream& outputStream = (argc == 4) ? (std::wostream&)std::wcout
                                                  : (std::wostream&)std::wofstream(argv[4]);

  auto startTime = std::chrono::system_clock::now();
  std::vector<Concurrency::task<void>> runningTasks;
  // each vector is float[0] -> float[dimensions]
  for (int i = 0; i < numPoints; i++) {
    float* query = new float[dimensions];
    std::copy(queries + i * dimensions, queries + i * dimensions + dimensions,
              query);

    std::cout << "******************" << std::endl
              << "Firing query: " << i << std::endl;
    web::json::value body = preparePostBody(numResults, query, dimensions, i);

    web::http::client::http_client client(bldr.to_uri());
    auto responseTask = client.request(web::http::methods::POST, L"", body)
                            .then([&](web::http::http_response response) {
                              std::wstring responseStr =
                                  response.extract_string().get();
                              outputStream << responseStr << std::endl;
                            });

    runningTasks.push_back(responseTask);
    delete[] query;
  }

  outputStream.flush();

  // wait for all tasks to complete.
  std::for_each(runningTasks.begin(), runningTasks.end(),
                [](Concurrency::task<void> rtask) { rtask.get(); });

  std::cout << "Completed " << numPoints << " queries in "
            << std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now() - startTime)
                   .count()
            << "s"
            << std::endl;
}
