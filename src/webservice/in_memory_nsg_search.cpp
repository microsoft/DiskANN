#include <cosine_similarity.h>
#include <webservice/in_memory_nsg_search.h>
#include <ctime>
#include <iomanip>
#include "efanna2e/util.h"

namespace NSG {
  const unsigned int DEFAULT_BEAM_WIDTH = 8;
  // const unsigned int L_MULTIPLIER = 10;
  // const unsigned int MAX_L = 300;
  const unsigned int DEFAULT_L = 704;


  InMemoryNSGSearch::InMemoryNSGSearch(const char* baseFile,
                                       const char* indexFile,
                                       const char* idsFile, Metric m)
      : _baseVectors(nullptr) {

	  float* baseData;
    load_data(baseFile, baseData, this->_numPoints, this->_dimensions);
    _baseVectors = NSG::data_align(baseData, _numPoints, _dimensions); 

    _nsgIndex = std::unique_ptr<NSG::IndexNSG<float>>(
        new NSG::IndexNSG<float>(_dimensions, _numPoints, m, nullptr));
    _nsgIndex->Load(indexFile);

    _ids = load_ids(idsFile);
  }

  NSGSearchResult InMemoryNSGSearch::search(const float*       query,
                                            const unsigned int dimensions,
                                            const unsigned int K) {
    std::vector<unsigned int> start_points;

    unsigned int*             indices = new unsigned int[K];


    auto startTime = std::chrono::high_resolution_clock::now();
    auto queryMagnitude = NSG::vectorMagnitude(query, dimensions);

	_nsgIndex->BeamSearch(query, _baseVectors, K,
                          /* (std::min)(K * L_MULTIPLIER, MAX_L)*/ DEFAULT_L,
                          indices, DEFAULT_BEAM_WIDTH, start_points);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - startTime)
                        .count();

    // indices has the indexes of the results. Select the results from the
    // ids_vector.
    NSGSearchResult searchResult(K, (unsigned int) duration);
    std::for_each(indices, indices + K, [&](const unsigned int& index) {
      searchResult.addResult(_ids[index]);
      searchResult.finalResultIndices.push_back(index); //TEMPORARY FOR IDENTIFYING RECALL
    });

    std::vector<float> similarityScores = NSG::compute_cosine_similarity(
        query, indices, _baseVectors, _dimensions, K, queryMagnitude);
    searchResult.distances = similarityScores;

	//TEMPORARY FOR IDENTIFYING RECALL

    delete[] indices;
    return searchResult;
  }

  InMemoryNSGSearch::~InMemoryNSGSearch() {
    delete[] _baseVectors;
  }

  void InMemoryNSGSearch::load_data(const char* filename, float*& data,
                                    unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
      std::cerr << "Could not open data file " << filename << std::endl;
      exit(-1);
    }
    in.read((char*) &dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();

    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    std::cout << "Reading " << num << " points...";
    data = new float[(size_t) num * (size_t) dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
      in.seekg(4, std::ios::cur);
      in.read((char*) (data + i * dim), dim * 4);
    }
    std::cout << "done." << std::endl;
    in.close();
  }

  std::vector<std::wstring> InMemoryNSGSearch::load_ids(const char* idsFile) {
    std::wifstream            in(idsFile);
    std::vector<std::wstring> ids;

    if (!in.is_open()) {
      std::cerr << "Could not open " << idsFile << std::endl;
    }

    std::wstring id;
    while (!in.eof()) {
      in >> id;
      ids.push_back(id);
    }

    std::cout << "Loaded " << ids.size() << " from " << idsFile << std::endl;
    return ids;
  }

}  // namespace NSG
