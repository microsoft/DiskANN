#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "cached_io.h"
#include "ann_exception.h"

float (*atofLambda)(const std::string&) = [](const std::string& str) {
  return (float) ::atof(str.c_str());
};
int8_t (*atoiLambda)(const std::string&) = [](const std::string& str) {
  return (int8_t)::atoi(str.c_str());
};
uint8_t (*atouLambda)(const std::string&) = [](const std::string& str) {
  return (uint8_t)::atoi(str.c_str());
};

const int ESTIMATED_VECTOR_COUNT = 100;

std::vector<std::string> splitLine(const std::string& str, const char delimiter,
                                   int suggestedCapacity) {
  std::vector<std::string> records;
  records.reserve(suggestedCapacity);

  boost::split(records, str, [delimiter](char c) { return c == delimiter; });

  return records;
}

template<typename T>
std::vector<T> getVector(const std::string& str,
                         T                  convertFn(const std::string& str),
                         int                suggestedSize) {
  auto idsAndVector = splitLine(str, '\t', suggestedSize);

  std::vector<T> typedVec;
  typedVec.reserve(suggestedSize);

  if (idsAndVector.size() >= 3) {
    auto strVec = splitLine(idsAndVector[2], ',', suggestedSize);
    std::transform(
        strVec.begin(), strVec.end(), std::back_inserter(typedVec),
        [&convertFn](const std::string& str) { return convertFn(str); });
  } else {
    std::cerr << "Found line " << str
              << " that didn't have standard delimiters." << std::endl;
  }
  return typedVec;
}

template<typename T>
uint32_t getDimension(const char* inFile, T convertFn(const std::string& val)) {
  std::ifstream fin(inFile);
  if (fin.is_open()) {
    std::string line;
    std::getline(fin, line);

    auto vector = getVector(line, convertFn, 1);
    return (uint32_t) vector.size();
  } else {
    char message[2048];
    std::cerr << "Could not open input file: " << inFile << ". "
              << strerror_s(message, 2048, errno) << std::endl;
    exit(1);
  }
}

uint32_t getRowCount(const char* infile) {
  std::ifstream fin(infile);
  std::string   line;

  uint32_t count = 0;
  while (std::getline(fin, line)) {
    if (line.length() != 0)
      count++;
  }

  return count;
}

template<typename T>
void writeBinary(const char* outfile, uint32_t numDimensions,
                 uint32_t numPoints, T* data) {
  std::cout << "Writing  " << numPoints
            << " records of dimension: " << numDimensions
            << " to file: " << outfile << "...";

  std::ofstream fout(outfile, std::ios::binary);
  if (fout.is_open()) {
    fout.write((const char*) &numPoints, sizeof(uint32_t));
    fout.write((const char*) &numDimensions, sizeof(uint32_t));

    fout.write((const char*) data, numDimensions * numPoints * sizeof(T));
    std::cout << "done." << std::endl;
  } else {
    char message[2048];
    std::cerr << "Could not open output file: " << outfile << ". "
              << strerror_s(message, 2048, errno) << std::endl;
  }
}

template<typename T>
void sanityCheck(const char* origfile, const char* binfile,
                 uint32_t numDimensions, uint32_t numPoints,
                 T convertFn(const std::string& str)) {
  uint32_t      nd, np;
  std::ifstream fin(binfile, std::ios::binary);
  std::ifstream orig(origfile);

  fin.read((char*) &np, sizeof(uint32_t));
  fin.read((char*) &nd, sizeof(uint32_t));

  assert(np == numPoints);
  if (np != numPoints) {
    throw diskann::ANNException(
        "Verifying binary write failed! np != numPoints", -1, __FUNCSIG__,
        __FILE__, __LINE__);
  }
  assert(nd == numDimensions);
  if (nd != numDimensions) {
    throw diskann::ANNException(
        "Verifying binary write failed! nd != numDimensions", -1, __FUNCSIG__,
        __FILE__, __LINE__);
  }

  uint64_t expectedDataSize = numPoints * numDimensions * sizeof(T);
  T*       data = (T*) malloc(expectedDataSize);
  fin.read((char*) data, expectedDataSize);


  uint32_t mismatchCount = 0;
  uint32_t vectorsToCheck =
      (uint32_t)(numPoints * 0.001) == 0
          ? numPoints
          : (uint32_t)(numPoints * 0.001);  // check 0.1% of points
  std::cout << "Checking first " << vectorsToCheck
            << " records for byte-byte match." << std::endl;

  for (uint32_t i = 0; i < vectorsToCheck; i++) {
    std::string line;
    std::getline(orig, line);

    auto origvec = getVector(line, convertFn, nd);
    assert(origvec.size() == nd);

    for (uint32_t j = 0; j < nd; j++) {
      if (origvec[j] != data[i * nd + j]) {
        std::cerr << "Found mismatch at line " << i << " orig: " << origvec[j]
                  << " bin val: " << data[i * nd + j] << std::endl;
        mismatchCount++;
      }
    }
  }

  if (mismatchCount == 0) {
    std::cout << "Successfully verified byte-byte conversion." << std::endl;
  } else {
    std::cout << "Conversion failed!!! Found " << mismatchCount
              << " mismatches in vector elements" << std::endl;
  }
  std::cout << "Successfully verified binary conversion." << std::endl;
}

template<typename T>
void convert(const char* infile, const char* outfile,
             T convertFn(const std::string& val)) {
  uint32_t numDimensions = getDimension(infile, convertFn);
  uint32_t numPoints = getRowCount(infile);

  T* data = new T[numDimensions * numPoints * sizeof(T)];

  int           index = 0;
  std::ifstream fin(infile);

  std::string line;
  while (std::getline(fin, line)) {
    auto vector = getVector(line, convertFn, numDimensions);
    if (vector.size() != 0) {
      assert(vector.size() == numDimensions);
      std::for_each(vector.begin(), vector.end(), [data, &index](const T& v) {
        data[index] = v;
        index++;
      });
    }

    if ((index / numDimensions) % 5000 == 0) {
      std::cout << "Processed " << index / numDimensions << " records."
                << std::endl;
    }
  }

  writeBinary(outfile, numDimensions, numPoints, data);
  delete[] data;
  sanityCheck<T>(infile, outfile, numDimensions, numPoints, convertFn);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <uint8|int8|float> <input_axon_file> <out_bin_file>"
              << std::endl;
    return -1;
  }

  std::cout << "Converting (" << argv[1] << ") axon file: " << argv[2]
            << " to randnsg bin format file " << argv[3] << std::endl;

  std::string datatype(argv[1]);

  if (datatype == "uint8") {
    convert<uint8_t>(argv[2], argv[3], atouLambda);
  } else if (datatype == "int8") {
    convert<int8_t>(argv[2], argv[3], atoiLambda);
  } else if (datatype == "float") {
    convert<float>(argv[2], argv[3], atofLambda);
  } else {
    std::cerr << "Unknown data type " << argv[1]
              << ". Supported data types are int8, uint8, and float."
              << std::endl;
  }
}