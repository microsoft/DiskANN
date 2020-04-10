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

const int MIN_POINTS_FOR_VALUE_VERIFICATION = 10000;

std::vector<std::string> splitLine(const std::string& str, const char delimiter,
                                   int suggestedCapacity) {
  std::vector<std::string> records;
  records.reserve(suggestedCapacity);

  boost::split(records, str, [delimiter](char c) { return c == delimiter; });

  return records;
}

template<typename T>
std::vector<T> getVector(const std::string& str,
                         T convertFn(const std::string& str),
                         int suggestedSize) {
  auto idsAndVector = splitLine(str, '\t', suggestedSize);

  std::vector<T> typedVec;
  typedVec.reserve(suggestedSize);
  std::vector<std::string> strVec;

  if (idsAndVector.size() >= 3) {
    strVec = splitLine(idsAndVector[2], ',', suggestedSize);
  } else if (idsAndVector.size() == 2) {
    strVec = splitLine(idsAndVector[1], ',', suggestedSize);
  } else if (idsAndVector.size() == 1) {
    strVec = splitLine(idsAndVector[0], ',', suggestedSize);
  } else {
    std::cerr << "Found line " << str
              << " that has non-standard number of delimiters: "
              << idsAndVector.size() << std::endl;
  }
  std::transform(
      strVec.begin(), strVec.end(), std::back_inserter(typedVec),
      [&convertFn](const std::string& str) { return convertFn(str); });

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
    if (line.length() != 0) {
      count++;
      if (count % 10000 == 0) {
        std::cout << "Counted " << count << " lines." << std::endl;
      }
    }
  }

  return count;
}

template<typename T>
void writeBinaryVec(const char* outfile, uint32_t numDimensions,
                    const std::vector<std::vector<T>>& allVecs) {
  uint32_t numPoints = (uint32_t) allVecs.size();
  std::cout << "Writing  " << numPoints
            << " records of dimension: " << numDimensions
            << " to file: " << outfile << "...";

  std::ofstream fout(outfile, std::ios::binary);
  if (fout.is_open()) {
    fout.write((const char*) &numPoints, sizeof(uint32_t));
    fout.write((const char*) &numDimensions, sizeof(uint32_t));

    for (auto vec : allVecs) {
      fout.write((const char*) vec.data(), numDimensions * sizeof(T));
    }
    std::cout << "done." << std::endl;
  } else {
    char message[2048];
    std::cerr << "Could not open output file: " << outfile << ". "
              << strerror_s(message, 2048, errno) << std::endl;
  }
}

// template<typename T>
// void writeBinary(const char* outfile, uint32_t numDimensions,
//                 uint32_t numPoints, T* data) {
//    std::cout << "Writing  " << numPoints
//              << " records of dimension: " << numDimensions
//              << " to file: " << outfile << "...";
//
//    std::ofstream fout(outfile, std::ios::binary);
//    if (fout.is_open()) {
//        fout.write((const char*) &numPoints, sizeof(uint32_t));
//        fout.write((const char*) &numDimensions, sizeof(uint32_t));
//
//        fout.write((const char*) data,
//                   ((uint64_t) numDimensions) * numPoints * sizeof(T));
//        std::cout << "done." << std::endl;
//    } else {
//        char message[2048];
//        std::cerr << "Could not open output file: " << outfile << ". "
//                  << strerror_s(message, 2048, errno) << std::endl;
//    }
//}

uint32_t numPointsToCheck(uint32_t numPoints) {
  uint32_t point01percent = (uint32_t)(0.01 * numPoints / 100);
  uint32_t point1percent = (uint32_t)(0.1 * numPoints / 100);
  uint32_t onepercent = (uint32_t)(1 * numPoints / 100);

  return point01percent > MIN_POINTS_FOR_VALUE_VERIFICATION
             ? point01percent
             : point1percent > MIN_POINTS_FOR_VALUE_VERIFICATION
                   ? point1percent
                   : onepercent > MIN_POINTS_FOR_VALUE_VERIFICATION ? onepercent
                                                                    : numPoints;
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

  uint32_t mismatchCount = 0;
  uint32_t vectorsToCheck = numPointsToCheck(numPoints);
  T*       data = new T[numDimensions * vectorsToCheck];
  fin.read((char*) data, sizeof(T) * numDimensions * vectorsToCheck);

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

  delete[] data;

  if (mismatchCount == 0) {
    std::cout << "Successfully verified byte-byte conversion." << std::endl;
  } else {
    std::cout << "Conversion failed!!! Found " << mismatchCount
              << " mismatches in vector elements" << std::endl;
  }
  std::cout << "Successfully verified binary conversion." << std::endl;
}

template<typename T>
std::vector<T> getVector(const std::string& str,
                         T convertFn(const std::string& val)) {
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
void convert(const char* infile, const char* outfile,
             T convertFn(const std::string& val), uint32_t suggestedSize) {
  uint32_t numDimensions = getDimension(infile, convertFn);
  uint32_t numPoints = 0;

  std::vector<std::vector<T>> allVecs;
  allVecs.reserve(suggestedSize > 0 ? suggestedSize : 1);

  // T* data = new T[((uint64_t) numDimensions) * numPoints];
  // int           index = 0;

  std::ifstream fin(infile);
  std::string   line;
  int           lc = 0;
  while (std::getline(fin, line)) {
    lc++;
    auto vector = getVector(line, convertFn, numDimensions);
    if (vector.size() != 0) {
      if (vector.size() != numDimensions) {
        std::stringstream stream;
        stream << "Found vector with dimension " << vector.size()
               << " instead of " << numDimensions << " at line: " << lc;
        throw diskann::ANNException(stream.str(), -1);
      }
      allVecs.push_back(vector);
      if (allVecs.size() % 10000 == 0) {
        std::cout << "Processed " << allVecs.size() << " records." << std::endl;
      }
      // std::for_each(vector.begin(), vector.end(),
      //              [data, &index](const T& v) {
      //                  data[index] = v;
      //                  index++;
      //              });
    }

    // if ((index / numDimensions) % 5000 == 0) {
    //    std::cout << "Processed " << index / numDimensions << "
    //    records."
    //              << std::endl;
    //}
  }

  // writeBinary(outfile, numDimensions, numPoints, data);
  writeBinaryVec(outfile, numDimensions, allVecs);
  // delete[] data;

  sanityCheck<T>(infile, outfile, numDimensions, (uint32_t) allVecs.size(),
                 convertFn);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <uint8|int8|float> <input_axon_file> <out_bin_file> "
                 "[num_of_points] (use zero to not specify)"
              << std::endl;
    return -1;
  }

  std::cout << "Converting (" << argv[1] << ") axon file: " << argv[2]
            << " with " << argv[4] << " points to diskann bin format file "
            << argv[3] << std::endl;

  std::string datatype(argv[1]);
  uint32_t    numPoints = (uint32_t)::atoi(argv[4]);

  if (datatype == "uint8") {
    convert<uint8_t>(argv[2], argv[3], atouLambda, numPoints);
  } else if (datatype == "int8") {
    convert<int8_t>(argv[2], argv[3], atoiLambda, numPoints);
  } else if (datatype == "float") {
    convert<float>(argv[2], argv[3], atofLambda, numPoints);
  } else {
    std::cerr << "Unknown data type " << argv[1]
              << ". Supported data types are int8, uint8, and float."
              << std::endl;
  }
}