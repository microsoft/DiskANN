#include <iostream>
#include <random>
#include "distance.h"

float distanceL2_I(const int8_t* a, const int8_t* b, size_t size) {
  float distance = 0;
  for (int i = 0; i < size; i++) {
    int16_t diff = ((int16_t) a[i] - (int16_t) b[i]);
    distance += diff * diff;
  }
  return distance;
}

int8_t* createVector(int size) {
  auto p = new int8_t[size];
  for (int i = 0; i < size; i++) {
    p[i] = 0;
  }
  return p;
}

uint8_t* getData(const char* infileName, uint8_t dataTypeSizeInBytes,
                 int32_t& count, int32_t& dimension) {
  std::ifstream infile(infileName, std::ios::binary);
  if (!infile.is_open()) {
    std::cerr << "Could not open input file: " << infileName << std::endl;
  }

  infile.read(reinterpret_cast<char*>(&count), sizeof(count));
  infile.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));

  std::cout << infileName << ": count: " << count
            << " dimensions: " << dimension << std::endl;

  uint64_t sizeToRead = (uint64_t) count * dataTypeSizeInBytes * dimension;
  uint8_t* bytes = new uint8_t[sizeToRead];
  infile.read((char*) bytes, sizeToRead);

  return bytes;
}

void concat_files(int argc, char** argv) {
  if (argc < 5) {
    std::cout << "Mode 1 requires 2 input files and 1 output file as argument."
              << std::endl
              << " This will concatenate the contents of the files specified "
                 "(in the same order) and save them to outfile."
              << std::endl;
    return;
  }

  int      dimension1, dimension2, count1, count2;
  uint8_t* data1 = getData(argv[2], 1, count1, dimension1);
  uint8_t* data2 = getData(argv[3], 1, count2, dimension2);

  if (dimension1 != dimension2) {
    std::cout << "Error! Cannot combine vectors of differing dimensions ("
              << dimension1 << "," << dimension2 << ")" << std::endl;
    return;
  }

  std::ofstream outFile(argv[4], std::ios::binary);
  if (!outFile.is_open()) {
    std::cerr << "Could not open output file: " << argv[4] << std::endl;
    return;
  }
  outFile << count1 + count2;
  outFile << dimension1;
  outFile.write((char*) data1, count1);
  outFile.write((char*) data2, count2);
  outFile.close();

  delete[] data1;
  delete[] data2;

  std::cout << "Concatenated " << (count1 + count2) << " " << dimension1
            << "-dimension vectors from files: " << argv[2] << " and "
            << argv[3] << " and saved to output file: " << argv[4] << std::endl;
}

void assignPtr(std::unique_ptr<float[]>& data) {
  float* ptr = new float[30];
  for (int i = 0; i < 30; i++) {
    ptr[i] = i * 1.0f;
  }

  data.reset(ptr);
  // std::make_unique<float[]>(30);
}

void uniquePtrAssignment() {
  std::unique_ptr<float[]> data;
  assignPtr(data);

  for (int i = 0; i < 30; i++) {
    std::cout << data[i] << " ";
    if (data[i] - i > 0.001) {
      std::cout << data[i] << "," << i << " screwed " << std::endl;
      break;
    }
  }
  std::cout << std::endl;
  std::cout << "safe." << std::endl;
}

void compareDistanceComputations() {
  int8_t vec1[] = {127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
                   127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
                   127, 127, 127, 127, 127, 127, 127, 127, 127, 127};
  int8_t vec2[] = {-128, -128, -128, -128, -128, -128, -128, -128,
                   -128, -128, -128, -128, -128, -128, -128, -128,
                   -128, -128, -128, -128, -128, -128, -128, -128,
                   -128, -128, -128, -128, -128, -128, -128, -128};

  diskann::DistanceL2Int8 dist;
  float                   dist1 = dist.compare(vec1, vec2, 8);
  float                   dist2 = distanceL2_I(vec1, vec2, 8);

  if (dist1 - dist2 > 0.01) {
    std::cout << "Test failed. AVX dist: " << dist1 << " normal dist: " << dist2
              << std::endl;
  } else {
    std::cout << "Two score are the same. " << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << std::string("Usage: ") << argv[0] << " <mode> [arguments]"
              << std::endl;
    std::cout << "Modes: 1 for file concat. Args <file1> <file2> <outfile>"
              << std::endl
              << "       2 for test unique_ptr assignment. No args."
              << std::endl;
    std::cout << "      3 for comparing distance computations (AVX and "
                 "normal). No args."
              << std::endl;
    return -1;
  }

  int mode = atoi(argv[1]);

  switch (mode) {
    case 1:
      concat_files(argc, argv);
      return 0;
      break;
    case 2:
      uniquePtrAssignment();
      break;
    case 3:
      compareDistanceComputations();
      break;
    default:
      std::cout << "Don't know what to do with mode parameter: " << argv[1]
                << std::endl;
  }
}
