// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <numeric>
#include <cassert>
#include <ctime>
#include <vector>

#include <omp.h>

#include "utils.h"
#include "distance.h"
#include "logger.h"

// 7-- OpenMP thread pool test

void testOpenMPThreadPool() {
  std::vector<uint32_t> vec(8192);
  std::iota(vec.begin(), vec.end(), 1);
  std::vector<uint32_t> squares(8192);
  std::vector<uint32_t> resource(8);

#pragma omp parallel for schedule(dynamic, 64) num_threads(8)
  for (int i = 0; i < vec.size(); i++) {
    squares[i] = vec[i] * vec[i];
    resource[omp_get_thread_num()] = omp_get_thread_num();
  }

  // Anything to prevent the above code from being optimized away.
  for (int i = 0; i < 8192; i++) {
    if (i % 1000 == 0) {
      std::cout << i << squares[i] << std::endl;
    }
  }
  std::cout << "Omp is running: " << omp_get_num_threads() << std::endl;

#pragma omp parallel for schedule(dynamic, 64) num_threads(8)
  for (int i = 0; i < vec.size(); i++) {
    squares[i] = vec[i] * vec[i];
    resource[omp_get_thread_num()] = omp_get_thread_num();
  }

  for (auto id : resource) {
    std::cout << id << " ";
  }
  std::cout << std::endl;
  std::cout << "After second omp, " << omp_get_num_threads() << std::endl;
}

// For flexibility in switching between AVX/AVX2.
bool Avx2SupportedCPU = true;

// Utility functions.
template<typename T>
void printArray(T* arr, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << (int) arr[i] << " ";
  }
  std::cout << std::endl;
}

template<>
void printArray(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}

// Lame but correct distance computations
float distanceL2_I(const int8_t* a, const int8_t* b, size_t size) {
  float distance = 0;
  for (int i = 0; i < size; i++) {
    int16_t diff = ((int16_t) a[i] - (int16_t) b[i]);
    distance += diff * diff;
  }
  return distance;
}

float distanceL2_F(const float* a, const float* b, size_t size) {
  float distance = 0;
  for (int i = 0; i < size; i++) {
    distance += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return distance;
}

float distanceCosine_I(const int8_t* a, const int8_t* b, size_t size) {
  float aMag = 0.0f, bMag = 0.0f, scalarProduct = 0.0f;
  for (int i = 0; i < size; i++) {
    aMag += ((int32_t) a[i]) * ((int32_t) a[i]);
    bMag += ((int32_t) b[i]) * ((int32_t) b[i]);
    scalarProduct += ((int32_t) a[i]) * ((int32_t) b[i]);
  }
  std::cout << "aMag: " << aMag << " bMag: " << bMag
            << " product: " << scalarProduct << std::endl;
  return 1.0f - (float) (scalarProduct / (sqrt(aMag) * sqrt(bMag)));
}
float distanceCosine_F(const float* a, const float* b, size_t size) {
  float aMag = 0.0f, bMag = 0.0f, scalarProduct = 0.0f;
  for (int i = 0; i < size; i++) {
    aMag += a[i] * a[i];
    bMag += b[i] * b[i];
    scalarProduct += a[i] * b[i];
  }
  std::cout << "aMag: " << aMag << " bMag: " << bMag
            << " product: " << scalarProduct << std::endl;

  return 1.0f - (float) (scalarProduct / (sqrt(aMag) * sqrt(bMag)));
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

  diskann::cout << infileName << ": count: " << count
                << " dimensions: " << dimension << std::endl;

  uint64_t sizeToRead = (uint64_t) count * dataTypeSizeInBytes * dimension;
  uint8_t* bytes = new uint8_t[sizeToRead];
  infile.read((char*) bytes, sizeToRead);

  return bytes;
}

void concat_files(int argc, char** argv) {
  if (argc < 5) {
    diskann::cout
        << "Mode 1 requires 2 input files and 1 output file as argument."
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
    diskann::cout << "Error! Cannot combine vectors of differing dimensions ("
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

  diskann::cout << "Concatenated " << (count1 + count2) << " " << dimension1
                << "-dimension vectors from files: " << argv[2] << " and "
                << argv[3] << " and saved to output file: " << argv[4]
                << std::endl;
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
    diskann::cout << data[i] << " ";
    if (data[i] - i > 0.001) {
      diskann::cout << data[i] << "," << i << " screwed " << std::endl;
      break;
    }
  }
  diskann::cout << std::endl;
  diskann::cout << "safe." << std::endl;
}

float     epsilon = 0.01;
const int A_SIZE = 100;
const int A_COUNT = 20;
void      compareDistanceComputationsInt() {
  // int8_t vec1[] = {127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
  //                 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
  //                 127, 127, 127, 127, 127, 127, 127, 127, 127, 127};
  // int8_t vec2[] = {-128, -128, -128, -128, -128, -128, -128, -128,
  //                 -128, -128, -128, -128, -128, -128, -128, -128,
  //                 -128, -128, -128, -128, -128, -128, -128, -128,
  //                 -128, -128, -128, -128, -128, -128, -128, -128};

  int8_t vec1[A_SIZE], vec2[A_SIZE];
  srand(time(0));
  bool neg = false;
  for (int i = 0; i < A_SIZE; i++) {
    auto a = rand() % 10;
    auto b = rand() % 10;
    vec1[i] = -a;  // < 0 ? -a : a;
    vec2[i] = -b;  // < 0 ? -b : b;

    neg = neg || vec1[i] < 0 || vec2[i] < 0;
  }
  if (!neg) {
    diskann::cout << "No negative numbers in test. " << std::endl;
  }

  std::cout << "Starting distance computation test for float." << std::endl;
  printArray(vec1, 16);
  printArray(vec2, 16);

  {
    // diskann::DistanceL2Int8 dist;
    diskann::AVXDistanceL2Int8 dist;
    float                      dist1 = dist.compare(vec1, vec2, 16);
    float                      dist2 = distanceL2_I(vec1, vec2, 16);

    if (abs(dist1 - dist2) > epsilon) {
      std::cout
          << "compareDistanceComputationsInt(): L2 Test failed. AVX dist: "
          << dist1 << " normal dist: " << dist2 << " difference > " << epsilon
          << std::endl;
    } else {
      std::cout << "Two scores are the same. " << std::endl;
    }
  }
  {
    diskann::DistanceCosineInt8 dist;
    float                       dist1 = dist.compare(vec1, vec2, 16);
    float                       dist2 = distanceCosine_I(vec1, vec2, 16);
    if (abs(dist1 - dist2) > epsilon) {
      std::cout
          << "compareDistanceComputationsInt(): Cosine Test failed. AVX dist: "
          << dist1 << " normal dist: " << dist2 << " difference > " << epsilon
          << std::endl;
    } else {
      std::cout << "Two scores are the same. " << std::endl;
    }
  }
}

void compareDistanceComputationsFloat() {
  srand(time(0));
  float a[A_SIZE], b[A_SIZE];
  for (int i = 0; i < A_SIZE; i++) {
    a[i] = (float) (rand() / 10E5);
    b[i] = (float) (rand() / 10E5);
  }

  printArray(a, 16);
  printArray(b, 16);

  std::cout << "Starting distance computation test for float." << std::endl;

  {
    diskann::AVXDistanceL2Float dist;
    float                       dist1 = dist.compare(a, b, 16);
    float                       dist2 = distanceL2_F(a, b, 16);
    if (abs(dist1 - dist2) > epsilon) {
      std::cout
          << "compareDistanceComputationsFloat(): L2 Test failed. AVX dist: "
          << dist1 << " normal dist: " << dist2 << " difference > " << epsilon
          << std::endl;
    } else {
      std::cout << "Two scores are the same." << std::endl;
    }
  }
  {
    diskann::DistanceCosineFloat dist;
    float                        dist1 = dist.compare(a, b, 16);
    float                        dist2 = distanceCosine_F(a, b, 16);
    if (abs(dist1 - dist2) > epsilon) {
      std::cout << "compareDistanceComputationsFloat(): Cosine Test failed. "
                   "AVX dist: "
                << dist1 << " normal dist: " << dist2 << " difference > "
                << epsilon << std::endl;
    } else {
      std::cout << "Two scores are the same." << std::endl;
    }
  }
}

void testStreamBufImpl() {
  std::vector<int> v(100);
  std::iota(v.begin(), v.end(), 1);

  diskann::cout << "Printing with diskann::cout" << std::endl;
#pragma omp parallel for schedule(dynamic, 64)
  for (int i = 0; i < v.size(); i++) {
    diskann::cout << std::to_string(i) + ",";
    if (i != 0 && i % 10 == 0) {
      diskann::cout << std::endl;
    }
  }

  //  diskann::cout << "Printing with diskann::cout" << std::endl;
  //#pragma omp parallel for schedule(dynamic, 64)
  //  for (int i = 0; i < v.size(); i++) {
  //    diskann::cout << std::to_string(i) + ",";
  //    if (i != 0 && i % 10 == 0) {
  //      diskann::cout << std::endl;
  //    }
  //  }
}

//--5. Testing MemBuf Impl
class ContentBuf : public std::basic_streambuf<char> {
 public:
  ContentBuf(char* ptr, size_t size) {
    setg(ptr, ptr, ptr + size);
  }
};
void testMemBufImpl(int argc, char** argv) {
  // Create a simple binary file.
  std::string   fileName = argv[2];
  std::ofstream output_file(fileName, std::ios::binary);
  uint32_t      n1 = 105, n2 = 18090;
  const int     F_ARRAY_LEN = 5;
  float         fs[] = {1.024f, 2.39021f, 4.532f, 7.980232f, 6.222f};
  const int     I_ARRAY_LEN = 7;
  uint8_t       arr[] = {12, 13, 14, 33, 222, 183, 99};
  float         f1 = 9283.1237f;

  output_file.write((const char*) &n1, sizeof(uint32_t));
  output_file.write((const char*) &n2, sizeof(uint32_t));
  output_file.write((const char*) &f1, sizeof(float));
  output_file.write((const char*) fs, F_ARRAY_LEN * sizeof(float));
  output_file.write((const char*) arr, I_ARRAY_LEN * sizeof(uint8_t));

  output_file.close();

  std::ifstream input_file(fileName, std::ios::binary | std::ios::ate);
  size_t        size = input_file.tellg();
  input_file.seekg(0);

  auto data = new char[size];
  input_file.read(data, size);

  diskann::cout << "Read " << size << " bytes from file" << std::endl;

  uint32_t int1, int2;
  float    f2;
  float*   fs1 = new float[F_ARRAY_LEN];
  uint8_t* arr1 = new uint8_t[I_ARRAY_LEN];
  memset(fs1, 0, sizeof(float) * F_ARRAY_LEN);
  memset(arr1, 0, sizeof(uint8_t) * I_ARRAY_LEN);

  // std::ifstream reader(argv[1], std::ios::binary);
  // reader.read((char *)&int1, sizeof(int));
  // reader.read((char*) &int2, sizeof(int));
  // reader.read((char*) &f2, sizeof(float));
  // reader.read((char*) fs1, sizeof(float) * F_ARRAY_LEN);

  ContentBuf               cb(data, size);
  std::basic_istream<char> reader(&cb);

  reader.read((char*) &int1, sizeof(uint32_t));
  reader.read((char*) &int2, sizeof(uint32_t));
  reader.read((char*) &f2, sizeof(float));
  reader.read((char*) fs1, F_ARRAY_LEN * sizeof(float));
  reader.read((char*) arr1, I_ARRAY_LEN * sizeof(uint8_t));

  diskann::cout << int1 << "," << int2 << "," << f2 << std::endl;
  for (int i = 0; i < F_ARRAY_LEN; i++) {
    diskann::cout << fs1[i] << ",";
  }
  diskann::cout << std::endl;
  for (int i = 0; i < I_ARRAY_LEN; i++) {
    diskann::cout << std::to_string(arr1[i]) << ",";
  }
  diskann::cout << std::endl;

  assert(int1 == n1);
  assert(int2 == n2);
  assert(abs(f1 - f2) < 0.0001);
}

void testSubstringImpl(int argc, char** argv) {
  assert(argc >= 4);

  std::string s1(argv[2]);
  std::string s2(argv[3]);
  assert(s1[0] == s2[0]);  // at least they share one char in common!

  size_t index = -1;
  size_t compareLen = s1.length() <= s2.length() ? s1.length() : s2.length();
  for (size_t i = 0; i < compareLen; i++) {
    if (s1[i] != s2[i]) {
      index = i - 1;
      break;
    }
  }
  diskann::cout << "Common substring at 0 is:" << s1.substr(0, index);
}

// DISKANN_DLLIMPORT std::basic_ostream<char> diskann::cout;
// DISKANN_DLLIMPORT std::basic_ostream<char> diskann::cerr;

int main(int argc, char** argv) {
  if (argc < 2) {
    diskann::cout
        << std::string("Usage: ") << argv[0] << " <mode> [arguments]"
        << std::endl
        << "Modes: 1 for file concat. Args <file1> <file2> <outfile>"
        << std::endl
        << "       2 for test unique_ptr assignment. No args." << std::endl
        << "      3 for comparing distance computations (AVX and "
           "normal). No args."
        << std::endl
        << "      4 for testing our streambuf() implemntation. No args"
        << std::endl
        << "      5 for testing membuf implementation. Args <outfile>"
        << std::endl
        << "      6 for testing substring implementation. Args <s1> <s2>"
        << std::endl
        << "      7 for testing OpenMP thread pool implementation. No args."
        << std::endl;
    return -1;
  }

  int mode = atoi(argv[1]);

  std::cout << "Testing" << std::endl;

  switch (mode) {
    case 1:
      concat_files(argc, argv);
      return 0;
      break;
    case 2:
      uniquePtrAssignment();
      break;
    case 3:
      compareDistanceComputationsInt();
      compareDistanceComputationsFloat();
      break;
    case 4:
      testStreamBufImpl();
      break;
    case 5:
      testMemBufImpl(argc, argv);
      break;
    case 6:
      testSubstringImpl(argc, argv);
      break;
    case 7:
      testOpenMPThreadPool();
      break;
    default:
      diskann::cout << "Don't know what to do with mode parameter: " << argv[1]
                    << std::endl;
  }
}
