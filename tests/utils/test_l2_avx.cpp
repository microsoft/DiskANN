#include <iostream>
#include <random>
#include <string>
#include <numeric>
#include "distance.h"
#include "logger.h"

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

float epsilon = 0.01;
const int   A_SIZE = 100;
void  compareDistanceComputationsInt() {
  //int8_t vec1[] = {127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
  //                 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
  //                 127, 127, 127, 127, 127, 127, 127, 127, 127, 127};
  //int8_t vec2[] = {-128, -128, -128, -128, -128, -128, -128, -128,
  //                 -128, -128, -128, -128, -128, -128, -128, -128,
  //                 -128, -128, -128, -128, -128, -128, -128, -128,
  //                 -128, -128, -128, -128, -128, -128, -128, -128};

  int8_t vec1[A_SIZE], vec2[A_SIZE];
  srand(time(0));
  bool neg = false;
  for (int i = 0; i < A_SIZE; i++) {
    auto a = rand() % 128;
    auto b = rand() % 128;
    vec1[i] = -a; // < 0 ? -a : a;
    vec2[i] = -b; // < 0 ? -b : b;

    neg = neg || vec1[i] < 0 || vec2[i] < 0;
  }
  if (!neg) {
    std::cout << "No negative numbers in test. " << std::endl;
  }


  //diskann::DistanceL2Int8 dist;
  diskann::AVXDistanceL2Int8 dist;
  float                   dist1 = dist.compare(vec1, vec2, 8);
  float                   dist2 = distanceL2_I(vec1, vec2, 8);

  if (abs(dist1 - dist2) > epsilon ) {
    std::cout << "compareDistanceComputationsInt(): Test failed. AVX dist: "
              << dist1 << " normal dist: " << dist2 << " difference > "
              << epsilon << std::endl;
  } else {
    std::cout << "Two scores are the same. " << std::endl;
  }
}

void compareDistanceComputationsFloat() {
  srand(time(0));
  float a[A_SIZE], b[A_SIZE];
  for (int i = 0; i < A_SIZE; i++) {
    a[i] = rand() / 10E5;
    b[i] = rand() / 10E5;
  }

  diskann::AVXDistanceL2Float dist;
  float                       dist1 = dist.compare(a, b, 100);
  float                       dist2 = distanceL2_F(a, b, 100);
  if (abs(dist1 - dist2) > epsilon) {
    std::cout << "compareDistanceComputationsFloat(): Test failed. AVX dist: "
              << dist1 << " normal dist: " << dist2 << " difference > "
              << epsilon << std::endl;
  } else {
    std::cout << "Two scores are the same." << std::endl;
  }
}
  



void testStreamBufImpl() {
  std::vector<int> v(100);
  std::iota(v.begin(), v.end(), 1);

  std::cout << "Printing with std::cout" << std::endl;
#pragma omp parallel for schedule(dynamic, 64)
  for (int i = 0; i < v.size(); i++) {
    std::cout << std::to_string(i) + ",";
    if (i != 0 && i % 10 == 0) {
      std::cout << std::endl;
    }
  }

  diskann::cout << "Printing with diskann::cout" << std::endl;
#pragma omp parallel for schedule(dynamic, 64)
  for (int i = 0; i < v.size(); i++) {
    diskann::cout << std::to_string(i) + ",";
    if (i != 0 && i % 10 == 0) {
      diskann::cout << std::endl;
    }
  }
}

DISKANN_DLLIMPORT std::basic_ostream<char> diskann::cout;
DISKANN_DLLIMPORT std::basic_ostream<char> diskann::cerr;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << std::string("Usage: ") << argv[0] << " <mode> [arguments]"
              << std::endl;
    std::cout << "Modes: 1 for file concat. Args <file1> <file2> <outfile>"
              << std::endl
              << "       2 for test unique_ptr assignment. No args."
              << std::endl;
    std::cout << "      3 for comparing distance computations (AVX and "
                 "normal). No args.";
    std::cout << "      4 for testing our streambuf() impleemntation."
                 "No args"
              << std::endl;
    return -1;
  }

  int mode = atoi(argv[1]);

  diskann::cout << "Testing" << std::endl;

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
    default:
      std::cout << "Don't know what to do with mode parameter: " << argv[1]
                << std::endl;
  }
}
