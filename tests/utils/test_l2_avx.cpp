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

//--5. Testing MemBuf Impl
class ContentBuf : public std::basic_streambuf<char> {
 public:
  ContentBuf(char* ptr, size_t size){
    setg(ptr, ptr, ptr + size);
  }
};
void testMemBufImpl(int argc, char** argv) {
  // Create a simple binary file.
  std::string        fileName = argv[2];
  std::ofstream output_file(fileName, std::ios::binary);
  uint32_t      n1 = 105, n2 = 18090;
  const int     F_ARRAY_LEN = 5;
  float         fs[] = {1.024f, 2.39021f, 4.532f, 7.980232f, 6.222f};
  const int     I_ARRAY_LEN = 7;
  _u8           arr[] = {12, 13, 14, 33, 222, 183, 99};
  float         f1 = 9283.1237f;

  output_file.write((const char*) &n1, sizeof(uint32_t));
  output_file.write((const char*) &n2, sizeof(uint32_t));
  output_file.write((const char*) &f1, sizeof(float));
  output_file.write((const char*) fs, F_ARRAY_LEN * sizeof(float));
  output_file.write((const char*) arr, I_ARRAY_LEN * sizeof(_u8));

  output_file.close();

  std::ifstream input_file(fileName, std::ios::binary | std::ios::ate);
  size_t        size = input_file.tellg();
  input_file.seekg(0);

  auto              data = new char[size];
  input_file.read(data, size);

  std::cout << "Read " << size << " bytes from file" << std::endl;
    
  uint32_t int1, int2;
  float    f2;
  float*   fs1 = new float[F_ARRAY_LEN];
  _u8*     arr1 = new _u8[I_ARRAY_LEN];
  memset(fs1, 0, sizeof(float) * F_ARRAY_LEN);
  memset(arr1, 0, sizeof(_u8) * I_ARRAY_LEN);

  //std::ifstream reader(argv[1], std::ios::binary);
  //reader.read((char *)&int1, sizeof(int));
  //reader.read((char*) &int2, sizeof(int));
  //reader.read((char*) &f2, sizeof(float));
  //reader.read((char*) fs1, sizeof(float) * F_ARRAY_LEN);

  ContentBuf cb(data, size);
  std::basic_istream<char> reader(&cb);

  reader.read((char*) &int1, sizeof(uint32_t));
  reader.read((char*) &int2, sizeof(uint32_t));
  reader.read((char*) &f2, sizeof(float));
  reader.read((char*) fs1, F_ARRAY_LEN * sizeof(float));
  reader.read((char*) arr1, I_ARRAY_LEN * sizeof(_u8));

  std::cout << int1 << "," << int2 << ","  << f2 << std::endl;
  for (int i = 0; i < F_ARRAY_LEN; i++) {
    std::cout << fs1[i] << ",";
  }
  std::cout << std::endl;
  for (int i = 0; i < I_ARRAY_LEN; i++) {
    std::cout << std::to_string(arr1[i]) << ",";
  }
  std::cout << std::endl;
  
  assert(int1 == n1);
  assert(int2 == n2);
  assert(abs(f1 - f2) < 0.0001);
}

void testSubstringImpl(int argc, char** argv) {
  assert(argc >= 4);

  std::string s1(argv[2]);
  std::string s2(argv[3]);
  assert(s1[0] == s2[0]); //at least they share one char in common!

  size_t index = -1;
  size_t compareLen = s1.length() <= s2.length() ? s1.length() : s2.length();
  for (size_t i = 0; i < compareLen; i++) {
    if (s1[i] != s2[i]) {
      index = i - 1;
      break;
    }
  }
  std::cout << "Common substring at 0 is:" << s1.substr(0, index);
}

DISKANN_DLLIMPORT std::basic_ostream<char> diskann::cout;
DISKANN_DLLIMPORT std::basic_ostream<char> diskann::cerr;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << std::string("Usage: ") << argv[0] << " <mode> [arguments]"
              << std::endl
              << "Modes: 1 for file concat. Args <file1> <file2> <outfile>"
              << std::endl
              << "       2 for test unique_ptr assignment. No args."
              << std::endl
              << "      3 for comparing distance computations (AVX and "
                 "normal). No args."
              << std::endl
              << "      4 for testing our streambuf() implemntation. No args"
              << std::endl
              << "      5 for testing membuf implementation. Args <outfile>"
              << std::endl
              << "      6 for testing substring implementation. Args <s1> <s2>"
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
      compareDistanceComputations();
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
    default:
      std::cout << "Don't know what to do with mode parameter: " << argv[1]
                << std::endl;
  }
}
