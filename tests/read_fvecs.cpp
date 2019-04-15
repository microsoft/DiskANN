#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  std::ifstream infile(argv[1], std::ifstream::binary);

  int   dim;
  float buf[1024];

  while (!infile.eof()) {
    infile.read((char*) &dim, sizeof(int));
    infile.read((char*) buf, sizeof(float) * dim);

    std::cout << dim << "\t";
    for (int i = 0; i < dim; ++i)
      std::cout << buf[i] << " ";
    std::cout << std::endl;
  }
}
