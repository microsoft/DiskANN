#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

void add_rand_unit_gaussian(std::ofstream& out, const int n, const int dim) {
  float*                     buf = new float[dim];
  std::random_device         rd{};
  std::mt19937               gen{rd()};
  std::normal_distribution<> dis{0, 1};

  for (int i = 0; i < n; ++i) {
    out.write((char*) &dim, sizeof(int));

    float norm = 0.0;
    for (int d = 0; d < dim; ++d) {
      buf[d] = dis(gen);
      norm += buf[d] * buf[d];
    }
    for (int d = 0; d < dim; ++d)
      buf[d] /= std::sqrt(norm);
    out.write((char*) buf, dim * sizeof(float));
  }
  delete[] buf;
}

void add_rand_unit_cube(std::ofstream& out, const int n, const int dim) {
  float*                           buf = new float[dim];
  std::random_device               rd{};
  std::mt19937                     gen{rd()};
  std::uniform_real_distribution<> dis(-5, 5);

  for (int i = 0; i < n; ++i) {
    out.write((char*) &dim, sizeof(int));

    //    float norm = 0.0;
    for (int d = 0; d < dim; ++d) {
      buf[d] = dis(gen);
      //      norm += buf[d] * buf[d];
    }
    //    for (int d = 0; d < dim; ++d)
    //    buf[d] /= std::sqrt(norm);
    out.write((char*) buf, dim * sizeof(float));
  }
  delete[] buf;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cout << "Usage: gen_grid <dim> <n> <output.fvecs> rand_type "
                 "(1=gaussian/2=cube)"
              << std::endl;
    return -1;
  }

  int dim = atoi(argv[1]);
  int n = atoi(argv[2]);
  int type = atoi(argv[4]);

  std::ofstream out(argv[3], std::ofstream::binary);
  if (type == 1)
    add_rand_unit_gaussian(out, n, dim);
  else
    add_rand_unit_cube(out, n, dim);
  out.close();
}
