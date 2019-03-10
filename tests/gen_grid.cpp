#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

void add_point(std::ofstream& out, const int dim, std::vector<float>& fixed_vals, const std::vector<float>& vals, bool rand=1)
{
  if (fixed_vals.size() == (size_t)dim-1) {
    for(auto iter: vals) {
      out.write((char*)&dim, sizeof(int));
      out.write((char*)(fixed_vals.data()), (dim-1)*sizeof(float));
      auto val = iter;
      out.write((char*)&val, sizeof(float));
    }
  }
  else {
    for(auto iter: vals) {
      fixed_vals.push_back(iter + 0.01 * std::rand()/RAND_MAX);
      add_point(out, dim, fixed_vals, vals);
      fixed_vals.pop_back();
    }
  }
}


int main (int argc, char** argv)
{
  if (argc != 6) {
    std::cout << "Usage: gen_grid <dim> <min> <max> <points/dim> <output.fvecs>" << std::endl;
    return -1;
  }

  int dim = atoi(argv[1]);
  float min = atof(argv[2]);
  float max = atof(argv[3]);
  int pts_per_dim = atof(argv[4]);

  std::ofstream out(argv[5], std::ofstream::binary);

  std::vector<float> vals, fixed_vals;
  for (int i=0; i<pts_per_dim; ++i)
    vals.push_back( min + i*(max-min)/(pts_per_dim-1));

  add_point(out, dim, fixed_vals, vals);

  if (fixed_vals.size() != 0) {
    std::cerr << "Something went wrong with recursion"<<std::endl;
  }
  
  out.close();
}
