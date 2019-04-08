
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void load_data(const char* filename, int*& data, size_t& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(std::string(filename), std::ios::binary);
  std::cout << "Filename: " << filename << "\n";
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*) &dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t             fsize = (size_t) ss;
  num = (unsigned) (fsize / (dim + 1) / 4);
  auto data_size = (size_t) num * (size_t) dim;
  std::cout << "data dimension: " << dim << std::endl;
  std::cout << "data num points: " << num << std::endl;
  data = new int[data_size];

  int* tmp_dim = new int;
  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.read((char*) tmp_dim, 4);
    in.read((char*) (data + i * dim), dim * 4);
  }
  in.close();
  std::cout << "data loaded \n";
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << argv[0] << " data_file1 data_file2 r" << std::endl;
    exit(-1);
  }
  int*     gold_std = NULL;
  int*     our_results = NULL;
  size_t   points_num;
  unsigned dim_gs;
  unsigned dim_or;
  load_data(argv[1], gold_std, points_num, dim_gs);
  load_data(argv[2], our_results, points_num, dim_or);
  size_t   recall = 0;
  size_t   total_recall = 0;
  uint32_t recall_at = std::atoi(argv[3]);

  unsigned mind = dim_gs;
  if ((dim_or < recall_at) || (recall_at > dim_gs)) {
    std::cout << "ground truth has size " << dim_gs << "; our set has "
              << dim_or << " points. Asking for recall " << recall_at
              << std::endl;
    return -1;
  }
  std::cout << "calculating recall " << recall_at << "@" << dim_or << std::endl;

  auto all_recall = new bool[points_num];
  for (unsigned i = 0; i < points_num; i++)
    all_recall[i] = false;

  auto this_point = new bool[dim_gs];
  for (size_t i = 0; i < points_num; i++) {
    for (unsigned j = 0; j < dim_gs; j++)
      this_point[j] = false;

    bool this_correct = true;
    for (size_t j1 = 0; j1 < recall_at; j1++)
      for (size_t j2 = 0; j2 < dim_or; j2++)
        if (gold_std[i * (size_t) dim_gs + j1] ==
            our_results[i * (size_t) dim_or + j2]) {
          if (this_point[j1] == false)
            total_recall++;
          this_point[j1] = true;
        }
    for (unsigned j1 = 0; j1 < dim_gs; j1++)
      if (this_point[j1] == false) {
        this_correct = false;
        break;
      }
    if (this_correct == true)
      recall++;
  }
  delete[] this_point;

  //  double avg_recall = (recall*1.0)/(points_num*1.0);
  std::cout << "avg. recall " << dim_gs << " at " << dim_or << " is "
            << 1.0 * (100.0 / recall_at) * ((total_recall * 1.0) / points_num)
            << std::endl;
}
