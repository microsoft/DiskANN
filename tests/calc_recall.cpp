
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

void load_data(const char* filename, unsigned*& data, size_t& num,
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
  data = new unsigned[data_size];

  unsigned* tmp_dim = new unsigned;
  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.read((char*) tmp_dim, 4);
    in.read((char*) (data + i * dim), dim * 4);
  }
  in.close();
  std::cout << "data loaded \n";
}
float calc_recall_set(unsigned num_queries, unsigned* gold_std, unsigned dim_gs,
                      unsigned* our_results, unsigned dim_or,
                      unsigned recall_at, unsigned subset_size) {
  std::cout << "dim_gs: " << dim_gs << ", dim_or: " << dim_or
            << ", recall_at: " << recall_at << " num_queries = " << num_queries
            << "\n";
  unsigned           total_recall = 0;
  std::set<unsigned> gt, res;

  for (size_t i = 0; i < num_queries; i++) {
    gt.clear();
    res.clear();
    unsigned* gt_vec = gold_std + dim_gs * i;
    unsigned* res_vec = our_results + dim_or * i;
    gt.insert(gt_vec, gt_vec + recall_at);
    res.insert(res_vec, res_vec + subset_size);
    unsigned cur_recall = 0;
    for (auto& v : gt) {
      if (res.find(v) != res.end()) {
        cur_recall++;
      }
    }
    // std::cout << " idx: " << i << ", interesection: " << cur_recall << "\n";
    total_recall += cur_recall;
  }
  return ((float) total_recall) / ((float) num_queries) *
         (100.0 / ((float) recall_at));
}

void print(int* data, int dim, int num_points, int num_lines) {
  if (num_lines <= num_points) {
    for (int i = 0; i < num_lines; i++) {
      for (int j = 0; j < dim; j++) {
        std::cout << *((data + i * dim) + j) << ",";
      }
      std::cout << std::endl;
    }
  } else {
    std::cerr << "Num lines to print: " << num_lines
              << " should be <= num_points(" << num_points << ")" << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cout << argv[0] << " data_file1 data_file2 r r2(optonal, equal to 1)"
              << std::endl;
    exit(-1);
  }
  unsigned* gold_std = NULL;
  unsigned* our_results = NULL;
  size_t    points_num;
  unsigned  dim_gs;
  unsigned  dim_or;
  load_data(argv[1], gold_std, points_num, dim_gs);
  load_data(argv[2], our_results, points_num, dim_or);

  size_t   recall = 0;
  size_t   total_recall = 0;
  uint32_t recall_at = std::atoi(argv[3]);
  uint32_t subset_size = dim_or;
  if (argc == 5)
    subset_size = std::atoi(argv[4]);

  unsigned mind = dim_gs;
  if ((dim_or < recall_at) || (recall_at > dim_gs)) {
    std::cout << "ground truth has size " << dim_gs << "; our set has "
              << dim_or << " points. Asking for recall " << recall_at
              << std::endl;
    return -1;
  }
  std::cout << "calculating recall " << recall_at << "@" << subset_size
            << std::endl;
  float recall_val = calc_recall_set(points_num, gold_std, dim_gs, our_results,
                                     dim_or, recall_at, subset_size);

  //  double avg_recall = (recall*1.0)/(points_num*1.0);
  std::cout << "avg. recall " << recall_at << " at " << subset_size << " is "
            << recall_val << "\n";
}
