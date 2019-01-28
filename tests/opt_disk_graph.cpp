#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

// round up X to the nearest multiple of Y
#define ROUND_UP(X, Y) ((((size_t)(X) / (Y)) + ((size_t)(X) % (Y) != 0)) * (Y))

#define SECTOR_LEN 512

void load_nsg(const char *filename, std::vector<std::vector<unsigned>> &graph,
              unsigned &width, unsigned &ep_) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *) &width, sizeof(unsigned));
  in.read((char *) &ep_, sizeof(unsigned));
  // width=100;
  while (!in.eof()) {
    unsigned k;
    in.read((char *) &k, sizeof(unsigned));
    if (in.eof())
      break;
    std::vector<unsigned> tmp(k);
    in.read((char *) tmp.data(), k * sizeof(unsigned));
    graph.push_back(tmp);
  }
}

void load_data(char *filename, float *&data, unsigned &num,
               unsigned &dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char *) &dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t             fsize = (size_t) ss;
  num = (unsigned) (fsize / (dim + 1) / 4);
  data = new float[(size_t) num * (size_t) dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char *) (data + i * dim), dim * 4);
  }
  in.close();
}

std::vector<size_t> optimize_and_write(
    char *filename, const std::vector<std::vector<unsigned>> &graph,
    const float *data, const unsigned width, const unsigned ep_,
    const unsigned npts, const unsigned ndims) {
  // contains offsets into output file for each node
  std::vector<size_t> offsets;
  std::ofstream       writer(filename, std::ios::binary | std::ios::out);
  unsigned            max_degree = 0;
  float               scale_factor = 127.0f;
  float               max_abs = std::numeric_limits<float>::min();
  for (size_t i = 0; i < (size_t) npts * (size_t) ndims; i++) {
    max_abs = std::max(max_abs, std::abs(data[i]));
  }
  std::cout << "max absolute value = " << max_abs << std::endl;
  scale_factor /= max_abs;
  std::cout << "scale factor = " << scale_factor << std::endl;
  // iterate over each node and find max degree
  for (auto nhood : graph) {
    max_degree = std::max((size_t) max_degree, nhood.size());
  }
  // compute max output size per node
  unsigned max_write_per_blk =
      max_degree * (sizeof(unsigned) + ndims * sizeof(int8_t)) +
      sizeof(unsigned);
  max_write_per_blk = ROUND_UP(max_write_per_blk, SECTOR_LEN);
  std::cout << "max_degree = " << max_degree << std::endl;
  std::cout << "max_write_per_blk = " << max_write_per_blk << std::endl;
  std::cout << "npts = " << npts << std::endl;
  std::cout << "ndims = " << ndims << std::endl;
  int8_t *  lower_prec = new int8_t[(size_t) npts * (size_t) ndims];
#pragma omp parallel for schedule(dynamic, 1048576)
  for (size_t i = 0; i < (size_t) npts * (size_t) ndims; i++) {
    lower_prec[i] = int8_t(data[i] * scale_factor);
    // handle underflow
    lower_prec[i] =
        data[i] < 0 && lower_prec[i] > 0 ? (int8_t) -127 : lower_prec[i];
    // handle overflow
    lower_prec[i] =
        data[i] > 0 && lower_prec[i] < 0 ? (int8_t) 127 : lower_prec[i];
  }
  std::cout << "converted data to int8_t" << std::endl;

  // per node layout:
  // [N][ID1][ID2][ID3]...[IDN][ID1 COORDS][ID2 COORDS]...[IDN COORDS]
  // NOTE: ID COORDS stored as int8_t
  char *write_buf = new char[max_write_per_blk];

  // write width, medoid
  memcpy(write_buf, (char *) &width, sizeof(unsigned));
  memcpy(write_buf + sizeof(unsigned), (char *) &ep_, sizeof(unsigned));
  unsigned write_size = ROUND_UP(2 * sizeof(unsigned), SECTOR_LEN);
  writer.write(write_buf, write_size);

  size_t cur_offset = write_size;
  // iterate over each node
  for (unsigned idx = 0; idx < graph.size(); idx++) {
    const auto &nhood = graph[idx];
    unsigned    nnbrs = nhood.size();
    write_size = 0;
    // record starting point
    offsets.push_back(cur_offset);
    memcpy(write_buf, (char *) &nnbrs, sizeof(unsigned));
    memcpy(write_buf + sizeof(unsigned), nhood.data(),
           nnbrs * sizeof(unsigned));
    write_size += ((nnbrs + 1) * sizeof(unsigned));
    for (unsigned nbr : nhood) {
      memcpy(write_buf + write_size, lower_prec + (nbr * ndims),
             ndims * sizeof(int8_t));
      write_size += ndims * sizeof(int8_t);
    }
    // align to sector
    write_size = ROUND_UP(write_size, SECTOR_LEN);
    writer.write(write_buf, write_size);
    cur_offset += write_size;
  }
  writer.close();
  return offsets;
}

void write_offsets(char *filename, const std::vector<size_t> &offsets) {
  std::ofstream writer(filename, std::ios::binary | std::ios::out);
  writer.write((char *) offsets.data(),
               (size_t)(offsets.size() * sizeof(size_t)));
  writer.close();
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << argv[0]
              << " data_file nsg_graph output_file output_file_offsets"
              << std::endl;
    exit(-1);
  }

  float *  data = NULL;
  unsigned npts, ndims;
  load_data(argv[1], data, npts, ndims);
  std::cout << "Data loaded\n";
  std::vector<std::vector<unsigned>> nsg_graph;
  unsigned                           width, ep_;
  load_nsg(argv[2], nsg_graph, width, ep_);
  std::cout << "NSG loaded\n";
  auto file_offsets =
      optimize_and_write(argv[3], nsg_graph, data, width, ep_, npts, ndims);
  std::cout << "Output file written\n";
  write_offsets(argv[4], file_offsets);
  return 0;
}
