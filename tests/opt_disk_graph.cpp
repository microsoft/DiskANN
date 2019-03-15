#include <efanna2e/util.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#define SECTOR_LEN 4096

void load_nsg(const char *filename, std::vector<std::vector<unsigned>> &graph,
              _u64 &medoid) {
  std::ifstream in(filename, std::ios::binary);
  unsigned      width_u32, medoid_u32;
  in.read((char *) &width_u32, sizeof(unsigned));  // ignored
  in.read((char *) &medoid_u32, sizeof(unsigned));
  medoid = (_u64) medoid_u32;
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
  in.close();
}

template<typename T>
float compute_scale_factor(const T *data, const _u64 npts, const _u64 ndims) {
  float scale_factor = 127.0f;
  float max_abs = std::numeric_limits<float>::min();
  for (_u64 i = 0; i < npts * ndims; i++) {
    max_abs = std::max(max_abs, (float) std::abs(data[i]));
  }

  std::cout << "max absolute value = " << max_abs << std::endl;
  scale_factor /= max_abs;
  std::cout << "scale factor = " << scale_factor << std::endl;
  return scale_factor;
}

template<typename OutType>
void write_header(std::ofstream &                           writer,
                  const std::vector<std::vector<unsigned>> &nsg,
                  const _u64 npts, const _u64 ndims, const _u64 medoid,
                  const float scale_factor) {
  std::vector<_u64> sizes(nsg.size());
  for (_u64 i = 0; i < npts; i++) {
    _u64 node_size =
        sizeof(unsigned) +
        (nsg[i].size() * (sizeof(unsigned) + ndims * sizeof(OutType)));
    sizes[i] = ROUND_UP(node_size, SECTOR_LEN);
  }

  _u64 write_size = 0;

  // # pts
  writer.write((char *) &npts, sizeof(_u64));
  write_size += sizeof(_u64);
  // # dims
  writer.write((char *) &ndims, sizeof(_u64));
  write_size += sizeof(_u64);
  // medoid id
  writer.write((char *) &medoid, sizeof(_u64));
  write_size += sizeof(_u64);
  // scale factor
  writer.write((char *) &scale_factor, sizeof(float));
  write_size += sizeof(float);
  // node sizes
  writer.write((char *) sizes.data(), npts * sizeof(_u64));
  write_size += (npts * sizeof(_u64));
  std::cout << "----Header----\n Actual size: " << writer.tellp();
  write_size = ROUND_UP(write_size, SECTOR_LEN);
  writer.seekp(write_size, std::ios::beg);
  std::cout << "B, Rounded size: " << writer.tellp() << std::endl;
}

template<typename InType, typename OutType>
void optimize_and_write(char *                                    filename,
                        const std::vector<std::vector<unsigned>> &graph,
                        const InType *data, const _u64 medoid, const _u64 npts,
                        const _u64 ndims, const bool scale) {
  std::cout << "sizeof(InType): " << sizeof(InType)
            << ", sizeof(OutType): " << sizeof(OutType) << std::endl;
  std::cout << "Output file: " << filename << std::endl;
  // contains sizes into output file for each node
  std::vector<size_t> sizes;
  std::ofstream       writer(filename, std::ios::binary | std::ios::out);

  // iterate over each node and find max degree
  _u64 max_degree = 0;
  for (auto nhood : graph) {
    max_degree = std::max((size_t) max_degree, nhood.size());
  }

  // compute max output size per node
  _u64 max_write_per_node =
      max_degree * (sizeof(unsigned) + ndims * sizeof(OutType)) +
      sizeof(unsigned);
  max_write_per_node = ROUND_UP(max_write_per_node, SECTOR_LEN);

  // block disk accesses
  _u64 blk_size = 32768;
  _u64 max_write_per_blk = blk_size * max_write_per_node;
  _u64 n_blks = ROUND_UP(npts, blk_size) / blk_size;

  std::cout << "# blks = " << n_blks << std::endl;
  std::cout << "max_degree = " << max_degree << std::endl;
  std::cout << "max_write_per_node = " << max_write_per_node << std::endl;
  std::cout << "max_write_per_blk = " << max_write_per_blk << std::endl;
  std::cout << "npts = " << npts << std::endl;
  std::cout << "ndims = " << ndims << std::endl;

  // per node layout:
  // [N][ID1][ID2][ID3]...[IDN][ID1 COORDS][ID2 COORDS]...[IDN COORDS]
  // NOTE: ID COORDS stored as int8_t
  char *write_buf = new char[max_write_per_blk];
  float scale_factor = 1.0f;
  if (scale) {
    scale_factor = compute_scale_factor<InType>(data, npts, ndims);
  }

  // write out header
  write_header<OutType>(writer, graph, npts, ndims, medoid, scale_factor);
  _u64 total_offset = 23625728;

  OutType *out_vec = new OutType[ndims];
  for (_u64 i = 0; i < n_blks; i++) {
    memset(write_buf, 0, max_write_per_blk / sizeof(unsigned));

    _u64 blk_start = blk_size * i;
    _u64 blk_end = std::min(npts, blk_size * (i + 1));
    _u64 blk_write_size = 0;

    // fill write_buf for block
    for (_u64 j = blk_start; j < blk_end; j++) {
      _u64     nnbrs = graph[j].size();
      unsigned nnbrs_u32 = (unsigned) nnbrs;
      _u64     blk_node_offset = blk_write_size;
      _u64     node_write_size = 0;
      // per node layout:
      // [N][ID1][ID2][ID3]...[IDN][ID1 COORDS][ID2 COORDS]...[IDN COORDS]
      memcpy(write_buf + blk_write_size + node_write_size, (char *) &nnbrs_u32,
             sizeof(unsigned));
      node_write_size += sizeof(unsigned);
      memcpy(write_buf + blk_write_size + node_write_size,
             (char *) graph[j].data(), nnbrs * sizeof(unsigned));
      node_write_size += nnbrs * sizeof(unsigned);

      // write coords of each nbr
      for (_u64 k = 0; k < nnbrs; k++) {
        _u64          nbr_id = graph[j][k];
        const InType *in_vec = data + (nbr_id * ndims);
        // scale/convert in_vec from InType to OutType, store in out_vec
        for (_u64 d = 0; d < ndims; d++) {
          out_vec[d] = (OutType)(in_vec[d] * scale_factor);
        }
        // copy out_vec to write_buf
        memcpy(write_buf + blk_write_size + node_write_size, (char *) out_vec,
               ndims * sizeof(OutType));
        node_write_size += (ndims * sizeof(OutType));
      }

      // round up node write size to sector boundary
      /*
      std::cout << j << "->" << total_offset << ":" << node_write_size
                << ", nnbrs: " << nnbrs << "\n";
                */
      node_write_size = ROUND_UP(node_write_size, SECTOR_LEN);
      blk_write_size += node_write_size;
      total_offset += node_write_size;
    }
    writer.write(write_buf, blk_write_size);
    std::cout << "block #" << i << " written\n";
  }

  // cleanup
  delete[] out_vec;
  delete[] write_buf;
  writer.close();
}

void write_sizes(char *filename, const std::vector<size_t> &sizes) {
  std::ofstream writer(filename, std::ios::binary | std::ios::out);
  writer.write((char *) sizes.data(), (size_t)(sizes.size() * sizeof(size_t)));
  writer.close();
}

void templated_load(const char *filename, void *&ptr, _u64 &npts, _u64 &ndims,
                    const std::string &format, const std::string &type) {
  unsigned npts_u32;
  unsigned ndims_u32;
  std::cout << "Reading format: " << format << std::endl;
  std::cout << "Reading type: " << type << std::endl;
  if (format == "bin") {
    if (type == "int8") {
      int8_t *local_ptr = nullptr;
      efanna2e::load_bin<int8_t>(filename, local_ptr, npts_u32, ndims_u32);
      ptr = (void *) local_ptr;
      npts = (_u64) npts_u32;
      ndims = (_u64) ndims_u32;
    } else if (type == "uint8") {
      uint8_t *local_ptr = nullptr;
      unsigned npts_u32;
      unsigned ndims_u32;
      efanna2e::load_bin<uint8_t>(filename, local_ptr, npts_u32, ndims_u32);
      ptr = (void *) local_ptr;
      npts = (_u64) npts_u32;
      ndims = (_u64) ndims_u32;
    } else if (type == "int16") {
      int16_t *local_ptr = nullptr;
      unsigned npts_u32;
      unsigned ndims_u32;
      efanna2e::load_bin<int16_t>(filename, local_ptr, npts_u32, ndims_u32);
      ptr = (void *) local_ptr;
      npts = (_u64) npts_u32;
      ndims = (_u64) ndims_u32;
    } else if (type == "float") {
      float *  local_ptr = nullptr;
      unsigned npts_u32;
      unsigned ndims_u32;
      efanna2e::load_bin<float>(filename, local_ptr, npts_u32, ndims_u32);
      ptr = (void *) local_ptr;
      npts = (_u64) npts_u32;
      ndims = (_u64) ndims_u32;
    } else {
      std::cerr << "Unsupported type: " << type << std::endl;
      exit(-1);
    }
  } else if (format == "fvecs") {
    if (type == "int8") {
      int8_t *local_ptr = nullptr;
      efanna2e::load_Tvecs<int8_t>(filename, local_ptr, npts_u32, ndims_u32);
      ptr = (void *) local_ptr;
      npts = (_u64) npts_u32;
      ndims = (_u64) ndims_u32;
    } else if (type == "uint8") {
      uint8_t *local_ptr = nullptr;
      unsigned npts_u32;
      unsigned ndims_u32;
      efanna2e::load_Tvecs<uint8_t>(filename, local_ptr, npts_u32, ndims_u32);
      ptr = (void *) local_ptr;
      npts = (_u64) npts_u32;
      ndims = (_u64) ndims_u32;
    } else if (type == "int16") {
      int16_t *local_ptr = nullptr;
      unsigned npts_u32;
      unsigned ndims_u32;
      efanna2e::load_Tvecs<int16_t>(filename, local_ptr, npts_u32, ndims_u32);
      ptr = (void *) local_ptr;
      npts = (_u64) npts_u32;
      ndims = (_u64) ndims_u32;
    } else if (type == "float") {
      float *  local_ptr = nullptr;
      unsigned npts_u32;
      unsigned ndims_u32;
      efanna2e::load_Tvecs<float>(filename, local_ptr, npts_u32, ndims_u32);
      ptr = (void *) local_ptr;
      npts = (_u64) npts_u32;
      ndims = (_u64) ndims_u32;
    } else {
      std::cerr << "Unsupported type: " << type << std::endl;
      exit(-1);
    }
  } else {
    std::cerr << "Unsupported format: " << format << std::endl;
    exit(-1);
  }
}

void templated_optimize_and_write(char *filename,
                                  const std::vector<std::vector<unsigned>> &nsg,
                                  void *data, _u64 npts, _u64 ndims,
                                  _u64 medoid, bool scale, std::string in_type,
                                  std::string out_type) {
  if (in_type == "int8" && out_type == "int8") {
    optimize_and_write<int8_t, int8_t>(filename, nsg, (int8_t *) data, medoid,
                                       npts, ndims, scale);
  } else if (in_type == "float" && out_type == "int8") {
    optimize_and_write<float, int8_t>(filename, nsg, (float *) data, medoid,
                                      npts, ndims, scale);
  } else if (in_type == "uint8" && out_type == "uint8") {
    optimize_and_write<uint8_t, uint8_t>(filename, nsg, (uint8_t *) data,
                                         medoid, npts, ndims, scale);
  } else {
    std::cerr << "Unsupported conversion: " << in_type << " -> " << out_type
              << std::endl;
    exit(-1);
  }
}

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cout << argv[0]
              << " data_file nsg_file output_file scale=1/0 fvecs/bin in_type "
                 "out_type\n in/out types: int8, uint8, int16, float"
              << std::endl;
    exit(-1);
  }

  void *      data = NULL;
  _u64        npts, ndims;
  bool        scale = ((unsigned) std::atoi(argv[4]) == 1);
  std::string format(argv[5]);
  std::string in_type(argv[6]);
  std::string out_type(argv[7]);

  templated_load(argv[1], data, npts, ndims, format, in_type);
  std::cout << "Data loaded\n";

  std::vector<std::vector<unsigned>> nsg;
  _u64                               medoid;
  load_nsg(argv[2], nsg, medoid);
  std::cout << "NSG loaded\n";

  templated_optimize_and_write(argv[3], nsg, data, npts, ndims, medoid, scale,
                               in_type, out_type);
  std::cout << "Output file written\n";
  if (format != "fvecs") {
    if (in_type == "int8") {
      delete[](int8_t *) data;
    } else if (in_type == "uint8") {
      delete[](uint8_t *) data;
    } else if (in_type == "int16") {
      delete[](int16_t *) data;
    } else if (in_type == "float") {
      delete[](float *) data;
    }
  } else {
    free(data);
  }
  return 0;
}