#include <algorithm>
#include <cassert>
#include <limits>
#include <parallel/algorithm>
#include <vector>
#include "util.h"

double quantize(int16_t input, double* map, double* bounds) {
  double*  pos = std::lower_bound(bounds, bounds + 256, (double) input);
  unsigned idx = (unsigned) (pos - bounds - 1);
  return map[idx];
}

uint8_t map_int16_to_uint8(int16_t input, double* map, double* bounds) {
  double*  pos = std::lower_bound(bounds, bounds + 256, (double) input);
  unsigned idx = (unsigned) (pos - bounds - 1);
  assert(input >= bounds[idx]);
  assert(input <= bounds[idx + 1]);
  return idx;
}

long double compute_dim_loss(int16_t* input, uint64_t npts, double* map,
                             double* bounds) {
  long double loss = 0.0;
  for (uint64_t i = 0; i < npts; i++) {
    double*  pos = std::lower_bound(bounds, bounds + 256, (double) input[i]);
    uint64_t idx = (uint64_t)(pos - bounds - 1);
    /*
    if(input[i] < bounds[idx]){
      std::cerr << "idx = " << idx << ", input[i] = " << input[i] << ",
    bounds[idx] = " << bounds[idx] << ", *pos = " << *pos << std::endl;
      for(uint64_t j=0;j<257;j++)
        std::cout << bounds[j] << " ";

      assert(false);
    }*/
    assert(input[i] >= bounds[idx]);
    assert(input[i] <= bounds[idx + 1]);
    double diff = (map[idx] - input[i]);
    loss += std::pow(diff, 2);
  }
  return loss / npts;
}

double compute_vector_loss(int16_t* input, uint8_t* mapped, double* maps,
                           uint64_t ndims) {
  double loss = 0;
  for (unsigned i = 0; i < ndims; i++) {
    double map_val = (maps + 256 * i)[mapped[i]];
    double dloss = ((double) input[i] - map_val);
    // std::cout << "input: " << input[i] << ", map: " << map_val << ", loss: "
    // << dloss << std::endl;
    loss += std::pow(dloss, 2);
  }
  return loss;
}

double compute_data_loss(int16_t* input, uint8_t* mapped, double* maps,
                         uint64_t npts, uint64_t ndims) {
  long double loss = 0;
#pragma omp   parallel for reduction(+ : loss) schedule(static, 32768)
  for (uint64_t i = 0; i < npts; i++) {
    double vec_loss =
        compute_vector_loss(input + i * ndims, mapped + i * ndims, maps, ndims);
    loss += vec_loss;
  }
  loss /= npts;
  return (double) loss;
}

void map_vector(int16_t* input, uint8_t* mapped, double* maps, double* bounds,
                uint64_t ndims) {
  for (unsigned i = 0; i < ndims; i++) {
    mapped[i] =
        map_int16_to_uint8(*(input + i), maps + i * 256, bounds + i * 257);
  }
}

void map_dataset(int16_t* input, uint8_t* mapped, double* maps, double* bounds,
                 uint64_t npts, uint64_t ndims) {
#pragma omp parallel for schedule(static, 32768)
  for (uint64_t i = 0; i < npts; i++) {
    map_vector(input + (i * ndims), mapped + (i * ndims), maps, bounds, ndims);
  }
}

// map_out: double[256]
// bounds_out: double[257]
void quantize(int16_t* input, uint64_t dim, uint64_t npts, uint64_t ndims,
              double* map_out, double* bounds_out) {
  // extract dimension
  // std::cout << "Starting dimension extraction" << std::endl;
  int16_t* dim_data = new int16_t[npts];
  for (uint64_t i = 0; i < npts; i++) {
    dim_data[i] = input[i * ndims + dim];
  }
  // std::cout << "Starting parallel sort" << std::endl;
  // alocate bin start/ends
  std::sort(dim_data, dim_data + npts);

  // compute pts per bin
  uint64_t bin_size = npts / 256;
  // std::cout << "Choosing bin_size = " << bin_size << std::endl;
  double start_out[256], end_out[256];

  // compute bin_starts and bin_ends
  for (uint64_t i = 0; i < 256; i++) {
    start_out[i] = dim_data[bin_size * i];
    end_out[i] = dim_data[bin_size * (i + 1)];
  }
  // extend min and max to supported min/max
  start_out[0] = std::numeric_limits<int16_t>::min();
  end_out[255] = std::numeric_limits<int16_t>::max();
  // compute median between min and max
  for (uint64_t i = 0; i < 255; i++) {
    double mid = (start_out[i + 1] + end_out[i]) / 2.0;
    end_out[i] = mid;
    start_out[i + 1] = mid;
  }
  memcpy(bounds_out, start_out, 256 * sizeof(double));
  bounds_out[256] = end_out[255];
  // std::cout << "Finished computing bin boundaries" << std::endl;

  // compute bin_avgs
  uint64_t end_idx = (npts / bin_size) * bin_size;
  for (uint64_t i = 0; i < end_idx; i++) {
    map_out[i / bin_size] += dim_data[i];
  }
  for (uint64_t i = end_idx; i < npts; i++) {
    map_out[255] += dim_data[i];
  }
  for (uint64_t i = 0; i < 255; i++) {
    map_out[i] /= bin_size;
  }
  map_out[255] /= (bin_size + (npts % bin_size));
  // std::cout << "Finished computing bin averages" << std::endl;
  /*
  for(uint64_t i=0;i<256;i++){
    std::cout << "Bin: " << i << ", min = " << bounds_out[i] << ", max = " <<
  bounds_out[i+1] << ", assigned = " << map_out[i] << std::endl;
  }*/
  double    dim_loss = compute_dim_loss(dim_data, npts, map_out, bounds_out);
#pragma omp critical
  std::cout << "Dim: " << dim << ", residual : " << dim_loss << std::endl;
  delete[] dim_data;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << argv[0] << " input_file npts ndims quant_info quant_data"
              << std::endl;
    exit(-1);
  }
  uint64_t npts = (uint64_t) std::atoi(argv[2]);
  uint64_t ndims = (uint64_t) std::atoi(argv[3]);
  int16_t* data = new int16_t[npts * ndims];
  std::cout << "npts = " << npts << ", ndims = " << ndims << std::endl;
  std::ifstream reader(argv[1], std::ios::binary);
  reader.read((char*) data, npts * ndims * sizeof(int16_t));
  reader.close();
  double*   bin_maps = new double[ndims * 256]();
  double*   bin_bounds = new double[ndims * 257]();
#pragma omp parallel for schedule(static, 1) num_threads(32)
  for (uint64_t i = 0; i < ndims; i++) {
    quantize(data, i, npts, ndims, bin_maps + (256 * i),
             bin_bounds + (257 * i));
  }
  std::ofstream writer(argv[4], std::ios::binary);
  writer.write((char*) bin_maps, ndims * 256 * sizeof(double));
  writer.write((char*) bin_bounds, ndims * 257 * sizeof(double));
  writer.close();
  std::cout << "Bin Maps written to " << std::string(argv[4])
            << "\n Format: [Row-Major] BIN_MAPS[N x 256] BIN_BOUNDS[N x 257]"
            << std::endl;
  uint8_t* mapped = new uint8_t[npts * ndims];
  map_dataset(data, mapped, bin_maps, bin_bounds, npts, ndims);
  std::cout << "Finished mapping int16 to uint8" << std::endl;
  writer.open(argv[5], std::ios::binary);
  writer.write((char*) mapped, npts * ndims * sizeof(uint8_t));
  writer.close();
  std::cout << "Mapped dataset written to " << std::string(argv[5])
            << "\n Format: [Row-Major] MAPPED[N x D]" << std::endl;
  std::cout << "Vector quantization residual: "
            << compute_data_loss(data, mapped, bin_maps, npts, ndims)
            << std::endl;
  delete[] data;
  delete[] mapped;
  delete[] bin_maps;
  delete[] bin_bounds;
}
