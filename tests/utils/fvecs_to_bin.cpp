#include <iostream>
#include "utils.h"

void block_convert(std::ifstream& reader, std::ofstream& writer,
                   float* read_buf, float* write_buf, _u64 npts, _u64 ndims) {
  reader.read((char*) read_buf,
              npts * (ndims * sizeof(float) + sizeof(unsigned)));
  for (_u64 i = 0; i < npts; i++) {
    memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1,
           ndims * sizeof(float));
  }
  writer.write((char*) write_buf, npts * ndims * sizeof(float));
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << argv[0] << " input_fvecs output_bin" << std::endl;
    exit(-1);
  }
  std::ifstream reader(argv[1], std::ios::binary | std::ios::ate);
  _u64          fsize = reader.tellg();
  reader.seekg(0, std::ios::beg);

  unsigned ndims_u32;
  reader.read((char*) &ndims_u32, sizeof(unsigned));
  reader.seekg(0, std::ios::beg);
  _u64 ndims = (_u64) ndims_u32;
  _u64 npts = fsize / ((ndims + 1) * sizeof(float));
  std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
            << std::endl;

  _u64 blk_size = 131072;
  _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
  std::cout << "# blks: " << nblks << std::endl;
  std::ofstream writer(argv[2], std::ios::binary);
  int           npts_s32 = (_s32) npts;
  int           ndims_s32 = (_s32) ndims;
  writer.write((char*) &npts_s32, sizeof(_s32));
  writer.write((char*) &ndims_s32, sizeof(_s32));
  float* read_buf = new float[npts * (ndims + 1)];
  float* write_buf = new float[npts * ndims];
  for (_u64 i = 0; i < nblks; i++) {
    _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
    block_convert(reader, writer, read_buf, write_buf, cblk_size, ndims);
    std::cout << "Block #" << i << " written" << std::endl;
  }

  delete[] read_buf;
  delete[] write_buf;

  reader.close();
  writer.close();
}
