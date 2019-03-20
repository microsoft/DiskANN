#include <iostream>
#include "efanna2e/util.h"

void block_convert(std::ifstream& reader, std::ofstream& writer,
                   unsigned* read_buf, _u8* write_buf, _u64 npts, _u64 ndims) {
  reader.read((char*) read_buf,
              npts * (ndims * sizeof(unsigned) + sizeof(unsigned)));
#pragma omp parallel for
  for (_u64 i = 0; i < npts; i++) {
    _u8*      out_vec = write_buf + i * ndims;
    unsigned* in_vec = (read_buf + i * (ndims + 1)) + 1;
    for (_u64 j = 0; j < ndims; j++) {
      out_vec[j] = (_u8) in_vec[j];
    }
  }
  writer.write((char*) write_buf, npts * ndims * sizeof(_u8));
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << argv[0] << " input_ivecs output_bin" << std::endl;
    exit(-1);
  }
  std::ifstream reader(argv[1], std::ios::binary | std::ios::ate);
  _u64          fsize = reader.tellg();
  reader.seekg(0, std::ios::beg);

  unsigned ndims_u32;
  reader.read((char*) &ndims_u32, sizeof(unsigned));
  reader.seekg(0, std::ios::beg);
  _u64 ndims = (_u64) ndims_u32;
  _u64 npts = fsize / ((ndims + 1) * sizeof(unsigned));
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
  unsigned* read_buf = new unsigned[npts * (ndims + 1)];
  _u8*      write_buf = new _u8[npts * ndims];
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