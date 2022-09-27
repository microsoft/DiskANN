// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"

#include <stdio.h>

#ifdef EXEC_ENV_OLS
#include "aligned_file_reader.h"
#endif

const uint32_t MAX_REQUEST_SIZE = 1024 * 1024 * 1024;  // 64MB
const uint32_t MAX_SIMULTANEOUS_READ_REQUESTS = 128;

#ifdef _WINDOWS
#include <intrin.h>

// Taken from:
// https://insufficientlycomplicated.wordpress.com/2011/11/07/detecting-intel-advanced-vector-extensions-avx-in-visual-studio/
bool cpuHasAvxSupport() {
  bool avxSupported = false;

  // Checking for AVX requires 3 things:
  // 1) CPUID indicates that the OS uses XSAVE and XRSTORE
  //     instructions (allowing saving YMM registers on context
  //     switch)
  // 2) CPUID indicates support for AVX
  // 3) XGETBV indicates the AVX registers will be saved and
  //     restored on context switch
  //
  // Note that XGETBV is only available on 686 or later CPUs, so
  // the instruction needs to be conditionally run.
  int cpuInfo[4];
  __cpuid(cpuInfo, 1);

  bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;
  bool cpuAVXSuport = cpuInfo[2] & (1 << 28) || false;

  if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
    // Check if the OS will save the YMM registers
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    avxSupported = (xcrFeatureMask & 0x6) || false;
  }

  return avxSupported;
}

bool cpuHasAvx2Support() {
  int cpuInfo[4];
  __cpuid(cpuInfo, 0);
  int n = cpuInfo[0];
  if (n >= 7) {
    __cpuidex(cpuInfo, 7, 0);
    static int avx2Mask = 0x20;
    return (cpuInfo[1] & avx2Mask) > 0;
  }
  return false;
}
#endif

#ifdef _WINDOWS
bool AvxSupportedCPU = cpuHasAvxSupport();
bool Avx2SupportedCPU = cpuHasAvx2Support();
#else
bool Avx2SupportedCPU = true;
bool AvxSupportedCPU = false;
#endif

namespace diskann {
  // Get the right distance function for the given metric.
  template<>
  diskann::Distance<float>* get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
      if (Avx2SupportedCPU) {
        diskann::cout << "L2: Using AVX2 distance computation DistanceL2Float"
                      << std::endl;
        return new diskann::DistanceL2Float();
      } else if (AvxSupportedCPU) {
        diskann::cout
            << "L2: AVX2 not supported. Using AVX distance computation"
            << std::endl;
        return new diskann::AVXDistanceL2Float();
      } else {
        diskann::cout << "L2: Older CPU. Using slow distance computation"
                      << std::endl;
        return new diskann::SlowDistanceL2Float();
      }
    } else if (m == diskann::Metric::COSINE) {
      diskann::cout << "Cosine: Using either AVX or AVX2 implementation"
                    << std::endl;
      return new diskann::DistanceCosineFloat();
    } else if (m == diskann::Metric::INNER_PRODUCT) {
      diskann::cout << "Inner product: Using AVX2 implementation "
                       "AVXDistanceInnerProductFloat"
                    << std::endl;
      return new diskann::AVXDistanceInnerProductFloat();
    } else if (m == diskann::Metric::FAST_L2) {
      diskann::cout << "Fast_L2: Using AVX2 implementation with norm "
                       "memoization DistanceFastL2<float>"
                    << std::endl;
      return new diskann::DistanceFastL2<float>();
    } else {
      std::stringstream stream;
      stream << "Only L2, cosine, and inner product supported for floating "
                "point vectors as of now. Email "
                "{gopalsr, harshasi, rakri}@microsoft.com if you need support "
                "for any other metric."
             << std::endl;
      diskann::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  diskann::Distance<int8_t>* get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
      if (Avx2SupportedCPU) {
        diskann::cout << "Using AVX2 distance computation DistanceL2Int8."
                      << std::endl;
        return new diskann::DistanceL2Int8();
      } else if (AvxSupportedCPU) {
        diskann::cout << "AVX2 not supported. Using AVX distance computation"
                      << std::endl;
        return new diskann::AVXDistanceL2Int8();
      } else {
        diskann::cout << "Older CPU. Using slow distance computation "
                         "SlowDistanceL2Int<int8_t>."
                      << std::endl;
        return new diskann::SlowDistanceL2Int<int8_t>();
      }
    } else if (m == diskann::Metric::COSINE) {
      diskann::cout << "Using either AVX or AVX2 for Cosine similarity "
                       "DistanceCosineInt8."
                    << std::endl;
      return new diskann::DistanceCosineInt8();
    } else {
      std::stringstream stream;
      stream << "Only L2 and cosine supported for signed byte vectors as of "
                "now. Email "
                "{gopalsr, harshasi, rakri}@microsoft.com if you need support "
                "for any other metric."
             << std::endl;
      diskann::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  diskann::Distance<uint8_t>* get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
#ifdef _WINDOWS
      diskann::cout
          << "WARNING: AVX/AVX2 distance function not defined for Uint8. Using "
             "slow version. "
             "Contact gopalsr@microsoft.com if you need AVX/AVX2 support."
          << std::endl;
#endif
      return new diskann::DistanceL2UInt8();
    } else if (m == diskann::Metric::COSINE) {
      diskann::cout
          << "AVX/AVX2 distance function not defined for Uint8. Using "
             "slow version SlowDistanceCosineUint8() "
             "Contact gopalsr@microsoft.com if you need AVX/AVX2 support."
          << std::endl;
      return new diskann::SlowDistanceCosineUInt8();
    } else {
      std::stringstream stream;
      stream << "Only L2 and cosine supported for unsigned byte vectors as of "
                "now. Email "
                "{gopalsr, harshasi, rakri}@microsoft.com if you need support "
                "for any other metric."
             << std::endl;
      diskann::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  void block_convert(std::ofstream& writr, std::ifstream& readr,
                     float* read_buf, _u64 npts, _u64 ndims) {
    readr.read((char*) read_buf, npts * ndims * sizeof(float));
    _u32 ndims_u32 = (_u32) ndims;
#pragma omp parallel for
    for (_s64 i = 0; i < (_s64) npts; i++) {
      float norm_pt = std::numeric_limits<float>::epsilon();
      for (_u32 dim = 0; dim < ndims_u32; dim++) {
        norm_pt +=
            *(read_buf + i * ndims + dim) * *(read_buf + i * ndims + dim);
      }
      norm_pt = std::sqrt(norm_pt);
      for (_u32 dim = 0; dim < ndims_u32; dim++) {
        *(read_buf + i * ndims + dim) = *(read_buf + i * ndims + dim) / norm_pt;
      }
    }
    writr.write((char*) read_buf, npts * ndims * sizeof(float));
  }

  void normalize_data_file(const std::string& inFileName,
                           const std::string& outFileName) {
    std::ifstream readr(inFileName, std::ios::binary);
    std::ofstream writr(outFileName, std::ios::binary);

    int npts_s32, ndims_s32;
    readr.read((char*) &npts_s32, sizeof(_s32));
    readr.read((char*) &ndims_s32, sizeof(_s32));

    writr.write((char*) &npts_s32, sizeof(_s32));
    writr.write((char*) &ndims_s32, sizeof(_s32));

    _u64 npts = (_u64) npts_s32, ndims = (_u64) ndims_s32;
    diskann::cout << "Normalizing FLOAT vectors in file: " << inFileName
                  << std::endl;
    diskann::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
                  << std::endl;

    _u64 blk_size = 131072;
    _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
    diskann::cout << "# blks: " << nblks << std::endl;

    float* read_buf = new float[npts * ndims];
    for (_u64 i = 0; i < nblks; i++) {
      _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
      block_convert(writr, readr, read_buf, cblk_size, ndims);
    }
    delete[] read_buf;

    diskann::cout << "Wrote normalized points to file: " << outFileName
                  << std::endl;
  }

#ifdef EXEC_ENV_OLS
  void get_bin_metadata(AlignedFileReader& reader, size_t& npts, size_t& ndim,
                        size_t offset) {
    std::vector<AlignedRead> readReqs;
    AlignedRead              readReq;
    uint32_t                 buf[2];  // npts/ndim are uint32_ts.

    readReq.buf = buf;
    readReq.offset = offset;
    readReq.len = 2 * sizeof(uint32_t);
    readReqs.push_back(readReq);

    IOContext& ctx = reader.get_ctx();
    reader.read(readReqs, ctx);  // synchronous
    if ((*(ctx.m_pRequestsStatus))[0] == IOContext::READ_SUCCESS) {
      npts = buf[0];
      ndim = buf[1];
      diskann::cout << "File has: " << npts << " points, " << ndim
                    << " dimensions at offset: " << offset << std::endl;
    } else {
      std::stringstream str;
      str << "Could not read binary metadata from index file at offset: "
          << offset << std::endl;
      throw diskann::ANNException(str.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<typename T>
  void load_bin(AlignedFileReader& reader, T*& data, size_t& npts, size_t& ndim,
                size_t offset) {
    // Code assumes that the reader is already setup correctly.
    get_bin_metadata(reader, npts, ndim, offset);
    data = new T[npts * ndim];

    size_t data_size = npts * ndim * sizeof(T);
    size_t write_offset = 0;
    size_t read_start = offset + 2 * sizeof(uint32_t);

    // BingAlignedFileReader can only read uint32_t bytes of data. So,
    // we limit ourselves even more to reading 1GB at a time.
    std::vector<AlignedRead> readReqs;
    while (data_size > 0) {
      AlignedRead readReq;
      readReq.buf = data + write_offset;
      readReq.offset = read_start + write_offset;
      readReq.len = data_size > MAX_REQUEST_SIZE ? MAX_REQUEST_SIZE : data_size;
      readReqs.push_back(readReq);
      // in the corner case, the loop will not execute
      data_size -= readReq.len;
      write_offset += readReq.len;
    }
    IOContext& ctx = reader.get_ctx();
    reader.read(readReqs, ctx);
    for (int i = 0; i < readReqs.size(); i++) {
      // Since we are making sync calls, no request will be in the
      // READ_WAIT state.
      if ((*(ctx.m_pRequestsStatus))[i] != IOContext::READ_SUCCESS) {
        std::stringstream str;
        str << "Could not read binary data from index file at offset: "
            << readReqs[i].offset << std::endl;
        throw diskann::ANNException(str.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
    }
  }
  template<typename T>
  void load_bin(AlignedFileReader& reader, std::unique_ptr<T[]>& data,
                size_t& npts, size_t& ndim, size_t offset) {
    T* ptr = nullptr;
    load_bin(reader, ptr, npts, ndim, offset);
    data.reset(ptr);
  }

  template<typename T>
  void copy_aligned_data_from_file(AlignedFileReader& reader, T*& data,
                                   size_t& npts, size_t& ndim,
                                   const size_t& rounded_dim, size_t offset) {
    if (data == nullptr) {
      diskann::cerr << "Memory was not allocated for " << data
                    << " before calling the load function. Exiting..."
                    << std::endl;
      throw diskann::ANNException(
          "Null pointer passed to copy_aligned_data_from_file()", -1,
          __FUNCSIG__, __FILE__, __LINE__);
    }

    size_t pts, dim;
    get_bin_metadata(reader, pts, dim, offset);

    if (ndim != dim || npts != pts) {
      std::stringstream ss;
      ss << "Either file dimension: " << dim
         << " is != passed dimension: " << ndim << " or file #pts: " << pts
         << " is != passed #pts: " << npts << std::endl;
      throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    // Instead of reading one point of ndim size and setting (rounded_dim - dim)
    // values to zero We'll set everything to zero and read in chunks of data at
    // the appropriate locations.
    size_t read_offset = offset + 2 * sizeof(uint32_t);
    memset(data, 0, npts * rounded_dim * sizeof(T));
    int                      i = 0;
    std::vector<AlignedRead> read_requests;

    while (i < npts) {
      int j = 0;
      read_requests.clear();
      while (j < MAX_SIMULTANEOUS_READ_REQUESTS && i < npts) {
        AlignedRead read_req;
        read_req.buf = data + i * rounded_dim;
        read_req.len = dim * sizeof(T);
        read_req.offset = read_offset + i * dim * sizeof(T);
        read_requests.push_back(read_req);
        i++;
        j++;
      }
      IOContext& ctx = reader.get_ctx();
      reader.read(read_requests, ctx);
      for (int k = 0; k < read_requests.size(); k++) {
        if ((*ctx.m_pRequestsStatus)[k] != IOContext::READ_SUCCESS) {
          throw diskann::ANNException(
              "Load data from file using AlignedReader failed.", -1,
              __FUNCSIG__, __FILE__, __LINE__);
        }
      }
    }
  }

  // Unlike load_bin, assumes that data is already allocated 'size' entries
  template<typename T>
  void read_array(AlignedFileReader& reader, T* data, size_t size,
                  size_t offset) {
    if (data == nullptr) {
      throw diskann::ANNException("read_array requires an allocated buffer.",
                                  -1);
      if (size * sizeof(T) > MAX_REQUEST_SIZE) {
        std::stringstream ss;
        ss << "Cannot read more than " << MAX_REQUEST_SIZE
           << " bytes. Current request size: " << std::to_string(size)
           << " sizeof(T): " << sizeof(T) << std::endl;
        throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
      std::vector<AlignedRead> read_requests;
      AlignedRead              read_req;
      read_req.buf = data;
      read_req.len = size * sizeof(T);
      read_req.offset = offset;
      read_requests.push_back(read_req);
      IOContext& ctx = reader.get_ctx();
      reader.read(read_requests, ctx);

      if ((*(ctx.m_pRequestsStatus))[0] != IOContext::READ_SUCCESS) {
        std::stringstream ss;
        ss << "Failed to read_array() of size: " << size * sizeof(T)
           << " at offset: " << offset << " from reader. " << std::endl;
        throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
    }
  }

  template<typename T>
  void read_value(AlignedFileReader& reader, T& value, size_t offset) {
    read_array(reader, &value, 1, offset);
  }

  template DISKANN_DLLEXPORT void load_bin<uint8_t>(
      AlignedFileReader& reader, std::unique_ptr<uint8_t[]>& data, size_t& npts,
      size_t& ndim, size_t offset);
  template DISKANN_DLLEXPORT void load_bin<int8_t>(
      AlignedFileReader& reader, std::unique_ptr<int8_t[]>& data, size_t& npts,
      size_t& ndim, size_t offset);
  template DISKANN_DLLEXPORT void load_bin<uint32_t>(
      AlignedFileReader& reader, std::unique_ptr<uint32_t[]>& data,
      size_t& npts, size_t& ndim, size_t offset);
  template DISKANN_DLLEXPORT void load_bin<uint64_t>(
      AlignedFileReader& reader, std::unique_ptr<uint64_t[]>& data,
      size_t& npts, size_t& ndim, size_t offset);
  template DISKANN_DLLEXPORT void load_bin<int64_t>(
      AlignedFileReader& reader, std::unique_ptr<int64_t[]>& data, size_t& npts,
      size_t& ndim, size_t offset);
  template DISKANN_DLLEXPORT void load_bin<float>(
      AlignedFileReader& reader, std::unique_ptr<float[]>& data, size_t& npts,
      size_t& ndim, size_t offset);

  template DISKANN_DLLEXPORT void load_bin<uint8_t>(AlignedFileReader& reader,
                                                    uint8_t*&          data,
                                                    size_t& npts, size_t& ndim,
                                                    size_t offset);
  template DISKANN_DLLEXPORT void load_bin<int64_t>(AlignedFileReader& reader,
                                                    int64_t*&          data,
                                                    size_t& npts, size_t& ndim,
                                                    size_t offset);
  template DISKANN_DLLEXPORT void load_bin<uint64_t>(AlignedFileReader& reader,
                                                     uint64_t*&         data,
                                                     size_t& npts, size_t& ndim,
                                                     size_t offset);
  template DISKANN_DLLEXPORT void load_bin<uint32_t>(AlignedFileReader& reader,
                                                     uint32_t*&         data,
                                                     size_t& npts, size_t& ndim,
                                                     size_t offset);
  template DISKANN_DLLEXPORT void load_bin<int32_t>(AlignedFileReader& reader,
                                                    int32_t*&          data,
                                                    size_t& npts, size_t& ndim,
                                                    size_t offset);

  template DISKANN_DLLEXPORT void copy_aligned_data_from_file<uint8_t>(
      AlignedFileReader& reader, uint8_t*& data, size_t& npts, size_t& dim,
      const size_t& rounded_dim, size_t offset);
  template DISKANN_DLLEXPORT void copy_aligned_data_from_file<int8_t>(
      AlignedFileReader& reader, int8_t*& data, size_t& npts, size_t& dim,
      const size_t& rounded_dim, size_t offset);
  template DISKANN_DLLEXPORT void copy_aligned_data_from_file<float>(
      AlignedFileReader& reader, float*& data, size_t& npts, size_t& dim,
      const size_t& rounded_dim, size_t offset);

  template DISKANN_DLLEXPORT void read_array<char>(AlignedFileReader& reader,
                                                   char* data, size_t size,
                                                   size_t offset);

  template DISKANN_DLLEXPORT void read_array<uint8_t>(AlignedFileReader& reader,
                                                      uint8_t*           data,
                                                      size_t             size,
                                                      size_t offset);
  template DISKANN_DLLEXPORT void read_array<int8_t>(AlignedFileReader& reader,
                                                     int8_t* data, size_t size,
                                                     size_t offset);
  template DISKANN_DLLEXPORT void read_array<uint32_t>(
      AlignedFileReader& reader, uint32_t* data, size_t size, size_t offset);
  template DISKANN_DLLEXPORT void read_array<float>(AlignedFileReader& reader,
                                                    float* data, size_t size,
                                                    size_t offset);

  template DISKANN_DLLEXPORT void read_value<uint8_t>(AlignedFileReader& reader,
                                                      uint8_t&           value,
                                                      size_t offset);
  template DISKANN_DLLEXPORT void read_value<int8_t>(AlignedFileReader& reader,
                                                     int8_t&            value,
                                                     size_t             offset);
  template DISKANN_DLLEXPORT void read_value<float>(AlignedFileReader& reader,
                                                    float&             value,
                                                    size_t             offset);
  template DISKANN_DLLEXPORT void read_value<uint32_t>(
      AlignedFileReader& reader, uint32_t& value, size_t offset);
  template DISKANN_DLLEXPORT void read_value<uint64_t>(
      AlignedFileReader& reader, uint64_t& value, size_t offset);

#endif

}  // namespace diskann
