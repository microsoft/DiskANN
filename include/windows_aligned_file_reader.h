#pragma once
#ifdef _WINDOWS
#include <Windows.h>
#include <fcntl.h>
#include <malloc.h>
#include <minwinbase.h>

#include <cstdio>
#include <mutex>
#include <thread>
#include "aligned_file_reader.h"
#include "tsl/robin_map.h"
#include "utils.h"
class WindowsAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t     file_sz;
  std::wstring filename;

 public:
  WindowsAlignedFileReader(){};
  virtual ~WindowsAlignedFileReader(){};

  virtual void register_thread();
  virtual void deregister_thread() {
  }
  virtual IOContext &get_ctx();

  // Open & close ops
  // Blocking calls
  virtual void open(const std::string &fname);
  virtual void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call for the calling thread, but can thread-safe
  virtual void read(std::vector<AlignedRead> &read_reqs, IOContext ctx);
};
#endif
