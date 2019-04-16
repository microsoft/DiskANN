#pragma once

#include <Windows.h>
#include <fcntl.h>
#include <malloc.h>
#include <minwinbase.h>

#include <cstdio>
#include <mutex>
#include <thread>
#include "aligned_file_reader.h"
#include "efanna2e/util.h"
#include "tsl/robin_map.h"

typedef struct{
  HANDLE                  fhandle = NULL;
  HANDLE                  iocp = NULL;
  std::vector<OVERLAPPED> reqs;
}IOContext;

class WindowsAlignedReader {
 public:
  uint64_t            file_sz;
  std::wstring        filename;

  tsl::robin_map<std::thread::id, IOContext> ctx_map;
  std::mutex                                 ctx_mut;

  WindowsAlignedReader(){};
  ~WindowsAlignedReader(){};

  void   register_thread();
  IOContext& get_ctx();

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call for the calling thread, but can thread-safe
  void read(std::vector<AlignedRead> &read_reqs);
};
