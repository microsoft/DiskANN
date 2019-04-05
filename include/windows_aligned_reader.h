#pragma once

#include <fcntl.h>
#include <Windows.h>
#include <minwinbase.h>
#include <malloc.h>

#include <cstdio>
#include <mutex>
#include <thread>
#include "efanna2e/util.h"
#include "tsl/robin_map.h"
#include "aligned_file_reader.h"

class WindowsAlignedReader {
  public:
  uint64_t   file_sz;
   std::wstring        filename;
  std::vector<HANDLE> handles;
  
  tsl::robin_map<std::thread::id, HANDLE> handle_map;
  std::mutex handle_mut;

  WindowsAlignedReader() {};
  ~WindowsAlignedReader() {};

  void register_thread();
  HANDLE get_handle();

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call for the calling thread, but can thread-safe
  void read(std::vector<AlignedRead> &read_reqs);
};
