// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#ifndef _WINDOWS
#include "aligned_file_reader.h"
#include "liburing.h"
#include <thread>

typedef struct io_uring IOUring;

class LinuxAsyncAlignedFileReader {
 private:
  tsl::robin_map<std::thread::id, IOUring> ring_map;
  std::mutex                               ctx_mut;
  uint64_t                                 file_sz;
  FileHandle                               file_desc;

 public:
  LinuxAsyncAlignedFileReader();
  ~LinuxAsyncAlignedFileReader();

  IOUring& get_ring();

  // register thread-id for a context
  void register_thread();

  // de-register thread-id for a context
  void deregister_thread();
  void deregister_all_threads();

  // Open & close ops
  // Blocking calls
  void open(const std::string& fname);
  void close();

  bool   submit_io(std::vector<AlignedRead>& read_reqs, IOUring& ring);
  bool   submit_io(void* buf, size_t len, size_t offset, IOUring& ring);
  size_t peek_io(IOUring& ring);
};

#endif
