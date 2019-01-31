#pragma once

#include <fcntl.h>
#include <libaio.h>
#include <unistd.h>
#include <cstdio>
#include <mutex>
#include <thread>
#include "tsl/robin_map.h"

// NOTE :: all 3 fields must be 512-aligned
struct AlignedRead {
  uint64_t offset;  // where to read from
  uint64_t len;     // how much to read
  void *   buf;     // where to read into

  AlignedRead() : offset(0), len(0), buf(nullptr) {
  }

  AlignedRead(uint64_t offset, uint64_t len, void *buf)
      : offset(offset), len(len), buf(buf) {
  }
};

class AlignedFileReader {
  tsl::robin_map<std::thread::id, io_context_t> ctx_map;
  std::mutex ctx_mut;

  // returns the thread-specific context
  // returns (io_context_t)(-1) if thread is not registered
  io_context_t get_ctx();

 public:
  uint64_t file_sz;
  int      file_desc;

  AlignedFileReader();
  ~AlignedFileReader();

  // register thread-id for a context
  void register_thread();

  // de-register thread-id for a context
  void deregister_thread();

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<AlignedRead> &read_reqs);
};
