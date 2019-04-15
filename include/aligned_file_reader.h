#pragma once

#include <fcntl.h>
#ifndef __NSG_WINDOWS__
#include <libaio.h>
#include <unistd.h>
typedef io_context_t Context;
#else
#include <Windows.h>
#include <minwinbase.h>
typedef _OVERLAPPED IoContext;
#endif
#include <malloc.h>

#include <cstdio>
#include <mutex>
#include <thread>
#include "efanna2e/util.h"
#include "tsl/robin_map.h"

// NOTE :: all 3 fields must be 512-aligned
struct AlignedRead {
  uint64_t offset;  // where to read from
  uint64_t len;     // how much to read
  void *   buf;     // where to read into
#ifdef __NSG_WINDOWS__
  OVERLAPPED overlapped; //Windows IO data structure
#endif

  AlignedRead() : offset(0), len(0), buf(nullptr) {
  #ifdef __NSG_WINDOWS__
	  memset(&overlapped, 0, sizeof(overlapped));
  #endif
  }

  AlignedRead(uint64_t offset, uint64_t len, void *buf)
      : offset(offset), len(len), buf(buf) {
    assert(IS_512_ALIGNED(offset));
    assert(IS_512_ALIGNED(len));
    assert(IS_512_ALIGNED(buf));
    // assert(malloc_usable_size(buf) >= len);
  }
};

class AlignedFileReader {
  tsl::robin_map<std::thread::id, IoContext> ctx_map;
  std::mutex                               ctx_mut;

  // returns the thread-specific context
  // returns (io_context_t)(-1) if thread is not registered
  IoContext get_ctx();

 public:
  uint64_t			file_sz;
  FileHandle		file_desc;

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
