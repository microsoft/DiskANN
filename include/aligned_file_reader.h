#pragma once

#include <vector>
#ifndef __NSG_WINDOWS__
  #include <libaio.h>
  #include <unistd.h>
  #include <fcntl.h>
  typedef io_context_t Context;
#else
  #include <Windows.h>
  #include <minwinbase.h>
  typedef struct{
    HANDLE                  fhandle = NULL;
    HANDLE                  iocp = NULL;
    std::vector<OVERLAPPED> reqs;
  } IOContext;
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


  AlignedRead() : offset(0), len(0), buf(nullptr) {
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
  protected:
    tsl::robin_map<std::thread::id, IOContext> ctx_map;
    std::mutex                                 ctx_mut;

  public: 

    // returns the thread-specific context
    // returns (io_context_t)(-1) if thread is not registered
    virtual IOContext& get_ctx() = 0;
  
	virtual ~AlignedFileReader() = 0 {};

    // register thread-id for a context
    virtual void register_thread() = 0;
    // de-register thread-id for a context
    virtual void deregister_thread() = 0;
    
    // Open & close ops
    // Blocking calls  
    virtual void open(const std::string&  fname) = 0;
    virtual void close() = 0;

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    virtual void read(std::vector<AlignedRead> &read_reqs) = 0;
};


