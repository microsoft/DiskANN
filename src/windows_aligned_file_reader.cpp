// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifdef _WINDOWS
#ifndef USE_BING_INFRA
#include "windows_aligned_file_reader.h"
#include <iostream>
#include "utils.h"

#define SECTOR_LEN 4096

void WindowsAlignedFileReader::open(const std::string& fname) {
  m_filename = std::wstring(fname.begin(), fname.end());
  this->register_thread();
}

void WindowsAlignedFileReader::close() {
  for (auto& k_v : ctx_map) {
    IOContext ctx = ctx_map[k_v.first];
    CloseHandle(ctx.fhandle);
  }
}

void WindowsAlignedFileReader::register_thread() {
  std::unique_lock<std::mutex> lk(this->ctx_mut);
  if (this->ctx_map.find(std::this_thread::get_id()) != ctx_map.end()) {
    diskann::cout << "Warning:: Duplicate registration for thread_id : "
                  << std::this_thread::get_id() << std::endl;
  }

  IOContext ctx;
  ctx.fhandle = CreateFile(m_filename.c_str(), GENERIC_READ, FILE_SHARE_READ,
                           NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_READONLY | FILE_FLAG_NO_BUFFERING |
                               FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
                           NULL);
  if (ctx.fhandle == INVALID_HANDLE_VALUE) {
    diskann::cout << "Error opening " << std::string(m_filename.begin(), m_filename.end())
                  << " -- error=" << GetLastError() << std::endl;
  }

  // create IOCompletionPort
  ctx.iocp = CreateIoCompletionPort(ctx.fhandle, ctx.iocp, 0, 0);

  // create MAX_DEPTH # of reqs
  for (_u64 i = 0; i < MAX_IO_DEPTH; i++) {
    OVERLAPPED os;
    memset(&os, 0, sizeof(OVERLAPPED));
    // os.hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
    ctx.reqs.push_back(os);
  }
  this->ctx_map.insert(std::make_pair(std::this_thread::get_id(), ctx));
}

IOContext& WindowsAlignedFileReader::get_ctx() {
  std::unique_lock<std::mutex> lk(this->ctx_mut);
  if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end()) {
    std::stringstream stream;
    stream << "unable to find IOContext for thread_id : "
           << std::this_thread::get_id() << "\n";
    throw diskann::ANNException(stream.str(), -2, __FUNCSIG__, __FILE__,
                                __LINE__);
  }
  IOContext& ctx = ctx_map[std::this_thread::get_id()];
  lk.unlock();
  return ctx;
}

void WindowsAlignedFileReader::read(std::vector<AlignedRead>& read_reqs,
                                    IOContext& ctx, bool async) {
  using namespace std::chrono_literals;
  // execute each request sequentially
  _u64 n_reqs = read_reqs.size();
  _u64 n_batches = ROUND_UP(n_reqs, MAX_IO_DEPTH) / MAX_IO_DEPTH;
  for (_u64 i = 0; i < n_batches; i++) {
    // reset all OVERLAPPED objects
    for (auto& os : ctx.reqs) {
      // HANDLE evt = os.hEvent;
      memset(&os, 0, sizeof(os));
      // os.hEvent = evt;

      /*
        if (ResetEvent(os.hEvent) == 0) {
          diskann::cerr << "ResetEvent failed" << std::endl;
          exit(-3);
        }
      */
    }

    // batch start/end
    _u64 batch_start = MAX_IO_DEPTH * i;
    _u64 batch_size =
        std::min((_u64)(n_reqs - batch_start), (_u64) MAX_IO_DEPTH);

    // fill OVERLAPPED and issue them
    for (_u64 j = 0; j < batch_size; j++) {
      AlignedRead& req = read_reqs[batch_start + j];
      OVERLAPPED&  os = ctx.reqs[j];

      _u64  offset = req.offset;
      _u64  nbytes = req.len;
      char* read_buf = (char*) req.buf;
      assert(IS_ALIGNED(read_buf, SECTOR_LEN));
      assert(IS_ALIGNED(offset, SECTOR_LEN));
      assert(IS_ALIGNED(nbytes, SECTOR_LEN));

      // fill in OVERLAPPED struct
      os.Offset = offset & 0xffffffff;
      os.OffsetHigh = (offset >> 32);

      BOOL ret = ReadFile(ctx.fhandle, read_buf, nbytes, NULL, &os);
      if (ret == FALSE) {
        auto error = GetLastError();
        if (error != ERROR_IO_PENDING) {
          diskann::cerr << "Error queuing IO -- " << error << "\n";
        }
      } else {
        diskann::cerr << "Error queueing IO -- ReadFile returned TRUE"
                      << std::endl;
      }
    }
    DWORD       n_read = 0;
    _u64        n_complete = 0;
    ULONG_PTR   completion_key = 0;
    OVERLAPPED* lp_os;
    while (n_complete < batch_size) {
      if (GetQueuedCompletionStatus(ctx.iocp, &n_read, &completion_key, &lp_os,
                                    INFINITE) != 0) {
        // successfully dequeued a completed I/O
        n_complete++;
      } else {
        // failed to dequeue OR dequeued failed I/O
        if (lp_os == NULL) {
          DWORD error = GetLastError();
          if (error != WAIT_TIMEOUT) {
            diskann::cerr << "GetQueuedCompletionStatus() failed with error = "
                          << error << std::endl;
            throw diskann::ANNException(
                "GetQueuedCompletionStatus failed with error: ", error,
                __FUNCSIG__, __FILE__, __LINE__);
          }
          // no completion packet dequeued ==> sleep for 5us and try again
          std::this_thread::sleep_for(5us);
        } else {
          // completion packet for failed IO dequeued
          auto              op_idx = lp_os - ctx.reqs.data();
          std::stringstream stream;
          stream << "I/O failed , offset: " << read_reqs[op_idx].offset
                 << "with error code: " << GetLastError() << std::endl;
          throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                      __LINE__);
        }
      }
    }
  }
}
#endif
#endif
