#pragma once

#include "aligned_file_reader.h"
#include <libaio.h>
#include <cassert>
#include <cstdio>
#include <iostream>
#include "efanna2e/util.h"
#include "tsl/robin_map.h"
#define MAX_EVENTS 1024

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb     iocb_t;

  void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs,
                  uint64_t n_retries = 5) {
    for (auto &req : read_reqs) {
      assert(IS_ALIGNED(req.len, 512));
      // std::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
      assert(IS_ALIGNED(req.offset, 512));
      assert(IS_ALIGNED(req.buf, 512));
      assert(malloc_usable_size(req.buf) >= req.len);
    }

    // break-up requests into chunks of size MAX_EVENTS each
    uint64_t n_iters = ROUND_UP(read_reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++) {
      uint64_t n_ops =
          std::min((uint64_t) read_reqs.size() - (iter * MAX_EVENTS),
                   (uint64_t) MAX_EVENTS);
      std::vector<iocb_t *>    cbs(n_ops, nullptr);
      std::vector<io_event_t>  evts(n_ops);
      std::vector<struct iocb> cb(n_ops);
      for (uint64_t j = 0; j < n_ops; j++) {
        io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * MAX_EVENTS].buf,
                      read_reqs[j + iter * MAX_EVENTS].len,
                      read_reqs[j + iter * MAX_EVENTS].offset);
      }
      // initialize `cbs` using `cb` array
      for (uint64_t i = 0; i < n_ops; i++) {
        cbs[i] = cb.data() + i;
      }

      uint64_t n_tries = 0;
      while (n_tries < n_retries) {
        // issue reads
        int64_t ret = io_submit(ctx, (int64_t) n_ops, cbs.data());
        // if requests didn't get accepted
        if (ret != (int64_t) n_ops) {
          std::cerr << "io_submit() failed; returned " << ret
                    << ", expected=" << n_ops << ", ernno=" << errno << "="
                    << ::strerror(-ret) << ", try #" << n_tries + 1;
          n_tries++;
          // try again
          continue;
        } else {
          // wait on io_getevents
          ret = io_getevents(ctx, (int64_t) n_ops, (int64_t) n_ops, evts.data(),
                             nullptr);
          // if requests didn't complete
          if (ret != (int64_t) n_ops) {
            std::cerr << "io_getevents() failed; returned " << ret
                      << ", expected=" << n_ops << ", ernno=" << errno << "="
                      << ::strerror(-ret) << ", try #" << n_tries + 1;
            n_tries++;
            // try again
            continue;
          } else {
            break;
          }
        }
      }
      assert(n_tries != n_retries);
      for (auto &req : read_reqs) {
        // corruption check
        assert(malloc_usable_size(req.buf) >= req.len);
      }
    }
    /*
    for(unsigned i=0;i<64;i++){
      std::cout << *((unsigned*)read_reqs[0].buf + i) << " ";
    }
    std::cout << std::endl;*/
  }
}

AlignedFileReader::AlignedFileReader() {
  this->file_desc = -1;
}

AlignedFileReader::~AlignedFileReader() {
  int64_t ret;
  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno
                  << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

io_context_t AlignedFileReader::get_ctx() {
  std::unique_lock<std::mutex> lk(AlignedFileReader::ctx_mut);
  // perform checks only in DEBUG mode
  if (AlignedFileReader::ctx_map.find(std::this_thread::get_id()) ==
      AlignedFileReader::ctx_map.end()) {
    std::cerr << "bad thread access; returning -1 as io_context_t" << std::endl;
    return (io_context_t) -1;
  } else {
    return AlignedFileReader::ctx_map[std::this_thread::get_id()];
  }
  lk.unlock();
}

void AlignedFileReader::register_thread() {
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(AlignedFileReader::ctx_mut);
  if (AlignedFileReader::ctx_map.find(my_id) !=
      AlignedFileReader::ctx_map.end()) {
    std::cerr << "multiple calls to register_thread from the same thread"
              << std::endl;
    return;
  }
  io_context_t ctx = 0;
  int          ret = io_setup(MAX_EVENTS, &ctx);
  if (ret != 0) {
    lk.unlock();
    assert(errno != EAGAIN);
    assert(errno != ENOMEM);
    std::cerr << "io_setup() failed; returned " << ret << ", errno=" << errno
              << ":" << ::strerror(errno) << std::endl;
  } else {
    std::cerr << "allocating ctx to thread-id:" << my_id << std::endl;
    AlignedFileReader::ctx_map[my_id] = ctx;
  }
  lk.unlock();
}

void AlignedFileReader::deregister_thread() {
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(AlignedFileReader::ctx_mut);
  assert(AlignedFileReader::ctx_map.find(my_id) !=
         AlignedFileReader::ctx_map.end());

  lk.unlock();
  io_context_t ctx = this->get_ctx();
  int          ret = io_destroy(ctx);
  assert(ret == 0);
  lk.lock();
  AlignedFileReader::ctx_map.erase(my_id);
  std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
  lk.unlock();
}

void AlignedFileReader::open(const std::string &fname) {
  int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
  this->file_desc = ::open(fname.c_str(), flags);
  // error checks
  assert(this->file_desc != -1);
  std::cerr << "Opened file : " << fname << std::endl;
}

void AlignedFileReader::close() {
  int64_t ret;

  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  assert(ret != -1);

  ret = ::close(this->file_desc);
  assert(ret != -1);
}

void AlignedFileReader::read(std::vector<AlignedRead> &read_reqs) {
  assert(this->file_desc != -1);
  io_context_t ctx = get_ctx();
  execute_io(ctx, this->file_desc, read_reqs);
}
