#pragma once

#include "aligned_file_reader.h"
#include <libaio.h>
#include <cassert>
#include <cstdio>
#include <iostream>
#include "tsl/robin_map.h"

#define MAX_EVENTS 256

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb     iocb_t;

  void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs,
                  uint64_t n_retries = 5) {
    uint64_t                 n_ops = read_reqs.size();
    std::vector<iocb_t *>    cbs(n_ops, nullptr);
    std::vector<io_event_t>  evts(n_ops);
    std::vector<struct iocb> cb(n_ops);
    for (uint64_t j = 0; j < n_ops; j++) {
      io_prep_pread(cb.data() + j, fd, read_reqs[j].buf, read_reqs[j].len,
                    read_reqs[j].offset);
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
                  << ::strerror(errno) << ", try #" << n_tries + 1;
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
                    << ::strerror(errno) << ", try #" << n_tries + 1;
          n_tries++;
          // try again
          continue;
        } else {
          break;
        }
      }
    }
    assert(n_tries != n_retries);
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
  assert(AlignedFileReader::ctx_map.find(my_id) ==
         AlignedFileReader::ctx_map.end());
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
  assert(this->fd != -1);
  io_context_t ctx = get_ctx();
  execute_io(ctx, this->file_desc, read_reqs);
}