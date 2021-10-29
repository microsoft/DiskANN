// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include "tsl/robin_map.h"
#include "utils.h"
#define MAX_EVENTS 1024

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb     iocb_t;

  // execute_io read file using the way fio does,
  // iops tested by fio with num-jobs=4 can reach the peak,
  // we can also create 4 io_context_t, and then parallel the execute_io.
  // Note:
  //    1, file should be opened with `O_DIRECT` flag;
  //    2, the bufs in read_reqs must be aligned to 512 bytes.
  void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs,
                  uint64_t n_retries = 0) {
#ifdef DEBUG
    for (auto &req : read_reqs) {
      assert(IS_ALIGNED(req.len, 512));
      // std::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
      assert(IS_ALIGNED(req.offset, 512));
      assert(IS_ALIGNED(req.buf, 512));
      // assert(malloc_usable_size(req.buf) >= req.len);
    }
#endif

    size_t wait_nr = 32;
    size_t total = read_reqs.size();
    size_t done = 0;
    size_t submitted = 0;
    size_t to_submit_num = MAX_EVENTS;

    std::vector<iocb_t *>    cbs(total, nullptr);
    std::vector<io_event_t>  evts(total);
    std::vector<struct iocb> cb(total);

    //#pragma omp parallel for
    for (auto i = 0; i < total; i++) {
      io_prep_pread(cb.data() + i,
                    fd,
                    read_reqs[i].buf,
                    read_reqs[i].len,
                    read_reqs[i].offset);
      cbs[i] = cb.data() + i;
    }

    while (done < total) {
      auto upper = total - submitted;
      if (to_submit_num > upper) {
        to_submit_num = upper;
      }
      if (to_submit_num > MAX_EVENTS) {
        to_submit_num = MAX_EVENTS;
      }

      if (to_submit_num > 0) {
          auto r_submit = io_submit(ctx, to_submit_num, cbs.data() + submitted);
          if (r_submit < 0) {
            std::cerr << "io_submit() failed; returned " << r_submit
                      << ", ernno=" << errno << "=" << ::strerror(-r_submit)
                      << std::endl;
            std::cout << "ctx: " << ctx << "\n";
            exit(-1);
      }

      submitted += r_submit;
      }

      auto pending = submitted - done;
      if (wait_nr > pending) {
        wait_nr = pending;
      }

      auto r_done = io_getevents(ctx, wait_nr, MAX_EVENTS, evts.data() + done, nullptr);
      if (r_done < 0) {
        std::cerr << "io_getevents() failed; returned " << r_done
                  << ", ernno=" << errno << "="
                  << ::strerror(-r_done)
                  << std::endl;
        exit(-1);
      }

      to_submit_num = r_done;
      done += r_done;
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
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

io_context_t &LinuxAlignedFileReader::get_ctx() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  // perform checks only in DEBUG mode
  if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end()) {
    std::cerr << "bad thread access; returning -1 as io_context_t" << std::endl;
    return this->bad_ctx;
  } else {
    return ctx_map[std::this_thread::get_id()];
  }
}

void LinuxAlignedFileReader::register_thread() {
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  if (ctx_map.find(my_id) != ctx_map.end()) {
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
    diskann::cout << "allocating ctx: " << ctx << " to thread-id:" << my_id
                  << std::endl;
    ctx_map[my_id] = ctx;
  }
  lk.unlock();
}

void LinuxAlignedFileReader::deregister_thread() {
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  assert(ctx_map.find(my_id) != ctx_map.end());

  lk.unlock();
  io_context_t ctx = this->get_ctx();
  io_destroy(ctx);
  //  assert(ret == 0);
  lk.lock();
  ctx_map.erase(my_id);
  std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
  lk.unlock();
}

void LinuxAlignedFileReader::open(const std::string &fname) {
  int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
  this->file_desc = ::open(fname.c_str(), flags);
  // error checks
  assert(this->file_desc != -1);
  std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                  io_context_t &ctx, bool async) {
  assert(this->file_desc != -1);
  //#pragma omp critical
  //	std::cout << "thread: " << std::this_thread::get_id() << ", crtx: " <<
  // ctx
  //<< "\n";
  execute_io(ctx, this->file_desc, read_reqs);
}
