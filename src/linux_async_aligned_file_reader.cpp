#include "linux_async_aligned_file_reader.h"

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include "aligned_file_reader.h"
#include "tsl/robin_map.h"
#include "utils.h"
#define MAX_EVENTS 1024

LinuxAsyncAlignedFileReader::LinuxAsyncAlignedFileReader() {
  this->file_desc = -1;
}

LinuxAsyncAlignedFileReader::~LinuxAsyncAlignedFileReader() {
  std::cout << "msq ~LinuxAsyncAlignedFileReader" << std::endl;
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

IOUring &LinuxAsyncAlignedFileReader::get_ring() {
  std::cout << "msq LinuxAsyncAlignedFileReader get_ring" << std::endl;
  std::unique_lock<std::mutex> lk(ctx_mut);
  // perform checks only in DEBUG mode
  if (ring_map.find(std::this_thread::get_id()) == ring_map.end()) {
    std::cerr << "bad thread access; returning -1 as io_context_t" << std::endl;
    exit(-1);
  }
  return ring_map[std::this_thread::get_id()];
}

void LinuxAsyncAlignedFileReader::register_thread() {
  std::cout << "msq LinuxAsyncAlignedFileReader register_thread" << std::endl;
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  if (ring_map.find(my_id) != ring_map.end()) {
    std::cerr << "multiple calls to register_thread from the same thread"
              << std::endl;
    return;
  }
  IOUring ring;
  auto    ret = io_uring_queue_init(MAX_EVENTS, &ring, 0);
  if (ret != 0) {
    lk.unlock();
    assert(errno != EAGAIN);
    assert(errno != ENOMEM);
    std::cerr << "io_setup() failed; returned " << ret << ", errno=" << errno
              << ":" << ::strerror(errno) << std::endl;
  } else {
    diskann::cout << "allocating ring: "
                  << " to thread-id:" << my_id << std::endl;
    ring_map[my_id] = ring;
  }
  lk.unlock();
}

void LinuxAsyncAlignedFileReader::deregister_thread() {
  std::cout << "msq LinuxAsyncAlignedFileReader deregister_thread" << std::endl;
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  assert(ring_map.find(my_id) != ring_map.end());

  lk.unlock();
  auto &ring = this->get_ring();
  io_uring_queue_exit(&ring);
  //  assert(ret == 0);
  lk.lock();
  ring_map.erase(my_id);
  std::cerr << "returned ring from thread-id:" << my_id << std::endl;
  lk.unlock();
}

void LinuxAsyncAlignedFileReader::deregister_all_threads() {
  std::cout << "msq LinuxAsyncAlignedFileReader deregister_all_threads"
            << std::endl;
  std::unique_lock<std::mutex> lk(ctx_mut);
  for (auto x = ring_map.begin(); x != ring_map.end(); x++) {
    auto &ring = x.value();
    io_uring_queue_exit(&ring);
  }
  ring_map.clear();
  //  lk.unlock();
}

void LinuxAsyncAlignedFileReader::open(const std::string &fname) {
  std::cout << "msq LinuxAsyncAlignedFileReader open" << std::endl;
  std::cout << "msq file name: " << fname << std::endl;
  // O_DIRECT | O_RDONLY | O_LARGEFILE;
  int flags = O_RDONLY | O_LARGEFILE;
  this->file_desc = ::open(fname.c_str(), flags);
  // error checks
  assert(this->file_desc != -1);
  std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAsyncAlignedFileReader::close() {
  std::cout << "msq LinuxAsyncAlignedFileReader close" << std::endl;
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

bool LinuxAsyncAlignedFileReader::submit_io(std::vector<AlignedRead> &read_reqs,
                                            IOUring                  &ring) {
  // std::cout << "msq LinuxAsyncAlignedFileReader submit_io" << std::endl;
  for (size_t i = 0; i < read_reqs.size(); i++) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    sqe->user_data = i;
    struct iovec iov = {
        .iov_base = read_reqs[i].buf,
        .iov_len = read_reqs[i].len,
    };
    // 添加到ring中
    io_uring_prep_readv(sqe, file_desc, &iov, 1, read_reqs[i].offset);
  }
  // 提交一批
  int ret = io_uring_submit(&ring);
  if (read_reqs.size() != ret) {
    std::cout << "msq submit io failed" << std::endl;
    return false;
  }
  return true;
}

bool LinuxAsyncAlignedFileReader::submit_io(void *buf, size_t len,
                                            size_t offset, IOUring &ring) {
  std::cout << "msq LinuxAsyncAlignedFileReader submit_io" << std::endl;
  struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
  sqe->user_data = 0;
  struct iovec iov = {
      .iov_base = buf,
      .iov_len = len,
  };
  // 添加到ring中
  io_uring_prep_readv(sqe, file_desc, &iov, 1, offset);
  // 提交一批
  int ret = io_uring_submit(&ring);
  if (1 != ret) {
    std::cout << "msq submit io failed" << std::endl;
    return false;
  }
  return true;
}

size_t LinuxAsyncAlignedFileReader::peek_io(IOUring &ring) {
  // std::cout << "msq LinuxAsyncAlignedFileReader peek_io" << std::endl;
  struct io_uring_cqe *cqe = nullptr;
  io_uring_peek_cqe(&ring, &cqe);
  size_t ret = -1;
  if (cqe) {
    ret = cqe->user_data;
    io_uring_cqe_seen(&ring, cqe);
  }
  return ret;
}