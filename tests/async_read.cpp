#include <liburing.h>
#include <fcntl.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "linux_async_aligned_file_reader.h"

int main() {
  LinuxAsyncAlignedFileReader reader;
  struct io_uring             ring;
  io_uring_queue_init(32, &ring, 0);

  // struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
  // int   fd = open("/data01/home/mengshangqi.123/DiskANN/build/tests/hello",
  //                 O_WRONLY | O_CREAT);
  char* x = "hello world";
  // {
  //   struct iovec iov = {
  //       .iov_base = x,
  //       .iov_len = strlen("Hello world"),
  //   };
  //   io_uring_prep_writev(sqe, fd, &iov, 1, 0);
  // }
  // io_uring_submit(&ring);

  // struct io_uring_cqe *cqe;

  // for (;;) {
  //   io_uring_peek_cqe(&ring, &cqe);
  //   if (!cqe) {
  //     puts("Waiting...");
  //     // accept 新连接，做其他事
  //   } else {
  //     puts("Finished.");
  //     break;
  //   }
  // }
  // io_uring_cqe_seen(&ring, cqe);
  // io_uring_queue_exit(&ring);
  char  str[1000];
  char* buf = reinterpret_cast<char*>(aligned_alloc(512, 1024));
  reader.open("/data01/home/mengshangqi.123/DiskANN/build/tests/hello");
  reader.submit_io(buf, strlen(x), 0, ring);
  while (reader.peek_io(ring) < 0) {
    std::cout << "------------- wait" << std::endl;
  }

  std::cout << buf << std::endl;
  free(buf);
  return 0;
}
