#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>
#include <libaio.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <chrono>
#include <functional>

#define MAX_EVENTS 1024

#define ROUND_UP(X, Y) \
  ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define TEST_FILE "/tmp/test_io"

struct AlignedRead {
  uint64_t offset;  // where to read from
  uint64_t len;     // how much to read
  void*    buf;     // where to read into

  AlignedRead() : offset(0), len(0), buf(nullptr) {
  }

  AlignedRead(uint64_t offset, uint64_t len, void* buf)
      : offset(offset), len(len), buf(buf) {
  }
};

namespace {

typedef struct io_event io_event_t;
typedef struct iocb     iocb_t;

void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs,
              uint64_t n_retries = 0) {

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

  void execute_io_old(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs,
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
      //

      for (uint64_t i = 0; i < n_ops; i++) {
        cbs[i] = cb.data() + i;
      }

      uint64_t n_tries = 0;
      while (n_tries <= n_retries) {
        // issue reads
        int64_t ret = io_submit(ctx, (int64_t) n_ops, cbs.data());
        // if requests didn't get accepted
        if (ret != (int64_t) n_ops) {
          std::cerr << "io_submit() failed; returned " << ret
                    << ", expected=" << n_ops << ", ernno=" << errno << "="
                    << ::strerror(-ret) << ", try #" << n_tries + 1;
          std::cout << "ctx: " << ctx << "\n";
          exit(-1);
        } else {
          // wait on io_getevents
          ret = io_getevents(ctx, (int64_t) n_ops, (int64_t) n_ops, evts.data(),
                             nullptr);
          // if requests didn't complete
          if (ret != (int64_t) n_ops) {
            std::cerr << "io_getevents() failed; returned " << ret
                      << ", expected=" << n_ops << ", ernno=" << errno << "="
                      << ::strerror(-ret) << ", try #" << n_tries + 1;
            exit(-1);
          } else {
            break;
          }
        }
      }
      // disabled since req.buf could be an offset into another buf
      /*
      for (auto &req : read_reqs) {
        // corruption check
        assert(malloc_usable_size(req.buf) >= req.len);
      }
      */
    }

    /*
    for(unsigned i=0;i<64;i++){
      std::cout << *((unsigned*)read_reqs[0].buf + i) << " ";
    }
    std::cout << std::endl;*/
  }

void
count_time(const std::function<void(void)>& func, std::string name) {
  auto begin = std::chrono::steady_clock::now();
  func();
  auto end = std::chrono::steady_clock::now();
  std::cout << "time cost of " << name << ": "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
            << std::endl;
}
} // namespace

int
main(int argc, char* argv[]) {
  size_t num = 1024 * 128;
  size_t l = 4096;

  auto fd = open(TEST_FILE, O_DIRECT | O_RDONLY);
  assert(fd > 0);

  std::vector<AlignedRead> reqs(num);

  for (size_t i = 0; i < num; i++) {
    reqs[i].offset = 0;
    reqs[i].len = l;
    auto r = posix_memalign(&reqs[i].buf, 512, l);
    assert(!r);
  }

  int num_jobs = 4;
  auto num_per_job = num / num_jobs;
  std::vector<io_context_t> ctxs(num_jobs, 0);
  std::vector<std::vector<AlignedRead>> sliced_reqs(num_jobs);
  for (auto i = 0; i < num_jobs; i++) {
    auto r = io_setup(MAX_EVENTS, &ctxs[i]);
    assert(!r);
    sliced_reqs[i] = std::vector<AlignedRead>(reqs.begin() + i * num_per_job,
                                              reqs.begin() + (i + 1) * num_per_job);
  }

  auto old_way = [&] {
      execute_io_old(ctxs[0], fd, reqs);
  };
  count_time(std::bind(old_way), "old way");

  auto fio_way = [&] {
      execute_io(ctxs[0], fd, reqs);
  };
  count_time(std::bind(fio_way), "fio way");

  auto fio_way_4_jobs = [&] {
      std::vector<std::thread> threads;
      for (auto i = 0; i < num_jobs; i++) {
          auto th = std::thread(std::bind(execute_io, ctxs[i], fd, sliced_reqs[i], 0));
          threads.push_back(std::move(th));
      }
      for (auto i = 0; i < num_jobs; i++) {
          threads[i].join();
      }
  };
  count_time(std::bind(fio_way_4_jobs), "fio way (num-jobs=4)");

  return 0;
}

