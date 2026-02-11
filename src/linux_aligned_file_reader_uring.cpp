// Copyright (c) KIOXIA Corporation. All rights reserved.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "linux_aligned_file_reader_uring.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include "tsl/robin_map.h"
#include "utils.h"
#define QUEUE_DEPTH 1024

void execute_io(IOContext ctx, int fd, std::vector<AlignedRead> &read_reqs, uint64_t n_retries = 0)
{
#ifdef DEBUG
    for (auto &req : read_reqs)
    {
        assert(IS_ALIGNED(req.len, 512));
        // std::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
        assert(IS_ALIGNED(req.offset, 512));
        assert(IS_ALIGNED(req.buf, 512));
        // assert(malloc_usable_size(req.buf) >= req.len);
    }
#endif

	// break-up requests into chunks of size QUEUE_DEPTH each
    uint64_t n_iters = ROUND_UP(read_reqs.size(), QUEUE_DEPTH) / QUEUE_DEPTH;
    for (uint64_t iter = 0; iter < n_iters; iter++)
    {
        uint64_t n_ops = std::min((uint64_t)read_reqs.size() - (iter * QUEUE_DEPTH), (uint64_t)QUEUE_DEPTH);

		// create n_ops io requests
        for (uint64_t j = 0; j < n_ops; j++)
        {
			struct io_uring_sqe *sqe = io_uring_get_sqe(ctx);
			if (!sqe){
                std::cerr << "io_uring_get_sqe() failed; ernno=" << errno;
                std::cout << "ctx: " << ctx << "\n";
                exit(-1);
			}
			io_uring_prep_read(sqe, fd, read_reqs[j + iter * QUEUE_DEPTH].buf, read_reqs[j + iter * QUEUE_DEPTH].len,
                          read_reqs[j + iter * QUEUE_DEPTH].offset);
		}

		uint64_t n_tries = 0;
        while (n_tries <= n_retries)
        {
			// send io requests here
            int64_t ret = io_uring_submit(ctx);
            // if requests didn't get accepted
            if (ret != (int64_t)n_ops)
            {
                std::cerr << "io_uring_submit() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno
                          << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
                std::cout << "ctx: " << ctx << "\n";
                exit(-1);
			}
            else
            {
				struct io_uring_cqe *cqes[QUEUE_DEPTH];
				unsigned int count = 0;
                // wait on io_uring
                int64_t ret = io_uring_wait_cqes(ctx, cqes, n_ops, nullptr, nullptr);

                // if requests didn't complete
                if (ret != 0)
                {
                    std::cerr << "io_uring_waite_cqes() failed; returned " << ret << ", expected=" << n_ops
                              << ", ernno=" << errno << "=" << ::strerror(-ret) << ", try #" << n_tries + 1 << std::endl;
                    exit(-1);
                }
                else
                {
					io_uring_cq_advance(ctx, n_ops);
                    break;
                }
            }
			n_tries++;
		}
    }
}

LinuxAlignedFileReader::LinuxAlignedFileReader()
{
    this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader()
{
    int64_t ret;
    // check to make sure file_desc is closed
    ret = ::fcntl(this->file_desc, F_GETFD);
    if (ret == -1)
    {
        if (errno != EBADF)
        {
            std::cerr << "close() not called" << std::endl;
            // close file desc
            ret = ::close(this->file_desc);
            // error checks
            if (ret == -1)
            {
                std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno)
                          << std::endl;
            }
        }
    }
}

IOContext &LinuxAlignedFileReader::get_ctx()
{
    std::unique_lock<std::mutex> lk(ctx_mut);
    // perform checks only in DEBUG mode
    if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end())
    {
        std::cerr << "bad thread access; returning -1 as io_uring" << std::endl;
        return this->bad_ctx;
    }
    else
    {
        return ctx_map[std::this_thread::get_id()];
    }
}

void LinuxAlignedFileReader::register_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    if (ctx_map.find(my_id) != ctx_map.end())
    {
        std::cerr << "multiple calls to register_thread from the same thread" << std::endl;
        return;
    }
    IOContext ctx = (struct io_uring *)malloc(sizeof(struct io_uring));
	int ret = io_uring_queue_init(QUEUE_DEPTH, ctx, 0);
    if (ret != 0)
    {
        lk.unlock();
        if (ret == -EAGAIN)
        {
            std::cerr << "io_setup() failed with EAGAIN" << std::endl;
        }
        else
        {
            std::cerr << "io_setup() failed; returned " << ret << ": " << ::strerror(-ret) << std::endl;
        }
    }
    else
    {
        diskann::cout << "allocating ctx: " << ctx << " to thread-id:" << my_id << std::endl;
        ctx_map[my_id] = ctx;
    }
    lk.unlock();
}

void LinuxAlignedFileReader::deregister_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    assert(ctx_map.find(my_id) != ctx_map.end());

    lk.unlock();
	IOContext ctx = this->get_ctx();
	io_uring_queue_exit(ctx);
	free(ctx);
    //  assert(ret == 0);
    lk.lock();
    ctx_map.erase(my_id);
    std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
    lk.unlock();
}

void LinuxAlignedFileReader::deregister_all_threads()
{
    std::unique_lock<std::mutex> lk(ctx_mut);
    for (auto x = ctx_map.begin(); x != ctx_map.end(); x++)
    {
		IOContext ctx = x.value();
		io_uring_queue_exit(ctx);
		free(ctx);
        //  assert(ret == 0);
        //  lk.lock();
        //  ctx_map.erase(my_id);
        //  std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
    }
    ctx_map.clear();
    //  lk.unlock();
}

void LinuxAlignedFileReader::open(const std::string &fname)
{
    int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
    this->file_desc = ::open(fname.c_str(), flags);
    // error checks
    assert(this->file_desc != -1);
    std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close()
{
    // check to make sure file_desc is closed
    ::fcntl(this->file_desc, F_GETFD);
    //  assert(ret != -1);

    ::close(this->file_desc);
    //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async)
{
    if (async == true)
    {
        diskann::cout << "Async currently not supported in linux." << std::endl;
    }
    assert(this->file_desc != -1);
    execute_io(ctx, this->file_desc, read_reqs);
}
