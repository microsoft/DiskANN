#include "aligned_file_reader.h"
#ifdef __APPLE__

#include "apple_aligned_file_reader.h"
#include "utils.h"

#define SECTOR_LEN 4096

AppleAlignedFileReader::AppleAlignedFileReader()
{
    this->file_desc = -1;
}

AppleAlignedFileReader::~AppleAlignedFileReader()
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

IOContext &AppleAlignedFileReader::get_ctx()
{
    std::unique_lock<std::mutex> lk(this->ctx_mut);
    // perform checks only in DEBUG mode
    if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end())
    {
        std::cerr << "bad thread access; returning -1 as io_context_t" << std::endl;
        throw;
    }
    IOContext &ctx = ctx_map[std::this_thread::get_id()];
    lk.unlock();
    return ctx;
}

void AppleAlignedFileReader::register_thread()
{
    auto current_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    if (ctx_map.find(current_id) != ctx_map.end())
    {
        std::cerr << "multiple calls to register_thread from the same thread" << std::endl;
        throw;
    }

    IOContext ctx;
    auto x = *static_cast<unsigned int *>(static_cast<void *>(&current_id));
    ctx.queue = dispatch_queue_create(std::to_string(x).c_str(), DISPATCH_QUEUE_SERIAL);
    ctx.grp = dispatch_group_create();
    ctx.channel = // the cleanup is handled by deregister_all_threads for
                  // readability reasons
        dispatch_io_create(DISPATCH_IO_RANDOM, this->file_desc, ctx.queue,
                           ^(int error){
                           });
    if (ctx.channel == NULL)
        throw;
    this->ctx_map.insert(std::make_pair(std::this_thread::get_id(), ctx));
}

void AppleAlignedFileReader::deregister_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    assert(ctx_map.find(my_id) != ctx_map.end());
    lk.unlock();

    IOContext ctx = this->get_ctx();

    dispatch_io_close(ctx.channel, DISPATCH_IO_STOP);
    dispatch_release(ctx.channel);
    dispatch_release(ctx.grp);

    lk.lock();
    ctx_map.erase(my_id);
    std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
    lk.unlock();
}

void AppleAlignedFileReader::deregister_all_threads()
{
    std::unique_lock<std::mutex> lk(ctx_mut);
    for (auto x = ctx_map.begin(); x != ctx_map.end(); x++)
    {
        IOContext ctx = x.value();
        dispatch_io_close(ctx.channel, DISPATCH_IO_STOP);
        dispatch_release(ctx.channel);
        dispatch_release(ctx.grp);
    }
    std::cerr << "apple aligned reader: cleared all queues and channels" << std::endl;
    ctx_map.clear();
}

void AppleAlignedFileReader::open(const std::string &fname)
{
    int flags = O_RDONLY;
    this->file_desc = ::open(fname.c_str(), flags);
    if (this->file_desc == -1)
    {
        std::cerr << "Failed to open file in apple file reader" << std::endl;
        throw;
    }
}

void AppleAlignedFileReader::close()
{
    ::close(this->file_desc);
}

void AppleAlignedFileReader::read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async)
{
    using namespace std::chrono_literals;

    dispatch_io_t channel = ctx.channel;
    dispatch_queue_t q = ctx.queue;
    dispatch_group_t group = ctx.grp;

    // execute each request sequentially
    uint64_t n_reqs = read_reqs.size();
    uint64_t n_batches = ROUND_UP(n_reqs, MAX_IO_DEPTH) / MAX_IO_DEPTH;
    for (uint64_t i = 0; i < n_batches; i++)
    {
        // batch start/end
        uint64_t batch_start = MAX_IO_DEPTH * i;
        uint64_t batch_size = std::min((uint64_t)(n_reqs - batch_start), (uint64_t)MAX_IO_DEPTH);

        // fill OVERLAPPED and issue them
        for (uint64_t j = 0; j < batch_size; j++)
        {
            AlignedRead &req = read_reqs[batch_start + j];
            uint64_t offset = req.offset;
            uint64_t nbytes = req.len;
            char *read_buf = (char *)req.buf;
            assert(IS_ALIGNED(read_buf, SECTOR_LEN));
            assert(IS_ALIGNED(offset, SECTOR_LEN));
            assert(IS_ALIGNED(nbytes, SECTOR_LEN));

            dispatch_group_enter(group);
            dispatch_io_read(channel, offset, nbytes, q, ^(bool done, dispatch_data_t data, int error) {
              if (error)
              {
                  diskann::cerr << "error " << error << " when reading"
                                << "\n";
                  dispatch_group_leave(group);
                  return;
              }

              if (data && done)
              {
                  if (nbytes == dispatch_data_get_size(data))
                  {
                      dispatch_data_apply(
                          data, (dispatch_data_applier_t) ^ (dispatch_data_t region, size_t offset, const void *buffer,
                                                             size_t size) { memcpy(read_buf, buffer, size); });
                      dispatch_group_leave(group);
                      return;
                  }
              }
            });
        }
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    }
}

#endif
