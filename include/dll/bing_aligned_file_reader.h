#pragma once
#ifdef _WINDOWS
#ifdef USE_BING_INFRA
#include "dll/IDiskPriorityIO.h"
#include "aligned_file_reader.h"

namespace diskann {
  class BingAlignedFileReader : public AlignedFileReader {
   private:
    ANNIndex::IDiskPriorityIO *m_pReader;
    std::string                m_filename;

   public:
    BingAlignedFileReader();
    virtual ~BingAlignedFileReader();

    virtual void register_thread();
    virtual void deregister_thread();

    virtual IOContext &get_ctx();

    // Open & close ops
    // Blocking calls
    virtual void open(const std::string &fname);
    virtual void close();

    // process batch of aligned requests in parallel
    // NOTE :: blocking call for the calling thread, but can thread-safe
    virtual void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx);

    void callback(std::shared_ptr<std::atomic<int>> pCounter, bool result);
  };
}  // namespace NSG
#endif
#endif
