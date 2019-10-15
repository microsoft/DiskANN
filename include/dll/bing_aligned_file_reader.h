#pragma once
#ifdef _WINDOWS
#ifdef BING_INFRA
#include "dll/IDiskPriorityIO.h"
#include "windows_aligned_file_reader.h"


class BingAlignedFileReader : public WindowsAlignedFileReader {
 private:
  ANNIndex::IDiskPriorityIO *m_pReader;

 public:
  BingAlignedFileReader(){};
  virtual ~BingAlignedFileReader(){};

  virtual void register_thread();
  virtual void deregister_thread() {
  }
  virtual IOContext &get_ctx();

  // Open & close ops
  // Blocking calls
  virtual void open(const std::string &fname);
  virtual void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call for the calling thread, but can thread-safe
  virtual void read(std::vector<AlignedRead> &read_reqs, IOContext ctx);
};
#endif
#endif
