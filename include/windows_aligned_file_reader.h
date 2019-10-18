#pragma once
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
#include <Windows.h>
#include <fcntl.h>
#include <malloc.h>
#include <minwinbase.h>

#include <cstdio>
#include <mutex>
#include <thread>
#include "aligned_file_reader.h"
#include "tsl/robin_map.h"
#include "utils.h"



class WindowsAlignedFileReader : public AlignedFileReader {
 private:
  std::wstring m_filename;

  protected:
  virtual IOContext createContext();

 public:
  WindowsAlignedFileReader(){};
  virtual ~WindowsAlignedFileReader(){};

    // Open & close ops
  // Blocking calls
  virtual void open(const std::string &fname);
  virtual void close();


  virtual void register_thread();
  virtual void deregister_thread() {
  }
  virtual IOContext &get_ctx();


  // process batch of aligned requests in parallel
  // NOTE :: blocking call for the calling thread, but can thread-safe
  virtual void read(std::vector<AlignedRead> &read_reqs, IOContext& ctx);
};
#endif //USE_BING_INFRA
#endif	//_WINDOWS
