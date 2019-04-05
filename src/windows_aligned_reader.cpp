#include "windows_aligned_reader.h"
#include <iostream>
#include "efanna2e/util.h"

#define SECTOR_LEN 4096

void WindowsAlignedReader::open(const std::string& fname) {
  filename = std::wstring(fname.begin(), fname.end());
  this->register_thread();
}

void WindowsAlignedReader::close() {
  for (auto& k_v : handle_map) {
    HANDLE fh = handle_map[k_v.first];
    CloseHandle(fh);
  }
}

void WindowsAlignedReader::register_thread() {
  std::unique_lock<std::mutex> lk(this->handle_mut);
  if (this->handle_map.find(std::this_thread::get_id()) != handle_map.end()) {
    std::cout << "Warning:: Duplicate registration for thread_id : "
              << std::this_thread::get_id() << "\n";
  }

  HANDLE fh = CreateFile(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                  OPEN_EXISTING,
                  FILE_ATTRIBUTE_READONLY | FILE_FLAG_NO_BUFFERING |
                      FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
                  NULL);
  if (fh == INVALID_HANDLE_VALUE) {
    std::cout << "Error opening " << filename.c_str()
              << " -- error=" << GetLastError() << std::endl;
  }
  this->handle_map.insert(std::make_pair(std::this_thread::get_id(), fh));
}

HANDLE WindowsAlignedReader::get_handle() {
  std::unique_lock<std::mutex> lk(this->handle_mut);
  if (handle_map.find(std::this_thread::get_id()) == handle_map.end()) {
    std::cerr << "unable to find HANDLE for thread_id : "
              << std::this_thread::get_id() << "\n";
    exit(-2);
  }
  HANDLE handle = handle_map[std::this_thread::get_id()];
  lk.unlock();
  return handle;
}

void WindowsAlignedReader::read(std::vector<AlignedRead>& read_reqs) {
  // execute each request sequentially
  HANDLE handle = get_handle();

  OVERLAPPED os;
  // std::cout << "::read -- " << read_reqs.size() << " requests\n";
  for (auto& req : read_reqs) {
    memset(&os, 0, sizeof(os));
    _u64  offset = req.offset;
    _u64  nbytes = req.len;
    char* read_buf = (char*) req.buf;
    assert(IS_ALIGNED(read_buf, SECTOR_LEN));
    assert(IS_ALIGNED(offset, SECTOR_LEN));
    assert(IS_ALIGNED(nbytes, SECTOR_LEN));

    // fill in OVERLAPPED struct
    os.Offset = offset & 0xffffffff;
    os.OffsetHigh = (offset >> 32);
    os.hEvent = NULL;

    DWORD nread = 0;
    BOOL  ret = ReadFile(handle, read_buf, nbytes, NULL, &os);
    if (ret == FALSE) {
      auto error = GetLastError();
      if (error != ERROR_IO_PENDING) {
        std::cerr << "Error queuing IO -- " << error << "\n";
      }
    } else {
      std::cerr << "Error queueing IO -- ReadFile returned TRUE\n";
    }
    BOOL res = GetOverlappedResult(handle, &os, &nread, TRUE);
    if (res == FALSE) {
      std::cerr << "Error code: " << GetLastError() << "\n";
    }
    // std::cout << "Read " << nread << " bytes at offset " << offset << "\n";
  }
}