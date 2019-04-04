#include "windows_aligned_reader.h"
#include "efanna2e/util.h"
#include <iostream>

#define SECTOR_LEN 4096

void WindowsAlignedReader::open(const std::string& fname) {
  std::wstring wname(fname.begin(), fname.end());
  fh = CreateFile(wname.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                         OPEN_EXISTING,
                         FILE_ATTRIBUTE_READONLY | FILE_FLAG_NO_BUFFERING |
                             FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
                         NULL);
  if (fh == INVALID_HANDLE_VALUE) {
    std::cout << "Error opening " << fname.c_str() << " -- error=" << GetLastError()
              << std::endl;
  }
}

void WindowsAlignedReader::close() {
  CloseHandle(fh);
}

void WindowsAlignedReader::read(std::vector<AlignedRead>& read_reqs) {
	// execute each request sequentially
  OVERLAPPED os;
  // std::cout << "::read -- " << read_reqs.size() << " requests\n";
  for (auto& req : read_reqs) {
    memset(&os, 0, sizeof(os));
    _u64 offset = req.offset;
    _u64 nbytes = req.len;
    char* read_buf = (char*)req.buf;
    assert(IS_ALIGNED(read_buf, SECTOR_LEN));
    assert(IS_ALIGNED(offset, SECTOR_LEN));
    assert(IS_ALIGNED(nbytes, SECTOR_LEN));

	// fill in OVERLAPPED struct
    os.Offset = offset & 0xffffffff;
    os.OffsetHigh = (offset >> 32);
    os.hEvent = NULL;

    DWORD nread = 0;
    BOOL  ret = ReadFile(fh, read_buf, nbytes, NULL, &os);
    if (ret == FALSE) {
      auto error = GetLastError();
      if (error != ERROR_IO_PENDING) {
        std::cerr << "Error queuing IO -- " << error << "\n";
      }
    } else {
      std::cerr << "Error queueing IO -- ReadFile returned TRUE\n";
    }
    BOOL res = GetOverlappedResult(fh, &os, &nread, TRUE);
    if (res == FALSE) {
      std::cerr << "Error code: " << GetLastError() << "\n";
    }
    // std::cout << "Read " << nread << " bytes at offset " << offset << "\n";
  }
}