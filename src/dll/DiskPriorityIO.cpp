#include <iostream>
#include <string>
#include <sstream>
#include <tchar.h>
#include "IDiskPriorityIO.h"
#include "DiskPriorityIO.h"

#include "utils.h"

#define SECTOR_LEN 4096
#define MAX_IO_DEPTH 128
#define NUM_IO_POLL_THREADS 1

namespace diskann {

  struct DiskAnnOverlapped : public OVERLAPPED {
    std::function<void(bool)> m_callback;
  };

  DiskPriorityIO::DiskPriorityIO(ANNIndex::DiskIOScenario scenario)
      : IDiskPriorityIO(scenario), m_fileHandle(nullptr), m_iocp(nullptr),
        m_currentThreadId(0), m_stopPolling(false),
        m_overlappedQueue(2 * MAX_IO_DEPTH), m_fileName("") {
  }

  DiskPriorityIO::~DiskPriorityIO() {
    std::cout << "In DiskPriorityIO destructor" << std::endl;
  }

  bool DiskPriorityIO::Initialize(const char* filePath,
                                  // Max read/write buffer size.
                                  unsigned __int32 maxIOSize,
                                  unsigned __int32 maxReadRetries,
                                  unsigned __int32 maxWriteRetries,
                                  unsigned __int16 threadPoolSize) {
    m_fileName = filePath;
    m_fileHandle = CreateFileA(
        filePath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
        FILE_ATTRIBUTE_READONLY | FILE_FLAG_NO_BUFFERING |
            FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
        NULL);

    if (m_fileHandle == INVALID_HANDLE_VALUE) {
      std::stringstream msgStream;
      msgStream << "Failed to open file: " << filePath
                << ". Error code: " << GetLastError();
      std::cerr << msgStream.str() << std::endl;
      throw std::exception(msgStream.str().c_str());
    }

    m_iocp =
        CreateIoCompletionPort(m_fileHandle, nullptr, 0, NUM_IO_POLL_THREADS);

    for (int i = 0; i < MAX_IO_DEPTH * 2; i++) {
      DiskAnnOverlapped* pOverlapped = new DiskAnnOverlapped();
      m_overlappedQueue.push(pOverlapped);
    }

    for (int i = 0; i < NUM_IO_POLL_THREADS; i++) {
      m_ioPollingThreads.emplace_back(
          std::thread(std::bind(&DiskPriorityIO::listenIOCP, this)));
    }

    return true;
  }

  unsigned __int32 DiskPriorityIO::ReadFile(unsigned __int64 offset,
                                            unsigned __int32 readSize,
                                            __int8*          buffer) {
    return 0;
  }
  bool DiskPriorityIO::ReadFileAsync(ANNIndex::AsyncReadRequest& readRequest) {
    DiskAnnOverlapped* os = nullptr;
    m_overlappedQueue.pop(os);
    if (os == nullptr) {
      os = new DiskAnnOverlapped();
    }

    os->OffsetHigh = (readRequest.m_offset >> 32);
    os->Offset = readRequest.m_offset & 0xffffffff;
    os->hEvent = nullptr;
    os->Internal = 0;
    os->InternalHigh = 0;
    os->m_callback = readRequest.m_callback;

    assert(os->Internal == 0);
    assert(os->InternalHigh == 0);
    assert(os->hEvent == nullptr);

    BOOL readSuccess = ::ReadFile(m_fileHandle, readRequest.m_buffer,
                                  readRequest.m_readSize, nullptr, os);

    if (!readSuccess && GetLastError() != ERROR_IO_PENDING) {
      m_overlappedQueue.push(os);
      std::cerr << "Failed to read file: " << m_fileName
                << " Error: " << std::hex << GetLastError() << std::endl;
      readRequest.m_callback(false);
      return false;
    } else {
      return true;
    }
  }

  void DiskPriorityIO::ShutDown() {
    m_stopPolling = true;
    for (auto& thrd : m_ioPollingThreads) {
      if (thrd.joinable()) {
        std::cout << "Trying to join thread " << thrd.get_id() << std::endl;
        thrd.join();
        std::cout << "Thread " << thrd.get_id() << " joined." << std::endl;
      }
    }
    if (m_fileHandle != nullptr) {
      CloseHandle(m_fileHandle);
    }

    DiskAnnOverlapped* ptr = nullptr;
    while (!m_overlappedQueue.empty()) {
      m_overlappedQueue.pop(ptr);
      if (ptr != nullptr) {
        delete ptr;
      }
    }
    if (m_iocp != nullptr) {
      CloseHandle(m_iocp);
    }
  }

  void DiskPriorityIO::listenIOCP() {
    DWORD              cBytesTransferred;
    ULONG_PTR          key;
    OVERLAPPED*        os;
    DiskAnnOverlapped* dOS;

    int i = 0;
    while (true) {
      if (m_stopPolling) {
        break;
      }

      BOOL ret = ::GetQueuedCompletionStatus(this->m_iocp, &cBytesTransferred,
                                             &key, &os, 2);
      i++;

      dOS = (DiskAnnOverlapped*) os;
      auto& callback = dOS->m_callback;

      if (ret == ERROR_IO_PENDING) {
        continue;
      }

      if (FALSE == ret) {
        if (os != nullptr) {
          m_overlappedQueue.push(dOS);
          callback(false);
        } else {
          // if os == nullptr, it means nothing was dequed, so we continue
          // loopin'
        }
      } else {
        m_overlappedQueue.push(dOS);
        callback(true);
      }
    }
  }

}  // namespace NSG