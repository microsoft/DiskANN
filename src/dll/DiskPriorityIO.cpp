#include <iostream>
#include <string>
#include <sstream>
#include <tchar.h>
#include "IDiskPriorityIO.h"
#include "DiskPriorityIO.h"

#include "utils.h"

#define SECTOR_LEN 4096
#define MAX_IO_DEPTH 64
#define NUM_IO_POLL_THREADS 4

namespace NSG {

  struct DiskAnnOverlapped : public OVERLAPPED {
    ANNIndex::AsyncReadRequest*          m_readRequest;
    ConcurrentQueue<DiskAnnOverlapped*>* m_overlappedQueue;

    DiskAnnOverlapped(ConcurrentQueue<DiskAnnOverlapped*>* overlappedQueue)
        : m_overlappedQueue(overlappedQueue), m_readRequest(nullptr) {
    }
  };

  void overlappedCompletionRoutine(DWORD errorCode, DWORD numBytesRead,
                                   LPOVERLAPPED pOverlapped) {
    DiskAnnOverlapped* pDiskAnnOS = ((DiskAnnOverlapped*) pOverlapped);
    pDiskAnnOS->m_overlappedQueue->push(pDiskAnnOS);

    bool ret = errorCode == ERROR_SUCCESS ? true : false;
    if (errorCode != ERROR_SUCCESS) {
      std::cerr << "Overlapped read request failed with error " << errorCode
                << std::endl;
    }
    pDiskAnnOS->m_readRequest->m_success = ret;
    pDiskAnnOS->m_readRequest->m_callback(ret);
  }

  DiskPriorityIO::DiskPriorityIO(ANNIndex::DiskIOScenario scenario)
      : IDiskPriorityIO(scenario), m_fileHandle(nullptr), m_iocp(nullptr),
        m_currentThreadId(0) {
  }

  DiskPriorityIO::~DiskPriorityIO() {
  }
  bool DiskPriorityIO::Initialize(const char* filePath,
                                  // Max read/write buffer size.
                                  unsigned __int32 maxIOSize,
                                  unsigned __int32 maxReadRetries,
                                  unsigned __int32 maxWriteRetries,
                                  unsigned __int16 threadPoolSize) {
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
      DiskAnnOverlapped* pOoverlapped =
          new DiskAnnOverlapped(&m_overlappedQueue);
      m_overlappedQueue.push(pOoverlapped);
    }

    for (int i = 0; i < NUM_IO_POLL_THREADS; i++) {
      m_ioPollingThreads.emplace_back(
          std::thread(std::bind(&DiskPriorityIO::listenIOCP, this)));
    }

    return true;
  }

  bool DiskPriorityIO::ReadFile(unsigned __int64 offset,
                                unsigned __int32 readSize, __int8* buffer) {
    return false;
  }
  bool DiskPriorityIO::ReadFileAsync(ANNIndex::AsyncReadRequest& readRequest) {
    DiskAnnOverlapped* os = m_overlappedQueue.pop();
    memset(os, 0, sizeof(DiskAnnOverlapped));

    os->m_overlappedQueue = &m_overlappedQueue;
    os->m_readRequest = &readRequest;

    os->Offset = readRequest.m_offset & 0xffffffff;
    os->OffsetHigh = (readRequest.m_offset >> 32);

    DWORD bytesRead = 0;
    BOOL  readSuccess = ::ReadFile(m_fileHandle, readRequest.m_buffer,
                                  readRequest.m_readSize, &bytesRead, os);

    if (!readSuccess && GetLastError() != ERROR_IO_PENDING) {
      m_overlappedQueue.push(os);
      readRequest.m_callback(false);
      return false;
    } else {
      return true;
    }
  }

  void DiskPriorityIO::ShutDown() {
    if (m_fileHandle != nullptr) {
      CloseHandle(m_fileHandle);
    }
    while (!m_overlappedQueue.empty()) {
      auto ptr = m_overlappedQueue.pop();
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

    while (true) {
      BOOL ret = ::GetQueuedCompletionStatus(this->m_iocp, &cBytesTransferred,
                                             &key, &os, INFINITE);
      std::cout << " Worker (" << GetCurrentThreadId() << ") Ret value: " << ret
                << std::endl;

      if (FALSE == ret || os == nullptr) {
        return;
      }
      if (ret == ERROR_IO_PENDING) {
        continue;
      }

      dOS = (DiskAnnOverlapped*) os;
      assert(cBytesTransferred == dOS->m_readRequest->m_readSize);

      auto& callback = dOS->m_readRequest->m_callback;
      dOS->m_overlappedQueue->push(dOS);
      callback(true);
    }
  }

}  // namespace NSG