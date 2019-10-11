#pragma once
#ifdef _WINDOWS

#include "Windows.h"
#include "ProcessThreadsApi.h"
#include "IDiskPriorityIO.h"

#include "concurrent_queue.h"

namespace NSG {

  struct DiskAnnOverlapped;

  class DiskPriorityIO : public ANNIndex::IDiskPriorityIO {
   public:
    DiskPriorityIO(ANNIndex::DiskIOScenario scenario);
    virtual ~DiskPriorityIO();
    virtual bool Initialize(const char* filePath,
                            // Max read/write buffer size.
                            unsigned __int32 maxIOSize = (1 << 20),
                            unsigned __int32 maxReadRetries = 2,
                            unsigned __int32 maxWriteRetries = 2,
                            unsigned __int16 threadPoolSize = 4);

    virtual bool ReadFile(unsigned __int64 offset, unsigned __int32 readSize,
                          __int8* buffer);
    virtual bool ReadFileAsync(ANNIndex::AsyncReadRequest& readRequest);

    virtual void ShutDown();

   protected:
    virtual void listenIOCP();

   private:
    const char*                         m_fileName;
    HANDLE                              m_fileHandle;
    HANDLE                              m_iocp;
    DWORD                               m_currentThreadId;
    ConcurrentQueue<DiskAnnOverlapped*> m_overlappedQueue;
    std::vector<std::thread>            m_ioPollingThreads;
  };
}  // namespace NSG

#endif
