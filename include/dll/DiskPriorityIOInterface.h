#pragma once
#include <thread>
#include <mutex>
#include "tsl/robin_map.h"
#include "windows_customizations.h"
#include "dll/IDiskPriorityIO.h"

namespace diskann {
  class DiskPriorityIOInterface : public ANNIndex::IDiskPriorityIO {
   public:
    DISKANN_DLLEXPORT DiskPriorityIOInterface(
        ANNIndex::DiskIOScenario scenario =
            ANNIndex::DiskIOScenario::DIS_HighPriorityUserRead);
    DISKANN_DLLEXPORT virtual ~DiskPriorityIOInterface();
    virtual bool Initialize(const char* filePath,
                            // Max read/write buffer size.
                            unsigned __int32 maxIOSize = (1 << 20),
                            unsigned __int32 maxReadRetries = 2,
                            unsigned __int32 maxWriteRetries = 2,
                            unsigned __int16 threadPoolSize = 4);

    virtual unsigned __int32 ReadFile(unsigned __int64 offset,
                                      unsigned __int32 readSize,
                                      __int8*          buffer);
    virtual bool ReadFileAsync(ANNIndex::AsyncReadRequest& readRequest);

    virtual void ShutDown();

   protected:
   private:
    ANNIndex::DiskIOScenario m_diskIOScenario;
    std::mutex               m_mutex;
    tsl::robin_map<std::thread::id, std::shared_ptr<IDiskPriorityIO>>
        m_diskPriorityIOs;
  };
}  // namespace diskann
