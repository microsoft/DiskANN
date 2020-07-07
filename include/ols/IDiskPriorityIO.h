#pragma once

#include <functional>

namespace ANNIndex {

  // All the scenarios defined in PredefinedDiskPriorities.h
  enum DiskIOScenario {
    DIS_BulkRead = 0,
    DIS_UserRead,
    DIS_HighPriorityUserRead,
    DIS_Count
  };

  struct AsyncReadRequest {
    unsigned __int64          m_offset;
    __int8*                   m_buffer;
    std::function<void(bool)> m_callback;
    unsigned __int32          m_readSize;
    // Carry items like counter for callback to process.
    void* m_payload;

    bool m_success;

    AsyncReadRequest()
        : m_offset(0), m_buffer(nullptr), m_readSize(0), m_payload(nullptr),
          m_success(false) {
    }
  };

  // Parameter names for disck priority configuration.
  static const char* paramEnableDiskPriority = "enableDiskPriority";

  static const char* paramMaxIOSize = "maxIOSize";
  static const char* paramMaxReadRetries = "maxReadRetries";
  static const char* paramMaxWriteRetries = "maxWriteRetries";
  static const char* paramThreadPoolSize = "threadPoolSize";

  class IDiskPriorityIO {
   public:
    IDiskPriorityIO(DiskIOScenario scenario) {
    }

    virtual ~IDiskPriorityIO() {
    }

    virtual bool Initialize(const char* filePath,
                            // Max read/write buffer size.
                            unsigned __int32 maxIOSize = (1 << 20),
                            unsigned __int32 maxReadRetries = 2,
                            unsigned __int32 maxWriteRetries = 2,
                            unsigned __int16 threadPoolSize = 4) = 0;

    virtual unsigned __int32 ReadFile(unsigned __int64 offset,
                                      unsigned __int32 readSize,
                                      __int8*          buffer) = 0;
    // TODO: need to figure out why ReadDataAsyncWithoutThreadPool didn't
    // callback.
    virtual bool ReadFileAsync(AsyncReadRequest& readRequest) = 0;

    virtual void ShutDown() = 0;
  };
}
