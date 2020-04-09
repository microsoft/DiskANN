#include "DiskPriorityIOInterface.h"
#include "DiskPriorityIO.h"

#include <sstream>
#include <iostream>

namespace diskann {

  DiskPriorityIOInterface::DiskPriorityIOInterface(
      ANNIndex::DiskIOScenario scenario)
      : ANNIndex::IDiskPriorityIO(scenario) {
    m_diskIOScenario = scenario;
  }
  DiskPriorityIOInterface::~DiskPriorityIOInterface() {
  }

  bool DiskPriorityIOInterface::Initialize(const char* filePath,
                                           // Max read/write buffer size.
                                           unsigned __int32 maxIOSize,
                                           unsigned __int32 maxReadRetries,
                                           unsigned __int32 maxWriteRetries,
                                           unsigned __int16 threadPoolSize) {
    auto                        threadId = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_diskPriorityIOs.find(threadId) != m_diskPriorityIOs.end()) {
      std::shared_ptr<IDiskPriorityIO> diskIOPtr(
          new DiskPriorityIO(m_diskIOScenario));
      m_diskPriorityIOs.insert(std::make_pair(threadId, diskIOPtr));
    } else {
      std::cerr << "Duplicate thread registration in "
                   "DPIOInterface::Initialize(). ID: "
                << threadId;
    }
    return true;
  }

  unsigned __int32 DiskPriorityIOInterface::ReadFile(unsigned __int64 offset,
                                                     unsigned __int32 readSize,
                                                     __int8*          buffer) {
    // Just as a sanity check.
    if (m_diskPriorityIOs.find(std::this_thread::get_id()) ==
        m_diskPriorityIOs.end()) {
      std::stringstream str;
      str << " Unregistered thread: " << std::this_thread::get_id()
          << " encountered in DPIOInterface::ReadFile";
      throw std::exception(str.str().c_str());
    }

    return 0;
  }

  bool DiskPriorityIOInterface::ReadFileAsync(
      ANNIndex::AsyncReadRequest& readRequest) {
    auto threadId = std::this_thread::get_id();
    auto iter = m_diskPriorityIOs.find(threadId);

    if (iter != m_diskPriorityIOs.end()) {
      return iter->second->ReadFileAsync(readRequest);
    } else {
      std::stringstream stream;
      stream << "Unregistered thread: " << threadId
             << " encountered in DPIOInterface::ReadFileAsync()";
      throw std::exception(stream.str().c_str());
    }
  }

  void DiskPriorityIOInterface::ShutDown() {
    // Note that this shutdown is called from the executing threads
    // So like everything else, we'll forward it to the correct
    // thread.
    auto threadId = std::this_thread::get_id();
    auto iter = m_diskPriorityIOs.find(threadId);
    if (iter != m_diskPriorityIOs.end()) {
      return iter->second->ShutDown();
    } else {
      std::stringstream stream;
      stream << "Unregistered thread: " << threadId
             << " encountered in DPIOInterface::ShutDown()";
      throw std::exception(stream.str().c_str());
    }
  }
}  // namespace diskann