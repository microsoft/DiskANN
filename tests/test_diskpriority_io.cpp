#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
#include "DiskPriorityIO.h"

#define NUM_THREADS 4
#define NUM_CHUNKS 16

std::string getTempFileName() {
  char tempPath[256];
  char tempFileName[MAX_PATH];

  GetTempPathA(256, tempPath);
  GetTempFileNameA(tempPath, "KNN", 0, tempFileName);

  return std::string(tempFileName);
}

void saveContents(const char* fileName, char* pBytes, int count) {
  std::fstream file;
  file.open(fileName, std::ios::binary | std::ios::out);
  file.write(pBytes, count);
  file.flush();
  file.close();
}

void processData(bool result) {
  // What do we do here?
  std::cout << result << std::endl;
}

int main(int argc, char** argv) {
  size_t fileSize = ((size_t) NUM_CHUNKS) * 1024 * 1024;
  char*  pByte = (char*) malloc(fileSize);
  char*  readVec = (char*) malloc(fileSize);
  memset(readVec, 0, fileSize);

  for (int i = 0; i < fileSize; i++) {
    *(pByte + i) = (char) (rand() % 8);
  }

  std::string s = getTempFileName();
  saveContents(s.c_str(), pByte, fileSize);

  std::cout << "Saved file of size " << fileSize << " to " << s << std::endl;

  int batchSize = fileSize / NUM_THREADS;

  for (int i = 0; i < NUM_THREADS; i++) {
    NSG::DiskPriorityIO iohandler(
        ANNIndex::DiskIOScenario::DIS_HighPriorityUserRead);
    iohandler.Initialize(s.c_str());

    HANDLE events[NUM_CHUNKS];
    int    chunkSize = batchSize / NUM_CHUNKS;

    std::vector<ANNIndex::AsyncReadRequest> requests;
    for (int j = 0; j < NUM_CHUNKS; j++) {
      ANNIndex::AsyncReadRequest request;
      request.m_buffer = new __int8[chunkSize];
      memset(request.m_buffer, 0, chunkSize);
      request.m_offset = i * batchSize + j * chunkSize;

      assert(request.m_offset + chunkSize < fileSize);
      request.m_readSize = chunkSize;
      requests.push_back(request);

      auto evnt = CreateEvent(NULL, FALSE, FALSE, NULL);
      events[j] = evnt;

      ANNIndex::AsyncReadRequest& rRequest = requests[j];
      rRequest.m_callback = [&evnt, &readVec, &rRequest](bool success) {
        memcpy(readVec + rRequest.m_offset, rRequest.m_buffer,
               rRequest.m_readSize);
        SetEvent(evnt);
      };

      iohandler.ReadFileAsync(rRequest);
    }

    WaitForMultipleObjectsEx(NUM_CHUNKS, events, TRUE, INFINITE, TRUE);

    for (auto evnt : events) {
      CloseHandle(evnt);
    }
    iohandler.ShutDown();
  }
}