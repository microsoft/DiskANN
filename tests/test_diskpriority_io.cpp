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

void processData(HANDLE evnt, __int8* readVec,
                 ANNIndex::AsyncReadRequest request, bool result) {
  // What do we do here?
  std::cout << "Entering callback: " << result << std::endl;
  memcpy(readVec + request.m_offset, request.m_buffer, request.m_readSize);
  SetEvent(evnt);
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
  std::cout << "Sending " << NUM_CHUNKS << " requests per batch. "
            << NUM_THREADS << " batches " << std::endl;


  for (int i = 0; i < NUM_THREADS; i++) {
    NSG::DiskPriorityIO iohandler(
        ANNIndex::DiskIOScenario::DIS_HighPriorityUserRead);
    iohandler.Initialize(s.c_str());

	HANDLE events[NUM_CHUNKS];
    std::vector<ANNIndex::AsyncReadRequest> requests;
    requests.reserve(NUM_CHUNKS);

    int    chunkSize = batchSize / NUM_CHUNKS;
    for (int j = 0; j < NUM_CHUNKS; j++) {
      ANNIndex::AsyncReadRequest& request = requests[j];
      request.m_buffer = new __int8[chunkSize];
      memset(request.m_buffer, 0, chunkSize);
      request.m_offset = i * batchSize + j * chunkSize;

      assert(request.m_offset + chunkSize <= fileSize);
      request.m_readSize = chunkSize;

      auto evnt = CreateEvent(NULL, FALSE, FALSE, NULL);
      events[j] = evnt;

      std::function<void(bool)> fptr = std::bind(
          &processData, evnt, readVec, request, std::placeholders::_1);
      request.m_callback = fptr;

      iohandler.ReadFileAsync(request);
    }

    WaitForMultipleObjectsEx(NUM_CHUNKS, events, TRUE, INFINITE, TRUE);

    for (auto evnt : events) {
      CloseHandle(evnt);
    }
    iohandler.ShutDown();
  }

  int k = 0;
  for (k = 0; k < fileSize; k++) {
    if (readVec[k] != pByte[k]) {
      std::cerr << "Expected " << pByte[k] << " got " << readVec[k]
                << " at index: " << k << std::endl;
      break;
    }
  }
  if (k == fileSize - 1) {
    std::cout << "Successfully verified read." << std::endl;
  } else {
    std::cout << "Read verification failed." << std::endl;
  }
}