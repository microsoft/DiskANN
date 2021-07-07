// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <sstream>
#include <mutex>

#ifdef EXEC_ENV_OLS
#include "IANNIndex.h"
#include "ANNLogging.h"
#endif

#include "ann_exception.h"

#ifndef EXEC_ENV_OLS
namespace ANNIndex {
  enum LogLevel {
    LL_Debug = 0,
    LL_Info,
    LL_Status,
    LL_Warning,
    LL_Error,
    LL_Assert,
    LL_Count
  };
};
#endif

namespace diskann {
  class ANNStreamBuf : public std::basic_streambuf<char> {
   public:
    DISKANN_DLLEXPORT explicit ANNStreamBuf(FILE* fp);
    DISKANN_DLLEXPORT ~ANNStreamBuf();

    DISKANN_DLLEXPORT bool is_open() const {
      return true;  // because stdout and stderr are always open.
    }
    DISKANN_DLLEXPORT void        close();
    DISKANN_DLLEXPORT virtual int underflow();
    DISKANN_DLLEXPORT virtual int overflow(int c);
    DISKANN_DLLEXPORT virtual int sync();

   private:
    FILE*              _fp;
    char*              _buf;
    int                _bufIndex;
    std::mutex         _mutex;
    ANNIndex::LogLevel _logLevel;

    int  flush();
    void logImpl(char* str, int numchars);

// Why the two buffer-sizes? If we are running normally, we are basically
// interacting with a character output system, so we short-circuit the
// output process by keeping an empty buffer and writing each character
// to stdout/stderr. But if we are running in OLS, we have to take all
// the text that is written to diskann::cout/diskann:cerr, consolidate it
// and push it out in one-shot, because the OLS infra does not give us
// character based output. Therefore, we use a larger buffer that is large
// enough to store the longest message, and continuously add characters
// to it. When the calling code outputs a std::endl or std::flush, sync()
// will be called and will output a log level, component name, and the text
// that has been collected. (sync() is also called if the buffer is full, so
// overflows/missing text are not a concern).
// This implies calling code _must_ either print std::endl or std::flush
// to ensure that the message is written immediately.
#ifdef EXEC_ENV_OLS
    static const int BUFFER_SIZE = 2048;
#else
    static const int BUFFER_SIZE = 0;
#endif

    ANNStreamBuf(const ANNStreamBuf&);
    ANNStreamBuf& operator=(const ANNStreamBuf&);
  };
}  // namespace diskann
