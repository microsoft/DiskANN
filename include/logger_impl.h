#pragma once

#include <sstream>
#include <mutex>
#include "bing\IANNIndex.h"
#include "bing\ANNLogging.h"

#include "ann_exception.h"

void ANNLogging(FILE* logger, ANNIndex::LogLevel level, const char* format,
                ...);

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


    //Why the two buffer-sizes? If we are running normally, we are basically
    //interacting with a character output system, so we short-circuit the 
    //output process by keeping an empty buffer and writing each character 
    //to stdout/stderr. But if we are running in OLS, we have to take all
    //the text that is written to diskann::cout/diskann:cerr, consolidate it
    //and push it out in one-shot, because the OLS infra does not give us
    //character based output. Therefore, we use a larger buffer that is large
    //enough to store the longest message, and continuously add characters 
    //to it. When the calling code outputs a std::endl or std::flush, sync() 
    //will be called and will output a log level, component name, and the text
    //that has been collected. (sync() is also called if the buffer is full, so
    //overflows/missing text are not a concern). 
    //This implies calling code _must_ either print std::endl or std::flush
    //to ensure that the message is written immediately. 
#ifdef EXEC_ENV_OLS
    static const int BUFFER_SIZE = 1024; 
#else 
    static const int BUFFER_SIZE = 0;
#endif

    ANNStreamBuf(const ANNStreamBuf&);
    ANNStreamBuf& operator=(const ANNStreamBuf&);
  };

  //  class logger {
  //   public:
  //    logger() {
  //    }
  //    explicit logger(FILE* fp) {
  //#ifndef EXEC_ENV_OLS
  //      if (fp == nullptr) {
  //        throw new diskann::ANNException(
  //            "Must specify a non-null file pointer for logging messages.",
  //            -1,
  //            __FUNCSIG__, __FILE__, __LINE__);
  //      }
  //#endif
  //      _fp = fp;
  //      _logLevel = _fp == stdout ? ANNIndex::LogLevel::LL_Info
  //                                : ANNIndex::LogLevel::LL_Error;
  //      _intFormat = "%ld";
  //    }
  //
  //    inline logger& operator<<(const std::string& value) {
  //      ANNLogging(_fp, _logLevel, value.c_str());
  //      return *this;
  //    }
  //    inline logger& operator<<(const char* str) {
  //      // strings can simply be passed as format-strings
  //      ANNLogging(_fp, _logLevel, str);
  //      return *this;
  //    }
  //    inline logger& operator<<(long value) {
  //      ANNLogging(_fp, _logLevel, _intFormat.c_str(), value);
  //      return *this;
  //    }
  //    inline logger& operator<<(double value) {
  //      ANNLogging(_fp, _logLevel, "%f", value);
  //      return *this;
  //    }
  //    inline logger& operator<<(const std::thread::id& threadId) {
  //      // This is a terrible way to implement a basic_ostream.
  //      // But we have three choices, none of them good.
  //      //  1. Derive from basic_ostream: This is a simple interface to
  //      implement,
  //      //      but we need to manage a stream_buf. Now that is unnecessary
  //      //      overhead for a simple logger that we need.
  //      //  2. Derive from basic_ofstream: This provides a stream_buf that we
  //      can
  //      //      use, but requires a filename to be provided. Funnily, cout is
  //      not
  //      //      basic_ofstream(stdout), but is a raw basic_ostream()
  //      //      implementation.
  //      //  3. So we fall back on our implementation. Do not derive from any
  //      of
  //      //      these classes, so when a new class (that provides an
  //      operator<<
  //      //      for basic_ostream) needs to use operator << with this class,
  //      we
  //      //      provide a hack of this form.
  //      // But the clincher for option 3 is that when we work in OLS
  //      environment,
  //      // we are forced to use their fprintf interface, which doesn't support
  //      // overloads. Hence this ugly code.
  //
  //      std::stringstream stream;
  //      stream << threadId;
  //      ANNLogging(_fp, _logLevel, stream.str().c_str());
  //      return *this;
  //    }
  //    //This is not thread-safe. It is possible for one thread to set this
  //    flag
  //    //and another one to come in and override it.
  //    inline logger& operator<<(const hex& hx) {
  //      _intFormat = "%X";
  //    }
  //
  //
  //   private:
  //    FILE*              _fp;
  //    ANNIndex::LogLevel _logLevel;
  //    std::string        _intFormat;
  //
  //    logger(const logger& other);
  //    logger& operator=(const logger& other);
  //  };
  //
  //  static logger cout(stdout), cerr(stderr);
  //  class hex {};
  //#ifdef _WINDOWS
  //  static const std::string endl("\\r\\n");
  //#else
  //  static const std::string endl("\\n");
  //#endif

}  // namespace diskann
