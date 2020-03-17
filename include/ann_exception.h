#pragma once
#include <string>
#include "windows_customizations.h"

#ifndef _WINDOWS
#define __FUNCSIG__ __PRETTY_FUNCTION__
#endif

namespace diskann {
  class ANNException {
   public:
    DISKANN_DLLEXPORT ANNException(const std::string& message, int errorCode);
    DISKANN_DLLEXPORT ANNException(const std::string& message, int errorCode,
                                   const std::string& funcSig,
                                   const std::string& fileName,
                                   unsigned int       lineNum);

    DISKANN_DLLEXPORT std::string message() const;

   private:
    int          _errorCode;
    std::string  _message;
    std::string  _funcSig;
    std::string  _fileName;
    unsigned int _lineNum;
  };
}  // namespace diskann
