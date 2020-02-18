#pragma once
#include <string>

namespace diskann {
  class ANNException {
   public:
    ANNException(const std::string& message, int errorCode);
    ANNException(const std::string& message, int errorCode,
                 const std::string& funcSig, const std::string& fileName,
                 unsigned int lineNum);

    std::string message() const;

   private:
    int          _errorCode;
    std::string  _message;
    std::string  _funcSig;
    std::string  _fileName;
    unsigned int _lineNum;
  };
}  // namespace diskann