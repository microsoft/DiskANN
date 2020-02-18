#include "ann_exception.h"
#include <sstream>

namespace diskann {
  ANNException::ANNException(const std::string& message, int errorCode)
      : _errorCode(errorCode), _message(message), _funcSig(""), _fileName(""),
        _lineNum(0) {
  }

  ANNException::ANNException(const std::string& message, int errorCode,
                             const std::string& funcSig,
                             const std::string& fileName, unsigned lineNum)
      : ANNException(message, errorCode) {
    _funcSig = funcSig;
    _fileName = fileName;
    _lineNum = lineNum;
  }

  std::string ANNException::message() const {
    std::stringstream sstream;

    sstream << "Exception: " << _message;
    if (_funcSig != "")
      sstream << ". occurred at: " << _funcSig;
    if (_fileName != "" && _lineNum != 0)
      sstream << " defined in file: " << _fileName << " at line: " << _lineNum;
    if (_errorCode != -1)
      sstream << ". OS error code: " << std::hex << _errorCode;

    return sstream.str();
  }

}  // namespace diskann