#pragma once
#include <stdexcept>

namespace diskann {

  class NotImplementedException : public std::logic_error {
   public:
    NotImplementedException()
        : std::logic_error("Function not yet implemented.") {
    }
  };
}
