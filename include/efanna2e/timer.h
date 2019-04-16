// borrowed from https://gist.github.com/tzutalin/fd0340a93bb8d998abb9
#include <chrono>

namespace NSG {
  class Timer {
    typedef std::chrono::high_resolution_clock clock_;
    std::chrono::time_point<clock_>            m_beg;

   public:
    Timer() : m_beg(clock_::now()) {
    }

    void reset() {
      m_beg = clock_::now();
    }

    // returns elapsed time in `us`
    uint32_t elapsed() const {
      return std::chrono::duration_cast<std::chrono::microseconds>(
                 clock_::now() - m_beg)
          .count();
    }
  };
}