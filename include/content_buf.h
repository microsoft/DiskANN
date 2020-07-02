#pragma once

#ifdef EXEC_ENV_OLS

namespace diskann {
  class ContentBuf : public std::basic_streambuf<char> {
   public:
    ContentBuf(char* p, size_t n) {
      setg(p, p, p + n);
    }

    virtual pos_type seekoff(
        off_type off, std::ios_base::seekdir dir,
        std::ios_base::openmode which = std::ios_base::in) {
      if (dir == std::ios_base::cur)
        gbump((int) off);
      else if (dir == std::ios_base::end)
        setg(eback(), egptr() + off, egptr());
      else if (dir == std::ios_base::beg)
        setg(eback(), eback() + off, egptr());
      return gptr() - eback();
    }
  };
}  // namespace diskann

#endif