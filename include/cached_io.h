#pragma once
#include <cstring>
#include <fstream>
#include <iostream>

// sequential cached reads
class cached_ifstream {
 public:
  cached_ifstream() {
  }
  cached_ifstream(const std::string& filename, uint64_t cache_size)
      : cache_size(cache_size), cur_off(0) {
    this->open(filename, cache_size);
  }
  ~cached_ifstream() {
    delete[] cache_buf;
    reader.close();
  }

  void open(const std::string& filename, uint64_t cache_size) {
    this->cache_size = cache_size;
    this->cur_off = 0;
    reader.open(filename, std::ios::binary | std::ios::ate);
    fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);
    assert(reader.is_open());
    assert(cache_size > 0);
    cache_buf = new char[cache_size];
    reader.read(cache_buf, cache_size);
    std::cout << "Opened: " << filename.c_str() << ", size: " << fsize
              << ", cache_size: " << cache_size << "\n";
  }

  void read(char* read_buf, uint64_t n_bytes) {
    assert(cache_buf != nullptr);
    assert(read_buf != nullptr);
    if (this->eof()) {
      // check EOF
      return;
    } else if (n_bytes <= (cache_size - cur_off)) {
      // case 1: cache contains all data
      memcpy(read_buf, cache_buf + cur_off, n_bytes);
      cur_off += n_bytes;
    } else {
      // case 2: cache contains some data
      uint64_t cached_bytes = cache_size - cur_off;
      memcpy(read_buf, cache_buf + cur_off, cached_bytes);

      // go to disk and fetch more data
      reader.read(cache_buf, cache_size);
      // reset cur off
      cur_off = 0;
      // copy remaining data to read_buf
      memcpy(read_buf + cached_bytes, cache_buf, n_bytes - cached_bytes);

      // increment cur_off
      cur_off = n_bytes - cached_bytes;
    }
  }

  bool eof() {
    // reader is EOF AND cur cache buf offset <=> last pos
    return reader.eof() && (cur_off == (fsize % cache_size));
  }

 private:
  // underlying ifstream
  std::ifstream reader;
  // # bytes to cache in one shot read
  uint64_t cache_size = 0;
  // underlying buf for cache
  char* cache_buf = nullptr;
  // offset into cache_buf for cur_pos
  uint64_t cur_off = 0;
  // file size
  uint64_t fsize = 0;
};

// sequential cached writes
class cached_ofstream {
 public:
  cached_ofstream(const std::string& filename, uint64_t cache_size)
      : cache_size(cache_size), cur_off(0) {
    writer.open(filename, std::ios::binary);
    assert(writer.is_open());
    assert(cache_size > 0);
    cache_buf = new char[cache_size];
    std::cout << "Opened: " << filename.c_str()
              << ", cache_size: " << cache_size << "\n";
  }

  ~cached_ofstream() {
    // dump any remaining data in memory
    if (cur_off > 0) {
      writer.write(cache_buf, cur_off);
      fsize += cur_off;
    }

    delete[] cache_buf;
    writer.close();
    std::cout << "Finished writing " << fsize << "B\n";
  }

  // writes n_bytes from write_buf to the underlying ofstream/cache
  void write(char* write_buf, uint64_t n_bytes) {
    assert(cache_buf != nullptr);
    if (n_bytes <= (cache_size - cur_off)) {
      // case 1: cache can take all data
      memcpy(cache_buf + cur_off, write_buf, n_bytes);
      cur_off += n_bytes;
    } else {
      // case 2: cache can take some data
      uint64_t cached_bytes = cache_size - cur_off;
      memcpy(cache_buf + cur_off, write_buf, cached_bytes);

      // go to disk and write all cache data
      writer.write(cache_buf, cache_size);
      fsize += cache_size;

      // memset all cache data
      memset(cache_buf, 0, cache_size);

      // reset cur off
      cur_off = 0;

      // copy remaining data from read_buf to cache_buf
      memcpy(cache_buf, write_buf + cached_bytes, n_bytes - cached_bytes);

      // increment cur_off
      cur_off = n_bytes - cached_bytes;
    }
  }

  void reset() {
	  writer.seekp(0);
  }

 private:
  // underlying ofstream
  std::ofstream writer;
  // # bytes to cache for one shot write
  uint64_t cache_size = 0;
  // underlying buf for cache
  char* cache_buf = nullptr;
  // offset into cache_buf for cur_pos
  uint64_t cur_off = 0;

  // file size
  uint64_t fsize = 0;
};
