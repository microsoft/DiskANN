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
    this->cur_off = 0;
    reader.open(filename, std::ios::binary | std::ios::ate);
    fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);
    assert(reader.is_open());
    assert(cache_size > 0);
    cache_size = (std::min)(cache_size, fsize);
    this->cache_size = cache_size;
    cache_buf = new char[cache_size];
    reader.read(cache_buf, cache_size);
    std::cout << "Opened: " << filename.c_str() << ", size: " << fsize
              << ", cache_size: " << cache_size << "\n";
  }

  size_t get_file_size() {
    return fsize;
  }
  void read(char* read_buf, uint64_t n_bytes) {
    assert(cache_buf != nullptr);
    assert(read_buf != nullptr);
    if (n_bytes <= (cache_size - cur_off)) {
      // case 1: cache contains all data
      memcpy(read_buf, cache_buf + cur_off, n_bytes);
      cur_off += n_bytes;
    } else {
      // case 2: cache contains some data
      uint64_t cached_bytes = cache_size - cur_off;
      if (n_bytes - cached_bytes > fsize - reader.tellg()) {
        std::cout << "Reading beyond end of file" << std::endl;
        exit(-1);
      }
      memcpy(read_buf, cache_buf + cur_off, cached_bytes);

      // go to disk and fetch more data
      reader.read(read_buf + cached_bytes, n_bytes - cached_bytes);
      // reset cur off
      cur_off = cache_size;

      uint64_t size_left = fsize - reader.tellg();

      if (size_left >= cache_size) {
        reader.read(cache_buf, cache_size);
        cur_off = 0;
      }

      // note that if size_left < cache_size, then cur_off = cache_size, so
      // subsequent reads will all be directly from file
    }
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
      this->flush_cache();
    }

    delete[] cache_buf;
    writer.close();
    std::cout << "Finished writing " << fsize << "B\n";
  }

  size_t get_file_size() {
    return fsize;
  }
  // writes n_bytes from write_buf to the underlying ofstream/cache
  void write(char* write_buf, uint64_t n_bytes) {
    assert(cache_buf != nullptr);
    if (n_bytes <= (cache_size - cur_off)) {
      // case 1: cache can take all data
      memcpy(cache_buf + cur_off, write_buf, n_bytes);
      cur_off += n_bytes;
    } else {
      // case 2: cache cant take all data
      // go to disk and write existing cache data
      writer.write(cache_buf, cur_off);
      fsize += cur_off;
      // write the new data to disk
      writer.write(write_buf, n_bytes);
      fsize += n_bytes;
      // memset all cache data and reset cur_off
      memset(cache_buf, 0, cache_size);
      cur_off = 0;
    }
  }

  void flush_cache() {
    assert(cache_buf != nullptr);
    writer.write(cache_buf, cur_off);
    fsize += cur_off;
    memset(cache_buf, 0, cache_size);
    cur_off = 0;
  }

  void reset() {
   flush_cache();
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
