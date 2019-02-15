#include <iostream>
#include "aligned_file_reader.h"
#include "efanna2e/util.h"
#include <ctime>
#include <random>
#include <thread>
#include <cassert>
#include <chrono>


AlignedFileReader reader;

void thread_fn(uint64_t size, uint64_t qlen, uint64_t filesize, uint64_t niters, double &time){
  std::cout << "Thread #" << std::this_thread::get_id() << " up" << std::endl;
  reader.register_thread();
  std::vector<AlignedRead> reads(qlen);
  for(auto &read_req : reads){
    efanna2e::alloc_aligned(&read_req.buf, size, 512);
    read_req.len = size;
  }

  std::random_device rd;
  std::mt19937_64 engine(rd());
  std::uniform_int_distribution<uint64_t> distr;
  uint64_t max_offset = filesize - size;
  for(auto &read_req : reads){
      read_req.offset = ROUND_UP(distr(engine) % max_offset, 4096);
    }

  auto iter_start = std::chrono::high_resolution_clock::now();
  for(uint64_t i=0;i < niters; i++){
/*
    for(auto &read_req : reads){
      read_req.offset = ROUND_UP(distr(engine) % max_offset, 512);
    }
*/
    reader.read(reads);
  }
  auto diff_time = std::chrono::high_resolution_clock::now() - iter_start;
  auto diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff_time);
  time = diff_ms.count();
  for(auto &read_req : reads){
    free(read_req.buf);
  }
  reader.deregister_thread();
  std::cout << "Thread #" << std::this_thread::get_id() << " down" << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout
        << argv[0]
        << " filename nthreads read_size niters_per_thread qlen_per_thread" << std::endl;
    exit(-1);
  }

  std::string filename(argv[1]);
  unsigned nthreads = (unsigned) std::atoi(argv[2]);
  uint64_t read_size = (uint64_t) std::atoi(argv[3]);
  uint64_t niters = (uint64_t) std::atoi(argv[4]);
  uint64_t qlen = (uint64_t) std::atoi(argv[5]);
  
  std::ifstream freader(filename, std::ios::binary | std::ios::ate);
  if(freader.fail()){
    std::cout << "Error opening file: " << filename << std::endl;
    exit(-1);
  }
  uint64_t filesize = freader.tellg();
  std::cout << "Filename: " << filename << ", size = " << filesize << " bytes" << std::endl; 
  freader.close();

  std::cout << "Attaching reader to file" << std::endl;
  reader.open(std::string(argv[1]));
  std::cout << "Successfully attached reader to file" << std::endl;

  std::vector<double> times(nthreads, 0.0);
  std::cout << "Spawning threads" << std::endl;
  std::vector<std::thread> threads;
  for(unsigned i=0;i<nthreads;i++){
    std::thread th(thread_fn, read_size, qlen, filesize, niters, std::ref(times[i]));
    threads.push_back(std::move(th));
  }
  // wait for threads to complete
  for(auto &thread : threads){
    thread.join();
  }
  std::cout << "All threads down" << std::endl;
  reader.close();
  std::cout << "Detached reader from file" << std::endl;
  std::cout << "Thread Statistics: " << std::endl;
  double total_iops = 0.0;
  for(unsigned i=0;i<nthreads;i++){
    double iops = niters * qlen / (times[i] / 1000.0);
    total_iops += iops;
    std::cout << "Thread #" << i << ", IOPS = " << iops << "/s" << std::endl;
  }
  std::cout << "Total IOPS = " << total_iops << "/s" << std::endl;

  return 0;
}
