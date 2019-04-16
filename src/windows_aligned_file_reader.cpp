#include <cassert>
#include <cstdio>
#include <iostream>
#include "FileAbstractions.h"
#include "aligned_file_reader.h"
#include "efanna2e/util.h"
#include "tsl/robin_map.h"

#include <fcntl.h>

#define MAX_EVENTS 1024

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb     iocb_t;

  void execute_io(IoContext ctx, FileHandle fd, std::vector<AlignedRead> &read_reqs, uint64_t n_retries = 5) {
	  for (auto &req : read_reqs) {
		  assert(IS_ALIGNED(req.len, 512));
		  assert(IS_ALIGNED(req.offset, 512));
		  assert(IS_ALIGNED(req.buf, 512));
	  }

	  LARGE_INTEGER offset;
	  offset.HighPart = offset.LowPart = offset.QuadPart = 0;
	  LARGE_INTEGER currentFp;
	  SetFilePointerEx(fd, offset, &currentFp, FILE_CURRENT);
	  

	  for (auto &req : read_reqs) {
		  DWORD num_bytes = 0;
		  req.overlapped.OffsetHigh = (uint32_t)((req.offset & 0xFFFFFFFF00000000LL) >> 32);
		  req.overlapped.Offset = (uint32_t)(req.offset & 0xFFFFFFFFLL);

		  if (FALSE == ReadFile(fd, req.buf, req.len, nullptr, &req.overlapped)) {
			  std::cerr << "***ERROR***: Read file failed because of: " << GetLastError() << std::endl;
			  exit(2);
		  }

		  GetOverlappedResult(fd, &(req.overlapped), &num_bytes, TRUE);
		  assert(num_bytes == req.len);

		  //RESET File pointer after reading. 
		  LARGE_INTEGER newFp;
		  SetFilePointerEx(fd, currentFp, &newFp, FILE_BEGIN);
		  assert(newFp.HighPart == currentFp.HighPart && newFp.LowPart == currentFp.LowPart && newFp.QuadPart == currentFp.QuadPart);
	  }
  }
}  // namespace

AlignedFileReader::AlignedFileReader() {
  this->file_desc = nullptr;
}

AlignedFileReader::~AlignedFileReader() {
  // Gopal. For now assume that the file is indeed open and try to close it.
	if ( this->file_desc != nullptr ){
		if (FALSE == CloseHandle(this->file_desc)) {
			std::cout << "Failed to close file handle in ~AlignedFileReader(). Error: "
				<< GetLastError() << std::endl;
		}
  }
}

IoContext AlignedFileReader::get_ctx() {
  IoContext context;
  memset(&context, 0, sizeof(IoContext));
  return context;
}

void AlignedFileReader::register_thread() {
	auto                         my_id = std::this_thread::get_id();
	std::unique_lock<std::mutex> lk(AlignedFileReader::ctx_mut);
	if (AlignedFileReader::ctx_map.find(my_id) !=
		AlignedFileReader::ctx_map.end()) {
		std::cerr << "multiple calls to register_thread from the same thread"
			<< std::endl;
		return;
	}
	IoContext ctx;
	memset(&ctx, 0, sizeof(ctx));
	std::cout << "allocating ctx to thread-id:" << my_id << std::endl;
	AlignedFileReader::ctx_map[my_id] = ctx;
	lk.unlock();
}

void AlignedFileReader::deregister_thread() {
	auto                         my_id = std::this_thread::get_id();
	std::unique_lock<std::mutex> lk(AlignedFileReader::ctx_mut);
	assert(AlignedFileReader::ctx_map.find(my_id) !=
		AlignedFileReader::ctx_map.end());

	lk.unlock();
	lk.lock();
	AlignedFileReader::ctx_map.erase(my_id);
	std::cout << "returned ctx from thread-id:" << my_id << std::endl;
	lk.unlock();
}

void AlignedFileReader::open(const std::string &fname) {
	this->file_desc = CreateFileA(fname.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
	if (this->file_desc == nullptr) {
		std::cerr << "Failed to open file " << fname.c_str() << " because of error: " << GetLastError() << std::endl;
		exit(1);
	}
	std::cout << "opened file: " << fname.c_str() << std::endl;
}

void AlignedFileReader::close() {
	//Gopal. assuming that the file is not closed. If it is, CloseHandle simply returns.
	CloseHandle(this->file_desc);
	this->file_desc = nullptr;
}

void AlignedFileReader::read(std::vector<AlignedRead> &read_reqs) {
	assert(this->file_desc != nullptr);
	IoContext ctx = get_ctx();
	execute_io(ctx, this->file_desc, read_reqs);
}
