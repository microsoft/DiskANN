#ifdef _WINDOWS
#ifdef USE_BING_INFRA

#include <string>
#include <sstream>
#include "bing_aligned_file_reader.h"
#include "dll/DiskPriorityIO.h"

namespace NSG {

  // TODO: Must refactor this and WAFR to avoid code
  // repeats

  BingAlignedFileReader::BingAlignedFileReader(){};
  BingAlignedFileReader::~BingAlignedFileReader(){};

  // Open & close ops
  // Blocking calls
  void BingAlignedFileReader::open(const std::string &fname) {
    m_filename = fname;
    this->register_thread();
  }
  void BingAlignedFileReader::close() {
  }

  void BingAlignedFileReader::register_thread() {
    std::unique_lock<std::mutex> lk(this->ctx_mut);
    if (this->ctx_map.find(std::this_thread::get_id()) != ctx_map.end()) {
      std::cout << "Warning:: Duplicate registration for thread_id : "
                << std::this_thread::get_id() << "\n";
    }

    IOContext context;
    context.m_pDiskIO =
        new DiskPriorityIO(ANNIndex::DiskIOScenario::DIS_HighPriorityUserRead);
    context.m_pDiskIO->Initialize(m_filename.c_str());

    for (_u64 i = 0; i < MAX_IO_DEPTH; i++) {
      ANNIndex::AsyncReadRequest req;
      memset(&req, 0, sizeof(ANNIndex::AsyncReadRequest));
      context.m_requests.push_back(req);
    }
    this->ctx_map.insert(std::make_pair(std::this_thread::get_id(), context));
  }

  void BingAlignedFileReader::deregister_thread() {
    auto& context = this->ctx_map.at(std::this_thread::get_id());

    context.m_pDiskIO->ShutDown();
    delete context.m_pDiskIO;

    this->ctx_map.erase(std::this_thread::get_id());
  }

  IOContext& BingAlignedFileReader::get_ctx() {
    if (this->ctx_map.find(std::this_thread::get_id()) == ctx_map.end()) {
      std::stringstream stream;
      stream << std::string("Thread ") << std::this_thread::get_id()
             << " is not registered.";
      throw std::exception(stream.str().c_str());
    }

    return this->ctx_map.at(std::this_thread::get_id());
  }

  void checkSize(const std::vector<AlignedRead> &read_reqs, IOContext& ctx) {
    if (read_reqs.size() > ctx.m_requests.size()) {
      int count = read_reqs.size() - ctx.m_requests.size();
      for (int i = 0; i < count; i++) {
        ANNIndex::AsyncReadRequest readReq;
        ctx.m_requests.push_back(readReq);
      }
    }
  }

  void BingAlignedFileReader::callback(std::shared_ptr<std::atomic<int>> pCounter, bool result) {
    (*pCounter)++;
    if (!result) {
      std::stringstream stream;
		//TODO: We must redo this request. But for now, just fail.
      stream << "Read request to file: " << m_filename << "failed.";
      throw std::exception(stream.str().c_str());
    } 
  }

  void BingAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                   IOContext                 &ctx) {
    checkSize(read_reqs, ctx);
    *(ctx.m_pCompleteCount) = 0;
    for (int i = 0; i < read_reqs.size(); i++) {
      ctx.m_requests[i].m_buffer = (__int8 *) read_reqs[i].buf;
      ctx.m_requests[i].m_offset = read_reqs[i].offset;
      ctx.m_requests[i].m_readSize = (unsigned int)read_reqs[i].len;

	  ctx.m_requests[i].m_callback =
          std::bind(&BingAlignedFileReader::callback, this, ctx.m_pCompleteCount, std::placeholders::_1);

	  ctx.m_pDiskIO->ReadFileAsync(ctx.m_requests[i]);
    }

	while ( *(ctx.m_pCompleteCount) < read_reqs.size()) {
      ;
    }
  }

}  // namespace NSG
#endif  // USE_BING_INFRA
#endif  //_WINDOWS
