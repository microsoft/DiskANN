#ifdef _WINDOWS
#ifdef USE_BING_INFRA

#include <string>
#include <sstream>
#include "bing_aligned_file_reader.h"
#include "dll/DiskPriorityIO.h"

namespace diskann {

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
      // context.m_pRequests->push_back(req);
      context.m_pRequests->push_back(req);
    }
    this->ctx_map.insert(std::make_pair(std::this_thread::get_id(), context));
  }

  void BingAlignedFileReader::deregister_thread() {
    auto &context = this->ctx_map.at(std::this_thread::get_id());

    context.m_pDiskIO->ShutDown();
    delete context.m_pDiskIO;

    this->ctx_map.erase(std::this_thread::get_id());
  }

  IOContext &BingAlignedFileReader::get_ctx() {
    if (this->ctx_map.find(std::this_thread::get_id()) == ctx_map.end()) {
      std::stringstream stream;
      stream << std::string("Thread ") << std::this_thread::get_id()
             << " is not registered.";
      throw std::exception(stream.str().c_str());
    }

    return this->ctx_map.at(std::this_thread::get_id());
  }

  void checkSize(const std::vector<AlignedRead> &read_reqs, IOContext &ctx) {
    if (read_reqs.size() > ctx.m_pRequests->size()) {
      auto count = read_reqs.size() - ctx.m_pRequests->size();
      for (int i = 0; i < count; i++) {
        ANNIndex::AsyncReadRequest readReq;
        ctx.m_pRequests->push_back(readReq);
      }
    }

    if (read_reqs.size() > ctx.m_pRequestsStatus->size()) {
      auto count = read_reqs.size() - ctx.m_pRequestsStatus->size();
      for (int i = 0; i < count; i++) {
        ctx.m_pRequestsStatus->push_back(IOContext::READ_WAIT);
      }
    }
  }

  void initializeRead(IOContext &ctx) {
    //*(ctx.m_pCompleteCount) = 0;
    // auto &statusVec = *(ctx.m_pRequestsStatus);
    // std::fill(statusVec.begin(), statusVec.end(), IOContext::READ_WAIT);
  }

  void BingAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                   IOContext &ctx, bool async) {
    checkSize(read_reqs, ctx);
    initializeRead(ctx);

    for (int i = 0; i < read_reqs.size(); i++) {
      (*ctx.m_pRequests)[i].m_buffer = (__int8 *) read_reqs[i].buf;
      (*ctx.m_pRequests)[i].m_offset = read_reqs[i].offset;
      (*ctx.m_pRequests)[i].m_readSize = (unsigned int) read_reqs[i].len;

      (*ctx.m_pRequestsStatus)[i] = IOContext::READ_WAIT;

      (*ctx.m_pRequests)[i].m_callback = [ctx, i, this](bool result) {
        if (result) {
          (*ctx.m_pRequestsStatus)[i] = IOContext::READ_SUCCESS;
        } else {
          std::stringstream stream;
          stream << "Read request to file: " << m_filename << "failed.";
          std::cout << stream.str() << std::endl;
          (*ctx.m_pRequestsStatus)[i] = IOContext::READ_FAILED;
        }
      };
      ctx.m_pDiskIO->ReadFileAsync((*ctx.m_pRequests)[i]);
    }

    if (!async) {
      auto &statusVec = (*ctx.m_pRequestsStatus);
      bool  mustWait = true;
      while (mustWait) {
        mustWait = false;
        for (auto &status : statusVec) {
          mustWait = mustWait || (status == IOContext::READ_WAIT);
          if (mustWait)
            break;  // for loop
        }
      }
    }
  }

}  // namespace diskann
#endif  // USE_BING_INFRA
#endif  //_WINDOWS
