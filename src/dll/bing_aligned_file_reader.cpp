#ifdef _WINDOWS
#include "bing_aligned_file_reader.h"


BingAlignedFileReader::BingAlignedFileReader(){};
BingAlignedFileReader::~BingAlignedFileReader(){};


//TODO: Must refactor this and WAFR to avoid code
//repeats
void BingAlignedFileReader::register_thread() {

}

void BingAlignedFileReader::deregister_thread() {
}

IOContext &BingAlignedFileReader::get_ctx() {
}

// Open & close ops
// Blocking calls
void BingAlignedFileReader::open(const std::string &fname) {
}
void BingAlignedFileReader::close() {
}

void BingAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                 IOContext                 ctx) {

}

#endif
