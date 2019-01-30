#ifndef EFANNA2E_FLASH_INDEX_NSG_H
#define EFANNA2E_FLASH_INDEX_NSG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include "tsl/robin_set.h"
#include "aligned_file_reader.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <stack>

namespace efanna2e {
  class FlashIndexNSG : public Index {
  public:
    explicit FlashIndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer);

    virtual ~FlashIndexNSG();

    virtual void Load(const char *filename)override;
    // empty function
    virtual void Save(const char *filename)override;

    // empty function
    virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

    // empty function
    virtual std::pair<int,int> Search(
			      const float *query,
			      const float *x,
			      size_t k,
			      const Parameters &parameters,
			      unsigned *indices) override;

    // implemented
    virtual std::pair<int,int> BeamSearch(
				  const float *query,
				  const float *x,
				  size_t k,
				  const Parameters &parameters,
				  unsigned *indices,
				  int beam_width) override;

  private:
    AlignedFileReader graph_reader;
    std::vector<size_t> node_offsets;
    Index *initializer_;

    unsigned width;
    unsigned ep_;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
  };
}

#endif //EFANNA2E_FLASH_INDEX_NSG_H
