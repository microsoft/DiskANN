#ifndef EFANNA2E_FLASH_INDEX_NSG_H
#define EFANNA2E_FLASH_INDEX_NSG_H

#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include "aligned_file_reader.h"
#include "index.h"
#include "neighbor.h"
#include "parameters.h"
#include "tsl/robin_set.h"
#include "util.h"

namespace efanna2e {
  class FlashIndexNSG : public Index {
   public:
    explicit FlashIndexNSG(const size_t dimension, const size_t n, Metric m,
                           Index *initializer);

    virtual ~FlashIndexNSG();

    // empty function
    virtual void Load(const char *filename) override;
    // empty function
    virtual void Save(const char *filename) override;

    // empty function
    virtual void Build(size_t n, const float *data,
                       const Parameters &parameters) override {
    }

    // empty function
    virtual std::pair<int, int> Search(const float *query, const float *x,
                                       size_t k, const Parameters &parameters,
                                       unsigned *indices) override;

    void load_embedded_index(const std::string &index_filename,
                             const std::string &node_size_fname);

    // implemented
    virtual std::pair<int, int> BeamSearch(const float *query, const float *x,
                                           size_t            k,
                                           const Parameters &parameters,
                                           unsigned *        indices,
                                           int beam_width) override;
    AlignedFileReader graph_reader;

   private:
    std::vector<size_t> node_offsets;
    std::vector<size_t> node_sizes;
    Index *             initializer_;

    unsigned    width;
    unsigned    ep_;
    float       scale_factor;  // for entire data
    unsigned    aligned_dim;   // ROUND_UP(dimension_, 8)
    SimpleNhood ep_nhood;
    size_t      node_size;
    size_t      data_len;
    size_t      neighbor_len;
  };
}

#endif  // EFANNA2E_FLASH_INDEX_NSG_H
