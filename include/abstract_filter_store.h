#pragma once
#include "types.h"
#include "windows_customizations.h"
#include <vector>

namespace diskann {
template <typename LabelT> class AbstractFilterStore {
public:
  DISKANN_DLLEXPORT virtual bool has_filter_support() const = 0;

  DISKANN_DLLEXPORT virtual bool
  point_has_label(location_t point_id, const LabelT label_id) const = 0;

  // Returns true if the index is filter-enabled and all files were loaded
  // correctly. false otherwise. Note that "false" can mean that the index
  // does not have filter support, or that some index files do not exist, or
  // that they exist and could not be opened.
  DISKANN_DLLEXPORT virtual bool load(const std::string &disk_index_file) = 0;

  DISKANN_DLLEXPORT virtual void
  generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                         const uint32_t nthreads) = 0;
};

} // namespace diskann
