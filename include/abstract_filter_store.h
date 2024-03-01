#pragma once
#include "common_includes.h"
#include "utils.h"
#include <any>

namespace diskann
{

enum class FilterMatchStrategy
{
    SET_INTERSECTION
};
// This class is responsible for filter actions in index, and should not be used outside.
template <typename label_type> class AbstractFilterStore
{
  public:
    DISKANN_DLLEXPORT AbstractFilterStore(const size_t num_points);
    virtual ~AbstractFilterStore() = default;

    // needs some internal lock + abstract implementation
    DISKANN_DLLEXPORT virtual bool detect_common_filters(
        uint32_t point_id, bool search_invocation, const std::vector<label_type> &incoming_labels,
        const FilterMatchStrategy strategy = FilterMatchStrategy::SET_INTERSECTION) = 0;

    DISKANN_DLLEXPORT virtual const std::vector<label_type> &get_labels_by_location(const location_t point_id) = 0;
    DISKANN_DLLEXPORT virtual void set_labels_to_location(const location_t location,
                                                          const std::vector<std::string> &labels) = 0;
    DISKANN_DLLEXPORT virtual void swap_labels(const location_t location_first, const location_t location_second) = 0;

    DISKANN_DLLEXPORT virtual const tsl::robin_set<label_type> &get_all_label_set() = 0;
    DISKANN_DLLEXPORT virtual void add_to_label_set(const label_type &label) = 0;
    // Throws: out of range exception
    DISKANN_DLLEXPORT virtual void add_label_to_location(const location_t point_id, const label_type label) = 0;
    // returns internal mapping for given raw_label
    DISKANN_DLLEXPORT virtual label_type get_numeric_label(const std::string &raw_label) = 0;

    // TODO: in future we may accept a set or vector of universal labels
    // DISKANN_DLLEXPORT virtual void set_universal_label(label_type universal_label) = 0;
    DISKANN_DLLEXPORT virtual void set_universal_label(const std::string &universal_labels) = 0;
    DISKANN_DLLEXPORT virtual std::pair<bool, label_type> get_universal_label() = 0;

    // takes raw label file and then genrate internal mapping file and keep the info of mapping
    DISKANN_DLLEXPORT virtual size_t populate_labels(const std::string &raw_labels_file,
                                                     const std::string &raw_universal_label) = 0;

    // save labels, labels_map and universal_label to files
    DISKANN_DLLEXPORT virtual void save(const std::string &save_path, const size_t total_points) = 0;

    // load labels, labels_map and universal_label to filter store variables & returns total number of points
    DISKANN_DLLEXPORT virtual size_t load(const std::string &load_path) = 0;

    // For dynamic filtered build, we compact the data and hence location_to_labels, we need the compacted version of
    // raw labels to compute GT correctly.
    DISKANN_DLLEXPORT virtual void save_raw_labels(const std::string &save_path, const size_t total_points) = 0;

  protected:
    // This is for internal use and only loads already parsed file
    DISKANN_DLLEXPORT virtual size_t load_labels(const std::string &labels_file) = 0;
    DISKANN_DLLEXPORT virtual void load_label_map(const std::string &labels_map_file) = 0;
    DISKANN_DLLEXPORT virtual void load_universal_labels(const std::string &universal_labels_file) = 0;
    DISKANN_DLLEXPORT virtual void save_labels(const std::string &save_path, const size_t total_points) = 0;
    DISKANN_DLLEXPORT virtual void save_label_map(const std::string &save_path) = 0;
    DISKANN_DLLEXPORT virtual void save_universal_label(const std::string &save_path) = 0;

  private:
    size_t _num_points;

    // mark Index as friend so it can access protected loads
    template <typename T, typename TagT, typename LabelT> friend class Index;
};

} // namespace diskann
