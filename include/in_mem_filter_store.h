#pragma once
#include <abstract_filter_store.h>

namespace diskann
{

// This class is responsible for filter actions in index, and should not be used outside.
template <typename label_type> class InMemFilterStore : public AbstractFilterStore<label_type>
{
  public:
    InMemFilterStore(const size_t num_points);
    ~InMemFilterStore() = default;

    // needs some internal lock
    bool detect_common_filters(uint32_t point_id, bool search_invocation,
                               const std::vector<label_type> &incoming_labels,
                               const FilterMatchStrategy filter_match_strategy) override;

    const std::vector<label_type> &get_labels_by_location(const location_t point_id) override;
    // const label_type get_universal_label

    // Dynamic Index
    void set_labels_to_location(const location_t location, const std::vector<std::string> &labels);
    void swap_labels(const location_t location_first, const location_t location_second) override;
    const tsl::robin_set<label_type> &get_all_label_set() override;
    void add_to_label_set(const label_type &label) override;
    // Throws: out of range exception
    void add_label_to_location(const location_t point_id, const label_type label) override;
    // returns internal mapping for given raw_label
    label_type get_numeric_label(const std::string &raw_label) override;

    // takes raw universal labels and map them internally.
    void set_universal_label(const std::string &raw_universal_labels) override;
    std::pair<bool, label_type> get_universal_label() override;

    // ideally takes raw label file and then genrate internal mapping file and keep the info of mapping
    size_t load_raw_labels(const std::string &raw_labels_file, const std::string &raw_universal_label) override;

    void save_labels(const std::string &save_path, const size_t total_points) override;
    // For dynamic filtered build, we compact the data and hence location_to_labels, we need the compacted version of
    // raw labels to compute GT correctly.
    void save_raw_labels(const std::string &save_path, const size_t total_points) override;
    void save_label_map(const std::string &save_path) override;
    void save_universal_label(const std::string &save_path) override;

    // The function is static so it remains the source of truth across the code. Returns label map
    DISKANN_DLLEXPORT static std::unordered_map<std::string, label_type> convert_label_to_numeric(
        const std::string &inFileName, const std::string &outFileName, const std::string &mapFileName,
        const std::string &raw_universal_labels);

  protected:
    // This is for internal use and only loads already parsed file, used by index in during load().
    size_t load_labels(const std::string &labels_file) override;
    void load_label_map(const std::string &labels_map_file) override;
    void load_universal_labels(const std::string &universal_labels_file) override;

  private:
    size_t _num_points;
    std::vector<std::vector<label_type>> _location_to_labels;
    tsl::robin_set<label_type> _labels;
    std::unordered_map<std::string, label_type> _label_map;

    // universal label
    bool _has_universal_label = false;
    label_type _universal_label;

    // no need of storing raw universal label ?
    // 1. _use_universal_label can be used to identify if universal label present or not
    // 2. from _label_map and _mapped_universal_label, we can know what is raw universal label. Hence seems duplicate
    // std::string _raw_universal_label;

    // populates _loaction_to labels and _labels from given label file
    size_t parse_label_file(const std::string &label_file);

    bool detect_common_filters_by_set_intersection(uint32_t point_id, bool search_invocation,
                                                   const std::vector<label_type> &incoming_labels);
};

} // namespace diskann
