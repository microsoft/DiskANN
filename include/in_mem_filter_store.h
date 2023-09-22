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
    const tsl::robin_set<label_type> &get_all_label_set() override;
    // Throws: out of range exception
    void add_label_to_location(const location_t point_id, label_type label) override;
    // returns internal mapping for given raw_label
    label_type get_converted_label(const std::string &raw_label) override;

    void update_medoid_by_label(const label_type &label, const uint32_t new_medoid) override;
    const uint32_t &get_medoid_by_label(const label_type &label) override;
    bool label_has_medoid(const label_type &label) override;
    void calculate_best_medoids(const size_t num_points_to_load, const uint32_t num_candidates) override;

    // takes raw universal labels and map them internally.
    void set_universal_labels(const std::vector<std::string> &raw_universal_labels) override;
    // const label_type get_universal_label() const;

    // ideally takes raw label file and then genrate internal mapping file and keep the info of mapping
    size_t load_raw_labels(const std::string &raw_labels_file) override;

    void save_labels(const std::string &save_path, const size_t total_points) override;
    // For dynamic filtered build, we compact the data and hence location_to_labels, we need the compacted version of
    // raw labels to compute GT correctly.
    void save_raw_labels(const std::string &save_path, const size_t total_points) override;
    void save_medoids(const std::string &save_path) override;
    void save_label_map(const std::string &save_path) override;
    void save_universal_label(const std::string &save_path) override;

    // The function is static so it remains the source of truth across the code. Returns label map
    DISKANN_DLLEXPORT static std::unordered_map<std::string, label_type> convert_labels_string_to_int(
        const std::string &inFileName, const std::string &outFileName, const std::string &mapFileName,
        const std::set<std::string> &raw_universal_labels);

  protected:
    // This is for internal use and only loads already parsed file, used by index in during load().
    size_t load_labels(const std::string &labels_file) override;
    size_t load_medoids(const std::string &labels_to_medoid_file) override;
    void load_label_map(const std::string &labels_map_file) override;
    void load_universal_labels(const std::string &universal_labels_file) override;

  private:
    size_t _num_points;
    std::vector<std::vector<label_type>> _location_to_labels;
    tsl::robin_set<label_type> _labels;
    std::unordered_map<std::string, label_type> _label_map;

    // medoids
    std::unordered_map<label_type, uint32_t> _label_to_medoid_id;
    std::unordered_map<uint32_t, uint32_t> _medoid_counts; // medoids only happen for filtered index

    // universal label
    bool _use_universal_label = false;
    tsl::robin_set<label_type> _mapped_universal_labels;
    std::set<std::string> _raw_universal_labels;

    // populates _loaction_to labels and _labels from given label file
    size_t parse_label_file(const std::string &label_file);

    bool detect_common_filters_by_set_intersection(uint32_t point_id, bool search_invocation,
                                                   const std::vector<label_type> &incoming_labels);
};

} // namespace diskann
