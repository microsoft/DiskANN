#pragma once
#include "distance.h"
#include "parameters.h"
#include "utils.h"
#include "types.h"
#include "index_config.h"
#include "index_build_params.h"
#include <any>

namespace diskann
{
struct consolidation_report
{
    enum status_code
    {
        SUCCESS = 0,
        FAIL = 1,
        LOCK_FAIL = 2,
        INCONSISTENT_COUNT_ERROR = 3
    };
    status_code _status;
    size_t _active_points, _max_points, _empty_slots, _slots_released, _delete_set_size, _num_calls_to_process_delete;
    double _time;

    consolidation_report(status_code status, size_t active_points, size_t max_points, size_t empty_slots,
                         size_t slots_released, size_t delete_set_size, size_t num_calls_to_process_delete,
                         double time_secs)
        : _status(status), _active_points(active_points), _max_points(max_points), _empty_slots(empty_slots),
          _slots_released(slots_released), _delete_set_size(delete_set_size),
          _num_calls_to_process_delete(num_calls_to_process_delete), _time(time_secs)
    {
    }
};

class AbstractIndex
{
  public:
    AbstractIndex()
    {
    }
    virtual ~AbstractIndex()
    {
    }
    virtual void build(const std::string &data_file, const size_t num_points_to_load,
                       IndexBuildParams &build_params) = 0;
    virtual void build(const DataType &data, const size_t num_points_to_load, const IndexWriteParameters &parameters,
                       const TagVector &tags) = 0;
    virtual void save(const char *filename, bool compact_before_save = false) = 0;

#ifdef EXEC_ENV_OLS
    virtual void load(AlignedFileReader &reader, uint32_t num_threads, uint32_t search_l) = 0;
#else
    virtual void load(const char *index_file, uint32_t num_threads, uint32_t search_l) = 0;
#endif

    virtual std::pair<uint32_t, uint32_t> search(const DataType &query, size_t K, uint32_t L, uint32_t *result_ids,
                                                 float *distances, std::string &filter_label) = 0;

    virtual int insert_point(const DataType &data_point, const TagType &tag) = 0;

    virtual int lazy_delete(const TagType &tag) = 0;

    virtual void lazy_delete(const TagVector &tags, TagVector &failed_tags) = 0;

    virtual void get_active_tags(TagRobinSet &active_tags) = 0;

    virtual void set_start_points_at_random(DataType radius, uint32_t random_seed = 0) = 0;

    virtual consolidation_report consolidate_deletes(const IndexWriteParameters &parameters) = 0;

    virtual void optimize_index_layout() = 0;

    // memory should be allocated for vec before calling this function
    virtual int get_vector_by_tag(TagType &tag, DataType &vec) = 0;
};
} // namespace diskann
