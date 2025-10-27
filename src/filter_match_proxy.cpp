#include "filter_match_proxy.h"

namespace diskann
{

template <typename LabelT>
bitmask_filter_match<LabelT>::bitmask_filter_match(
    simple_bitmask_buf& bitmask_filters,
    std::vector<std::uint64_t>& query_bitmask_buf,
    const std::vector<LabelT>& filter_labels,
    LabelT unv_label)
    : _bitmask_filters(bitmask_filters),
      _query_bitmask_buf(query_bitmask_buf)
{
    // _bitmask_size == 0 means no filter is set
    if (_bitmask_filters._bitmask_size > 0)
    {
        query_bitmask_buf.resize(_bitmask_filters._bitmask_size, 0);
        _bitmask_full_val._mask = query_bitmask_buf.data();

        for (const auto& filter_label : filter_labels)
        {
            auto bitmask_val = simple_bitmask::get_bitmask_val(filter_label);
            _bitmask_full_val.merge_bitmask_val(bitmask_val);
        }

        // if unv isn't set, it will be default value 0
        auto bitmask_val = simple_bitmask::get_bitmask_val(unv_label);
        _bitmask_full_val.merge_bitmask_val(bitmask_val);
    }
}

template <typename LabelT>
bool bitmask_filter_match<LabelT>::contain_filtered_label(uint32_t id)
{
    simple_bitmask bm(_bitmask_filters.get_bitmask(id), _bitmask_filters._bitmask_size);

    return bm.test_full_mask_val(_bitmask_full_val);
}

template <typename LabelT>
integer_label_filter_match<LabelT>::integer_label_filter_match(
    integer_label_vector& label_vector,
    const std::vector<LabelT>& filter_labels,
    LabelT unv_label)
    : _label_vector(label_vector),
      _filter_labels(filter_labels),
      _unv_label(unv_label)
{
}

template <typename LabelT>
bool integer_label_filter_match<LabelT>::contain_filtered_label(uint32_t id)
{
    // if unv isn't set, it will be default value 0, and there will be no match
    return _label_vector.check_label_exists(id, _filter_labels) 
        || _label_vector.check_label_exists(id, _unv_label);
}

template <typename LabelT>
label_filter_match_holder<LabelT>::label_filter_match_holder(simple_bitmask_buf& bitmask_filters,
    std::vector<std::uint64_t>& query_bitmask_buf,
    integer_label_vector& label_vector,
    const std::vector<LabelT>& filter_labels,
    LabelT unv_label,
    bool use_integer_labels)
    : _bitmask_filter_match(bitmask_filters, query_bitmask_buf, filter_labels, unv_label),
      _integer_label_filter_match(label_vector, filter_labels, unv_label),
      _use_integer_labels(use_integer_labels)
{
}

template <typename LabelT>
bool label_filter_match_holder<LabelT>::contain_filtered_label(uint32_t id)
{
    if (_use_integer_labels)
    {
        return _integer_label_filter_match.contain_filtered_label(id);
    }
    else
    {
        return _bitmask_filter_match.contain_filtered_label(id);
    }
}

template class bitmask_filter_match<uint16_t>;
template class bitmask_filter_match<uint32_t>;
template class integer_label_filter_match<uint16_t>;
template class integer_label_filter_match<uint32_t>;
template class label_filter_match_holder<uint16_t>;
template class label_filter_match_holder<uint32_t>;

}