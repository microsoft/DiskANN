// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "index_search_context.h"

namespace diskann
{
template <typename LabelT>
void IndexSearchContext<LabelT>::set_filter_label(const LabelT &filter_label, bool use_filter)
{
    _filter_label = filter_label;
    _use_filter = use_filter;
}

template <typename LabelT> void IndexSearchContext<LabelT>::set_state(State state)
{
    _result_state = state;
}

template <typename LabelT> State IndexSearchContext<LabelT>::get_state() const
{
    return _result_state;
}

template <typename LabelT> LabelT IndexSearchContext<LabelT>::get_filter_label() const
{
    return _filter_label;
}

template <typename LabelT> uint32_t IndexSearchContext<LabelT>::get_io_limit() const
{
    return _io_limit;
}

template <typename LabelT> bool IndexSearchContext<LabelT>::use_filter() const
{
    return _use_filter;
}

template <typename LabelT> bool IndexSearchContext<LabelT>::is_success() const
{
    return _result_state == State::Success;
}

template <typename LabelT> bool IndexSearchContext<LabelT>::check_timeout()
{
    if (_time_limit_in_microseconds > 0 && _time_limit_in_microseconds < _timer.elapsed())
    {
        set_state(State::FailureTimeout);
        return true;
    }

    return false;
}

template <typename LabelT> QueryStats &IndexSearchContext<LabelT>::get_stats()
{
    return _stats;
}

template DISKANN_DLLEXPORT class IndexSearchContext<uint16_t>;
template DISKANN_DLLEXPORT class IndexSearchContext<uint32_t>;
template DISKANN_DLLEXPORT class IndexSearchContext<uint64_t>;

} // namespace diskann
