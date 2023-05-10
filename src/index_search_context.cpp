// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "index_search_context.h"

namespace diskann
{
template <typename LabelT> void IndexSearchContext<LabelT>::SetLabel(LabelT label, bool use_filter)
{
    _label = label;
    _use_filter = use_filter;
}

template <typename LabelT> void IndexSearchContext<LabelT>::SetState(State state)
{
    _result_state = state;
}

template <typename LabelT> State IndexSearchContext<LabelT>::GetState() const
{
    return _result_state;
}

template <typename LabelT> LabelT IndexSearchContext<LabelT>::GetLabel() const
{
    return _label;
}

template <typename LabelT> uint32_t IndexSearchContext<LabelT>::GetIOLimit() const
{
    return _io_limit;
}

template <typename LabelT> bool IndexSearchContext<LabelT>::UseFilter() const
{
    return _use_filter;
}

template <typename LabelT> bool IndexSearchContext<LabelT>::IsSuccess() const
{
    return _result_state == State::Success;
}

template <typename LabelT> bool IndexSearchContext<LabelT>::CheckTimeout()
{
    if (_time_limit_in_microseconds > 0 && _time_limit_in_microseconds < _timer.elapsed())
    {
        SetState(State::FailureTimeout);
        return true;
    }

    return false;
}

template <typename LabelT> QueryStats &IndexSearchContext<LabelT>::GetStats()
{
    return _stats;
}

template DISKANN_DLLEXPORT class IndexSearchContext<uint16_t>;
template DISKANN_DLLEXPORT class IndexSearchContext<uint32_t>;

} // namespace diskann
