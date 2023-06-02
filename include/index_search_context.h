// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "percentile_stats.h"
#include "timer.h"
#include "windows_customizations.h"

namespace diskann
{

/// <summary>
/// Search state
/// </summary>
enum State : uint8_t
{
    Unknown = 0,
    Success = 1,
    Failure = 2,
    FailureTimeout = 3,   // Fail because of timeout
    FailureException = 4, // Fail because of Exception
    StateCount = 5        // The number of state
};

/// <summary>
/// Use this class to pass in searching parameters and pass out searching result
/// </summary>

template <typename LabelT = uint32_t> class IndexSearchContext
{
  public:
    DISKANN_DLLEXPORT IndexSearchContext(uint32_t time_limit_in_microseconds = 0u, uint32_t io_limit = UINT32_MAX)
        : _time_limit_in_microseconds(time_limit_in_microseconds), _io_limit(io_limit), _result_state(State::Unknown)
    {
        _use_filter = false;
        _filter_label = (LabelT)0;
    }

    DISKANN_DLLEXPORT void set_filter_label(const LabelT &filter_label, bool use_filter);

    void set_state(State state);

    DISKANN_DLLEXPORT State get_state() const;

    LabelT get_filter_label() const;

    uint32_t get_io_limit() const;

    bool use_filter() const;

    DISKANN_DLLEXPORT bool is_success() const;

    bool check_timeout();

    DISKANN_DLLEXPORT QueryStats &get_stats();

  private:
    uint32_t _time_limit_in_microseconds;
    uint32_t _io_limit;
    State _result_state;
    bool _use_filter;
    LabelT _filter_label;
    Timer _timer;
    QueryStats _stats;
};

} // namespace diskann
