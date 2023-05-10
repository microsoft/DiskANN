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
    FailureTimeout = 3,      // Fail because of timeout
    FailureException = 4,    // Fail because of Exception
    FailureInvalidLabel = 5, // Fail because of invalid label
    StateCount = 6           // The number of state
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
        _label = (LabelT)0;
    }

    DISKANN_DLLEXPORT void SetLabel(LabelT label, bool use_filter);

    void SetState(State state);

    DISKANN_DLLEXPORT State GetState() const;

    LabelT GetLabel() const;

    uint32_t GetIOLimit() const;

    bool UseFilter() const;

    DISKANN_DLLEXPORT bool IsSuccess() const;

    bool CheckTimeout();

    DISKANN_DLLEXPORT QueryStats &GetStats();

  private:
    uint32_t _time_limit_in_microseconds;
    uint32_t _io_limit;
    State _result_state;
    bool _use_filter;
    LabelT _label;
    Timer _timer;
    QueryStats _stats;
};

} // namespace diskann
