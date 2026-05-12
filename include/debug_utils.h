// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <vector>
#include <limits>

namespace diskann
{

// Reason why a node visited during ANN graph traversal was or was not included
// in the final result set.
enum class FilterReason : uint8_t
{
    InResult       = 0, // Node was kept in the final top-K result
    DistanceTooLarge,   // Node was visited but its distance was too large for top-K
    LabelMismatch,      // Node was skipped because its label did not match the filter
};

// Collects per-node traversal information during a debug ANN search.
// Populated by iterate_to_fixed_point / cached_beam_search when a non-null
// pointer is passed. Each parallel vector entry corresponds to one node
// encountered during traversal.
struct DebugTraversalInfo
{
    std::vector<uint32_t> ids;            // Internal location index of each encountered node
    std::vector<float>    distances;      // PQ/exact distance to query; FLT_MAX when label-rejected
    std::vector<uint8_t>  label_rejected; // 1 if skipped due to label mismatch, 0 if evaluated

    void clear()
    {
        ids.clear();
        distances.clear();
        label_rejected.clear();
    }

    void record_label_rejected(uint32_t id)
    {
        ids.push_back(id);
        distances.push_back(std::numeric_limits<float>::max());
        label_rejected.push_back(1);
    }

    void record_visited(uint32_t id, float dist)
    {
        ids.push_back(id);
        distances.push_back(dist);
        label_rejected.push_back(0);
    }
};

} // namespace diskann
