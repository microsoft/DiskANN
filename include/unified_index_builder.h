// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "distance.h"
#include "unified_index_format.h"
#include "windows_customizations.h"

namespace diskann
{

// All parameters required to build a unified-format index file.
//
// One struct, runtime-typed (no template). The data_type field selects which
// concrete `Index<T>` is instantiated internally; coords are read from
// `data_file_path` in `.bin` format (the legacy DiskANN file layout).
struct UnifiedBuildContext
{
    // --- Input data ---
    std::string data_file_path; // .bin file holding N points x dim coords of `data_type`
    DataTypeTag data_type = DataTypeTag::Float;
    diskann::Metric metric = diskann::Metric::L2;

    // --- Graph build parameters (Vamana) ---
    uint32_t R = 64;            // max degree
    uint32_t L = 100;           // search list size during build
    float alpha = 1.2f;         // pruning alpha
    uint32_t num_threads = 0;   // 0 = use omp_get_num_procs()

    // --- PQ parameters ---
    // pq_dim == 0          => no PQ (memory-only unified file; SSD load will reject).
    // 0 < pq_dim < dim     => train PQ with `pq_dim` chunks on a sampled subset and
    //                         emit pivots + codes into the unified file.
    // pq_dim >= dim        => train PQ with `dim` chunks (chunk size 1, full-precision
    //                         per dimension). Clamped so the SSD load path -- which
    //                         requires HAS_PQ -- can always load the produced file.
    uint32_t pq_dim = 0;
    double pq_sampling_rate = 0.1; // fraction of points to sample for pivot training (clamped server-side)

    // --- Optional filtered-index inputs ---
    std::string label_file;       // per-point labels (.txt), empty = unfiltered
    std::string universal_label;  // string to treat as "any label"
    bool use_integer_labels = false;

    // --- Output ---
    std::string output_path; // destination unified container file
};

// Builds a unified-format index file end-to-end: trains the Vamana graph from
// the input data file, optionally trains PQ on a sampled subset, then writes
// graph + medoids + (optional) PQ + (optional) labels into the unified
// container at `ctx.output_path`.
//
// Class shape (instead of free function) leaves room for future stateful build
// modes (incremental build, multi-pass, etc.). For now `build()` is the only
// method.
class unified_index_builder
{
  public:
    unified_index_builder();
    ~unified_index_builder();

    // Throws ANNException on failure (file open, mismatched dims, build crash,
    // PQ training error, etc.). Returns successfully when the unified file is
    // fully written and closed.
    void build(const UnifiedBuildContext &ctx);
};

} // namespace diskann
