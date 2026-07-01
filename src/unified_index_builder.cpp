// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "boost/dynamic_bitset.hpp"

#include "unified_index_builder.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "ann_exception.h"
#include "index.h"
#include "parameters.h"
#include "pq.h"
#include "utils.h"

namespace diskann
{

namespace
{

// Read a temp file (produced by generate_quantized_data) fully into a byte
// buffer. Used to pull PQ pivots / codes back in so they can be embedded into
// the unified container via UnifiedIndexWriter::write_pq.
std::vector<uint8_t> slurp_file(const std::string &path)
{
    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(path, std::ios::binary | std::ios::ate);
    const std::streamoff sz = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(static_cast<size_t>(sz));
    in.read(reinterpret_cast<char *>(out.data()), sz);
    return out;
}

// Discover (npts, dim) from a DiskANN .bin file: a 4-byte int32 npts followed
// by a 4-byte int32 dim, then npts*dim*sizeof(T) bytes of coords.
void read_bin_metadata(const std::string &path, size_t &npts_out, size_t &dim_out)
{
    diskann::get_bin_metadata(path, npts_out, dim_out, 0);
}

// Build a Vamana Index<T> over the data file and write the unified container,
// optionally embedding PQ pivots + codes. All disk artifacts other than the
// final unified file are cleaned up before this function returns.
template <typename T> void build_impl(const UnifiedBuildContext &ctx)
{
    size_t npts = 0, dim = 0;
    read_bin_metadata(ctx.data_file_path, npts, dim);
    if (npts == 0 || dim == 0)
    {
        throw ANNException("unified_index_builder: empty or unreadable data file", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    }

    // -----------------------------------------------------------------------
    // 1) Build Vamana graph in memory via Index<T>.
    // -----------------------------------------------------------------------
    auto write_params = std::make_shared<IndexWriteParameters>(
        IndexWriteParametersBuilder(ctx.L, ctx.R)
            .with_alpha(ctx.alpha)
            .with_num_threads(ctx.num_threads)
            .build());

    Index<T, uint32_t, uint32_t> idx(ctx.metric, dim, npts, write_params, /*search_params=*/nullptr,
                                      /*num_frozen_pts=*/0, /*dynamic=*/false, /*enable_tags=*/false,
                                      /*concurrent_consolidate=*/false, /*pq_dist_build=*/false,
                                      /*num_pq_chunks=*/0, /*use_opq=*/false,
                                      /*filtered_index=*/!ctx.label_file.empty());

    if (!ctx.label_file.empty())
    {
        IndexFilterParams filter_params = IndexFilterParamsBuilder()
                                              .with_label_file(ctx.label_file)
                                              .with_universal_label(ctx.universal_label)
                                              .with_save_path_prefix(ctx.output_path + ".legacy_tmp")
                                              .build();
        idx.build(ctx.data_file_path, npts, filter_params);
    }
    else
    {
        idx.build(ctx.data_file_path.c_str(), npts, std::vector<uint32_t>());
    }

    // -----------------------------------------------------------------------
    // 2) PQ training (sampled subset).
    // -----------------------------------------------------------------------
    // The unified SSD load path (unified_index_ssd::load_storage) requires the
    // HAS_PQ regions to be present, so we emit PQ whenever PQ is requested
    // (pq_dim > 0) -- including the pq_dim == dim case, which yields chunk
    // size 1 (a full-precision-per-dimension PQ). Previously pq_dim >= dim
    // skipped PQ generation, producing a file that could not be loaded as an
    // SSD index. Clamp the chunk count to dim so an over-large pq_dim can't
    // silently skip PQ either (generate_pq_pivots rejects num_pq_chunks > dim).
    // TODO: revisit -- a cleaner design would make PQ truly optional on the
    // SSD path (serve full-precision coords when HAS_PQ is unset).
    std::vector<uint8_t> pq_pivots_bytes;
    std::vector<uint8_t> pq_codes_bytes;
    const bool train_pq = (ctx.pq_dim > 0);
    if (train_pq)
    {
        const size_t pq_chunks = std::min(static_cast<size_t>(ctx.pq_dim), dim);
        const std::string temp_prefix = ctx.output_path + ".pq_tmp";
        const std::string temp_pivots = temp_prefix + ".pq_pivots.bin";
        const std::string temp_codes = temp_prefix + ".pq_codes.bin";

        double p_val = ctx.pq_sampling_rate;
        if (p_val <= 0.0 || p_val > 1.0)
            p_val = 0.1; // safety fallback
        // For tiny datasets ensure at least a few hundred points train.
        if (npts > 0)
        {
            const double min_p = std::min(1.0, 256.0 / static_cast<double>(npts));
            if (p_val < min_p)
                p_val = min_p;
        }

        diskann::generate_quantized_data<T>(ctx.data_file_path, temp_pivots, temp_codes, ctx.metric, p_val,
                                            pq_chunks, /*use_opq=*/false, /*codebook_prefix=*/"");

        try
        {
            pq_pivots_bytes = slurp_file(temp_pivots);
            pq_codes_bytes = slurp_file(temp_codes);
        }
        catch (...)
        {
            std::remove(temp_pivots.c_str());
            std::remove(temp_codes.c_str());
            throw;
        }
        std::remove(temp_pivots.c_str());
        std::remove(temp_codes.c_str());
    }

    // -----------------------------------------------------------------------
    // 3) Emit the unified container.
    // -----------------------------------------------------------------------
    idx.save_unified(ctx.output_path.c_str(), pq_pivots_bytes, pq_codes_bytes);
}

} // namespace

unified_index_builder::unified_index_builder() = default;
unified_index_builder::~unified_index_builder() = default;

void unified_index_builder::build(const UnifiedBuildContext &ctx)
{
    if (ctx.data_file_path.empty())
        throw ANNException("UnifiedBuildContext: data_file_path is empty", -1, __FUNCSIG__, __FILE__, __LINE__);
    if (ctx.output_path.empty())
        throw ANNException("UnifiedBuildContext: output_path is empty", -1, __FUNCSIG__, __FILE__, __LINE__);

    switch (ctx.data_type)
    {
    case DataTypeTag::Float:
        build_impl<float>(ctx);
        break;
    case DataTypeTag::Uint8:
        build_impl<uint8_t>(ctx);
        break;
    case DataTypeTag::Int8:
        build_impl<int8_t>(ctx);
        break;
    default:
        throw ANNException("unified_index_builder: unsupported data_type", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
}

} // namespace diskann
