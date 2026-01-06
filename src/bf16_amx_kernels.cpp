// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "bf16_amx_kernels.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>

#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
#include <immintrin.h>
#endif

namespace diskann
{

static inline float bf16_dot_scalar(const bfloat16 *a, const bfloat16 *b, uint32_t length)
{
    float dot = 0.0f;
#ifndef _WINDOWS
#pragma omp simd reduction(+ : dot) aligned(a, b : 8)
#endif
    for (int32_t i = 0; i < (int32_t)length; i++)
    {
        dot += a[i].to_float() * b[i].to_float();
    }
    return dot;
}

#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))

static inline uint64_t xgetbv_u32(uint32_t index)
{
    uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return (static_cast<uint64_t>(edx) << 32) | eax;
}

static inline bool cpu_has_osxsave()
{
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return false;
    return (ecx & (1u << 27)) != 0; // OSXSAVE
}

static inline bool cpu_has_amx_bf16_hw()
{
    if (__get_cpuid_max(0, nullptr) < 7)
        return false;

    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);

    // Structured Extended Feature Flags Enumeration Leaf (CPUID.07H:EDX)
    // - AMX_BF16: EDX[22]
    // - AMX_TILE: EDX[24]
    const bool has_amx_bf16 = (edx & (1u << 22)) != 0;
    const bool has_amx_tile = (edx & (1u << 24)) != 0;
    return has_amx_bf16 && has_amx_tile;
}

static inline bool os_xcr0_allows_amx_state()
{
    if (!cpu_has_osxsave())
        return false;

    // XCR0 bits:
    // - 17: XTILECFG
    // - 18: XTILEDATA
    const uint64_t xcr0 = xgetbv_u32(0);
    const uint64_t kAmxMask = (1ULL << 17) | (1ULL << 18);
    return (xcr0 & kAmxMask) == kAmxMask;
}

// Linux xstate permission request.
// Keep local constants to avoid depending on kernel UAPI headers.
static constexpr unsigned long kXfeatureXtilecfg = 17;
static constexpr unsigned long kXfeatureXtiledData = 18;
static constexpr unsigned long kXfeatureMaskXtilecfg = (1UL << kXfeatureXtilecfg);
static constexpr unsigned long kXfeatureMaskXtiledData = (1UL << kXfeatureXtiledData);
static constexpr unsigned long kArchGetXcompPerm = 0x1022;
static constexpr unsigned long kArchReqXcompPerm = 0x1023;

static inline bool request_linux_amx_perm_this_thread()
{
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, kArchGetXcompPerm, &bitmask);
    if (status != 0)
        return false;

    if ((bitmask & kXfeatureMaskXtiledData) != 0)
        return true;

    status = syscall(SYS_arch_prctl, kArchReqXcompPerm, kXfeatureXtiledData);
    if (status != 0)
        return false;

    bitmask = 0;
    status = syscall(SYS_arch_prctl, kArchGetXcompPerm, &bitmask);
    if (status != 0)
        return false;

    return (bitmask & kXfeatureMaskXtiledData) != 0;
}

static inline bool amx_bf16_runtime_available_impl()
{
    if (!cpu_has_amx_bf16_hw())
        return false;
    if (!os_xcr0_allows_amx_state())
        return false;

    // Linux additionally requires per-thread permission before first AMX use.
    return request_linux_amx_perm_this_thread();
}

#else

static inline bool amx_bf16_runtime_available_impl()
{
    return false;
}

#endif

bool amxbf16_kernels_compiled()
{
#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
    return true;
#else
    return false;
#endif
}

bool amxbf16_runtime_available()
{
#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
    static thread_local int state = 0; // 0 unknown, 1 ok, -1 no
    if (state == 0)
        state = amx_bf16_runtime_available_impl() ? 1 : -1;
    return state == 1;
#else
    return false;
#endif
}

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)

static inline void bf16_dot_amx_query_batch_impl(const bfloat16 *base,
                                                 const bfloat16 *query,
                                                 const uint32_t n_vecs,
                                                 const uint32_t dim,
                                                 float *out)
{
    // Process 32 bf16 elements per AMX dpbf16ps step.
    constexpr uint32_t kStep = 32;
    const uint32_t blockCount = dim / kStep;
    const uint32_t tailCount = dim % kStep;

    // Tile config cache (per-thread) parameterized by A_rows.
    // A: [A_rows x 64B], B: [16 x 4B], C: [A_rows x 4B]
    alignas(64) static thread_local unsigned char cfg[64];
    static thread_local int prevA = -1;

    const int A_rows = static_cast<int>(n_vecs);
    const int N = 1;
    const int A_colsb = static_cast<int>(kStep * 2); // 64
    const int B_colsb = N * 4;                       // 4 bytes per row (2 bf16)
    const int B_rows = static_cast<int>(kStep / 2);  // 16 rows
    const int C_rows = A_rows;
    const int C_colsb = N * 4; // 4

    if (prevA != A_rows)
    {
        std::memset(cfg, 0, sizeof(cfg));
        cfg[0] = 1;

        // tile0: A
        cfg[16] = (unsigned char)A_colsb;
        cfg[48] = (unsigned char)A_rows;

        // tile1: B
        cfg[18] = (unsigned char)B_colsb;
        cfg[49] = (unsigned char)B_rows;

        // tile2: C
        cfg[20] = (unsigned char)C_colsb;
        cfg[50] = (unsigned char)C_rows;

        _tile_loadconfig((void *)cfg);
        prevA = A_rows;
    }

    _tile_zero(2);

    const int a_stride = static_cast<int>(dim * sizeof(bfloat16));

    for (uint32_t blk = 0; blk < blockCount; ++blk)
    {
        const uint32_t elem_off = blk * kStep;
        _tile_loadd(0, (const void *)(base + elem_off), a_stride);
        _tile_loadd(1, (const void *)(query + elem_off), 4);
        _tile_dpbf16ps(2, 0, 1);
    }

    // Store results: N=1, stride=4 bytes => out[0..A_rows-1]
    _tile_stored(2, (void *)out, 4);

    // Tail correction (dim % 32)
    if (tailCount != 0)
    {
        const uint32_t base_elem = blockCount * kStep;
        for (uint32_t r = 0; r < n_vecs; ++r)
        {
            out[r] += bf16_dot_scalar(base + r * dim + base_elem, query + base_elem, tailCount);
        }
    }
}

#endif

float bf16_dot_f32_accum_amx(const bfloat16 *a, const bfloat16 *b, uint32_t length)
{
#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
    if (!amxbf16_runtime_available())
        return bf16_dot_scalar(a, b, length);

    // Avoid AMX overhead for tiny vectors.
    if (length < 256)
        return bf16_dot_scalar(a, b, length);

    float out = 0.0f;
    bf16_dot_amx_query_batch_impl(a, b, 1, length, &out);
    return out;
#else
    return bf16_dot_scalar(a, b, length);
#endif
}

void bf16_dot_f32_accum_amx_batch(const bfloat16 *base,
                                 const bfloat16 *query,
                                 uint32_t n_vecs,
                                 uint32_t dim,
                                 float *out)
{
#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
    if (!amxbf16_runtime_available())
    {
        for (uint32_t i = 0; i < n_vecs; ++i)
            out[i] = bf16_dot_scalar(base + i * dim, query, dim);
        return;
    }

    // Avoid AMX overhead for tiny batches or tiny dims.
    if (n_vecs == 0)
        return;
    if (dim < 256)
    {
        for (uint32_t i = 0; i < n_vecs; ++i)
            out[i] = bf16_dot_scalar(base + i * dim, query, dim);
        return;
    }

    // Kernel supports up to 16 rows per tile.
    constexpr uint32_t kMaxRows = 16;
    uint32_t offset = 0;
    while (offset < n_vecs)
    {
        const uint32_t cur = std::min(kMaxRows, n_vecs - offset);
        bf16_dot_amx_query_batch_impl(base + offset * dim, query, cur, dim, out + offset);
        offset += cur;
    }
#else
    for (uint32_t i = 0; i < n_vecs; ++i)
        out[i] = bf16_dot_scalar(base + i * dim, query, dim);
#endif
}

void bf16_dot_f32_accum_amx_matmul(const bfloat16 *base,
                                  const bfloat16 *queries,
                                  uint32_t n_base,
                                  uint32_t n_queries,
                                  uint32_t dim,
                                  float *out)
{
    if (n_base == 0 || n_queries == 0)
        return;

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
    if (!amxbf16_runtime_available() || dim < 256)
    {
        for (uint32_t i = 0; i < n_base; ++i)
            for (uint32_t j = 0; j < n_queries; ++j)
                out[i * n_queries + j] = bf16_dot_scalar(base + i * dim, queries + j * dim, dim);
        return;
    }

    // We compute a (Mb x Nb) tile of C = A * B where:
    // - A is [Mb x 32 bf16] (64 bytes per row)
    // - B is [16 x (Nb*4) bytes] where each column is one query (packed as bf16), and 16 rows correspond to 32 bf16.
    // - C is [Mb x (Nb*4) bytes] storing fp32 accumulators.
    // This is a direct generalization of the existing N=1 kernel.
    constexpr uint32_t kStep = 32;
    const uint32_t blockCount = dim / kStep;
    const uint32_t tailCount = dim % kStep;

    constexpr uint32_t kMaxMb = 16; // tile row capacity (bytes_per_row=64)
    constexpr uint32_t kMaxNb = 16; // keep B_colsb = Nb*4 within 64 bytes

    alignas(64) static thread_local unsigned char cfg[64];
    static thread_local int prevMb = -1;
    static thread_local int prevNb = -1;

    // Pack buffer for a B tile: [16 x (Nb*4) bytes].
    // We will pack 32 bf16 from each query into 16 rows (each row holds 2 bf16 per query => 4 bytes).
    alignas(64) static thread_local uint8_t bpack[16 * 64];
    alignas(64) static thread_local float cbuf[16 * 16];

    for (uint32_t i0 = 0; i0 < n_base; i0 += kMaxMb)
    {
        const uint32_t Mb = std::min<uint32_t>(kMaxMb, n_base - i0);
        const int A_rows = static_cast<int>(Mb);
        const int A_colsb = static_cast<int>(kStep * sizeof(bfloat16)); // 64

        for (uint32_t j0 = 0; j0 < n_queries; j0 += kMaxNb)
        {
            const uint32_t Nb = std::min<uint32_t>(kMaxNb, n_queries - j0);
            const int N = static_cast<int>(Nb);
            const int B_colsb = N * 4;
            const int B_rows = static_cast<int>(kStep / 2); // 16
            const int C_rows = A_rows;
            const int C_colsb = N * 4;

            if (prevMb != A_rows || prevNb != N)
            {
                std::memset(cfg, 0, sizeof(cfg));
                cfg[0] = 1;
                // tile0: A
                cfg[16] = (unsigned char)A_colsb;
                cfg[48] = (unsigned char)A_rows;
                // tile1: B
                cfg[18] = (unsigned char)B_colsb;
                cfg[49] = (unsigned char)B_rows;
                // tile2: C
                cfg[20] = (unsigned char)C_colsb;
                cfg[50] = (unsigned char)C_rows;

                _tile_loadconfig((void *)cfg);
                prevMb = A_rows;
                prevNb = N;
            }

            _tile_zero(2);

            const int a_stride = static_cast<int>(dim * sizeof(bfloat16));

            for (uint32_t blk = 0; blk < blockCount; ++blk)
            {
                const uint32_t elem_off = blk * kStep;

                // Pack B for this block.
                // Layout: bpack[row][col] where row in [0..15], col in bytes [0..B_colsb).
                // For each query q, we take 32 bf16 starting at elem_off. For each pair (2 bf16)
                // we write 4 bytes into bpack[row] at offset q*4.
                const bfloat16 *qptr = queries + (j0 * dim) + elem_off;
                for (uint32_t r = 0; r < 16; ++r)
                {
                    uint8_t *dst_row = bpack + r * 64;
                    for (uint32_t q = 0; q < Nb; ++q)
                    {
                        const uint16_t *src16 = reinterpret_cast<const uint16_t *>(qptr + q * dim + (r * 2));
                        std::memcpy(dst_row + q * 4, src16, 4);
                    }
                }

                _tile_loadd(0, (const void *)(base + (i0 * dim) + elem_off), a_stride);
                _tile_loadd(1, (const void *)bpack, 64);
                _tile_dpbf16ps(2, 0, 1);
            }

            // Store C tile. C is [Mb x Nb] fp32 laid out with row stride (Nb*4 bytes).
            _tile_stored(2, (void *)cbuf, (int)(Nb * sizeof(float)));

            // Write out with tail correction.
            for (uint32_t ii = 0; ii < Mb; ++ii)
            {
                for (uint32_t jj = 0; jj < Nb; ++jj)
                {
                    float v = cbuf[ii * Nb + jj];
                    if (tailCount != 0)
                    {
                        const uint32_t base_elem = blockCount * kStep;
                        v += bf16_dot_scalar(base + (i0 + ii) * dim + base_elem, queries + (j0 + jj) * dim + base_elem,
                                             tailCount);
                    }
                    out[(i0 + ii) * n_queries + (j0 + jj)] = v;
                }
            }
        }
    }
#else
    for (uint32_t i = 0; i < n_base; ++i)
        for (uint32_t j = 0; j < n_queries; ++j)
            out[i * n_queries + j] = bf16_dot_scalar(base + i * dim, queries + j * dim, dim);
#endif
}

void bf16_dot_f32_accum_amx_matmul_gather(const bfloat16 *data,
                                         uint32_t data_stride,
                                         const uint32_t *base_ids,
                                         uint32_t n_base,
                                         const uint32_t *query_ids,
                                         uint32_t n_queries,
                                         uint32_t dim,
                                         float *out)
{
    if (n_base == 0 || n_queries == 0)
        return;

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
    if (!amxbf16_runtime_available() || dim < 256)
    {
        for (uint32_t i = 0; i < n_base; ++i)
        {
            const bfloat16 *a = data + (size_t)base_ids[i] * (size_t)data_stride;
            for (uint32_t j = 0; j < n_queries; ++j)
            {
                const bfloat16 *b = data + (size_t)query_ids[j] * (size_t)data_stride;
                out[i * n_queries + j] = bf16_dot_scalar(a, b, dim);
            }
        }
        return;
    }

    constexpr uint32_t kStep = 32;
    const uint32_t blockCount = dim / kStep;
    const uint32_t tailCount = dim % kStep;

    constexpr uint32_t kMaxMb = 16;
    constexpr uint32_t kMaxNb = 16;

    alignas(64) static thread_local unsigned char cfg[64];
    static thread_local int prevMb = -1;
    static thread_local int prevNb = -1;

    alignas(64) static thread_local uint8_t apack[16 * 64];
    alignas(64) static thread_local uint8_t bpack[16 * 64];
    alignas(64) static thread_local float cbuf[16 * 16];

    for (uint32_t i0 = 0; i0 < n_base; i0 += kMaxMb)
    {
        const uint32_t Mb = std::min<uint32_t>(kMaxMb, n_base - i0);
        const int A_rows = static_cast<int>(Mb);
        const int A_colsb = static_cast<int>(kStep * sizeof(bfloat16)); // 64

        for (uint32_t j0 = 0; j0 < n_queries; j0 += kMaxNb)
        {
            const uint32_t Nb = std::min<uint32_t>(kMaxNb, n_queries - j0);
            const int N = static_cast<int>(Nb);
            const int B_colsb = N * 4;
            const int B_rows = static_cast<int>(kStep / 2); // 16
            const int C_rows = A_rows;
            const int C_colsb = N * 4;

            if (prevMb != A_rows || prevNb != N)
            {
                std::memset(cfg, 0, sizeof(cfg));
                cfg[0] = 1;
                // tile0: A
                cfg[16] = (unsigned char)A_colsb;
                cfg[48] = (unsigned char)A_rows;
                // tile1: B
                cfg[18] = (unsigned char)B_colsb;
                cfg[49] = (unsigned char)B_rows;
                // tile2: C
                cfg[20] = (unsigned char)C_colsb;
                cfg[50] = (unsigned char)C_rows;

                _tile_loadconfig((void *)cfg);
                prevMb = A_rows;
                prevNb = N;
            }

            _tile_zero(2);

            for (uint32_t blk = 0; blk < blockCount; ++blk)
            {
                const uint32_t elem_off = blk * kStep;

                // Pack A rows for this block: apack[row][0..63] = 32 bf16
                for (uint32_t r = 0; r < Mb; ++r)
                {
                    const bfloat16 *src = data + (size_t)base_ids[i0 + r] * (size_t)data_stride + elem_off;
                    std::memcpy(apack + r * 64, src, 64);
                }

                // Pack B for this block.
                for (uint32_t r = 0; r < 16; ++r)
                {
                    uint8_t *dst_row = bpack + r * 64;
                    for (uint32_t q = 0; q < Nb; ++q)
                    {
                        const bfloat16 *srcq =
                            data + (size_t)query_ids[j0 + q] * (size_t)data_stride + elem_off + (r * 2);
                        const uint16_t *src16 = reinterpret_cast<const uint16_t *>(srcq);
                        std::memcpy(dst_row + q * 4, src16, 4);
                    }
                }

                _tile_loadd(0, (const void *)apack, 64);
                _tile_loadd(1, (const void *)bpack, 64);
                _tile_dpbf16ps(2, 0, 1);
            }

            _tile_stored(2, (void *)cbuf, (int)(Nb * sizeof(float)));

            for (uint32_t ii = 0; ii < Mb; ++ii)
            {
                const bfloat16 *a_full = data + (size_t)base_ids[i0 + ii] * (size_t)data_stride;
                for (uint32_t jj = 0; jj < Nb; ++jj)
                {
                    float v = cbuf[ii * Nb + jj];
                    if (tailCount != 0)
                    {
                        const uint32_t base_elem = blockCount * kStep;
                        const bfloat16 *b_full = data + (size_t)query_ids[j0 + jj] * (size_t)data_stride;
                        v += bf16_dot_scalar(a_full + base_elem, b_full + base_elem, tailCount);
                    }
                    out[(i0 + ii) * n_queries + (j0 + jj)] = v;
                }
            }
        }
    }
#else
    (void)data_stride;
    for (uint32_t i = 0; i < n_base; ++i)
    {
        const bfloat16 *a = data + (size_t)base_ids[i] * (size_t)data_stride;
        for (uint32_t j = 0; j < n_queries; ++j)
        {
            const bfloat16 *b = data + (size_t)query_ids[j] * (size_t)data_stride;
            out[i * n_queries + j] = bf16_dot_scalar(a, b, dim);
        }
    }
#endif
}

} // namespace diskann
