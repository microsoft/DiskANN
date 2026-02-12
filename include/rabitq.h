#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace diskann
{
namespace rabitq
{

enum class Metric : uint32_t
{
    L2 = 0,
    INNER_PRODUCT = 1,
};

#pragma pack(push, 1)
struct SignBitFactors
{
    float or_minus_c_l2sqr = 0;
    float dp_multiplier = 0;
};

struct SignBitFactorsWithError : SignBitFactors
{
    float f_error = 0;
};

struct ExtraBitsFactors
{
    float f_add_ex = 0;
    float f_rescale_ex = 0;
};
#pragma pack(pop)

static_assert(sizeof(SignBitFactors) == 8, "Unexpected padding in SignBitFactors");
static_assert(sizeof(SignBitFactorsWithError) == 12, "Unexpected padding in SignBitFactorsWithError");
static_assert(sizeof(ExtraBitsFactors) == 8, "Unexpected padding in ExtraBitsFactors");

size_t compute_code_size(size_t d, size_t nb_bits);

// Encodes a single vector (assumes any rotation/centering is already applied externally).
// Layout matches Faiss standard RaBitQ format:
// - sign bits: (d+7)/8 bytes
// - base factors: SignBitFactors (nb_bits==1) or SignBitFactorsWithError (nb_bits>1)
// - ex_code: (d*(nb_bits-1)+7)/8 bytes (only when nb_bits>1)
// - ex_factors: ExtraBitsFactors (only when nb_bits>1)
void encode_vector(const float* x, size_t d, Metric metric, size_t nb_bits, uint8_t* out_code);

// Approximate IP scorer.
// Returns an approximate inner product (higher is better) computed from a multi-bit RaBitQ code.
// For nb_bits==1, this returns a 1-bit estimate (still usable for prefilter).
float approx_inner_product_from_code(const uint8_t* code, const float* query, size_t d, size_t nb_bits);

} // namespace rabitq
} // namespace diskann
