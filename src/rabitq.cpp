#include "rabitq.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <vector>

namespace diskann
{
namespace rabitq
{

namespace
{
constexpr float kTightStart[9] = {0.0f, 0.15f, 0.20f, 0.52f, 0.59f, 0.71f, 0.75f, 0.77f, 0.81f};
constexpr double kEps = 1e-5;

inline float l2_sqr(const float* x, size_t d)
{
    double sum = 0.0;
    for (size_t i = 0; i < d; ++i)
        sum += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    return static_cast<float>(sum);
}

inline float inner_product(const float* a, const float* b, size_t d)
{
    double sum = 0.0;
    for (size_t i = 0; i < d; ++i)
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return static_cast<float>(sum);
}

inline void set_bit_standard(uint8_t* code, size_t bit_index)
{
    const size_t byte_idx = bit_index / 8;
    const size_t bit_offset = bit_index % 8;
    code[byte_idx] |= static_cast<uint8_t>(1u << bit_offset);
}

inline bool extract_bit_standard(const uint8_t* code, size_t bit_index)
{
    const size_t byte_idx = bit_index / 8;
    const size_t bit_offset = bit_index % 8;
    return (code[byte_idx] >> bit_offset) & 1u;
}

inline int extract_code_inline(const uint8_t* ex_code, size_t index, size_t ex_bits)
{
    size_t bit_pos = index * ex_bits;
    int code_value = 0;

    for (size_t bit = 0; bit < ex_bits; ++bit)
    {
        const size_t byte_idx = bit_pos / 8;
        const size_t bit_idx = bit_pos % 8;
        if (ex_code[byte_idx] & (1u << bit_idx))
            code_value |= (1u << bit);
        bit_pos++;
    }

    return code_value;
}

SignBitFactorsWithError compute_vector_factors(const float* x, size_t d, Metric metric, bool compute_error)
{
    // Mirrors Faiss rabitq_utils::compute_vector_factors but with centroid == nullptr.
    // or_minus_c == x
    float norm_L2sqr = 0.0f;
    float or_L2sqr = 0.0f;
    float dp_oO = 0.0f;

    for (size_t j = 0; j < d; ++j)
    {
        const float x_val = x[j];
        const float or_minus_c = x_val;
        const float sq = or_minus_c * or_minus_c;
        norm_L2sqr += sq;
        or_L2sqr += x_val * x_val;

        const bool xb = (or_minus_c > 0.0f);
        dp_oO += xb ? or_minus_c : -or_minus_c;
    }

    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    constexpr float kConstEpsilon = 1.9f;

    const float sqrt_norm_L2 = std::sqrt(norm_L2sqr);
    const float inv_norm_L2 = (norm_L2sqr < epsilon) ? 1.0f : (1.0f / sqrt_norm_L2);

    const float inv_d_sqrt = (d == 0) ? 1.0f : (1.0f / std::sqrt(static_cast<float>(d)));
    const float normalized_dp = dp_oO * inv_norm_L2 * inv_d_sqrt;
    const float inv_dp_oO = (std::abs(normalized_dp) < epsilon) ? 1.0f : (1.0f / normalized_dp);

    SignBitFactorsWithError factors;
    if (metric == Metric::INNER_PRODUCT)
        factors.or_minus_c_l2sqr = (norm_L2sqr - or_L2sqr);
    else
        factors.or_minus_c_l2sqr = norm_L2sqr;

    factors.dp_multiplier = inv_dp_oO * sqrt_norm_L2;

    if (compute_error)
    {
        const float xu_cb_norm_sqr = static_cast<float>(d) * 0.25f;
        const float ip_resi_xucb = 0.5f * dp_oO;

        float tmp_error = 0.0f;
        if (std::abs(ip_resi_xucb) > epsilon)
        {
            const float ratio_sq = (norm_L2sqr * xu_cb_norm_sqr) / (ip_resi_xucb * ip_resi_xucb);
            if (ratio_sq > 1.0f)
            {
                if (d == 1)
                    tmp_error = sqrt_norm_L2 * kConstEpsilon * std::sqrt(ratio_sq - 1.0f);
                else
                    tmp_error = sqrt_norm_L2 * kConstEpsilon *
                        std::sqrt((ratio_sq - 1.0f) / static_cast<float>(d - 1));
            }
        }

        factors.f_error = (metric == Metric::L2) ? (2.0f * tmp_error) : (1.0f * tmp_error);
    }

    return factors;
}

float compute_optimal_scaling_factor(const float* o_abs, size_t d, size_t nb_bits)
{
    const size_t ex_bits = nb_bits - 1;
    assert(ex_bits >= 1 && ex_bits <= 8);

    const int kNEnum = 10;
    const int max_code = (1 << ex_bits) - 1;

    float max_o = *std::max_element(o_abs, o_abs + d);
    if (!(max_o > 0.0f))
        return 1.0f;

    const float t_end = static_cast<float>(max_code + kNEnum) / max_o;
    const float t_start = t_end * kTightStart[ex_bits];

    std::vector<float> inv_o_abs(d);
    for (size_t i = 0; i < d; ++i)
        inv_o_abs[i] = 1.0f / o_abs[i];

    std::vector<int> cur_o_bar(d);
    float sqr_denominator = static_cast<float>(d) * 0.25f;
    float numerator = 0.0f;

    for (size_t i = 0; i < d; ++i)
    {
        int cur = static_cast<int>((t_start * o_abs[i]) + kEps);
        cur_o_bar[i] = cur;
        sqr_denominator += static_cast<float>(cur * cur + cur);
        numerator += (cur + 0.5f) * o_abs[i];
    }

    float inv_sqrt_denom = 1.0f / std::sqrt(sqr_denominator);

    std::vector<std::pair<float, size_t>> pq_storage;
    pq_storage.reserve(d);

    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, std::greater<>> next_t(
        std::greater<>(), std::move(pq_storage));

    for (size_t i = 0; i < d; ++i)
    {
        float t_next = static_cast<float>(cur_o_bar[i] + 1) * inv_o_abs[i];
        if (t_next < t_end)
            next_t.emplace(t_next, i);
    }

    float max_ip = 0.0f;
    float t = 0.0f;

    while (!next_t.empty())
    {
        const float cur_t = next_t.top().first;
        const size_t update_id = next_t.top().second;
        next_t.pop();

        cur_o_bar[update_id]++;
        const int update_o_bar = cur_o_bar[update_id];

        const float delta = 2.0f * update_o_bar;
        sqr_denominator += delta;
        numerator += o_abs[update_id];

        const float old_denom = sqr_denominator - delta;
        inv_sqrt_denom = inv_sqrt_denom * (1.0f - 0.5f * delta / (old_denom + delta * 0.5f));

        const float cur_ip = numerator * inv_sqrt_denom;
        if (cur_ip > max_ip)
        {
            max_ip = cur_ip;
            t = cur_t;
        }

        if (update_o_bar < max_code)
        {
            float t_next = static_cast<float>(update_o_bar + 1) * inv_o_abs[update_id];
            if (t_next < t_end)
                next_t.emplace(t_next, update_id);
        }
    }

    return (t > 0.0f) ? t : 1.0f;
}

void pack_multibit_codes(const int* tmp_code, uint8_t* ex_code, size_t d, size_t nb_bits)
{
    const size_t ex_bits = nb_bits - 1;
    assert(ex_bits >= 1 && ex_bits <= 8);

    const size_t total_bits = d * ex_bits;
    const size_t output_size = (total_bits + 7) / 8;
    std::memset(ex_code, 0, output_size);

    size_t bit_pos = 0;
    for (size_t i = 0; i < d; ++i)
    {
        const int code_value = tmp_code[i];
        for (size_t bit = 0; bit < ex_bits; ++bit)
        {
            const size_t byte_idx = bit_pos / 8;
            const size_t bit_idx = bit_pos % 8;
            if (code_value & (1 << bit))
                ex_code[byte_idx] |= static_cast<uint8_t>(1u << bit_idx);
            ++bit_pos;
        }
    }
}

void compute_ex_factors(const float* residual, const int* tmp_code, size_t d, size_t ex_bits, float norm,
                        double ipnorm, ExtraBitsFactors& ex_factors, Metric metric)
{
    float ipnorm_inv = static_cast<float>(1.0 / ipnorm);
    if (!std::isnormal(ipnorm_inv))
        ipnorm_inv = 1.0f;

    const float cb = -(static_cast<float>(1 << ex_bits) - 0.5f);

    std::vector<float> xu_cb(d);
    for (size_t i = 0; i < d; ++i)
        xu_cb[i] = static_cast<float>(tmp_code[i]) + cb;

    const float l2_sqr_val = norm * norm;
    const float ip_resi_xucb = inner_product(residual, xu_cb.data(), d);

    if (metric == Metric::L2)
    {
        ex_factors.f_add_ex = l2_sqr_val;
        ex_factors.f_rescale_ex = ipnorm_inv * -2.0f * norm;
    }
    else
    {
        // centroid == nullptr in this DiskANN integration, so centroid correction term is 0.
        ex_factors.f_add_ex = 1;
        ex_factors.f_rescale_ex = ipnorm_inv * -norm;
        (void)ip_resi_xucb;
    }
}

void quantize_ex_bits(const float* residual, size_t d, size_t nb_bits, uint8_t* ex_code, ExtraBitsFactors& ex_factors,
                      Metric metric)
{
    const size_t ex_bits = nb_bits - 1;
    assert(ex_bits >= 1 && ex_bits <= 8);

    // Normalize residual
    const float norm2 = l2_sqr(residual, d);
    const float norm = std::sqrt(norm2);
    const float inv_norm = (norm2 > 0.0f) ? (1.0f / norm) : 1.0f;

    std::vector<float> o_abs(d);
    std::vector<int> tmp_code(d);

    // abs(normalized residual)
    for (size_t i = 0; i < d; ++i)
        o_abs[i] = std::abs(residual[i] * inv_norm);

    const float t = compute_optimal_scaling_factor(o_abs.data(), d, nb_bits);
    const int max_code = (1 << ex_bits) - 1;

    // Quantize and apply sign bit flipping behavior (encode magnitude with sign stored separately)
    // total_code = (sign << ex_bits) + ex_code
    for (size_t i = 0; i < d; ++i)
    {
        int q = static_cast<int>((t * o_abs[i]) + kEps);
        q = std::clamp(q, 0, max_code);

        // If residual is negative, flip code via complement (matches Faiss logic via sign bit storage)
        if (residual[i] <= 0.0f)
            q = max_code - q;

        tmp_code[i] = q;
    }

    pack_multibit_codes(tmp_code.data(), ex_code, d, nb_bits);

    // ipnorm from quantized normalized residual vs abs residual. Faiss uses a specific normalization;
    // here we approximate with a stable fallback.
    // We keep ipnorm > 0 to avoid NaNs.
    double ipnorm = 0.0;
    {
        const float cb = -(static_cast<float>(1 << ex_bits) - 0.5f);
        std::vector<float> xu_cb(d);
        for (size_t i = 0; i < d; ++i)
            xu_cb[i] = static_cast<float>(tmp_code[i]) + cb;
        ipnorm = inner_product(o_abs.data(), xu_cb.data(), d);
        if (!(ipnorm > 0.0))
            ipnorm = 1.0;
    }

    compute_ex_factors(residual, tmp_code.data(), d, ex_bits, norm, ipnorm, ex_factors, metric);
}

} // namespace

size_t compute_code_size(size_t d, size_t nb_bits)
{
    assert(nb_bits >= 1 && nb_bits <= 9);

    const size_t ex_bits = nb_bits - 1;

    const size_t base_size = (d + 7) / 8 + (ex_bits == 0 ? sizeof(SignBitFactors) : sizeof(SignBitFactorsWithError));

    size_t ex_size = 0;
    if (ex_bits > 0)
        ex_size = (d * ex_bits + 7) / 8 + sizeof(ExtraBitsFactors);

    return base_size + ex_size;
}

void encode_vector(const float* x, size_t d, Metric metric, size_t nb_bits, uint8_t* out_code)
{
    assert(x != nullptr);
    assert(out_code != nullptr);
    assert(nb_bits >= 1 && nb_bits <= 9);

    const size_t ex_bits = nb_bits - 1;
    const size_t code_size = compute_code_size(d, nb_bits);
    std::memset(out_code, 0, code_size);

    uint8_t* sign_bits = out_code;

    const bool compute_error = (ex_bits > 0);
    const SignBitFactorsWithError factors_data = compute_vector_factors(x, d, metric, compute_error);

    if (ex_bits == 0)
    {
        auto* base_factors = reinterpret_cast<SignBitFactors*>(out_code + (d + 7) / 8);
        base_factors->or_minus_c_l2sqr = factors_data.or_minus_c_l2sqr;
        base_factors->dp_multiplier = factors_data.dp_multiplier;
    }
    else
    {
        auto* full_factors = reinterpret_cast<SignBitFactorsWithError*>(out_code + (d + 7) / 8);
        *full_factors = factors_data;
    }

    // Sign bits
    std::vector<float> residual(d);
    for (size_t j = 0; j < d; ++j)
    {
        residual[j] = x[j];
        if (x[j] > 0.0f)
            set_bit_standard(sign_bits, j);
    }

    if (ex_bits > 0)
    {
        uint8_t* ex_code = out_code + (d + 7) / 8 + sizeof(SignBitFactorsWithError);
        auto* ex_factors = reinterpret_cast<ExtraBitsFactors*>(ex_code + (d * ex_bits + 7) / 8);
        quantize_ex_bits(residual.data(), d, nb_bits, ex_code, *ex_factors, metric);
    }
}

float approx_inner_product_from_code(const uint8_t* code, const float* query, size_t d, size_t nb_bits)
{
    assert(code != nullptr);
    assert(query != nullptr);
    assert(nb_bits >= 1 && nb_bits <= 9);

    const size_t ex_bits = nb_bits - 1;

    const uint8_t* sign_bits = code;

    // For this integration we assume query is already rotated/centered as needed.
    const float qr_to_c_L2sqr = l2_sqr(query, d);
    const float qr_norm_L2sqr = qr_to_c_L2sqr;

    float ex_ip = 0.0f;

    const float inv_d_sqrt = (d == 0) ? 1.0f : (1.0f / std::sqrt(static_cast<float>(d)));

    if (ex_bits == 0)
    {
        // Very simple 1-bit estimator: reconstructed value per dim is (bit-0.5)*2*inv_d_sqrt*dp_multiplier
        const auto* fac = reinterpret_cast<const SignBitFactors*>(code + (d + 7) / 8);
        const float scale = fac->dp_multiplier * 2.0f * inv_d_sqrt;

        for (size_t i = 0; i < d; ++i)
        {
            const float bit = extract_bit_standard(sign_bits, i) ? 1.0f : 0.0f;
            const float reconstructed = (bit - 0.5f) * scale;
            ex_ip += query[i] * reconstructed;
        }

        return ex_ip;
    }

    const auto* base_fac = reinterpret_cast<const SignBitFactorsWithError*>(code + (d + 7) / 8);
    const uint8_t* ex_code = code + (d + 7) / 8 + sizeof(SignBitFactorsWithError);
    const auto* ex_fac = reinterpret_cast<const ExtraBitsFactors*>(ex_code + (d * ex_bits + 7) / 8);

    const float cb = -(static_cast<float>(1 << ex_bits) - 0.5f);
    for (size_t i = 0; i < d; ++i)
    {
        const bool sign_bit = extract_bit_standard(sign_bits, i);
        const int ex_code_val = extract_code_inline(ex_code, i, ex_bits);
        int total_code = (sign_bit ? 1 : 0) << ex_bits;
        total_code += ex_code_val;
        const float reconstructed = static_cast<float>(total_code) + cb;
        ex_ip += query[i] * reconstructed;
    }

    // Faiss multi-bit distance formula (metric==IP) transformed yields an IP-like score.
    // dist = qr_to_c_L2sqr + ex_fac.f_add_ex + ex_fac.f_rescale_ex * ex_ip
    // ip_score = -0.5 * (dist - qr_norm_L2sqr)
    const float dist = qr_to_c_L2sqr + ex_fac->f_add_ex + ex_fac->f_rescale_ex * ex_ip;
    const float ip_score = -0.5f * (dist - qr_norm_L2sqr);

    (void)base_fac; // base factors not needed for full multi-bit path here

    return ip_score;
}

} // namespace rabitq
} // namespace diskann
