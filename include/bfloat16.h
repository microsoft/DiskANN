#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace diskann
{
// Minimal IEEE-754 bfloat16 (bf16) implementation.
// Stores the top 16 bits of a float32, with round-to-nearest-even on conversion.
struct bfloat16
{
    uint16_t value = 0;

    constexpr bfloat16() = default;
    explicit constexpr bfloat16(uint16_t v) : value(v)
    {
    }

    // Convenience constructor for generic code that does bfloat16(f).
    explicit bfloat16(float f) : value(from_float(f).value) {}

    // Convenience constructor for generic code that does static_cast<bfloat16>(double_expr).
    explicit bfloat16(double f) : value(from_float(static_cast<float>(f)).value) {}

    static inline bfloat16 from_float(float f)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));

        // Round-to-nearest-even: add 0x7FFF + LSB of the truncated part.
        const uint32_t lsb = (bits >> 16) & 1u;
        bits += 0x7FFFu + lsb;
        return bfloat16(static_cast<uint16_t>(bits >> 16));
    }

    inline float to_float() const
    {
        uint32_t bits = static_cast<uint32_t>(value) << 16;
        float f = 0.0f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    inline operator float() const
    {
        return to_float();
    }
};

// bfloat16 is not a built-in floating point type, but for most DiskANN code
// paths it should be treated as "floating-point-like".
template <typename T> struct is_floating_point_like : std::is_floating_point<T>
{
};

template <> struct is_floating_point_like<diskann::bfloat16> : std::true_type
{
};

template <typename T> inline constexpr bool is_floating_point_like_v = is_floating_point_like<T>::value;

} // namespace diskann
