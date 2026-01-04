#pragma once

#include <cstdint>
#include <cstring>

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

} // namespace diskann
