#ifndef INCLUDE_STREAMVBYTE_ZIGZAG_H_
#define INCLUDE_STREAMVBYTE_ZIGZAG_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Convert N signed integers to N unsigned integers, using zigzag encoding.
 */
void zigzag_encode(const int32_t* in, uint32_t* out, size_t N);

/**
 * Convert N signed integers to N unsigned integers, using zigzag delta encoding.
 */
void zigzag_delta_encode(const int32_t* in, uint32_t* out, size_t N, int32_t prev);

/**
 * Convert N unsigned integers to N signed integers, using zigzag encoding.
 */
void zigzag_decode(const uint32_t* in, int32_t* out, size_t N);

/**
 * Convert N unsigned integers to N signed integers, using zigzag delta encoding.
 */
void zigzag_delta_decode(const uint32_t* in, int32_t* out, size_t N, int32_t prev);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_STREAMVBYTE_ZIGZAG_H_ */
