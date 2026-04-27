#ifndef VARINTENCODE_H_
#define VARINTENCODE_H_

#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

// Encode an array of a given length read from in to bout in varint format.
// Returns the number of bytes written.
size_t vbyte_encode(const uint32_t *in, size_t length, uint8_t *bout);

// Encode an array of a given length read from in to bout in varint format with differential
// coding starting at value prev. (Setting prev to 0 is a good default.)
//
// Returns the number of bytes written.
size_t vbyte_encode_delta(const uint32_t *in, size_t length, uint8_t *bout, uint32_t prev);

#if defined(__cplusplus)
}
#endif

#endif /* VARINTENCODE_H_ */
