#ifndef INCLUDE_STREAMVBYTEDELTA_H_
#define INCLUDE_STREAMVBYTEDELTA_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Encode an array of a given length read from in to out in StreamVByte format.
// Returns the number of bytes written.
// The number of values being stored (length) is not encoded in the compressed stream,
// the caller is responsible for keeping a record of this length. The pointer "in" should
// point to "length" values of size uint32_t, there is no alignment requirement on the out
// pointer. This version uses differential coding (coding differences between values) starting
// at prev (you can often set prev to zero). For safety, the out pointer should point to at least
// streamvbyte_max_compressedbyte(length) bytes (see streamvbyte.h)
size_t streamvbyte_delta_encode(const uint32_t* in, uint32_t length, uint8_t* out, uint32_t prev);

// Read "length" 32-bit integers in StreamVByte format from in, storing the result in out.
// Returns the number of bytes read.
// We may read up to STREAMVBYTE_PADDING extra bytes from the input buffer (these bytes are
// read but never used). The caller is responsible for knowing how many integers ("length")
// are to be read: this information ought to be stored somehow. There is no alignment requirement
// on the "in" pointer. The out pointer should point to length * sizeof(uint32_t) bytes. This
// version uses differential coding (coding differences between values) starting at prev
// (you can often set prev to zero).
size_t streamvbyte_delta_decode(const uint8_t* in, uint32_t* out, uint32_t length, uint32_t prev);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_STREAMVBYTEDELTA_H_ */
