#ifndef INCLUDE_STREAMVBYTE_H_
#define INCLUDE_STREAMVBYTE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STREAMVBYTE_PADDING 16

// Encode an array of a given length read from in to out in varint format.
// Returns the number of bytes written.
// The number of values being stored (length) is not encoded in the compressed stream,
// the caller is responsible for keeping a record of this length.
// The pointer "in" should point to "length" values of size uint32_t
// there is no alignment requirement on the out pointer
// For safety, the out pointer should point to at least streamvbyte_max_compressedbyte(length)
// bytes.
// Uses 1,2,3 or 4 bytes per value + the decoding keys.
size_t streamvbyte_encode(const uint32_t* in, uint32_t length, uint8_t* out);

// same as streamvbyte_encode but 0,1,2 or 4 bytes per value (plus decoding keys) instead of
// using 1,2,3 or 4 bytes. This might be useful when there's a lot of zeros in the input array.
size_t streamvbyte_encode_0124(const uint32_t* in, uint32_t length, uint8_t* out);

// return the maximum number of compressed bytes given length input integers
// in the worst case we overestimate data bytes required by four, see below
// for a function you can run upfront over your data to compute allocations
// It includes the STREAMVBYTE_PADDING bytes.
static inline size_t streamvbyte_max_compressedbytes(const uint32_t length) {
   // number of control bytes:
   size_t cb = (length + 3) / 4;
   // maximum number of control bytes:
   size_t db = (size_t)length * sizeof(uint32_t);
   return cb + db + STREAMVBYTE_PADDING;
}

// return the exact number of compressed bytes given length input integers
// runtime in O(n) wrt. in; use streamvbyte_max_compressedbyte if you
// care about speed more than potentially over-allocating memory
// Our decoding functions may read (but not use) STREAMVBYTE_PADDING extra bytes beyond
// the compressed data: the user needs to ensure that this region is allocated, and it
// is not included by streamvbyte_compressedbytes.
size_t streamvbyte_compressedbytes(const uint32_t* in, uint32_t length);

// return the exact number of compressed bytes given length input integers
// runtime in O(n) wrt. in; use streamvbyte_max_compressedbyte if you
// care about speed more than potentially over-allocating memory
// Our decoding functions may read (but not use) STREAMVBYTE_PADDING extra bytes beyond
// the compressed data: the user needs to ensure that this region is allocated, and it
// is not included by streamvbyte_compressedbytes.
size_t streamvbyte_compressedbytes_0124(const uint32_t* in, uint32_t length);

// Read "length" 32-bit integers in varint format from in, storing the result in out.
// Returns the number of bytes read. We may read up to STREAMVBYTE_PADDING extra bytes
// from the input buffer (these bytes are read but never used).
// The caller is responsible for knowing how many integers ("length") are to be read:
// this information ought to be stored somehow.
// There is no alignment requirement on the "in" pointer.
// The out pointer should point to length * sizeof(uint32_t) bytes.
size_t streamvbyte_decode(const uint8_t* in, uint32_t* out, uint32_t length);

// Same as streamvbyte_decode but is meant to be used for streams encoded with
// streamvbyte_encode_0124.
size_t streamvbyte_decode_0124(const uint8_t* in, uint32_t* out, uint32_t length);

// Validate an encoded stream.
// This can be used to validate that data received from an untrusted source (disk, network,
// etc...) has a valid length stored alongside it.
// "inLength" is the size of the encoded data "in", and "outLength" is the expected number
// of integers that were compressed.
bool streamvbyte_validate_stream(const uint8_t* in, size_t inLength, uint32_t outLength);

// Same as streamvbyte_validate_stream but is meant to be used for streams encoded with
// streamvbyte_encode_0124.
bool streamvbyte_validate_stream_0124(const uint8_t* in, size_t inLength, uint32_t outLength);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_STREAMVBYTE_H_ */
