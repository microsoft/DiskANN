
#include "streamvbyte_shuffle_tables_encode.h"
#include "streamvbyte_isadetection.h"

#ifdef STREAMVBYTE_X64

#ifdef __clang__
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wdeclaration-after-statement"
#endif

STREAMVBYTE_TARGET_SSE41
static __m128i Delta(__m128i curr, __m128i prev) {
  return _mm_sub_epi32(curr, _mm_alignr_epi8(curr, prev, 12));
}
STREAMVBYTE_UNTARGET_REGION

// based on code by aqrit  (streamvbyte_encode_SSE41)
STREAMVBYTE_TARGET_SSE41
static size_t streamvbyte_encode_SSE41_d1_init (const uint32_t* in, uint32_t count, uint8_t* out, uint32_t prev) {
  __m128i Prev = _mm_set1_epi32((int32_t)prev);
	uint32_t keyLen = (count >> 2) + (((count & 3) + 3) >> 2); // 2-bits per each rounded up to byte boundary
	uint8_t *restrict keyPtr = &out[0];
	uint8_t *restrict dataPtr = &out[keyLen]; // variable length data after keys

	const __m128i mask_01 = _mm_set1_epi8(0x01);
	const __m128i mask_7F00 = _mm_set1_epi16(0x7F00);

	for (const uint32_t* end = &in[(count & ~7U)]; in != end; in += 8)
	{
		__m128i rawr0, r0, rawr1, r1, r2, r3;
		size_t keys;

		rawr0 = _mm_loadu_si128((const __m128i*)&in[0]);
    r0 = Delta(rawr0, Prev);
    Prev = rawr0;
		rawr1 = _mm_loadu_si128((const __m128i*)&in[4]);
    r1 = Delta(rawr1, Prev);
    Prev = rawr1;

		r2 = _mm_min_epu8(mask_01, r0);
		r3 = _mm_min_epu8(mask_01, r1);
		r2 = _mm_packus_epi16(r2, r3);
		r2 = _mm_min_epi16(r2, mask_01); // convert 0x01FF to 0x0101
		r2 = _mm_adds_epu16(r2, mask_7F00); // convert: 0x0101 to 0x8001, 0xFF01 to 0xFFFF
		keys = (size_t)_mm_movemask_epi8(r2);

		r2 = _mm_loadu_si128((const __m128i*)&shuf_lut[(keys << 4) & 0x03F0]);
		r3 = _mm_loadu_si128((const __m128i*)&shuf_lut[(keys >> 4) & 0x03F0]);
		r0 = _mm_shuffle_epi8(r0, r2);
		r1 = _mm_shuffle_epi8(r1, r3);

		_mm_storeu_si128((__m128i *)dataPtr, r0);
		dataPtr += len_lut[keys & 0xFF];
		_mm_storeu_si128((__m128i *)dataPtr, r1);
		dataPtr += len_lut[keys >> 8];

		*((uint16_t*)keyPtr) = (uint16_t)keys;
		keyPtr += 2;
	}
  prev = (uint32_t)_mm_extract_epi32(Prev,3);
	// do remaining
	uint32_t key = 0;
	for(size_t i = 0; i < (count & 7); i++)
	{
		uint32_t dw = in[i] - prev; prev = in[i];
		uint32_t symbol = (dw > 0x000000FF) + (dw > 0x0000FFFF) + (dw > 0x00FFFFFF);
		key |= symbol << (i + i);
		memcpy(dataPtr, &dw, 4);
		dataPtr += 1 + symbol;
	}
	memcpy(keyPtr, &key, ((count & 7) + 3) >> 2);

	return (size_t)(dataPtr - out);
}
STREAMVBYTE_UNTARGET_REGION
#endif
