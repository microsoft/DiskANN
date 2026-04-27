#include "streamvbyte_isadetection.h"
#include "streamvbyte_shuffle_tables_encode.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wdeclaration-after-statement"
#endif

#ifdef STREAMVBYTE_X64
// contributed by aqrit

static size_t svb_data_bytes_scalar(const uint32_t* in, uint32_t length);

STREAMVBYTE_TARGET_SSE41
static inline size_t svb_control_SSE41 (__m128i lo, __m128i hi) {
    const __m128i mask_01 = _mm_set1_epi8(0x01);
    const __m128i mask_7F00 = _mm_set1_epi16(0x7F00);

    __m128i m0, m1;
    size_t keys;

    m0 = _mm_min_epu8(mask_01, lo);
    m1 = _mm_min_epu8(mask_01, hi);
    m0 = _mm_packus_epi16(m0, m1);
    m0 = _mm_min_epi16(m0, mask_01); // convert 0x01FF to 0x0101
    m0 = _mm_adds_epu16(m0, mask_7F00); // convert: 0x0101 to 0x8001, 0xFF01 to 0xFFFF
    keys = (size_t)_mm_movemask_epi8(m0);
    return keys;
}
STREAMVBYTE_UNTARGET_REGION

STREAMVBYTE_TARGET_SSE41
static size_t svb_data_bytes_SSE41 (const uint32_t* in, uint32_t count) {
    size_t dataLen = 0;

    for (const uint32_t* end = &in[(count & ~7U)]; in != end; in += 8)
    {
        __m128i r0, r1;
        size_t keys;

        r0 = _mm_loadu_si128((const __m128i *) &in[0]);
        r1 = _mm_loadu_si128((const __m128i *) &in[4]);

        keys = svb_control_SSE41(r0, r1);
        dataLen += len_lut[keys & 0xFF];
        dataLen += len_lut[keys >> 8];
    }

    dataLen += svb_data_bytes_scalar(in, count & 7);
    return dataLen;
}
STREAMVBYTE_UNTARGET_REGION

STREAMVBYTE_TARGET_SSE41
static size_t streamvbyte_encode_SSE41 (const uint32_t* in, uint32_t count, uint8_t* out) {
	uint32_t keyLen = (count >> 2) + (((count & 3) + 3) >> 2); // 2-bits per each rounded up to byte boundary
	uint8_t *restrict keyPtr = &out[0];
	uint8_t *restrict dataPtr = &out[keyLen]; // variable length data after keys

	for (const uint32_t* end = &in[(count & ~7U)]; in != end; in += 8)
	{
		__m128i r0, r1, r2, r3;
		size_t keys;

		r0 = _mm_loadu_si128((const __m128i*)&in[0]);
		r1 = _mm_loadu_si128((const __m128i*)&in[4]);

		keys = svb_control_SSE41(r0, r1);

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

	// do remaining
	uint32_t key = 0;
	for(size_t i = 0; i < (count & 7); i++)
	{
		uint32_t dw = in[i];
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
