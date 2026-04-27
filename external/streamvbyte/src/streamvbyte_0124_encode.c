#include "streamvbyte.h"
#include "streamvbyte_isadetection.h"
#include "streamvbyte_shuffle_tables_0124_encode.h"

#include <string.h> // for memcpy

#ifdef __clang__
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wdeclaration-after-statement"
#endif

static uint8_t svb_encode_data(uint32_t val, uint8_t *__restrict__ *dataPtrPtr) {
  uint8_t *dataPtr = *dataPtrPtr;
  uint8_t code;

  if (val == 0) { // 0 bytes
    code = 0;
  } else if (val < (1 << 8)) { // 1 bytes
    *dataPtr = (uint8_t)(val);
    *dataPtrPtr += 1;
    code = 1;
  } else if (val < (1 << 16)) { // 2 bytes
    memcpy(dataPtr, &val, 2);   // assumes little endian
    *dataPtrPtr += 2;
    code = 2;
  } else { // 4 bytes
    memcpy(dataPtr, &val, sizeof(uint32_t));
    *dataPtrPtr += sizeof(uint32_t);
    code = 3;
  }
  return code;
}

static uint8_t *svb_encode_scalar(const uint32_t *in,
                                  uint8_t *__restrict__ keyPtr,
                                  uint8_t *__restrict__ dataPtr,
                                  uint32_t count) {
  if (count == 0)
    return dataPtr; // exit immediately if no data

  uint8_t shift = 0; // cycles 0, 2, 4, 6, 0, 2, 4, 6, ...
  uint8_t key = 0;
  for (uint32_t c = 0; c < count; c++) {
    if (shift == 8) {
      shift = 0;
      *keyPtr++ = key;
      key = 0;
    }
    uint32_t val = in[c];
    uint8_t code = svb_encode_data(val, &dataPtr);
    key |= code << shift;
    shift += 2;
  }

  *keyPtr = key;  // write last key (no increment needed)
  return dataPtr; // pointer to first unused data byte
}

#ifdef STREAMVBYTE_X64
STREAMVBYTE_TARGET_SSE41
static size_t streamvbyte_encode4(__m128i in, uint8_t *outData, uint8_t *outCode) {
  const __m128i Ones = _mm_set1_epi32(0x01010101);
  const __m128i GatherBits = _mm_set1_epi32(0x08040102);
  const __m128i CodeTable = _mm_set_epi32(0x03030303, 0x03030303, 0x03030303, 0x02020100);
  const __m128i GatherBytes = _mm_set_epi32(0, 0, 0x0D090501, 0x0D090501);
  const __m128i Aggregators = _mm_set_epi32(0, 0, 0x01010101, 0x10400104);

  __m128i m0, m1;
  m0 = _mm_min_epu8(in, Ones); // set byte to 1 if it is not zero
  m0 = _mm_madd_epi16(m0, GatherBits); // gather bits 8,16,24 to bits 8,9,10
  m1 = _mm_shuffle_epi8(CodeTable, m0); // translate to a 2-bit encoded symbol
  m1 = _mm_shuffle_epi8(m1, GatherBytes); // gather bytes holding symbols; 2 copies
  m1 = _mm_madd_epi16(m1, Aggregators); // sum dword_1, pack dword_0

  size_t code = (size_t)_mm_extract_epi8(m1, 1);
  size_t length = lengthTable[code];

  __m128i* shuf = (__m128i*)(((uint8_t*)encodingShuffleTable) + code * 16);
  __m128i out = _mm_shuffle_epi8(in, _mm_loadu_si128(shuf)); // todo: aligned access

  _mm_storeu_si128((__m128i *)outData, out);
  *outCode = (uint8_t)code;
  return length;
}
STREAMVBYTE_UNTARGET_REGION

STREAMVBYTE_TARGET_SSE41
static size_t streamvbyte_encode_quad(const uint32_t *in, uint8_t *outData, uint8_t *outKey) {
  __m128i vin = _mm_loadu_si128((const __m128i *) in );
  return streamvbyte_encode4(vin, outData, outKey);
}
STREAMVBYTE_UNTARGET_REGION

#endif

size_t streamvbyte_encode_0124(const uint32_t *in, uint32_t count, uint8_t *out) {
  uint8_t *keyPtr = out;
  uint32_t keyLen = (count + 3) / 4;  // 2-bits rounded to full byte
  uint8_t *dataPtr = keyPtr + keyLen; // variable byte data after all keys
#ifdef STREAMVBYTE_X64
  if(streamvbyte_sse41()) {
    uint32_t count_quads = count / 4;
    count -= 4 * count_quads;
    for (uint32_t c = 0; c < count_quads; c++) {
      dataPtr += streamvbyte_encode_quad(in, dataPtr, keyPtr);
      keyPtr++;
      in += 4;
    }
  }
#endif
  return (size_t)(svb_encode_scalar(in, keyPtr, dataPtr, count) - out);

}
