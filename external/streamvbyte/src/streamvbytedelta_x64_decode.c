#include <string.h> // for memcpy
#include "streamvbyte_shuffle_tables_decode.h"
#include "streamvbyte_isadetection.h"
#ifdef STREAMVBYTE_X64

#ifdef __clang__
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wdeclaration-after-statement"
#endif

STREAMVBYTE_TARGET_SSE41
static inline __m128i svb_decode_sse41(uint32_t key,
                                  const uint8_t *__restrict__ *dataPtrPtr) {
  uint8_t len = lengthTable[key];
  __m128i Data = _mm_loadu_si128((const __m128i *)*dataPtrPtr);
  __m128i Shuf = *(__m128i *)&shuffleTable[key];

  Data = _mm_shuffle_epi8(Data, Shuf);
  *dataPtrPtr += len;

  return Data;
}
STREAMVBYTE_UNTARGET_REGION
#define BroadcastLastXMM 0xFF // bits 0-7 all set to choose highest element
STREAMVBYTE_TARGET_SSE41
static inline void svb_write_sse41(uint32_t *out, __m128i Vec) {
  _mm_storeu_si128((__m128i *)out, Vec);
}
STREAMVBYTE_UNTARGET_REGION

STREAMVBYTE_TARGET_SSE41
static __m128i svb_write_sse41_d1(uint32_t *out, __m128i Vec, __m128i Prev) {
  __m128i Add = _mm_slli_si128(Vec, 4); // Cycle 1: [- A B C] (already done)
  Prev = _mm_shuffle_epi32(Prev, BroadcastLastXMM); // Cycle 2: [P P P P]
  Vec = _mm_add_epi32(Vec, Add);                    // Cycle 2: [A AB BC CD]
  Add = _mm_slli_si128(Vec, 8);                     // Cycle 3: [- - A AB]
  Vec = _mm_add_epi32(Vec, Prev);                   // Cycle 3: [PA PAB PBC PCD]
  Vec = _mm_add_epi32(Vec, Add); // Cycle 4: [PA PAB PABC PABCD]

  svb_write_sse41(out, Vec);
  return Vec;
}
STREAMVBYTE_UNTARGET_REGION


STREAMVBYTE_TARGET_SSE41
static inline __m128i svb_write_16bit_sse41_d1(uint32_t *out, __m128i Vec,
                                          __m128i Prev) {
  __m128i High16To32 = _mm_set_epi64x(0xFFFF0F0EFFFF0D0CLL,
                                      0xFFFF0B0AFFFF0908LL);
  // vec == [A B C D E F G H] (16 bit values)
  __m128i Add = _mm_slli_si128(Vec, 2);             // [- A B C D E F G]
  Prev = _mm_shuffle_epi32(Prev, BroadcastLastXMM); // [P P P P] (32-bit)
  Vec = _mm_add_epi32(Vec, Add);                    // [A AB BC CD DE FG GH]
  Add = _mm_slli_si128(Vec, 4);                     // [- - A AB BC CD DE EF]
  Vec = _mm_add_epi32(Vec, Add);        // [A AB ABC ABCD BCDE CDEF DEFG EFGH]
  __m128i V1 = _mm_cvtepu16_epi32(Vec); // [A AB ABC ABCD] (32-bit)
  V1 = _mm_add_epi32(V1, Prev);         // [PA PAB PABC PABCD] (32-bit)
  __m128i V2 =
      _mm_shuffle_epi8(Vec, High16To32); // [BCDE CDEF DEFG EFGH] (32-bit)
  V2 = _mm_add_epi32(V1, V2); // [PABCDE PABCDEF PABCDEFG PABCDEFGH] (32-bit)
  svb_write_sse41(out, V1);
  svb_write_sse41(out + 4, V2);
  return V2;
}
STREAMVBYTE_UNTARGET_REGION

STREAMVBYTE_TARGET_SSE41
static const uint8_t *svb_decode_sse41_d1_init(uint32_t *out,
                                      const uint8_t *__restrict__ keyPtr,
                                      const uint8_t *__restrict__ dataPtr,
                                      uint64_t count, uint32_t prev) {
  uint64_t keybytes = count / 4; // number of key bytes
  if (keybytes >= 8) {
    __m128i Prev = _mm_set1_epi32((int32_t)prev);
    __m128i Data;

    int64_t Offset = -(int64_t)keybytes / 8 + 1;

    const uint64_t *keyPtr64 = (const uint64_t *)keyPtr - Offset;
    uint64_t nextkeys;
    memcpy(&nextkeys, keyPtr64 + Offset, sizeof(nextkeys));
    for (; Offset != 0; ++Offset) {
      uint64_t keys = nextkeys;
      memcpy(&nextkeys, keyPtr64 + Offset + 1, sizeof(nextkeys));
      // faster 16-bit delta since we only have 8-bit values
      if (!keys) { // 32 1-byte ints in a row

        Data = _mm_cvtepu8_epi16(_mm_lddqu_si128((const __m128i *)(dataPtr)));
        Prev = svb_write_16bit_sse41_d1(out, Data, Prev);
        Data = _mm_cvtepu8_epi16(_mm_lddqu_si128((const __m128i *)(dataPtr + 8)));
        Prev = svb_write_16bit_sse41_d1(out + 8, Data, Prev);
        Data = _mm_cvtepu8_epi16(_mm_lddqu_si128((const __m128i *)(dataPtr + 16)));
        Prev = svb_write_16bit_sse41_d1(out + 16, Data, Prev);
        Data = _mm_cvtepu8_epi16(_mm_lddqu_si128((const __m128i *)(dataPtr + 24)));
        Prev = svb_write_16bit_sse41_d1(out + 24, Data, Prev);
        out += 32;
        dataPtr += 32;
        continue;
      }

      Data = svb_decode_sse41(keys & 0x00FF, &dataPtr);
      Prev = svb_write_sse41_d1(out, Data, Prev);
      Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
      Prev = svb_write_sse41_d1(out + 4, Data, Prev);

      keys >>= 16;
      Data = svb_decode_sse41((keys & 0x00FF), &dataPtr);
      Prev = svb_write_sse41_d1(out + 8, Data, Prev);
      Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
      Prev = svb_write_sse41_d1(out + 12, Data, Prev);

      keys >>= 16;
      Data = svb_decode_sse41((keys & 0x00FF), &dataPtr);
      Prev = svb_write_sse41_d1(out + 16, Data, Prev);
      Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
      Prev = svb_write_sse41_d1(out + 20, Data, Prev);

      keys >>= 16;
      Data = svb_decode_sse41((keys & 0x00FF), &dataPtr);
      Prev = svb_write_sse41_d1(out + 24, Data, Prev);
      Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
      Prev = svb_write_sse41_d1(out + 28, Data, Prev);

      out += 32;
    }
    {
      uint64_t keys = nextkeys;
      // faster 16-bit delta since we only have 8-bit values
      if (!keys) { // 32 1-byte ints in a row
        Data = _mm_cvtepu8_epi16(_mm_lddqu_si128((const __m128i *)(dataPtr)));
        Prev = svb_write_16bit_sse41_d1(out, Data, Prev);
        Data = _mm_cvtepu8_epi16(_mm_lddqu_si128((const __m128i *)(dataPtr + 8)));
        Prev = svb_write_16bit_sse41_d1(out + 8, Data, Prev);
        Data = _mm_cvtepu8_epi16(_mm_lddqu_si128((const __m128i *)(dataPtr + 16)));
        Prev = svb_write_16bit_sse41_d1(out + 16, Data, Prev);
        Data = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(dataPtr + 24)));
        Prev = svb_write_16bit_sse41_d1(out + 24, Data, Prev);
        out += 32;
        dataPtr += 32;

      } else {

        Data = svb_decode_sse41(keys & 0x00FF, &dataPtr);
        Prev = svb_write_sse41_d1(out, Data, Prev);
        Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
        Prev = svb_write_sse41_d1(out + 4, Data, Prev);

        keys >>= 16;
        Data = svb_decode_sse41((keys & 0x00FF), &dataPtr);
        Prev = svb_write_sse41_d1(out + 8, Data, Prev);
        Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
        Prev = svb_write_sse41_d1(out + 12, Data, Prev);

        keys >>= 16;
        Data = svb_decode_sse41((keys & 0x00FF), &dataPtr);
        Prev = svb_write_sse41_d1(out + 16, Data, Prev);
        Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
        Prev = svb_write_sse41_d1(out + 20, Data, Prev);

        keys >>= 16;
        Data = svb_decode_sse41((keys & 0x00FF), &dataPtr);
        Prev = svb_write_sse41_d1(out + 24, Data, Prev);
        Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
        Prev = svb_write_sse41_d1(out + 28, Data, Prev);

        out += 32;
      }
    }
    prev = out[-1];
  }
  uint64_t consumedkeys = keybytes - (keybytes & 7);
  return svb_decode_scalar_d1_init(out, keyPtr + consumedkeys, dataPtr,
                                   count & 31, prev);
}
STREAMVBYTE_UNTARGET_REGION
#endif
