#include "streamvbyte_isadetection.h"
#include "streamvbyte_shuffle_tables_decode.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wcast-align"
#endif

#ifdef STREAMVBYTE_X64
STREAMVBYTE_TARGET_SSE41
static inline __m128i svb_decode_sse41(uint32_t key,
                                  const uint8_t *__restrict__ *dataPtrPtr) {
  uint8_t len;
  __m128i Data = _mm_loadu_si128((const __m128i *)*dataPtrPtr);
  uint8_t *pshuf = (uint8_t *) &shuffleTable[key];
  __m128i Shuf = *(__m128i *)pshuf;
#ifdef AVOIDLENGTHLOOKUP
  // this avoids the dependency on lengthTable,
  // see https://github.com/lemire/streamvbyte/issues/12
  len = pshuf[12 + (key >> 6)] + 1;
#else
  len = lengthTable[key];
#endif
  Data = _mm_shuffle_epi8(Data, Shuf);
  *dataPtrPtr += len;
  return Data;
}
STREAMVBYTE_UNTARGET_REGION


STREAMVBYTE_TARGET_SSE41
static inline void svb_write_sse41(uint32_t *out, __m128i Vec) {
  _mm_storeu_si128((__m128i *)out, Vec);
}
STREAMVBYTE_UNTARGET_REGION



STREAMVBYTE_TARGET_SSE41
static inline const uint8_t *svb_decode_sse41_simple(uint32_t *out,
                                     const uint8_t *__restrict__ keyPtr,
                                     const uint8_t *__restrict__ dataPtr,
                                     uint64_t count) {

  uint64_t keybytes = count / 4; // number of key bytes
  __m128i Data;
  if (keybytes >= 8) {

    int64_t Offset = -(int64_t)keybytes / 8 + 1;

    const uint64_t *keyPtr64 = (const uint64_t *)keyPtr - Offset;
    uint64_t nextkeys;
    memcpy(&nextkeys, keyPtr64 + Offset, sizeof(nextkeys));
    for (; Offset != 0; ++Offset) {
      uint64_t keys = nextkeys;
      memcpy(&nextkeys, keyPtr64 + Offset + 1, sizeof(nextkeys));

      Data = svb_decode_sse41((keys & 0xFF), &dataPtr);
      svb_write_sse41(out, Data);
      Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
      svb_write_sse41(out + 4, Data);

      keys >>= 16;
      Data = svb_decode_sse41((keys & 0xFF), &dataPtr);
      svb_write_sse41(out + 8, Data);
      Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
      svb_write_sse41(out + 12, Data);

      keys >>= 16;
      Data = svb_decode_sse41((keys & 0xFF), &dataPtr);
      svb_write_sse41(out + 16, Data);
      Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
      svb_write_sse41(out + 20, Data);

      keys >>= 16;
      Data = svb_decode_sse41((keys & 0xFF), &dataPtr);
      svb_write_sse41(out + 24, Data);
      Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
      svb_write_sse41(out + 28, Data);

      out += 32;
    }
    {
      uint64_t keys = nextkeys;

      Data = svb_decode_sse41((keys & 0xFF), &dataPtr);
      svb_write_sse41(out, Data);
      Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
      svb_write_sse41(out + 4, Data);

      keys >>= 16;
      Data = svb_decode_sse41((keys & 0xFF), &dataPtr);
      svb_write_sse41(out + 8, Data);
      Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
      svb_write_sse41(out + 12, Data);

      keys >>= 16;
      Data = svb_decode_sse41((keys & 0xFF), &dataPtr);
      svb_write_sse41(out + 16, Data);
      Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
      svb_write_sse41(out + 20, Data);

      keys >>= 16;
      Data = svb_decode_sse41((keys & 0xFF), &dataPtr);
      svb_write_sse41(out + 24, Data);
      Data = svb_decode_sse41((keys & 0xFF00) >> 8, &dataPtr);
      svb_write_sse41(out + 28, Data);

      out += 32;
    }
  }
  return dataPtr;
}
STREAMVBYTE_UNTARGET_REGION
#endif
