#include "streamvbytedelta.h"
#include "streamvbyte_isadetection.h"


#ifdef STREAMVBYTE_X64
#include "streamvbyte_shuffle_tables.h"
size_t streamvbyte_encode4(__m128i in, uint8_t *outData, uint8_t *outCode);
#endif

#include <string.h> // for memcpy

#ifdef __clang__
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wdeclaration-after-statement"
#endif

static uint8_t svb_encode_data(uint32_t val, uint8_t *__restrict__ *dataPtrPtr) {
  uint8_t *dataPtr = *dataPtrPtr;
  uint8_t code;

  if (val < (1 << 8)) { // 1 byte
    *dataPtr = (uint8_t)(val);
    *dataPtrPtr += 1;
    code = 0;
  } else if (val < (1 << 16)) { // 2 bytes
    memcpy(dataPtr, &val, 2);   // assumes little endian
    *dataPtrPtr += 2;
    code = 1;
  } else if (val < (1 << 24)) { // 3 bytes
    memcpy(dataPtr, &val, 3);   // assumes little endian
    *dataPtrPtr += 3;
    code = 2;
  } else { // 4 bytes
    memcpy(dataPtr, &val, sizeof(uint32_t));
    *dataPtrPtr += sizeof(uint32_t);
    code = 3;
  }

  return code;
}

static uint8_t *svb_encode_scalar_d1_init(const uint32_t *in,
                                          uint8_t *__restrict__ keyPtr,
                                          uint8_t *__restrict__ dataPtr,
                                          uint32_t count, uint32_t prev) {
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
    uint32_t val = in[c] - prev;
    prev = in[c];
    uint8_t code = svb_encode_data(val, &dataPtr);
    key |= code << shift;
    shift += 2;
  }

  *keyPtr = key;  // write last key (no increment needed)
  return dataPtr; // pointer to first unused data byte
}

#ifdef STREAMVBYTE_X64
// from streamvbyte.c
size_t streamvbyte_encode_quad(__m128i in, uint8_t *outData, uint8_t *outCode);

static __m128i Delta(__m128i curr, __m128i prev) {
  return _mm_sub_epi32(curr, _mm_alignr_epi8(curr, prev, 12));
}

static uint8_t *svb_encode_vector_d1_init(const uint32_t *in,
                                          uint8_t *__restrict__ keyPtr,
                                          uint8_t *__restrict__ dataPtr,
                                          uint32_t count, uint32_t prev) {

  uint8_t *outData = dataPtr;
  uint8_t *outKey = keyPtr;

  uint32_t count4 = count / 4;
  __m128i Prev = _mm_set1_epi32((int32_t)prev);

  for (uint32_t c = 0; c < count4; c++) {
    __m128i vin = _mm_loadu_si128((const __m128i *)(in + 4 * c));
    __m128i deltain = Delta(vin, Prev);
    Prev = vin;
    outData += streamvbyte_encode4(deltain, outData, outKey);
    outKey++;
  }
  prev = (uint32_t)_mm_extract_epi32(Prev, 3); // we grab the last*/
  outData = svb_encode_scalar_d1_init(in + 4 * count4, outKey, outData,
                                      count - 4 * count4, prev);
  // outData = svb_encode_scalar_d1_init(in, outKey, outData, count, prev);
  return outData;
}

#endif

size_t streamvbyte_delta_encode(const uint32_t *in, uint32_t count, uint8_t *out,
                                uint32_t prev) {
  uint8_t *keyPtr = out;             // keys come immediately after 32-bit count
  uint32_t keyLen = (count + 3) / 4; // 2-bits rounded to full byte
  uint8_t *dataPtr = keyPtr + keyLen; // variable byte data after all keys
#ifdef STREAMVBYTE_X64
  if(streamvbyte_sse41()) {
    return (size_t)(svb_encode_vector_d1_init(in, keyPtr, dataPtr, count, prev) - out);
  }
#endif
  return (size_t)(svb_encode_scalar_d1_init(in, keyPtr, dataPtr, count, prev) - out);
}

#ifdef STREAMVBYTE_X64
static inline __m128i svb_decode_sse41(uint32_t key,
                                  const uint8_t *__restrict__ *dataPtrPtr) {
  uint8_t len = lengthTable[key];
  __m128i Data = _mm_loadu_si128((const __m128i *)*dataPtrPtr);
  __m128i Shuf = *(__m128i *)&shuffleTable[key];

  Data = _mm_shuffle_epi8(Data, Shuf);
  *dataPtrPtr += len;

  return Data;
}
#define BroadcastLastXMM 0xFF // bits 0-7 all set to choose highest element

static inline void svb_write_sse41(uint32_t *out, __m128i Vec) {
  _mm_storeu_si128((__m128i *)out, Vec);
}

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

#ifndef _MSC_VER
static __m128i High16To32 = {0xFFFF0B0AFFFF0908, 0xFFFF0F0EFFFF0D0C};
#else
static __m128i High16To32 = {8,  9,  -1, -1, 10, 11, -1, -1,
                             12, 13, -1, -1, 14, 15, -1, -1};
#endif

static inline __m128i svb_write_16bit_sse41_d1(uint32_t *out, __m128i Vec,
                                          __m128i Prev) {
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
#endif

static inline uint32_t svb_decode_data(const uint8_t **dataPtrPtr, uint8_t code) {
  const uint8_t *dataPtr = *dataPtrPtr;
  uint32_t val;

  if (code == 0) { // 1 byte
    val = (uint32_t)*dataPtr;
    dataPtr += 1;
  } else if (code == 1) { // 2 bytes
    val = 0;
    memcpy(&val, dataPtr, 2); // assumes little endian
    dataPtr += 2;
  } else if (code == 2) { // 3 bytes
    val = 0;
    memcpy(&val, dataPtr, 3); // assumes little endian
    dataPtr += 3;
  } else { // code == 3
    memcpy(&val, dataPtr, 4);
    dataPtr += 4;
  }

  *dataPtrPtr = dataPtr;
  return val;
}

static const uint8_t *svb_decode_scalar_d1_init(uint32_t *outPtr,
                                         const uint8_t *keyPtr,
                                         const uint8_t *dataPtr, uint32_t count,
                                         uint32_t prev) {
  if (count == 0)
    return dataPtr; // no reads or writes if no data

  uint8_t shift = 0;
  uint32_t key = *keyPtr++;

  for (uint32_t c = 0; c < count; c++) {
    if (shift == 8) {
      shift = 0;
      key = *keyPtr++;
    }
    uint32_t val = svb_decode_data(&dataPtr, (key >> shift) & 0x3);
    val += prev;
    *outPtr++ = val;
    prev = val;
    shift += 2;
  }

  return dataPtr; // pointer to first unused byte after end
}

#ifdef STREAMVBYTE_X64
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
#endif

size_t streamvbyte_delta_decode(const uint8_t *in, uint32_t *out,
                                uint32_t count, uint32_t prev) {
  uint32_t keyLen = ((count + 3) / 4); // 2-bits per key (rounded up)
  const uint8_t *keyPtr = in;
  const uint8_t *dataPtr = keyPtr + keyLen; // data starts at end of keys
#ifdef STREAMVBYTE_X64
  if(streamvbyte_sse41()) {
    return (size_t)(svb_decode_sse41_d1_init(out, keyPtr, dataPtr, count, prev) - in);
  }
#endif
  return (size_t)(svb_decode_scalar_d1_init(out, keyPtr, dataPtr, count, prev) - in);
}
