#include "streamvbyte_isadetection.h"
#include "streamvbyte_shuffle_tables_encode.h"
#ifdef STREAMVBYTE_ARM
static const uint8_t pgatherlo[] = {12, 8, 4, 0, 12, 8, 4, 0}; // apparently only used in streamvbyte_encode4
#define concat (1 | 1 << 10 | 1 << 20 | 1 << 30)
#define sum (1 | 1 << 8 | 1 << 16 | 1 << 24)
static const  uint32_t pAggregators[2] = {concat, sum}; // apparently only used in streamvbyte_encode4

static inline size_t streamvbyte_encode4(uint32x4_t data, uint8_t *__restrict__ outData, uint8_t *__restrict__ outCode) {

  const uint8x8_t gatherlo = vld1_u8(pgatherlo);
  const uint32x2_t Aggregators = vld1_u32(pAggregators);

  // lane code is 3 - (saturating sub) (clz(data)/8)
  uint32x4_t clzbytes = vshrq_n_u32(vclzq_u32(data), 3);
  uint32x4_t lanecodes = vqsubq_u32(vdupq_n_u32(3), clzbytes);

  // nops
  uint8x16_t lanebytes = vreinterpretq_u8_u32(lanecodes);
#ifdef __aarch64__
  uint8x8_t lobytes = vqtbl1_u8( lanebytes, gatherlo );
#else
  uint8x8x2_t twohalves = {{vget_low_u8(lanebytes), vget_high_u8(lanebytes)}};

  // shuffle lsbytes into two copies of an int
  uint8x8_t lobytes = vtbl2_u8(twohalves, gatherlo);
#endif

  uint32x2_t mulshift = vreinterpret_u32_u8(lobytes);

  uint32_t codeAndLength[2];
  vst1_u32(codeAndLength, vmul_u32(mulshift, Aggregators));

  uint32_t code = codeAndLength[0] >> 24;
  size_t length = 4 + (codeAndLength[1] >> 24);

  // shuffle in 8-byte chunks
  uint8x16_t databytes = vreinterpretq_u8_u32(data);
  uint8x16_t encodingShuffle = vld1q_u8((uint8_t *) &encodingShuffleTable[code]);
#ifdef __aarch64__
  vst1q_u8(outData, vqtbl1q_u8(databytes, encodingShuffle));
#else
  uint8x8x2_t datahalves = {{vget_low_u8(databytes), vget_high_u8(databytes)}};
  vst1_u8(outData, vtbl2_u8(datahalves, vget_low_u8(encodingShuffle)));
  vst1_u8(outData + 8, vtbl2_u8(datahalves, vget_high_u8(encodingShuffle)));
#endif

  *outCode = (uint8_t) code;
  return length;
}

static inline size_t streamvbyte_encode_quad(const uint32_t *__restrict__ in, uint8_t *__restrict__ outData, uint8_t *__restrict__ outCode) {
  uint32x4_t inq = vld1q_u32(in);

  return streamvbyte_encode4(inq, outData, outCode);
}
#endif
