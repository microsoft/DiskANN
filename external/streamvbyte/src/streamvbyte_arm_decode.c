

#include "streamvbyte_isadetection.h"
#ifdef STREAMVBYTE_ARM
#include "streamvbyte_shuffle_tables_decode.h"
#ifdef __aarch64__
typedef uint8x16_t decode_t;
#else
typedef uint8x8x2_t decode_t;
#endif
static inline decode_t  _decode_neon(const uint8_t key,
					const uint8_t * restrict *dataPtrPtr) {

  uint8_t len;
  uint8_t *pshuf = (uint8_t *)&shuffleTable[key];
  uint8x16_t decodingShuffle = vld1q_u8(pshuf);

  uint8x16_t compressed = vld1q_u8(*dataPtrPtr);
#ifdef AVOIDLENGTHLOOKUP
  // this avoids the dependency on lengthTable,
  // see https://github.com/lemire/streamvbyte/issues/12
  len = pshuf[12 + (key >> 6)] + 1;
#else
  len = lengthTable[key];
#endif
#ifdef __aarch64__
  uint8x16_t data = vqtbl1q_u8(compressed, decodingShuffle);
#else
  uint8x8x2_t codehalves = {{vget_low_u8(compressed), vget_high_u8(compressed)}};

  uint8x8x2_t data = {{vtbl2_u8(codehalves, vget_low_u8(decodingShuffle)),
		       vtbl2_u8(codehalves, vget_high_u8(decodingShuffle))}};
#endif
  *dataPtrPtr += len;
  return data;
}

static void streamvbyte_decode_quad( const uint8_t * restrict *dataPtrPtr, uint8_t key, uint32_t * restrict out ) {
  decode_t data =_decode_neon( key, dataPtrPtr );
#ifdef __aarch64__
  vst1q_u8((uint8_t *) out, data);
#else
  vst1_u8((uint8_t *) out, data.val[0]);
  vst1_u8((uint8_t *) (out + 2), data.val[1]);
#endif
}

static const uint8_t *svb_decode_vector(uint32_t *out, const uint8_t *keyPtr, const uint8_t *dataPtr, uint32_t count) {
  for(uint32_t i = 0; i < count/4; i++)
    streamvbyte_decode_quad( &dataPtr, keyPtr[i], out + 4*i );

  return dataPtr;
}

static uint64_t svb_validate_vector(const uint8_t **keyPtrPtr,
                                    uint32_t *countPtr) {
  // Reduce the count by how many we'll process
  const uint32_t count = *countPtr & ~7U;
  const uint8_t *keyPtr = *keyPtrPtr;
  *countPtr &= 7;
  *keyPtrPtr += count / 4;

  // Deal with each of the 4 keys in a separate lane
  const int32x4_t shifts = {0, -2, -4, -6};
  const uint32x4_t mask = vdupq_n_u32(3);
  uint32x4_t acc0 = vdupq_n_u32(0);
  uint32x4_t acc1 = vdupq_n_u32(0);

  // Unrolling more than twice doesn't seem to improve performance
  for (uint32_t c = 0; c < count; c += 8) {
    uint32x4_t shifted0 = vshlq_u32(vdupq_n_u32(*keyPtr++), shifts);
    acc0 = vaddq_u32(acc0, vandq_u32(shifted0, mask));
    uint32x4_t shifted1 = vshlq_u32(vdupq_n_u32(*keyPtr++), shifts);
    acc1 = vaddq_u32(acc1, vandq_u32(shifted1, mask));
  }

  // Accumulate the sums and add the +1 for each element (count)
  uint64x2_t sum0 = vpaddlq_u32(acc0);
  uint64x2_t sum1 = vpaddlq_u32(acc1);
  return sum0[0] + sum0[1] + sum1[0] + sum1[1] + count;
}
#endif
