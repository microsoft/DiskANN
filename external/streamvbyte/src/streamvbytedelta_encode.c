#include "streamvbytedelta.h"
#include "streamvbyte_isadetection.h"

#include <string.h> // for memcpy

#ifdef STREAMVBYTE_X64
#include "streamvbytedelta_x64_encode.c"
#endif

#ifdef __clang__
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

size_t streamvbyte_delta_encode(const uint32_t *in, uint32_t count, uint8_t *out,
                                uint32_t prev) {
#ifdef STREAMVBYTE_X64
  if(streamvbyte_sse41()) {
    return streamvbyte_encode_SSE41_d1_init(in,count,out,prev);
  }
#endif
  uint8_t *keyPtr = out;             // keys come immediately after 32-bit count
  uint32_t keyLen = (count + 3) / 4; // 2-bits rounded to full byte
  uint8_t *dataPtr = keyPtr + keyLen; // variable byte data after all keys
  return (size_t)(svb_encode_scalar_d1_init(in, keyPtr, dataPtr, count, prev) - out);
}
