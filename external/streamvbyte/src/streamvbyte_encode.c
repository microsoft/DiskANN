#include "streamvbyte.h"
#include "streamvbyte_isadetection.h"

#include <string.h> // for memcpy

#ifdef __clang__
#pragma clang diagnostic ignored "-Wdeclaration-after-statement"
#endif

#ifdef STREAMVBYTE_X64
#include "streamvbyte_x64_encode.c"
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


#ifdef STREAMVBYTE_IS_ARM64
#include "streamvbyte_arm_encode.c"
#endif

static size_t svb_data_bytes_scalar(const uint32_t* in, uint32_t length) {
   size_t db = 0;
   for (uint32_t c = 0; c < length; c++) {
      uint32_t val = in[c];
      
      uint32_t bytes = 1 + (val > 0x000000FF) + (val > 0x0000FFFF) + (val > 0x00FFFFFF);
      db += bytes;
   }
   return db;
}

static size_t svb_data_bytes_0124_scalar(const uint32_t* in, uint32_t length) {
   size_t db = 0;
   for (uint32_t c = 0; c < length; c++) {
      uint32_t val = in[c];

      uint32_t bytes = (val > 0x00000000) + (val > 0x000000FF) + (val > 0x0000FFFF) * 2;
      db += bytes;
   }
   return db;
}

size_t streamvbyte_compressedbytes(const uint32_t* in, uint32_t length) {
   // number of control bytes:
   size_t cb = (length + 3) / 4;

#ifdef STREAMVBYTE_X64
   if (streamvbyte_sse41()) {
      return cb + svb_data_bytes_SSE41(in, length);
   }
#endif
   return cb + svb_data_bytes_scalar(in, length);
}

size_t streamvbyte_compressedbytes_0124(const uint32_t* in, uint32_t length) {
   // number of control bytes:
   size_t cb = (length + 3) / 4;

   return cb + svb_data_bytes_0124_scalar(in, length);
}


// Encode an array of a given length read from in to out in streamvbyte format.
// Returns the number of bytes written.
size_t streamvbyte_encode(const uint32_t *in, uint32_t count, uint8_t *out) {
#ifdef STREAMVBYTE_X64
  if(streamvbyte_sse41()) {
    return streamvbyte_encode_SSE41(in,count,out);
  }
#endif
  uint8_t *keyPtr = out;
  uint32_t keyLen = (count + 3) / 4;  // 2-bits rounded to full byte
  uint8_t *dataPtr = keyPtr + keyLen; // variable byte data after all keys

#if defined(STREAMVBYTE_IS_ARM64)

  uint32_t count_quads = count / 4;
  count -= 4 * count_quads;

  for (uint32_t c = 0; c < count_quads; c++) {
    dataPtr += streamvbyte_encode_quad(in, dataPtr, keyPtr);
    keyPtr++;
    in += 4;
  }

#endif

  return (size_t)(svb_encode_scalar(in, keyPtr, dataPtr, count) - out);
}
