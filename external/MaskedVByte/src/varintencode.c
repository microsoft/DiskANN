#include "varintencode.h"


size_t vbyte_encode_delta(const uint32_t *in, size_t length, uint8_t *bout, uint32_t prev) {
	uint8_t *initbout = bout;
	for (size_t k = 0; k < length; ++k) {
		const uint32_t val = in[k] - prev;
		prev = in[k];
		if (val < (1U << 7)) {
			*bout = val & 0x7F;
			++bout;
		} else if (val < (1U << 14)) {
			*bout = (uint8_t)((val & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(val >> 7);
			++bout;
		} else if (val < (1U << 21)) {
			*bout = (uint8_t)((val & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(((val >> 7) & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(val >> 14);
			++bout;
		} else if (val < (1U << 28)) {
			*bout = (uint8_t)((val & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(((val >> 7) & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(((val >> 14) & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(val >> 21);
			++bout;
		} else {
			*bout = (uint8_t)((val & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(((val >> 7) & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(((val >> 14) & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(((val >> 21) & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(val >> 28);
			++bout;
		}
	}
	return bout - initbout;
}

size_t vbyte_encode(const uint32_t *in, size_t length, uint8_t *bout) {
	uint8_t *initbout = bout;
	for (size_t k = 0; k < length; ++k) {
		const uint32_t val = in[k];

		if (val < (1U << 7)) {
			*bout = val & 0x7F;
			++bout;
		} else if (val < (1U << 14)) {
			*bout = (uint8_t)((val & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(val >> 7);
			++bout;
		} else if (val < (1U << 21)) {
			*bout = (uint8_t)((val & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(((val >> 7) & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(val >> 14);
			++bout;
		} else if (val < (1U << 28)) {
			*bout = (uint8_t)((val & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(((val >> 7) & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(((val >> 14) & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(val >> 21);
			++bout;
		} else {
			*bout = (uint8_t)((val & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(((val >> 7) & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(((val >> 14) & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(((val >> 21) & 0x7F) | (1U << 7));
			++bout;
			*bout = (uint8_t)(val >> 28);
			++bout;
		}
	}
	return bout - initbout;
}





