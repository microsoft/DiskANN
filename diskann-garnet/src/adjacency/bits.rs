/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Little-endian, LSB-first bitstream for fixed-row Golomb–Rice.

use super::CodecError;

pub(super) struct Writer<'a> {
    out: &'a mut [u8],
    position: usize,
}

impl<'a> Writer<'a> {
    pub(super) fn new(out: &'a mut [u8]) -> Self {
        Self { out, position: 0 }
    }

    // The caller has already computed the exact byte-rounded capacity.
    pub(super) fn write(&mut self, mut value: u64, mut width: u8) {
        while width != 0 {
            let offset = self.position % 8;
            let take = width.min((8 - offset) as u8);
            let byte = &mut self.out[self.position / 8];
            if offset == 0 {
                *byte = 0;
            }
            *byte |= ((value as u8) & ((1u16 << take) - 1) as u8) << offset;
            value >>= take;
            width -= take;
            self.position += usize::from(take);
        }
    }

    pub(super) fn write_unary(&mut self, quotient: u64) -> Result<(), CodecError> {
        let zeros = usize::try_from(quotient).map_err(|_| CodecError::SizeOverflow)?;
        let terminator = self
            .position
            .checked_add(zeros)
            .ok_or(CodecError::SizeOverflow)?;
        let byte = terminator / 8;
        if byte >= self.out.len() {
            return Err(CodecError::OutputTooSmall);
        }
        // The unwritten high bits of a partially initialized byte are zero.
        self.out[self.position.div_ceil(8)..terminator.div_ceil(8)].fill(0);
        self.position = terminator;
        self.write(1, 1);
        Ok(())
    }
}

pub(super) struct Reader<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    pub(super) fn new(input: &'a [u8]) -> Self {
        Self { input, position: 0 }
    }

    pub(super) fn read(&mut self, mut width: u8) -> Result<u64, CodecError> {
        let end = self
            .position
            .checked_add(usize::from(width))
            .ok_or(CodecError::SizeOverflow)?;
        if end.div_ceil(8) > self.input.len() {
            return Err(CodecError::Malformed("truncated Rice remainder"));
        }
        let mut value = 0;
        let mut shift = 0;
        while width != 0 {
            let offset = self.position % 8;
            let take = width.min((8 - offset) as u8);
            let mask = ((1u16 << take) - 1) as u8;
            let byte = (self.input[self.position / 8] >> offset) & mask;
            value |= u64::from(byte) << shift;
            shift += take;
            width -= take;
            self.position += usize::from(take);
        }
        Ok(value)
    }

    pub(super) fn read_unary(&mut self, limit: u64) -> Result<u64, CodecError> {
        let mut quotient = 0;
        loop {
            let byte = *self
                .input
                .get(self.position / 8)
                .ok_or(CodecError::Malformed("unterminated Rice quotient"))?;
            let offset = self.position % 8;
            let zeros = ((byte >> offset).trailing_zeros() as usize).min(8 - offset);
            quotient = u64::checked_add(quotient, zeros as u64)
                .filter(|&n| n <= limit)
                .ok_or(CodecError::Malformed("Rice quotient overflows u64"))?;
            let terminated = zeros < 8 - offset;
            self.position = self
                .position
                .checked_add(zeros + usize::from(terminated))
                .ok_or(CodecError::SizeOverflow)?;
            if terminated {
                return Ok(quotient);
            }
        }
    }

    pub(super) fn finish(self) -> Result<(), CodecError> {
        let bits = self.position % 8;
        if bits != 0 && self.input[self.position / 8] >> bits != 0 {
            return Err(CodecError::Malformed("nonzero Rice padding"));
        }
        super::finished(&self.input[self.position.div_ceil(8)..])
    }
}
