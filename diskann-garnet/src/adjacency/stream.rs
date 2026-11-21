/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tinyid v1 D1 StreamVByte64: four 1/2/4/8-byte gaps per control byte.
//! A partial group ends with one zero byte per absent gap.

use super::{CodecError, ListCodec, add_gap, finished, take};

pub(super) struct Stream;

impl ListCodec for Stream {
    fn max_payload_len(max_degree: usize) -> usize {
        max_degree
            .saturating_mul(8)
            .saturating_add(max_degree.div_ceil(4))
            .saturating_add((4 - max_degree % 4) % 4)
    }

    fn encode(ids: &[u64], out: &mut [u8], _widths: &mut [u8]) -> Result<usize, CodecError> {
        let groups = ids.len().div_ceil(4);
        let (controls, data) = out
            .split_at_mut_checked(groups)
            .ok_or(CodecError::OutputTooSmall)?;
        #[cfg(target_arch = "x86_64")]
        if super::x86::available() {
            // SAFETY: AVX2+SSSE3 were checked above.
            return unsafe { super::x86::encode_stream(ids, controls, data) }.map(|n| groups + n);
        }
        encode_scalar(ids, controls, data).map(|n| groups + n)
    }

    fn decode(mut payload: &[u8], out: &mut [u64]) -> Result<(), CodecError> {
        let controls = take(&mut payload, out.len().div_ceil(4))?;
        #[cfg(target_arch = "x86_64")]
        if super::x86::available() {
            // SAFETY: AVX2+SSSE3 were checked above.
            return unsafe { super::x86::decode_byte_groups(controls, payload, 0, out, true) };
        }
        decode_scalar(controls, payload, 0, out, true)
    }
}

pub(super) const fn width(control: u8, lane: usize) -> usize {
    1 << ((control >> (2 * lane)) & 3)
}

pub(super) fn encode_scalar(
    ids: &[u64],
    controls: &mut [u8],
    data: &mut [u8],
) -> Result<usize, CodecError> {
    let mut previous = 0;
    let mut offset = 0;
    for (group, control) in ids.chunks(4).zip(controls) {
        *control = 0;
        for (lane, &id) in group.iter().enumerate() {
            let gap = id - previous;
            previous = id;
            let bytes: usize = match gap {
                0..=0xff => 1,
                0x100..=0xffff => 2,
                0x1_0000..=0xffff_ffff => 4,
                _ => 8,
            };
            *control |= (bytes.trailing_zeros() as u8) << (2 * lane);
            put_gap(data, &mut offset, gap, bytes)?;
        }
        let padding = 4 - group.len();
        data.get_mut(offset..offset + padding)
            .ok_or(CodecError::OutputTooSmall)?
            .fill(0);
        offset += padding;
    }
    Ok(offset)
}

pub(super) fn put_gap(
    data: &mut [u8],
    offset: &mut usize,
    gap: u64,
    bytes: usize,
) -> Result<(), CodecError> {
    data.get_mut(*offset..*offset + bytes)
        .ok_or(CodecError::OutputTooSmall)?
        .copy_from_slice(&gap.to_le_bytes()[..bytes]);
    *offset += bytes;
    Ok(())
}

pub(super) fn decode_scalar(
    controls: &[u8],
    mut payload: &[u8],
    mut previous: u64,
    out: &mut [u64],
    synthetic: bool,
) -> Result<(), CodecError> {
    for (&control, group) in controls.iter().zip(out.chunks_mut(4)) {
        check_tail(control, group.len())?;
        for (lane, id) in group.iter_mut().enumerate() {
            let gap = take_gap(&mut payload, width(control, lane))?;
            add_gap(&mut previous, gap, id)?;
        }
        if synthetic
            && take(&mut payload, 4 - group.len())?
                .iter()
                .any(|&gap| gap != 0)
        {
            return Err(CodecError::Malformed("nonzero synthetic StreamVByte gap"));
        }
    }
    finished(payload)
}

pub(super) fn check_tail(control: u8, lanes: usize) -> Result<(), CodecError> {
    if usize::from(control) >> (2 * lanes) != 0 {
        return Err(CodecError::Malformed("nonzero unused gap selectors"));
    }
    Ok(())
}

pub(super) fn take_gap(payload: &mut &[u8], bytes: usize) -> Result<u64, CodecError> {
    let src = take(payload, bytes)?;
    // These four lengths are known to both byte-width books.
    Ok(match bytes {
        1 => u64::from(src[0]),
        2 => u64::from(u16::from_le_bytes(src.try_into().unwrap())),
        4 => u64::from(u32::from_le_bytes(src.try_into().unwrap())),
        8 => u64::from_le_bytes(src.try_into().unwrap()),
        _ => unreachable!(),
    })
}
