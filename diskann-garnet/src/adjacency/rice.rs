/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tinyid v1 fixed-row DeltaRice: ULEB128 anchor, one k byte, then
//! byte-rounded Golomb–Rice residuals. Empty rows have no payload.

use super::{
    CodecError, ListCodec, add_gap, bits, finished, put_byte, put_uleb, take_byte, take_uleb,
};

pub(super) struct Rice;

impl ListCodec for Rice {
    fn max_payload_len(max_degree: usize) -> usize {
        match max_degree {
            0 => 0,
            1 => 11,
            // k=63 costs 64 bits per residual plus at most one quotient bit
            // for the entire row: the residuals telescope to at most u64::MAX.
            degree => (degree - 1).saturating_mul(8).saturating_add(12),
        }
    }

    fn encode(ids: &[u64], out: &mut [u8], _widths: &mut [u8]) -> Result<usize, CodecError> {
        let Some(&first) = ids.first() else {
            return Ok(0);
        };
        let mut cursor = 0;
        put_uleb(out, &mut cursor, first)?;
        let (k, len) = select_parameter(ids)?;
        // The bit writer keeps its cursor in usize.
        len.checked_mul(8).ok_or(CodecError::SizeOverflow)?;
        put_byte(out, &mut cursor, k)?;
        let end = cursor.checked_add(len).ok_or(CodecError::SizeOverflow)?;
        let encoded = out.get_mut(cursor..end).ok_or(CodecError::OutputTooSmall)?;
        let mut writer = bits::Writer::new(encoded);
        let mask = (1u64 << k) - 1;
        for pair in ids.windows(2) {
            let gap = pair[1] - pair[0];
            writer.write_unary(gap >> k)?;
            writer.write(gap & mask, k);
        }
        Ok(end)
    }

    fn decode(mut payload: &[u8], out: &mut [u64]) -> Result<(), CodecError> {
        let Some((first, tail)) = out.split_first_mut() else {
            return finished(payload);
        };
        let mut previous = take_uleb(&mut payload)?;
        *first = previous;
        let k = take_byte(&mut payload)?;
        if k > 63 || (tail.is_empty() && k != 0) {
            return Err(CodecError::Malformed("invalid Rice parameter"));
        }
        let mut reader = bits::Reader::new(payload);
        for id in tail {
            let quotient = reader.read_unary((u64::MAX - previous) >> k)?;
            let gap = (quotient << k) | reader.read(k)?;
            add_gap(&mut previous, gap, id)?;
        }
        reader.finish()
    }
}

fn select_parameter(ids: &[u64]) -> Result<(u8, usize), CodecError> {
    if ids.len() < 2 {
        return Ok((0, 0));
    }
    let n = (ids.len() - 1) as u128;
    let span = u128::from(ids[ids.len() - 1] - ids[0]);
    // At each skipped k, span >> k >= 4*n+16. The sum of the per-gap
    // quotients is then at least 3*n+17 (rounding down loses fewer than n).
    // Raising k saves at least half that many quotient bits and adds n
    // remainder bits, so it strictly improves even the byte-rounded cost.
    let lower = (span / (4 * n + 16)) as u64;
    let start = (64 - lower.leading_zeros()) as u8;
    let mut best = (0, u128::MAX);
    let mut previous_bits = u128::MAX;
    for k in start..64 {
        let quotient_bits: u128 = ids
            .windows(2)
            .map(|pair| u128::from((pair[1] - pair[0]) >> k))
            .sum();
        let bits = quotient_bits + n * (u128::from(k) + 1);
        // Each gap's quotient savings on incrementing k are nonincreasing.
        // Once the unrounded cost stops decreasing, no larger k can win.
        if bits >= previous_bits {
            break;
        }
        previous_bits = bits;
        let bytes = bits.div_ceil(8);
        if bytes < best.1 {
            best = (k, bytes);
        }
    }
    Ok((
        best.0,
        usize::try_from(best.1).map_err(|_| CodecError::SizeOverflow)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_parameter_matches_exhaustive_byte_cost() {
        for scale in [0, 1, 256, 1 << 28, 1 << 48, u64::MAX] {
            for count in [1, 2, 3, 7, 8, 9, 64, 512] {
                let ids: Vec<u64> = (0..count)
                    .map(|i| (u128::from(scale) * i as u128 / count as u128) as u64)
                    .collect();
                let expected = (0..64u8)
                    .map(|k| {
                        let bits: u128 = ids
                            .windows(2)
                            .map(|pair| u128::from((pair[1] - pair[0]) >> k) + u128::from(k) + 1)
                            .sum();
                        (bits.div_ceil(8) as usize, k)
                    })
                    .min()
                    .unwrap();
                let (k, len) = select_parameter(&ids).unwrap();
                assert_eq!((len, k), expected, "scale={scale}, count={count}");
            }
        }
    }
}
