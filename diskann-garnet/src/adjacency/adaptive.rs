/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tinyid v1 full-u64 adaptation of GINGER AdaptiveGap. Each row chooses a
//! width book, an optional ULEB128 anchor, and dense or sparse controls.
//! Groups contain up to four real gaps, with each group's bits byte-rounded.

use super::{
    CodecError, ListCodec, add_gap, finished, put_byte, put_uleb, stream, take, take_byte,
    take_uleb, uleb_len,
};

pub(super) struct Adaptive;

pub(super) const BOOKS: [[u8; 4]; 16] = [
    [8, 16, 32, 64],
    [12, 16, 20, 28],
    [8, 12, 20, 28],
    [4, 12, 20, 28],
    [8, 16, 20, 28],
    [8, 12, 16, 28],
    [16, 20, 24, 28],
    [10, 14, 18, 26],
    [6, 14, 22, 26],
    [6, 10, 18, 26],
    [14, 18, 22, 26],
    [6, 14, 18, 26],
    [2, 4, 8, 24],
    [8, 12, 16, 24],
    [12, 16, 20, 24],
    [6, 10, 14, 22],
];
const ANCHOR: u8 = 0x20;
const SPARSE: u8 = 1;

// Widths are 0..=64; selector 4 marks an ineligible book.
const SELECTORS: [[u8; 65]; BOOKS.len()] = {
    let mut table = [[4; 65]; BOOKS.len()];
    let mut book = 0;
    while book < BOOKS.len() {
        let mut bits = 0;
        let mut selector = 0;
        while bits <= 64 && selector < 4 {
            if bits <= BOOKS[book][selector] as usize {
                table[book][bits] = selector as u8;
                bits += 1;
            } else {
                selector += 1;
            }
        }
        book += 1;
    }
    table
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Plan {
    book: usize,
    anchor: bool,
    sparse: bool,
    control_bytes: usize,
    total_bytes: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct Score {
    pub(super) bytes: usize,
    pub(super) nonzero: usize,
}

impl ListCodec for Adaptive {
    fn max_payload_len(max_degree: usize) -> usize {
        // Unanchored book 0 is always an eligible candidate.
        max_degree
            .saturating_mul(8)
            .saturating_add(max_degree.div_ceil(4))
            .saturating_add(usize::from(max_degree != 0))
    }

    fn encode(ids: &[u64], out: &mut [u8], widths: &mut [u8]) -> Result<usize, CodecError> {
        #[cfg(target_arch = "x86_64")]
        let simd = super::x86::available();
        #[cfg(not(target_arch = "x86_64"))]
        let simd = false;
        encode(ids, out, widths, simd)
    }

    fn decode(payload: &[u8], out: &mut [u64]) -> Result<(), CodecError> {
        #[cfg(target_arch = "x86_64")]
        let simd = super::x86::available();
        #[cfg(not(target_arch = "x86_64"))]
        let simd = false;
        decode(payload, out, simd)
    }
}

fn score_scalar(widths: &[u8], max_width: u8) -> [Score; BOOKS.len()] {
    let mut scores = [Score::default(); BOOKS.len()];
    for (book, score) in scores.iter_mut().enumerate() {
        if max_width > BOOKS[book][3] {
            continue;
        }
        for group in widths.chunks(4) {
            let bits: usize = group
                .iter()
                .map(|&w| {
                    let selector = SELECTORS[book][usize::from(w)];
                    usize::from(BOOKS[book][usize::from(selector)])
                })
                .sum();
            score.bytes += bits.div_ceil(8);
            score.nonzero += usize::from(bits > group.len() * usize::from(BOOKS[book][0]));
        }
    }
    scores
}

fn select(widths: &[u8], first: u64, simd: bool) -> Plan {
    let mut best = None;
    for anchor in [false, true] {
        if anchor && widths.len() < 2 {
            continue;
        }
        let widths = &widths[usize::from(anchor)..];
        let groups = widths.len().div_ceil(4);
        let max_width = widths.iter().copied().max().unwrap_or(0);
        let scores = score(widths, max_width, simd);
        for (book, score) in scores.into_iter().enumerate() {
            if max_width > BOOKS[book][3] {
                continue;
            }
            let sparse = groups <= 16 && 2 + score.nonzero < groups;
            let control_bytes = if sparse { 2 + score.nonzero } else { groups };
            let total_bytes =
                1 + if anchor { uleb_len(first) } else { 0 } + control_bytes + score.bytes;
            if best.is_none_or(|p: Plan| total_bytes < p.total_bytes) {
                best = Some(Plan {
                    book,
                    anchor,
                    sparse,
                    control_bytes,
                    total_bytes,
                });
            }
        }
    }
    best.expect("book 0 covers every u64 gap")
}

fn score(widths: &[u8], max_width: u8, _simd: bool) -> [Score; BOOKS.len()] {
    #[cfg(target_arch = "x86_64")]
    if _simd && max_width <= 28 {
        // SAFETY: the caller checked AVX2+SSSE3; all widths fit COSTS.
        return unsafe { super::x86::score_adaptive(widths) };
    }
    score_scalar(widths, max_width)
}

fn encode(ids: &[u64], out: &mut [u8], widths: &mut [u8], simd: bool) -> Result<usize, CodecError> {
    let Some(&first) = ids.first() else {
        return Ok(0);
    };
    let widths = widths
        .get_mut(..ids.len())
        .ok_or(CodecError::WidthScratchTooSmall)?;
    let mut previous = 0;
    for (&id, w) in ids.iter().zip(widths.iter_mut()) {
        *w = (64 - (id - previous).leading_zeros()) as u8;
        previous = id;
    }
    let plan = select(widths, first, simd);
    let out = out
        .get_mut(..plan.total_bytes)
        .ok_or(CodecError::OutputTooSmall)?;
    let tag = (u8::from(plan.anchor) << 5) | ((plan.book as u8) << 1) | u8::from(plan.sparse);
    let mut cursor = 0;
    put_byte(out, &mut cursor, tag)?;
    if plan.anchor {
        put_uleb(out, &mut cursor, first)?;
    }
    let (controls, mut payload) = out[cursor..].split_at_mut(plan.control_bytes);
    let mut mask = 0u16;
    let mut sparse_cursor = 2;
    let start = usize::from(plan.anchor);
    let mut previous = if plan.anchor { first } else { 0 };
    for (index, (group, widths)) in ids[start..]
        .chunks(4)
        .zip(widths[start..].chunks(4))
        .enumerate()
    {
        let mut control = 0;
        let mut bits = 0;
        let mut selected = [0; 4];
        for (lane, &width) in widths.iter().enumerate() {
            let selector = SELECTORS[plan.book][usize::from(width)];
            let width = BOOKS[plan.book][usize::from(selector)];
            control |= selector << (2 * lane);
            bits += usize::from(width);
            selected[lane] = width;
        }
        if plan.sparse {
            if control != 0 {
                mask |= 1 << index;
                controls[sparse_cursor] = control;
                sparse_cursor += 1;
            }
        } else {
            controls[index] = control;
        }
        let (encoded, remaining) = payload.split_at_mut(bits.div_ceil(8));
        payload = remaining;
        if plan.book == 0 {
            let mut offset = 0;
            for (&id, width) in group.iter().zip(selected) {
                stream::put_gap(encoded, &mut offset, id - previous, usize::from(width / 8))?;
                previous = id;
            }
        } else {
            let mut gaps = [0; 4];
            for (&id, gap) in group.iter().zip(&mut gaps) {
                *gap = id - previous;
                previous = id;
            }
            let packed = pack(gaps, selected, simd);
            encoded.copy_from_slice(&packed.to_le_bytes()[..encoded.len()]);
        }
    }
    if plan.sparse {
        controls[..2].copy_from_slice(&mask.to_le_bytes());
    }
    debug_assert!(payload.is_empty());
    Ok(plan.total_bytes)
}

fn pack(gaps: [u64; 4], widths: [u8; 4], _simd: bool) -> u128 {
    #[cfg(target_arch = "x86_64")]
    if _simd {
        // SAFETY: the caller checked AVX2+SSSE3; all gaps have sub-byte widths.
        return unsafe { super::x86::pack_adaptive(gaps, widths) };
    }
    let mut packed = 0;
    let mut shift = 0;
    for (gap, width) in gaps.into_iter().zip(widths) {
        packed |= u128::from(gap) << shift;
        shift += width;
    }
    packed
}

fn decode(mut payload: &[u8], out: &mut [u64], _simd: bool) -> Result<(), CodecError> {
    if out.is_empty() {
        return finished(payload);
    }
    let tag = take_byte(&mut payload)?;
    if tag & 0xc0 != 0 {
        return Err(CodecError::Malformed("reserved AdaptiveGap tag bits"));
    }
    let book = usize::from((tag >> 1) & 15);
    let (previous, out) = if tag & ANCHOR != 0 {
        if out.len() < 2 {
            return Err(CodecError::Malformed("AdaptiveGap anchor on a singleton"));
        }
        let previous = take_uleb(&mut payload)?;
        out[0] = previous;
        (previous, &mut out[1..])
    } else {
        (0, out)
    };
    let groups = out.len().div_ceil(4);
    let mut dense;
    let controls = if tag & SPARSE != 0 {
        if groups > 16 {
            return Err(CodecError::Malformed(
                "too many AdaptiveGap sparse controls",
            ));
        }
        let mut mask = u16::from_le_bytes(take(&mut payload, 2)?.try_into().unwrap());
        let used = mask.count_ones() as usize;
        if u32::from(mask) >> groups != 0 || used + 2 >= groups {
            return Err(CodecError::Malformed("invalid AdaptiveGap sparse mask"));
        }
        dense = [0; 16];
        for &value in take(&mut payload, used)? {
            if value == 0 {
                return Err(CodecError::Malformed("zero AdaptiveGap sparse control"));
            }
            dense[mask.trailing_zeros() as usize] = value;
            mask &= mask - 1;
        }
        &dense[..groups]
    } else {
        take(&mut payload, groups)?
    };
    #[cfg(target_arch = "x86_64")]
    if _simd {
        // SAFETY: the caller checked AVX2+SSSE3 and book/control bounds.
        return unsafe {
            if book == 0 {
                super::x86::decode_byte_groups(controls, payload, previous, out, false)
            } else {
                super::x86::decode_adaptive(book, controls, payload, previous, out)
            }
        };
    }
    if book == 0 {
        return stream::decode_scalar(controls, payload, previous, out, false);
    }
    decode_subbyte_scalar(book, controls, payload, previous, out)
}

pub(super) fn decode_subbyte_scalar(
    book: usize,
    controls: &[u8],
    mut payload: &[u8],
    mut previous: u64,
    out: &mut [u64],
) -> Result<(), CodecError> {
    for (&control, group) in controls.iter().zip(out.chunks_mut(4)) {
        stream::check_tail(control, group.len())?;
        let widths =
            (0..group.len()).map(|lane| BOOKS[book][usize::from((control >> (2 * lane)) & 3)]);
        let bits: usize = widths.clone().map(usize::from).sum();
        let bytes = take(&mut payload, bits.div_ceil(8))?;
        let mut packed = [0; 16];
        packed[..bytes.len()].copy_from_slice(bytes);
        let mut packed = u128::from_le_bytes(packed);
        for (id, width) in group.iter_mut().zip(widths) {
            let gap = packed as u64 & ((1u64 << width) - 1);
            packed >>= width;
            add_gap(&mut previous, gap, id)?;
        }
        if packed != 0 {
            return Err(CodecError::Malformed("nonzero AdaptiveGap group padding"));
        }
    }
    finished(payload)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn avx2_and_scalar_match_at_scoring_group_and_density_boundaries() {
        if !crate::adjacency::x86::available() {
            return;
        }
        let mut widths = [0; 512];
        let mut scalar = [0xff; 5000];
        let mut simd = [0xff; 5000];
        for count in [0, 1, 2, 3, 4, 5, 59, 60, 61, 63, 64, 65, 511, 512] {
            for max in [0, 1, 3, 5, 11, 17, 22, 26, 28, 29, 33, 48, 56] {
                let mut previous = 0;
                let ids: Vec<_> = (0..count)
                    .map(|i| {
                        previous += if i % 13 == 0 {
                            1u64 << max
                        } else {
                            (i * 7 % 31) as u64
                        };
                        previous
                    })
                    .collect();
                let a = encode(&ids, &mut scalar, &mut widths, false).unwrap();
                let b = encode(&ids, &mut simd, &mut widths, true).unwrap();
                assert_eq!(&scalar[..a], &simd[..b], "count={count} max={max}");
                let mut output = vec![0; count];
                for backend in [false, true] {
                    decode(&simd[..b], &mut output, backend).unwrap();
                    assert_eq!(ids, output);
                }
            }
        }
    }

    #[test]
    fn avx2_subbyte_packing_and_decoding_match_every_control() {
        if !crate::adjacency::x86::available() {
            return;
        }
        for (book, menu) in BOOKS.iter().enumerate().skip(1) {
            for control in u8::MIN..=u8::MAX {
                let widths =
                    std::array::from_fn(|lane| menu[usize::from((control >> (2 * lane)) & 3)]);
                let gaps = widths.map(|width| (1u64 << width) - 1);
                let packed = pack(gaps, widths, false);
                assert_eq!(pack(gaps, widths, true), packed);
                let len = widths
                    .into_iter()
                    .map(usize::from)
                    .sum::<usize>()
                    .div_ceil(8);
                let mut payload = [0; 16];
                payload[0] = (book as u8) << 1;
                payload[1] = control;
                payload[2..][..len].copy_from_slice(&packed.to_le_bytes()[..len]);
                let mut expected = [0; 4];
                let mut actual = [0; 4];
                decode(&payload[..2 + len], &mut expected, false).unwrap();
                decode(&payload[..2 + len], &mut actual, true).unwrap();
                assert_eq!(actual, expected, "book={book} control={control}");
            }
        }
    }
}
