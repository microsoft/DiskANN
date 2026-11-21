/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tinyid's AVX2 byte shuffles, prefix sums and AdaptiveGap book scoring.
//! Loads and stores use bounded slices; neither format requires guard bytes.

use std::arch::x86_64::*;

use super::{
    CodecError,
    adaptive::{self, BOOKS, Score},
    finished, stream, take,
};

pub(super) fn available() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("ssse3")
}

#[target_feature(enable = "avx2,ssse3")]
fn load16(bytes: &[u8; 16]) -> __m128i {
    // SAFETY: the reference provides 16 readable bytes, without an alignment requirement.
    unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
}

#[target_feature(enable = "avx2,ssse3")]
fn load32(bytes: &[u8; 32]) -> __m256i {
    // SAFETY: the reference provides 32 readable bytes, without an alignment requirement.
    unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) }
}

#[target_feature(enable = "avx2,ssse3")]
fn store32(out: &mut [u64; 4], value: __m256i) {
    // SAFETY: the reference provides 32 writable bytes, without an alignment requirement.
    unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), value) }
}

struct EmitTable<const N: usize> {
    shuffle: [[u8; 16]; N],
    len: [u8; N],
}

const fn make_emit<const N: usize>(lane_bytes: usize) -> EmitTable<N> {
    let mut table = EmitTable {
        shuffle: [[0x80; 16]; N],
        len: [0; N],
    };
    let mut tag = 0;
    while tag < N {
        let mut lane = 0;
        let mut len = 0;
        while lane < 16 / lane_bytes {
            let width = stream::width(tag as u8, lane);
            if width > lane_bytes {
                // Qword selectors cannot index the narrow dword table.
                break;
            }
            let mut byte = 0;
            while byte < width {
                table.shuffle[tag][len] = (lane * lane_bytes + byte) as u8;
                len += 1;
                byte += 1;
            }
            lane += 1;
        }
        table.len[tag] = len as u8;
        tag += 1;
    }
    table
}

static NARROW: EmitTable<256> = make_emit(4);
static PAIR: EmitTable<16> = make_emit(8);

#[target_feature(enable = "avx2,ssse3")]
fn narrow_tag(dwords: __m128i) -> u8 {
    // Byte 0 never matters. Pack byte presence flags into words with a
    // low byte of 0/255 and a high byte nonzero iff byte 2 or 3 is present.
    // Clamp to 256 and add 0x7f00: the sign bits are the 00/01/10 selectors.
    let present = _mm_min_epu8(dwords, _mm_set1_epi32(0x0101_0100));
    let pairs = _mm_packus_epi16(present, present);
    let levels = _mm_min_epu16(pairs, _mm_set1_epi16(256));
    _mm_movemask_epi8(_mm_add_epi16(levels, _mm_set1_epi16(0x7f00))) as u8
}

#[target_feature(enable = "avx2,ssse3")]
fn put16(out: &mut [u8], value: __m128i, n: usize) {
    if let Some(chunk) = out.first_chunk_mut::<16>() {
        // SAFETY: chunk provides all 16 bytes. Extra stores are inside scratch.
        unsafe { _mm_storeu_si128(chunk.as_mut_ptr().cast(), value) };
    } else {
        // SAFETY: integers have no invalid bit patterns and both types are 16 bytes.
        let bytes = unsafe { std::mem::transmute::<__m128i, [u8; 16]>(value) };
        out[..n].copy_from_slice(&bytes[..n]);
    }
}

#[target_feature(enable = "avx2,ssse3")]
fn emit_stream(gaps: __m256i, wide: u8, out: &mut [u8]) -> Result<(u8, usize), CodecError> {
    if wide == 0xaa {
        let out = out
            .first_chunk_mut::<32>()
            .ok_or(CodecError::OutputTooSmall)?;
        // SAFETY: out provides all 32 writable bytes.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), gaps) };
        return Ok((0xff, 32));
    }
    let dwords = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(
        gaps,
        _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0),
    ));
    let narrow = narrow_tag(dwords);
    if wide == 0 {
        let idx = usize::from(narrow);
        let n = usize::from(NARROW.len[idx]);
        if n > out.len() {
            return Err(CodecError::OutputTooSmall);
        }
        put16(
            out,
            _mm_shuffle_epi8(dwords, load16(&NARROW.shuffle[idx])),
            n,
        );
        return Ok((narrow, n));
    }
    let tag = narrow | wide | (wide >> 1);
    let lo = usize::from(tag & 15);
    let hi = usize::from(tag >> 4);
    let a = usize::from(PAIR.len[lo]);
    let b = usize::from(PAIR.len[hi]);
    if a + b > out.len() {
        return Err(CodecError::OutputTooSmall);
    }
    let masks = _mm256_set_m128i(load16(&PAIR.shuffle[hi]), load16(&PAIR.shuffle[lo]));
    let packed = _mm256_shuffle_epi8(gaps, masks);
    put16(out, _mm256_castsi256_si128(packed), a);
    put16(&mut out[a..], _mm256_extracti128_si256::<1>(packed), b);
    Ok((tag, a + b))
}

/// `ids` is sorted; `controls` has one slot per four IDs.
#[target_feature(enable = "avx2,ssse3")]
pub(super) fn encode_stream(
    ids: &[u64],
    controls: &mut [u8],
    data: &mut [u8],
) -> Result<usize, CodecError> {
    let Some((&first, &last)) = ids.first().zip(ids.last()) else {
        return Ok(0);
    };
    let compact = last - first <= u64::from(u32::MAX);
    let mut first_wide = u8::from(first > u64::from(u32::MAX)) << 1;
    let zero = _mm256_setzero_si256();
    let mut previous = zero;
    let mut offset = 0;
    for (group, control) in ids.chunks(4).zip(controls) {
        let current = match *group {
            [a] => _mm256_set1_epi64x(a as i64),
            [a, b] => _mm256_setr_epi64x(a as i64, b as i64, b as i64, b as i64),
            [a, b, c] => _mm256_setr_epi64x(a as i64, b as i64, c as i64, c as i64),
            // SAFETY: a full group has 32 readable bytes.
            [_, _, _, _] => unsafe { _mm256_loadu_si256(group.as_ptr().cast()) },
            _ => unreachable!(),
        };
        let bridge = _mm256_permute2x128_si256::<0x21>(previous, current);
        let predecessor = _mm256_alignr_epi8::<8>(current, bridge);
        let gaps = _mm256_sub_epi64(current, predecessor);
        previous = current;
        let wide = if compact {
            // All gaps after the first fit in a dword for compact sorted rows.
            first_wide
        } else {
            let zero_dwords = _mm256_cmpeq_epi32(gaps, zero);
            !(_mm256_movemask_ps(_mm256_castsi256_ps(zero_dwords)) as u8) & 0xaa
        };
        first_wide = 0;
        let (tag, n) = emit_stream(gaps, wide, &mut data[offset..])?;
        *control = tag;
        offset += n;
    }
    Ok(offset)
}

struct Gather {
    shuffle: [u8; 32],
    split: u8,
    len: u8,
}

static GATHER: [Gather; 256] = {
    let mut table = [const {
        Gather {
            shuffle: [0x80; 32],
            split: 0,
            len: 0,
        }
    }; 256];
    let mut tag = 0;
    while tag < 256 {
        let mut lane = 0;
        let mut start = 0;
        while lane < 4 {
            if lane == 2 {
                table[tag].split = start as u8;
                start = 0;
            }
            let width = stream::width(tag as u8, lane);
            let mut byte = 0;
            while byte < width {
                table[tag].shuffle[lane * 8 + byte] = (start + byte) as u8;
                byte += 1;
            }
            start += width;
            lane += 1;
        }
        table[tag].len = table[tag].split + start as u8;
        tag += 1;
    }
    table
};

/// Select a 16-byte load wholly inside the data and the adjustment to its
/// shuffle selectors. Only data shorter than 16 bytes needs a staging copy.
#[target_feature(enable = "avx2,ssse3")]
fn byte_window(data: &[u8], at: usize) -> (__m128i, usize) {
    let start = at.min(data.len() - 16);
    let bytes = data[start..].first_chunk::<16>().unwrap();
    (load16(bytes), at - start)
}

#[target_feature(enable = "avx2,ssse3")]
fn add_prefix(gaps: __m256i, previous: __m256i) -> __m256i {
    let pairs = _mm256_add_epi64(gaps, _mm256_slli_si256::<8>(gaps));
    let low = _mm256_blend_epi32::<0xf0>(
        _mm256_setzero_si256(),
        _mm256_permute4x64_epi64::<0x55>(pairs),
    );
    _mm256_add_epi64(_mm256_add_epi64(pairs, low), previous)
}

/// Shared 1/2/4/8-byte AVX2 decoder for StreamVByte and AdaptiveGap book 0.
/// The latter omits synthetic tail gaps, but complete groups are identical.
#[target_feature(enable = "avx2,ssse3")]
pub(super) fn decode_byte_groups(
    controls: &[u8],
    data: &[u8],
    previous: u64,
    out: &mut [u64],
    synthetic: bool,
) -> Result<(), CodecError> {
    let (groups, tail) = out.as_chunks_mut::<4>();
    let (head_controls, tail_controls) = controls.split_at(groups.len());
    let mut small = [0; 16];
    let windows = if data.len() < 16 && !groups.is_empty() {
        small[..data.len()].copy_from_slice(data);
        &small
    } else {
        data
    };
    let mut previous = _mm256_set1_epi64x(previous as i64);
    let sign = _mm256_set1_epi64x(i64::MIN);
    let mut carries = _mm256_setzero_si256();
    let mut offset = 0;
    for (&control, group) in head_controls.iter().zip(groups) {
        let plan = &GATHER[usize::from(control)];
        let end = offset + usize::from(plan.len);
        if end > data.len() {
            return Err(CodecError::Malformed("truncated byte-width gap group"));
        }
        let (lo, lo_shift) = byte_window(windows, offset);
        let (hi, hi_shift) = byte_window(windows, offset + usize::from(plan.split));
        let shifts = _mm256_set_m128i(_mm_set1_epi8(hi_shift as i8), _mm_set1_epi8(lo_shift as i8));
        let shuffle = _mm256_add_epi8(load32(&plan.shuffle), shifts);
        let gaps = _mm256_shuffle_epi8(_mm256_set_m128i(hi, lo), shuffle);
        let ids = add_prefix(gaps, previous);
        // (previous + gap) wraps exactly when the resulting ID is below gap.
        let wrapped = _mm256_cmpgt_epi64(_mm256_xor_si256(gaps, sign), _mm256_xor_si256(ids, sign));
        carries = _mm256_or_si256(carries, wrapped);
        previous = _mm256_permute4x64_epi64::<0xff>(ids);
        store32(group, ids);
        offset = end;
    }
    if _mm256_testz_si256(carries, carries) == 0 {
        return Err(CodecError::Malformed("adjacency ID overflows u64"));
    }
    stream::decode_scalar(
        tail_controls,
        &data[offset..],
        _mm256_extract_epi64::<0>(previous) as u64,
        tail,
        synthetic,
    )
}

// One lookup scores a gap for all sixteen books. The first half holds
// selected widths in two-bit units, the second half nonzero-selector flags.
static COSTS: [[u8; 32]; 32] = {
    let mut costs = [[0; 32]; 32];
    let mut width = 0;
    while width <= 28 {
        let mut book = 0;
        while book < BOOKS.len() {
            let mut selector = 0;
            while selector < 3 && width > BOOKS[book][selector] as usize {
                selector += 1;
            }
            if width <= BOOKS[book][selector] as usize {
                costs[width][book] = BOOKS[book][selector] / 2;
                costs[width][16 + book] = (selector != 0) as u8;
            }
            book += 1;
        }
        width += 1;
    }
    costs
};

#[target_feature(enable = "avx2,ssse3")]
fn group_cost(sum: __m256i) -> __m256i {
    // Two rounded-up averages with zero compute ceil(sum / 4): round widths
    // to bytes and reduce four selector flags to one nonzero-group flag.
    let zero = _mm256_setzero_si256();
    _mm256_avg_epu8(_mm256_avg_epu8(sum, zero), zero)
}

/// All required widths are <= 28. Ineligible books' scores are ignored.
#[target_feature(enable = "avx2,ssse3")]
pub(super) fn score_adaptive(widths: &[u8]) -> [Score; 16] {
    let mut scores = [Score::default(); 16];
    // At most 16 bytes per four-gap group; flush before the u8 totals wrap.
    for block in widths.chunks(15 * 4) {
        let (groups, tail) = block.as_chunks::<4>();
        let mut total = _mm256_setzero_si256();
        for &[a, b, c, d] in groups {
            let ab = _mm256_add_epi8(gap_cost(a), gap_cost(b));
            let cd = _mm256_add_epi8(gap_cost(c), gap_cost(d));
            total = _mm256_add_epi8(total, group_cost(_mm256_add_epi8(ab, cd)));
        }
        if !tail.is_empty() {
            let mut sum = _mm256_setzero_si256();
            for &width in tail {
                sum = _mm256_add_epi8(sum, gap_cost(width));
            }
            total = _mm256_add_epi8(total, group_cost(sum));
        }
        // SAFETY: byte arrays accept all bit patterns and both types are 32 bytes.
        let totals = unsafe { std::mem::transmute::<__m256i, [u8; 32]>(total) };
        for (book, score) in scores.iter_mut().enumerate() {
            score.bytes += usize::from(totals[book]);
            score.nonzero += usize::from(totals[16 + book]);
        }
    }
    scores
}

#[target_feature(enable = "avx2,ssse3")]
fn gap_cost(width: u8) -> __m256i {
    load32(&COSTS[usize::from(width & 31)])
}

#[target_feature(enable = "avx2,ssse3")]
pub(super) fn pack_adaptive([a, b, c, d]: [u64; 4], widths: [u8; 4]) -> u128 {
    // Two gaps fit in a qword; pack both pairs in parallel, then join them.
    let odd = _mm_sllv_epi64(
        _mm_set_epi64x(d as i64, b as i64),
        _mm_set_epi64x(i64::from(widths[2]), i64::from(widths[0])),
    );
    let pairs = _mm_or_si128(_mm_set_epi64x(c as i64, a as i64), odd);
    // SAFETY: u64 arrays accept all bit patterns and both types are 16 bytes.
    let [lo, hi] = unsafe { std::mem::transmute::<__m128i, [u64; 2]>(pairs) };
    u128::from(lo) | (u128::from(hi) << (widths[0] + widths[1]))
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct SubBytePlan {
    shuffle: [u8; 16],
    shifts: [u32; 4],
    masks: [u32; 4],
    bits: u8,
}

static SUB_BYTE: [[SubBytePlan; 256]; 15] = {
    let empty = SubBytePlan {
        shuffle: [0; 16],
        shifts: [0; 4],
        masks: [0; 4],
        bits: 0,
    };
    let mut table = [[empty; 256]; 15];
    let mut book = 1;
    while book < 16 {
        let mut control = 0;
        while control < 256 {
            let mut plan = empty;
            let mut lane = 0;
            while lane < 4 {
                let width = BOOKS[book][(control >> (2 * lane)) & 3];
                let byte = plan.bits / 8;
                let shift = plan.bits % 8;
                // Every field fits its gathered dword and the 16-byte window.
                assert!(shift + width <= 32 && byte + 4 <= 16);
                let mut i = 0;
                while i < 4 {
                    plan.shuffle[lane * 4 + i] = byte + i as u8;
                    i += 1;
                }
                plan.shifts[lane] = shift as u32;
                plan.masks[lane] = (1u32 << width) - 1;
                plan.bits += width;
                lane += 1;
            }
            table[book - 1][control] = plan;
            control += 1;
        }
        book += 1;
    }
    table
};

#[target_feature(enable = "avx2,ssse3")]
pub(super) fn decode_adaptive(
    book: usize,
    controls: &[u8],
    mut payload: &[u8],
    previous: u64,
    out: &mut [u64],
) -> Result<(), CodecError> {
    let plans = &SUB_BYTE[book - 1];
    let (groups, tail) = out.as_chunks_mut::<4>();
    let (head_controls, tail_controls) = controls.split_at(groups.len());
    let mut current = _mm256_set1_epi64x(previous as i64);
    let mut staged = [0; 32];
    let mut staged_remaining = 0;
    // At most 2^30 groups of gaps below 2^28 per run. Running IDs can
    // wrap at most once, so a single exit comparison detects every wrap.
    for (controls, groups) in head_controls
        .chunks(1 << 30)
        .zip(groups.chunks_mut(1 << 30))
    {
        let before = _mm256_extract_epi64::<0>(current) as u64;
        for (&control, out) in controls.iter().zip(groups) {
            let plan = &plans[usize::from(control)];
            let bits = usize::from(plan.bits);
            let len = bits.div_ceil(8);
            let window = if let Some(window) = payload.first_chunk::<16>() {
                load16(window)
            } else {
                if staged_remaining == 0 {
                    staged_remaining = payload.len();
                    staged[..staged_remaining].copy_from_slice(payload);
                }
                let start = staged_remaining - payload.len();
                load16(staged[start..].first_chunk::<16>().unwrap())
            };
            let encoded = take(&mut payload, len)?;
            if bits % 8 != 0 && encoded[len - 1] >> (bits % 8) != 0 {
                return Err(CodecError::Malformed("nonzero AdaptiveGap group padding"));
            }
            let fields = _mm_shuffle_epi8(window, load16(&plan.shuffle));
            // SAFETY: shifts and masks provide 16 readable bytes each.
            let shifts = unsafe { _mm_loadu_si128(plan.shifts.as_ptr().cast()) };
            let masks = unsafe { _mm_loadu_si128(plan.masks.as_ptr().cast()) };
            let gaps = _mm_and_si128(_mm_srlv_epi32(fields, shifts), masks);
            let pairs = _mm_add_epi32(gaps, _mm_slli_si128::<4>(gaps));
            let prefix = _mm256_cvtepu32_epi64(_mm_add_epi32(pairs, _mm_slli_si128::<8>(pairs)));
            store32(out, _mm256_add_epi64(prefix, current));
            current = _mm256_add_epi64(current, _mm256_permute4x64_epi64::<0xff>(prefix));
        }
        if (_mm256_extract_epi64::<0>(current) as u64) < before {
            return Err(CodecError::Malformed("adjacency ID overflows u64"));
        }
    }
    if tail.is_empty() {
        return finished(payload);
    }
    adaptive::decode_subbyte_scalar(
        book,
        tail_controls,
        payload,
        _mm256_extract_epi64::<0>(current) as u64,
        tail,
    )
}
