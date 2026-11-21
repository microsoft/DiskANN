/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::{AdjacencyCodec, CodecError};
use AdjacencyCodec::{AdaptiveGap, Identity, Rice, StreamVByte64};

const CODECS: [AdjacencyCodec; 4] = [Identity, StreamVByte64, AdaptiveGap, Rice];

fn encode(codec: AdjacencyCodec, ids: &mut [u64], max_degree: usize) -> Vec<u8> {
    let mut record = vec![0xc7; codec.max_encoded_len(max_degree)];
    let len = codec
        .encode(ids, max_degree, &mut record, &mut vec![0; max_degree])
        .unwrap();
    record.truncate(len);
    record
}

#[test]
fn selector_and_identity_layout_are_compatible() {
    assert_eq!(AdjacencyCodec::default(), Identity);
    for (id, codec) in CODECS.into_iter().enumerate() {
        assert_eq!(AdjacencyCodec::try_from(id as u8), Ok(codec));
        assert_eq!(codec as u8, id as u8);
    }
    assert_eq!(
        AdjacencyCodec::try_from(4),
        Err(CodecError::UnknownCodec(4))
    );

    let mut ids = [0x0102_0304_0506_0708, 1];
    let mut record = encode(Identity, &mut ids, 3);
    let mut expected = ids
        .into_iter()
        .chain([0, 2])
        .flat_map(u64::to_ne_bytes)
        .collect::<Vec<_>>();
    assert_eq!(record, expected);
    // Historical records do not guarantee the contents of padding slots.
    record[16..24].fill(0xff);
    let mut out = [u64::MAX; 3];
    assert_eq!(AdjacencyCodec::decode(&record, 3, &mut out), Ok(2));
    assert_eq!(out, [ids[0], ids[1], u64::MAX]);
    expected[24..].copy_from_slice(&4u64.to_ne_bytes());
    assert_eq!(
        AdjacencyCodec::decode(&expected, 3, &mut out),
        Err(CodecError::InvalidDegree)
    );
}

#[test]
fn tinyid_v1_payload_vectors() {
    let vectors: [(AdjacencyCodec, &[u64], &[u8]); 3] = [
        (
            StreamVByte64,
            &[0, 1, 257, 65537, 65537],
            &[1, 5, 0x50, 0, 0, 1, 0, 1, 0, 255, 0, 0, 0, 0],
        ),
        (AdaptiveGap, &[1, 2, 3, 4], &[2, 4, 24, 0, 0x55]),
        (Rice, &[1, 2, 3, 4], &[3, 4, 1, 0, 0x2a]),
    ];
    for (codec, ids, wire) in vectors {
        assert_eq!(encode(codec, &mut ids.to_vec(), 16), wire);
        let mut out = [0; 16];
        let count = AdjacencyCodec::decode(wire, 16, &mut out).unwrap();
        assert_eq!(&out[..count], ids);
        assert_eq!(encode(codec, &mut [], 0), [codec as u8, 0]);
        assert_eq!(AdjacencyCodec::decode(&[codec as u8, 0], 0, &mut []), Ok(0));
    }
}

#[test]
fn equal_and_oversized_compressed_lengths_are_escaped() {
    for (codec, degree, shift, unescaped_delta) in [
        (StreamVByte64, 14, 32, 0),
        (StreamVByte64, 13, 32, 1),
        (AdaptiveGap, 20, 56, 0),
        (AdaptiveGap, 24, 56, 1),
    ] {
        let mut ids: Vec<u64> = (1..=degree as u64).map(|id| id << shift).collect();
        let identity_len = Identity.max_encoded_len(degree);
        let mut record = encode(codec, &mut ids, degree);
        assert_eq!(record.len(), identity_len + unescaped_delta + 1);
        assert_eq!(record.last(), Some(&0));
        let mut out = vec![0; degree];
        assert_eq!(
            AdjacencyCodec::decode(&record, degree, &mut out),
            Ok(degree)
        );
        assert_eq!(out, ids);

        // A caller may supply exactly the encoded record length with no guards.
        let mut exact = vec![0x9b; record.len()];
        let mut widths = vec![0; degree];
        assert_eq!(
            codec.encode(&mut ids, degree, &mut exact, &mut widths),
            Ok(record.len())
        );
        assert_eq!(record, exact);
        assert_eq!(
            codec.encode(
                &mut ids,
                degree,
                &mut exact[..record.len() - 1],
                &mut widths
            ),
            Err(CodecError::OutputTooSmall)
        );
        *record.last_mut().unwrap() = 1;
        assert_eq!(
            AdjacencyCodec::decode(&record, degree, &mut out),
            Err(CodecError::Malformed("invalid size suffix"))
        );
    }
}

#[test]
fn roundtrips_cover_full_u64_and_degree_bounds() {
    for codec in CODECS {
        for max_degree in [0, 1, 2, 3, 4, 5, 16, 127, 128, 255, 511, 512, 1025] {
            for count in [0, max_degree / 2, max_degree] {
                let mut ids: Vec<u64> = (0..count)
                    .map(|i| match i % 5 {
                        0 => u64::MAX,
                        1 => 0,
                        2 => (1 << 63) - 1,
                        3 => 1 << 63,
                        _ => (i as u64) << 32,
                    })
                    .collect();
                let original = ids.clone();
                let encoded = encode(codec, &mut ids, max_degree);
                let mut out = vec![0; max_degree];
                assert_eq!(
                    AdjacencyCodec::decode(&encoded, max_degree, &mut out),
                    Ok(count)
                );
                assert_eq!(&out[..count], ids);
                if codec == Identity {
                    assert_eq!(ids, original);
                } else {
                    assert!(ids.is_sorted());
                    assert_ne!(encoded.len(), Identity.max_encoded_len(max_degree));
                }
            }
        }
    }
}

#[test]
fn decoder_rejects_malformed_payloads_and_counts() {
    let bad: &[&[u8]] = &[
        &[],                                                         // missing codec
        &[0, 0],                // identity has no tagged encoding
        &[4, 0],                // unknown codec
        &[1, 0x80, 0],          // overlong zero count
        &[1, 33],               // count > D
        &[1, 0, 0],             // trailing byte on empty row
        &[1, 1],                // truncated controls
        &[1, 4, 0, 1, 2, 3],    // truncated gap group
        &[1, 1, 4, 1, 0, 0, 0], // nonzero unused selector
        &[1, 1, 0, 1, 2, 0, 0], // nonzero synthetic gap
        &[1, 4, 3, 255, 255, 255, 255, 255, 255, 255, 255, 1, 0, 0], // ID overflow
        &[2, 1, 0xc0, 0, 0],    // reserved AdaptiveGap bits
        &[2, 1, 0x20, 0, 0],    // anchor on singleton
        &[2, 65, 1],            // invalid count here; sparse too long on larger D
        &[2, 16, 1, 0, 1],      // sparse mask beyond group count
        &[2, 16, 1, 1, 0, 0],   // sparse control explicitly zero
        &[2, 1, 24, 0, 0x81],   // nonzero partial-group bit padding
        &[2, 4, 20, 1, 0, 0, 0, 0, 0, 0, 0, 0x80], // nonzero full-group padding
        &[
            2, 5, 56, 255, 255, 255, 255, 255, 255, 255, 255, 255, 1, 0, 1,
        ], // anchor + gap overflows
        &[3, 1, 0, 1],          // nonzero singleton k
        &[3, 2, 0, 64],         // out-of-range k
        &[3, 2, 0, 63, 4],      // quotient 2 cannot fit at k=63
        &[3, 2, 0, 63, 1],      // unterminated remainder
        &[3, 2, 0, 0, 0],       // unterminated unary
        &[3, 2, 0, 0, 0x81],    // nonzero residual padding
        &[3, 2, 0, 0, 1, 0],    // trailing residual byte
    ];
    let mut out = [0; 32];
    for record in bad {
        assert!(
            AdjacencyCodec::decode(record, 32, &mut out).is_err(),
            "accepted {record:?}"
        );
    }
    assert!(AdjacencyCodec::decode(&[2, 65, 1], 65, &mut [0; 65]).is_err());
    assert_eq!(
        AdjacencyCodec::decode(&[3, 2, 0, 0, 1], 32, &mut [0; 1]),
        Err(CodecError::OutputTooSmall)
    );
    assert_eq!(
        AdaptiveGap.encode(&mut [1, 2], 1, &mut [0; 32], &mut [0; 2]),
        Err(CodecError::InvalidDegree)
    );
    assert_eq!(
        AdaptiveGap.encode(&mut [1, 2], 2, &mut [0; 32], &mut [0; 1]),
        Err(CodecError::WidthScratchTooSmall)
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn stream_avx2_and_scalar_match_all_controls_and_tail_sizes() {
    use super::{stream, x86};

    if !x86::available() {
        return;
    }
    for tag in u8::MIN..=u8::MAX {
        for count in 1..=11 {
            let mut previous = 0;
            let ids: Vec<_> = (0..count)
                .map(|i| {
                    let selector = (tag >> (2 * (i % 4))) & 3;
                    previous += 1u64 << [0, 8, 16, 32][usize::from(selector)];
                    previous
                })
                .collect();
            let mut scalar_controls = [0; 3];
            let mut avx2_controls = [0; 3];
            let groups = ids.len().div_ceil(4);
            let scalar_controls = &mut scalar_controls[..groups];
            let avx2_controls = &mut avx2_controls[..groups];
            let mut scalar = [0x75; 96];
            let mut avx2 = [0x91; 96];
            let expected = stream::encode_scalar(&ids, scalar_controls, &mut scalar).unwrap();
            // SAFETY: AVX2+SSSE3 were checked above.
            let actual =
                unsafe { x86::encode_stream(&ids, avx2_controls, &mut avx2[..expected]) }.unwrap();
            assert_eq!(scalar_controls, avx2_controls, "tag={tag} count={count}");
            assert_eq!(
                &scalar[..expected],
                &avx2[..actual],
                "tag={tag} count={count}"
            );
            let mut out = [0; 11];
            let out = &mut out[..count];
            stream::decode_scalar(scalar_controls, &scalar[..expected], 0, out, true).unwrap();
            assert_eq!(&*out, ids.as_slice());
            out.fill(0);
            // SAFETY: AVX2+SSSE3 were checked above.
            unsafe { x86::decode_byte_groups(avx2_controls, &avx2[..actual], 0, out, true) }
                .unwrap();
            assert_eq!(&*out, ids.as_slice());
        }
    }
}
