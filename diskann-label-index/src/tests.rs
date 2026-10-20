/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    bloom::{append_label_rows, BLOOM_FORMAT},
    encode_bloom_label_index_jsonl, encode_label_index_jsonl,
    format::{
        write_u32, write_u64, BITSLICE_FORMAT, COUNTED_BLOOM_INDEX_VERSION, LABEL_INDEX_MAGIC,
        LABEL_INDEX_VERSION, MAX_LABEL_COUNT, MIXED_BLOOM_INDEX_VERSION,
        PREFIX_EXACT_BLOOM_INDEX_VERSION, SUMMARIZED_BLOOM_INDEX_VERSION,
    },
    BloomFilterConfig, BloomQueryConfig, BloomVisitScratch, EncodedLabelIndex, EncodedLabelQuery,
    FilterExpressionType,
};
use diskann::graph::ext::labeled::{CandidateLabelProvider, QueryLabelProvider};
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    sync::Arc,
};

fn sample_jsonl() -> &'static str {
    concat!(
        "{\"doc_id\":0,\"A\":true,\"group\":\"x\"}\n",
        "{\"doc_id\":1,\"B\":true,\"group\":\"x\"}\n",
        "{\"doc_id\":2,\"A\":true,\"B\":true,\"score\":2}\n",
        "{\"doc_id\":3,\"labels\":[\"C\",\"D\"]}\n",
    )
}

fn round_trip() -> EncodedLabelIndex {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bin");
    std::fs::write(&input, sample_jsonl()).unwrap();
    encode_label_index_jsonl(&input, &output).unwrap();
    EncodedLabelIndex::load(output).unwrap()
}

fn bloom_round_trip(config: BloomFilterConfig) -> EncodedLabelIndex {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    std::fs::write(&input, sample_jsonl()).unwrap();
    encode_bloom_label_index_jsonl(&input, &output, config).unwrap();
    EncodedLabelIndex::load(output).unwrap()
}

fn compile(
    index: &EncodedLabelIndex,
    clauses: &[&str],
    expression_type: FilterExpressionType,
) -> EncodedLabelQuery<'static> {
    index.query(clauses, expression_type).unwrap()
}

fn matching_ids(query: &EncodedLabelQuery, num_vectors: u32) -> Vec<u32> {
    (0..num_vectors)
        .filter(|&vec_id| query.is_match(vec_id))
        .collect()
}

fn assert_send_sync_static<T: Send + Sync + 'static>(_: &T) {}

struct PrefixExactOffsets {
    exact_count_offset: usize,
    flags_offset: usize,
    mapping_start: usize,
    payload_start: usize,
    summary_words_offset: usize,
    summary_start: usize,
    summary_end: usize,
    bit_count: usize,
    hash_count: usize,
    exact_count: usize,
}

fn prefix_exact_offsets(bytes: &[u8]) -> PrefixExactOffsets {
    let read_u32_at =
        |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    let read_u64_at =
        |offset: usize| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;

    let label_count = read_u64_at(24);
    let mut offset = 32;
    for _ in 0..label_count {
        let length = read_u32_at(offset);
        offset += 4 + length;
    }
    let bit_count = read_u32_at(offset);
    let hash_count = read_u32_at(offset + 4);
    let exact_count_offset = offset + 8;
    let exact_count = read_u32_at(exact_count_offset);
    let flags_offset = offset + 12;
    let words_per_row = read_u64_at(offset + 16);
    let mapping_start = offset + 24;
    let payload_start =
        mapping_start + (label_count - exact_count) * hash_count * std::mem::size_of::<u32>();
    let row_count = exact_count + bit_count;
    let payload_end = payload_start + row_count * words_per_row * 8;
    let summary_words_offset = payload_end;
    let summary_words_per_row = read_u64_at(summary_words_offset);
    let summary_start = summary_words_offset + 8;
    let summary_end = summary_start + row_count * summary_words_per_row * 8;
    PrefixExactOffsets {
        exact_count_offset,
        flags_offset,
        mapping_start,
        payload_start,
        summary_words_offset,
        summary_start,
        summary_end,
        bit_count,
        hash_count,
        exact_count,
    }
}

struct MixedV4Offsets {
    exact_count: usize,
    exact_ids: usize,
    exact_bits: usize,
    exact_summary_words: usize,
    exact_summary: usize,
}

fn mixed_v4_offsets(bytes: &[u8]) -> MixedV4Offsets {
    let read_u32_at =
        |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    let read_u64_at =
        |offset: usize| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
    let label_count = read_u64_at(24);
    let mut offset = 32;
    for _ in 0..label_count {
        let length = read_u32_at(offset);
        offset += 4 + length;
    }
    let bit_count = read_u32_at(offset);
    let words_per_row = read_u64_at(offset + 8);
    let bloom_end = offset + 16 + bit_count * words_per_row * 8;
    let summary_words_per_row = read_u64_at(bloom_end);
    let summary_end = bloom_end + 8 + bit_count * summary_words_per_row * 8;
    let exact_count = summary_end + label_count * 4;
    let exact_rows = read_u32_at(exact_count);
    let exact_ids = exact_count + 4;
    let exact_bits = exact_ids + exact_rows * 4;
    let exact_summary_words = exact_bits + exact_rows * words_per_row * 8;
    let exact_summary = exact_summary_words + 8;
    MixedV4Offsets {
        exact_count,
        exact_ids,
        exact_bits,
        exact_summary_words,
        exact_summary,
    }
}

fn legacy_bloom_bytes(version: u32) -> Vec<u8> {
    assert!(matches!(
        version,
        LABEL_INDEX_VERSION | COUNTED_BLOOM_INDEX_VERSION | SUMMARIZED_BLOOM_INDEX_VERSION
    ));
    let labels = ["A", "B"];
    let config = BloomFilterConfig::new(8, 2).unwrap();
    let mut label_rows = Vec::new();
    for label in labels {
        append_label_rows(label, config, &mut label_rows);
    }
    let mut bits = vec![0u64; config.bit_count() as usize];
    for (label_id, vector_ids) in [(0usize, &[0u32][..]), (1, &[1u32, 2][..])] {
        let start = label_id * config.hash_count() as usize;
        for &row in &label_rows[start..start + config.hash_count() as usize] {
            for &vector_id in vector_ids {
                bits[row as usize] |= 1u64 << vector_id;
            }
        }
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&LABEL_INDEX_MAGIC);
    write_u32(&mut bytes, version).unwrap();
    write_u32(&mut bytes, BLOOM_FORMAT).unwrap();
    write_u64(&mut bytes, 3).unwrap();
    write_u64(&mut bytes, labels.len() as u64).unwrap();
    for label in labels {
        write_u32(&mut bytes, label.len() as u32).unwrap();
        bytes.extend_from_slice(label.as_bytes());
    }
    write_u32(&mut bytes, config.bit_count()).unwrap();
    write_u32(&mut bytes, config.hash_count()).unwrap();
    write_u64(&mut bytes, 1).unwrap();
    for word in &bits {
        write_u64(&mut bytes, *word).unwrap();
    }
    if version == SUMMARIZED_BLOOM_INDEX_VERSION {
        write_u64(&mut bytes, 1).unwrap();
        for word in bits {
            write_u64(&mut bytes, u64::from(word != 0)).unwrap();
        }
    }
    if version != LABEL_INDEX_VERSION {
        write_u32(&mut bytes, 1).unwrap();
        write_u32(&mut bytes, 2).unwrap();
    }
    bytes
}

fn mixed_v4_bytes() -> Vec<u8> {
    let labels = ["A", "B"];
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&LABEL_INDEX_MAGIC);
    write_u32(&mut bytes, MIXED_BLOOM_INDEX_VERSION).unwrap();
    write_u32(&mut bytes, BLOOM_FORMAT).unwrap();
    write_u64(&mut bytes, 3).unwrap();
    write_u64(&mut bytes, labels.len() as u64).unwrap();
    for label in labels {
        write_u32(&mut bytes, label.len() as u32).unwrap();
        bytes.extend_from_slice(label.as_bytes());
    }
    write_u32(&mut bytes, 1).unwrap();
    write_u32(&mut bytes, 1).unwrap();
    write_u64(&mut bytes, 1).unwrap();
    write_u64(&mut bytes, 0b111).unwrap();
    write_u64(&mut bytes, 1).unwrap();
    write_u64(&mut bytes, 1).unwrap();
    write_u32(&mut bytes, 1).unwrap();
    write_u32(&mut bytes, 2).unwrap();
    write_u32(&mut bytes, 1).unwrap();
    write_u32(&mut bytes, 1).unwrap();
    write_u64(&mut bytes, 0b110).unwrap();
    write_u64(&mut bytes, 1).unwrap();
    write_u64(&mut bytes, 1).unwrap();
    bytes
}

#[test]
fn dense_round_trip_supports_dnf() {
    let index = round_trip();
    let query = compile(&index, &["A&B", "C&D"], FilterExpressionType::DNF);
    assert_eq!(matching_ids(&query, index.num_vectors()), vec![2, 3]);
    assert_eq!(query.count_matches(), 2);
}

#[test]
fn dense_round_trip_supports_cnf() {
    let index = round_trip();
    let query = compile(
        &index,
        &["A|B", "group=x|score=2"],
        FilterExpressionType::CNF,
    );
    assert_eq!(matching_ids(&query, index.num_vectors()), vec![0, 1, 2]);
    assert_eq!(query.count_matches(), 3);
}

#[test]
fn bloom_round_trip_supports_dnf_and_cnf() {
    let index = bloom_round_trip(BloomFilterConfig::new(1024, 8).unwrap());

    let dnf = compile(&index, &["A&B", "C&D"], FilterExpressionType::DNF);
    assert_eq!(matching_ids(&dnf, index.num_vectors()), vec![2, 3]);
    assert_eq!(dnf.count_matches(), 2);

    let cnf = compile(
        &index,
        &["A|B", "group=x|score=2"],
        FilterExpressionType::CNF,
    );
    assert_eq!(matching_ids(&cnf, index.num_vectors()), vec![0, 1, 2]);
    assert_eq!(cnf.count_matches(), 3);
}

#[test]
fn compiled_bloom_query_implements_library_candidate_provider() {
    let index = bloom_round_trip(BloomFilterConfig::default());
    let query = compile(&index, &["A"], FilterExpressionType::DNF);
    assert_eq!(CandidateLabelProvider::match_upper_bound(&query), Some(2));
    assert!(QueryLabelProvider::is_match(&query, 0));
    let mut candidates = Vec::new();
    CandidateLabelProvider::visit_candidates(&query, |id| candidates.push(id));
    assert_eq!(candidates, matching_ids(&query, index.num_vectors()));
}

#[test]
fn bloom_round_trip_has_no_false_negatives() {
    let index = bloom_round_trip(BloomFilterConfig::default());
    for (label, expected_ids) in [
        ("A", &[0, 2][..]),
        ("B", &[1, 2][..]),
        ("C", &[3][..]),
        ("D", &[3][..]),
        ("group=x", &[0, 1][..]),
        ("score=2", &[2][..]),
    ] {
        let query = compile(&index, &[label], FilterExpressionType::DNF);
        let actual = matching_ids(&query, index.num_vectors());
        for expected_id in expected_ids {
            assert!(
                actual.contains(expected_id),
                "Bloom query for {label} rejected true vector {expected_id}"
            );
        }
    }
}

#[test]
fn bloom_false_positives_follow_approximate_membership() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    std::fs::write(&input, "\"A\"\n\"B\"\n").unwrap();
    encode_bloom_label_index_jsonl(&input, &output, BloomFilterConfig::new(1, 1).unwrap()).unwrap();
    let index = EncodedLabelIndex::load(output).unwrap();

    let query = compile(&index, &["B"], FilterExpressionType::DNF);
    assert_eq!(matching_ids(&query, index.num_vectors()), vec![0, 1]);
    assert_eq!(query.count_matches(), 2);
}

#[test]
fn prefix_exact_bloom_rows_remove_false_positives_and_bloom_contamination() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    std::fs::write(&input, "\"A\"\n\"B\"\n\"B\"\n").unwrap();
    let config = BloomFilterConfig::new(1, 1)
        .unwrap()
        .with_exact_label_count(1)
        .unwrap();
    encode_bloom_label_index_jsonl(&input, &output, config).unwrap();
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(
        &bytes[8..12],
        &PREFIX_EXACT_BLOOM_INDEX_VERSION.to_le_bytes()
    );

    let index = EncodedLabelIndex::load(output).unwrap();
    assert_eq!(index.num_exact_labels(), 1);
    assert_eq!(
        matching_ids(
            &compile(&index, &["B"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![1, 2]
    );
    assert_eq!(
        matching_ids(
            &compile(&index, &["A"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![0]
    );
}

#[test]
fn prefix_exact_bloom_sorts_count_then_label_bytes_deterministically() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    std::fs::write(&input, "\"B\"\n\"A\"\n").unwrap();
    let config = BloomFilterConfig::new(1, 1)
        .unwrap()
        .with_exact_label_count(1)
        .unwrap();
    encode_bloom_label_index_jsonl(&input, &output, config).unwrap();
    let index = EncodedLabelIndex::load(output).unwrap();

    assert_eq!(
        matching_ids(
            &compile(&index, &["A"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![1]
    );
    assert_eq!(
        matching_ids(
            &compile(&index, &["B"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![0]
    );
}

#[test]
fn prefix_exact_bloom_specializes_two_label_dnf_point_checks() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    std::fs::write(&input, "[\"A\",\"B\"]\n\"A\"\n\"A\"\n\"C\"\n").unwrap();
    let config = BloomFilterConfig::new(128, 4)
        .unwrap()
        .with_exact_label_count(2)
        .unwrap();
    encode_bloom_label_index_jsonl(&input, &output, config).unwrap();
    let index = EncodedLabelIndex::load(output).unwrap();

    let exact_pair = compile(&index, &["A&B"], FilterExpressionType::DNF);
    assert!(exact_pair.uses_exact_and2_point_plan());
    assert_eq!(exact_pair.exact_and2_rows(), Some((1, 0)));
    assert!(!QueryLabelProvider::cache_rejections(&exact_pair));
    assert_eq!(matching_ids(&exact_pair, index.num_vectors()), vec![0]);
    assert_eq!(exact_pair.count_matches(), 1);

    assert!(!compile(&index, &["A"], FilterExpressionType::DNF).uses_exact_and2_point_plan());
    let mixed_pair = compile(&index, &["A&C"], FilterExpressionType::DNF);
    assert!(!mixed_pair.uses_exact_and2_point_plan());
    assert!(QueryLabelProvider::cache_rejections(&mixed_pair));
    assert!(!compile(&index, &["A|B"], FilterExpressionType::CNF).uses_exact_and2_point_plan());
}

#[test]
fn prefix_exact_bloom_persists_contiguous_exact_and_bloom_rows() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    std::fs::write(&input, "\"A\"\n\"A\"\n\"B\"\n").unwrap();
    let config = BloomFilterConfig::new(8, 2)
        .unwrap()
        .with_exact_label_count(1)
        .unwrap();
    encode_bloom_label_index_jsonl(&input, &output, config).unwrap();
    let bytes = std::fs::read(&output).unwrap();
    let offsets = prefix_exact_offsets(&bytes);
    assert_eq!(offsets.exact_count, 1);
    assert_eq!(offsets.bit_count, 8);
    assert_eq!(offsets.hash_count, 2);

    let read_u32_at =
        |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let read_u64_at =
        |offset: usize| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    let mut expected_rows = Vec::new();
    append_label_rows("B", config, &mut expected_rows);
    assert_eq!(
        [
            read_u32_at(offsets.mapping_start),
            read_u32_at(offsets.mapping_start + 4)
        ],
        [expected_rows[0], expected_rows[1]]
    );

    assert_eq!(read_u64_at(offsets.payload_start), 0b011);
    for logical_row in 0..offsets.bit_count {
        let physical_row = offsets.exact_count + logical_row;
        let expected = u64::from(expected_rows.contains(&(logical_row as u32))) << 2;
        assert_eq!(
            read_u64_at(offsets.payload_start + physical_row * 8),
            expected
        );
    }
}

#[test]
fn prefix_exact_bloom_supports_zero_and_clamped_exact_counts() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    std::fs::write(&input, "\"A\"\n\"B\"\n").unwrap();

    let no_exact = dir.path().join("no-exact.bin");
    encode_bloom_label_index_jsonl(&input, &no_exact, BloomFilterConfig::new(8, 2).unwrap())
        .unwrap();
    assert_eq!(
        EncodedLabelIndex::load(no_exact)
            .unwrap()
            .num_exact_labels(),
        0
    );

    let all_exact = dir.path().join("all-exact.bin");
    encode_bloom_label_index_jsonl(
        &input,
        &all_exact,
        BloomFilterConfig::new(8, 2)
            .unwrap()
            .with_exact_label_count(100)
            .unwrap(),
    )
    .unwrap();
    let bytes = std::fs::read(&all_exact).unwrap();
    let offsets = prefix_exact_offsets(&bytes);
    assert_eq!(offsets.exact_count, 2);
    assert_eq!(offsets.mapping_start, offsets.payload_start);
    for row in offsets.exact_count..offsets.exact_count + offsets.bit_count {
        assert_eq!(
            u64::from_le_bytes(
                bytes[offsets.payload_start + row * 8..offsets.payload_start + row * 8 + 8]
                    .try_into()
                    .unwrap()
            ),
            0
        );
    }
    assert_eq!(
        EncodedLabelIndex::load(all_exact)
            .unwrap()
            .num_exact_labels(),
        2
    );
}

#[test]
fn prefix_exact_bloom_rejects_invalid_sections() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    let documents = (0..65)
        .map(|id| if id == 64 { "\"COLD\"\n" } else { "\"HOT\"\n" })
        .collect::<String>();
    std::fs::write(&input, documents).unwrap();
    let config = BloomFilterConfig::new(128, 4)
        .unwrap()
        .with_exact_label_count(1)
        .unwrap();
    encode_bloom_label_index_jsonl(&input, &output, config).unwrap();
    let bytes = std::fs::read(&output).unwrap();
    let offsets = prefix_exact_offsets(&bytes);
    assert_eq!(offsets.exact_count, 1);
    assert_eq!(offsets.hash_count, 4);

    let excessive_count = dir.path().join("excessive-exact-count.bin");
    let mut data = bytes.clone();
    data[offsets.exact_count_offset..offsets.exact_count_offset + 4]
        .copy_from_slice(&3u32.to_le_bytes());
    std::fs::write(&excessive_count, data).unwrap();
    assert!(EncodedLabelIndex::load(excessive_count).is_err());

    let invalid_flags = dir.path().join("invalid-flags.bin");
    let mut data = bytes.clone();
    data[offsets.flags_offset..offsets.flags_offset + 4].copy_from_slice(&1u32.to_le_bytes());
    std::fs::write(&invalid_flags, data).unwrap();
    assert!(EncodedLabelIndex::load(invalid_flags).is_err());

    let invalid_mapping = dir.path().join("invalid-mapping.bin");
    let mut data = bytes.clone();
    data[offsets.mapping_start..offsets.mapping_start + 4]
        .copy_from_slice(&(offsets.bit_count as u32).to_le_bytes());
    std::fs::write(&invalid_mapping, data).unwrap();
    assert!(EncodedLabelIndex::load(invalid_mapping).is_err());

    let duplicate_mapping = dir.path().join("duplicate-mapping.bin");
    let mut data = bytes.clone();
    let first = data[offsets.mapping_start..offsets.mapping_start + 4].to_vec();
    data[offsets.mapping_start + 4..offsets.mapping_start + 8].copy_from_slice(&first);
    std::fs::write(&duplicate_mapping, data).unwrap();
    assert!(EncodedLabelIndex::load(duplicate_mapping).is_err());

    let invalid_payload_padding = dir.path().join("invalid-exact-padding.bin");
    let mut data = bytes.clone();
    data[offsets.payload_start + 8..offsets.payload_start + 16]
        .copy_from_slice(&(1u64 << 1).to_le_bytes());
    std::fs::write(&invalid_payload_padding, data).unwrap();
    assert!(EncodedLabelIndex::load(invalid_payload_padding).is_err());

    let invalid_summary_length = dir.path().join("invalid-summary-length.bin");
    let mut data = bytes.clone();
    data[offsets.summary_words_offset..offsets.summary_words_offset + 8]
        .copy_from_slice(&2u64.to_le_bytes());
    std::fs::write(&invalid_summary_length, data).unwrap();
    assert!(EncodedLabelIndex::load(invalid_summary_length).is_err());

    let invalid_summary_padding = dir.path().join("invalid-summary-padding.bin");
    let mut data = bytes.clone();
    data[offsets.summary_start..offsets.summary_start + 8]
        .copy_from_slice(&(1u64 << 2).to_le_bytes());
    std::fs::write(&invalid_summary_padding, data).unwrap();
    assert!(EncodedLabelIndex::load(invalid_summary_padding).is_err());

    let truncated = dir.path().join("truncated-summary.bin");
    std::fs::write(&truncated, &bytes[..offsets.summary_end - 1]).unwrap();
    assert!(EncodedLabelIndex::load(truncated).is_err());
}

#[test]
fn mixed_bloom_summary_agrees_with_point_lookups_across_words() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    let documents = (0..130)
        .map(|id| {
            if id % 17 == 0 {
                "[\"HOT\",\"SPARSE\"]\n"
            } else if id % 5 == 0 {
                "[\"HOT\",\"OTHER\"]\n"
            } else {
                "\"HOT\"\n"
            }
        })
        .collect::<String>();
    std::fs::write(&input, documents).unwrap();
    let config = BloomFilterConfig::new(128, 4)
        .unwrap()
        .with_exact_label_count(1)
        .unwrap();
    encode_bloom_label_index_jsonl(&input, &output, config).unwrap();
    let index = EncodedLabelIndex::load(output).unwrap();

    for (clauses, form) in [
        (vec!["HOT&SPARSE"], FilterExpressionType::DNF),
        (
            vec!["HOT|missing", "SPARSE|OTHER"],
            FilterExpressionType::CNF,
        ),
    ] {
        let query = index.query(&clauses, form).unwrap();
        let expected = matching_ids(&query, index.num_vectors());
        let mut visited = Vec::new();
        query.visit_matches(|id| visited.push(id));
        assert_eq!(visited, expected);
        assert_eq!(query.count_matches(), expected.len() as u64);

        let mut scratch = BloomVisitScratch::default();
        let mut with_scratch = Vec::new();
        query.visit_matches_with_scratch(&mut scratch, |id| with_scratch.push(id));
        assert_eq!(with_scratch, expected);
        assert_eq!(
            query.count_matches_with_scratch(&mut scratch),
            expected.len() as u64
        );
        with_scratch.clear();
        query.visit_matches_with_scratch(&mut scratch, |id| with_scratch.push(id));
        assert_eq!(with_scratch, expected);
    }
}

#[test]
fn thread_local_bloom_scratch_falls_back_for_reentrant_visits() {
    let index = bloom_round_trip(
        BloomFilterConfig::new(128, 4)
            .unwrap()
            .with_exact_label_count(1)
            .unwrap(),
    );
    let query = compile(&index, &["A"], FilterExpressionType::DNF);
    let expected = matching_ids(&query, index.num_vectors());
    let mut visited = Vec::new();
    let mut nested = Vec::new();
    query.visit_matches(|id| {
        visited.push(id);
        if nested.is_empty() {
            query.visit_matches(|nested_id| nested.push(nested_id));
        }
    });
    assert_eq!(visited, expected);
    assert_eq!(nested, expected);
}

#[test]
fn thread_local_bloom_scratch_is_independent_across_threads() {
    let index = bloom_round_trip(
        BloomFilterConfig::new(128, 4)
            .unwrap()
            .with_exact_label_count(1)
            .unwrap(),
    );
    let query = Arc::new(compile(&index, &["A&B", "C&D"], FilterExpressionType::DNF));
    let expected = matching_ids(&query, index.num_vectors());
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let query = Arc::clone(&query);
            std::thread::spawn(move || {
                for _ in 0..100 {
                    let mut actual = Vec::new();
                    query.visit_matches(|id| actual.push(id));
                    assert_eq!(actual, vec![2, 3]);
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(matching_ids(&query, index.num_vectors()), expected);
}

#[test]
fn counted_bloom_bounds_distinct_vectors_for_dnf_and_cnf() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    std::fs::write(
        &input,
        "[\"A\",\"A\",\"B\"]\n[\"B\",\"C\"]\n\"A\"\n[\"C\",\"D\"]\n",
    )
    .unwrap();
    encode_bloom_label_index_jsonl(&input, &output, BloomFilterConfig::default()).unwrap();
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(
        &bytes[8..12],
        &PREFIX_EXACT_BLOOM_INDEX_VERSION.to_le_bytes()
    );
    let index = EncodedLabelIndex::load(&output).unwrap();
    assert!(index.has_bloom_word_summary());
    let dnf = compile(&index, &["A&B", "C&D"], FilterExpressionType::DNF);
    assert_eq!(dnf.match_upper_bound(), Some(3));
    let cnf = compile(&index, &["A|B", "C|D"], FilterExpressionType::CNF);
    assert_eq!(cnf.match_upper_bound(), Some(3));
    assert_eq!(
        compile(&index, &["A"], FilterExpressionType::DNF).match_upper_bound(),
        Some(2)
    );
    assert_eq!(
        compile(&index, &["missing&B", "D"], FilterExpressionType::DNF).match_upper_bound(),
        Some(1)
    );
    assert_eq!(
        compile(&index, &["missing", "A"], FilterExpressionType::CNF).match_upper_bound(),
        Some(0)
    );
}

#[test]
fn legacy_bloom_versions_remain_loadable() {
    let dir = tempfile::tempdir().unwrap();
    for version in [
        LABEL_INDEX_VERSION,
        COUNTED_BLOOM_INDEX_VERSION,
        SUMMARIZED_BLOOM_INDEX_VERSION,
    ] {
        let path = dir.path().join(format!("legacy-v{version}.bin"));
        std::fs::write(&path, legacy_bloom_bytes(version)).unwrap();
        let index = EncodedLabelIndex::load(&path).unwrap();
        assert_eq!(index.num_exact_labels(), 0);
        assert!(compile(&index, &["A"], FilterExpressionType::DNF).is_match(0));
        assert!(compile(&index, &["B"], FilterExpressionType::DNF).is_match(2));
        assert_eq!(
            index.has_bloom_word_summary(),
            version == SUMMARIZED_BLOOM_INDEX_VERSION
        );
        assert_eq!(
            compile(&index, &["B"], FilterExpressionType::DNF).match_upper_bound(),
            (version != LABEL_INDEX_VERSION).then_some(2)
        );
    }

    let mixed = dir.path().join("mixed-v4.bin");
    std::fs::write(&mixed, mixed_v4_bytes()).unwrap();
    let index = EncodedLabelIndex::load(mixed).unwrap();
    assert_eq!(index.num_exact_labels(), 1);
    assert_eq!(
        matching_ids(
            &compile(&index, &["B"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![1, 2]
    );
    assert_eq!(
        matching_ids(
            &compile(&index, &["A"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![0, 1, 2]
    );
}

#[test]
fn mixed_v4_rejects_invalid_exact_sections() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = mixed_v4_bytes();
    let offsets = mixed_v4_offsets(&bytes);

    for (name, data) in [
        ("zero-count", {
            let mut data = bytes.clone();
            data[offsets.exact_count..offsets.exact_count + 4].copy_from_slice(&0u32.to_le_bytes());
            data
        }),
        ("invalid-id", {
            let mut data = bytes.clone();
            data[offsets.exact_ids..offsets.exact_ids + 4].copy_from_slice(&2u32.to_le_bytes());
            data
        }),
        ("invalid-padding", {
            let mut data = bytes.clone();
            data[offsets.exact_bits..offsets.exact_bits + 8]
                .copy_from_slice(&(1u64 << 3).to_le_bytes());
            data
        }),
        ("invalid-summary-length", {
            let mut data = bytes.clone();
            data[offsets.exact_summary_words..offsets.exact_summary_words + 8]
                .copy_from_slice(&2u64.to_le_bytes());
            data
        }),
        ("invalid-summary-padding", {
            let mut data = bytes.clone();
            data[offsets.exact_summary..offsets.exact_summary + 8]
                .copy_from_slice(&(1u64 << 1).to_le_bytes());
            data
        }),
    ] {
        let path = dir.path().join(format!("{name}.bin"));
        std::fs::write(&path, data).unwrap();
        assert!(EncodedLabelIndex::load(path).is_err(), "{name}");
    }
}

#[test]
fn prefix_exact_bloom_always_stores_word_summary() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    std::fs::write(&input, "\"A\"\n".repeat(65)).unwrap();
    encode_bloom_label_index_jsonl(&input, &output, BloomFilterConfig::new(1, 1).unwrap()).unwrap();
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(
        &bytes[8..12],
        &PREFIX_EXACT_BLOOM_INDEX_VERSION.to_le_bytes()
    );
    assert!(EncodedLabelIndex::load(output)
        .unwrap()
        .has_bloom_word_summary());
}

#[test]
fn counted_bloom_rejects_missing_invalid_and_trailing_counts() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    std::fs::write(&input, sample_jsonl()).unwrap();
    encode_bloom_label_index_jsonl(&input, &output, BloomFilterConfig::default()).unwrap();
    let bytes = std::fs::read(output).unwrap();

    for (name, contents) in [
        ("missing", bytes[..bytes.len() - 4].to_vec()),
        ("zero", {
            let mut data = bytes.clone();
            let last = data.len() - 4;
            data[last..].copy_from_slice(&0u32.to_le_bytes());
            data
        }),
        ("too-many", {
            let mut data = bytes.clone();
            let last = data.len() - 4;
            data[last..].copy_from_slice(&5u32.to_le_bytes());
            data
        }),
        ("trailing", {
            let mut data = bytes.clone();
            data.push(0);
            data
        }),
    ] {
        let path = dir.path().join(format!("{name}.bin"));
        std::fs::write(&path, contents).unwrap();
        assert!(EncodedLabelIndex::load(&path).is_err(), "{name}");
    }
}

#[test]
fn bloom_configuration_is_validated() {
    let default = BloomFilterConfig::default();
    assert_eq!(default.bit_count(), 128);
    assert_eq!(default.hash_count(), 4);
    assert_eq!(default.exact_label_count(), 0);
    assert_eq!(
        BloomFilterConfig::new(256, 3)
            .unwrap()
            .with_exact_label_count(100)
            .unwrap()
            .exact_label_count(),
        100
    );

    assert!(BloomFilterConfig::new(0, 1).is_err());
    assert!(BloomFilterConfig::new(128, 0).is_err());
    assert!(BloomFilterConfig::new(4, 5).is_err());
    assert!(BloomFilterConfig::new(128, 65).is_err());
    assert!(BloomFilterConfig::new(128, 4)
        .unwrap()
        .with_exact_label_count(1_000_001)
        .is_err());
}

#[test]
fn bloom_query_percentages_are_validated_and_applied() {
    let default = BloomQueryConfig::default();
    assert_eq!(default.summary_cache_bound_percent(), 1.0);
    assert_eq!(default.summary_scan_max_word_percent(), 10.0);
    assert!(default.should_scan_summarized_words(10, 100));
    assert!(!default.should_scan_summarized_words(11, 100));

    assert!(BloomQueryConfig::new(0.999).is_err());
    assert!(BloomQueryConfig::new(100.001).is_err());
    assert!(BloomQueryConfig::new(f64::NAN).is_err());
    assert!(default.with_summary_scan_max_word_percent(0.999).is_err());
    assert!(default.with_summary_scan_max_word_percent(100.001).is_err());
    assert!(default
        .with_summary_scan_max_word_percent(f64::INFINITY)
        .is_err());
    assert!(default
        .with_summary_scan_max_word_percent(100.0)
        .unwrap()
        .should_scan_summarized_words(100, 100));

    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    let documents = (0..200)
        .map(|id| if id < 2 { "\"RARE\"\n" } else { "\"COMMON\"\n" })
        .collect::<String>();
    std::fs::write(&input, documents).unwrap();
    encode_bloom_label_index_jsonl(&input, &output, BloomFilterConfig::default()).unwrap();
    let index = EncodedLabelIndex::load(output).unwrap();

    let one_percent = index
        .query_with_config(
            &["RARE"],
            FilterExpressionType::DNF,
            BloomQueryConfig::new(1.0).unwrap(),
        )
        .unwrap();
    assert!(one_percent.should_cache_summary_matches());

    let common = index
        .query_with_config(
            &["COMMON"],
            FilterExpressionType::DNF,
            BloomQueryConfig::new(1.0).unwrap(),
        )
        .unwrap();
    assert!(!common.should_cache_summary_matches());

    let full_cache = index
        .query_with_config(
            &["COMMON"],
            FilterExpressionType::DNF,
            BloomQueryConfig::new(100.0).unwrap(),
        )
        .unwrap();
    assert!(full_cache.should_cache_summary_matches());
}

#[test]
fn count_matches_agrees_with_point_lookups_across_word_boundaries() {
    for num_vectors in [1u32, 63, 64, 65, 129, 257, 513] {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let labels = (0..num_vectors)
            .map(|vector_id| {
                if vector_id % 3 == 0 {
                    "[\"A\",\"B\"]\n"
                } else if vector_id % 3 == 1 {
                    "\"B\"\n"
                } else {
                    "\"C\"\n"
                }
            })
            .collect::<String>();
        std::fs::write(&input, labels).unwrap();

        for bloom in [false, true] {
            let output = dir
                .path()
                .join(if bloom { "bloom.bin" } else { "bitslice.bin" });
            if bloom {
                encode_bloom_label_index_jsonl(
                    &input,
                    &output,
                    BloomFilterConfig::new(128, 4).unwrap(),
                )
                .unwrap();
            } else {
                encode_label_index_jsonl(&input, &output).unwrap();
            }
            let index = EncodedLabelIndex::load(&output).unwrap();
            for (clauses, form) in [
                (vec!["A&B", "C&missing"], FilterExpressionType::DNF),
                (vec!["A|missing", "B|C"], FilterExpressionType::CNF),
                (vec!["missing"], FilterExpressionType::DNF),
            ] {
                let query = index.query(&clauses, form).unwrap();
                let expected = matching_ids(&query, num_vectors);
                let mut visited = Vec::new();
                query.visit_matches(|id| visited.push(id));
                assert_eq!(
                    visited, expected,
                    "{num_vectors} vectors, Bloom: {bloom}, form: {form:?}"
                );
                assert_eq!(
                    query.count_matches(),
                    visited.len() as u64,
                    "{num_vectors} vectors, Bloom: {bloom}, form: {form:?}"
                );
            }
        }
    }
}

#[test]
fn bloom_visitor_skips_empty_simd_blocks_without_losing_sparse_ids() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    let rare_ids = [0, 256, 512, 1024];
    let documents = (0..1025)
        .map(|id| {
            if rare_ids.contains(&id) {
                "[\"rare\",\"common\"]\n"
            } else {
                "\"common\"\n"
            }
        })
        .collect::<String>();
    std::fs::write(&input, documents).unwrap();
    encode_bloom_label_index_jsonl(&input, &output, BloomFilterConfig::default()).unwrap();
    let index = EncodedLabelIndex::load(output).unwrap();

    for (clauses, form) in [
        (vec!["rare&common"], FilterExpressionType::DNF),
        (
            vec!["rare|missing", "common|missing"],
            FilterExpressionType::CNF,
        ),
    ] {
        let query = index.query(&clauses, form).unwrap();
        let mut visited = Vec::new();
        query.visit_matches(|id| visited.push(id));
        assert_eq!(visited, matching_ids(&query, index.num_vectors()));
        assert!(rare_ids.iter().all(|id| visited.contains(id)));
    }
}

#[test]
fn query_accepts_owned_strings() {
    let index = round_trip();
    let clauses = vec!["A&B".to_string(), "C&D".to_string()];
    let query = index.query(&clauses, FilterExpressionType::DNF).unwrap();
    assert_eq!(matching_ids(&query, index.num_vectors()), vec![2, 3]);
}

#[test]
fn unknown_labels_follow_normal_form_semantics() {
    let index = round_trip();
    let dnf = compile(&index, &["missing"], FilterExpressionType::DNF);
    assert!(matching_ids(&dnf, index.num_vectors()).is_empty());

    let cnf = compile(&index, &["missing|A"], FilterExpressionType::CNF);
    assert_eq!(matching_ids(&cnf, index.num_vectors()), vec![0, 2]);
}

#[test]
fn summarized_bloom_rejects_invalid_summary_length_padding_and_truncation() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    let documents = (0..65)
        .map(|id| if id % 2 == 0 { "\"A\"\n" } else { "\"B\"\n" })
        .collect::<String>();
    std::fs::write(&input, documents).unwrap();
    encode_bloom_label_index_jsonl(&input, &output, BloomFilterConfig::default()).unwrap();
    let bytes = std::fs::read(&output).unwrap();
    let offsets = prefix_exact_offsets(&bytes);

    let invalid_length = dir.path().join("invalid-summary-length.bin");
    let mut data = bytes.clone();
    data[offsets.summary_words_offset..offsets.summary_words_offset + 8]
        .copy_from_slice(&2u64.to_le_bytes());
    std::fs::write(&invalid_length, data).unwrap();
    assert!(EncodedLabelIndex::load(invalid_length).is_err());

    let invalid_padding = dir.path().join("invalid-summary-padding.bin");
    let mut data = bytes.clone();
    data[offsets.summary_start..offsets.summary_start + 8]
        .copy_from_slice(&(1u64 << 2).to_le_bytes());
    std::fs::write(&invalid_padding, data).unwrap();
    assert!(EncodedLabelIndex::load(invalid_padding).is_err());

    let truncated = dir.path().join("truncated-summary.bin");
    let mut data = bytes[..offsets.summary_end - 8].to_vec();
    data.extend_from_slice(&bytes[offsets.summary_end..]);
    std::fs::write(&truncated, data).unwrap();
    assert!(EncodedLabelIndex::load(truncated).is_err());
}

#[test]
fn compiled_query_remains_usable_after_index_drop() {
    let query = {
        let index = round_trip();
        Arc::new(
            index
                .query(&["A&B", "C&D"], FilterExpressionType::DNF)
                .unwrap(),
        )
    };
    assert_send_sync_static(query.as_ref());
    assert_eq!(matching_ids(&query, 4), vec![2, 3]);
}

#[test]
fn query_rejects_empty_input_and_invalid_clauses() {
    let index = round_trip();
    assert!(index.query::<&str>(&[], FilterExpressionType::DNF).is_err());
    assert!(index.query(&[""], FilterExpressionType::DNF).is_err());
    assert!(index.query(&["A&&B"], FilterExpressionType::DNF).is_err());
    assert!(index.query(&["A|B"], FilterExpressionType::DNF).is_err());
}

#[test]
fn raw_and_object_jsonl_forms_are_supported() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bin");
    std::fs::write(
        &input,
        concat!(
            "\"solo\"\n",
            "[\"left\",\"right\"]\n",
            "{\"doc_id\":4,\"enabled\":true,\"group\":\"g\",\"count\":2,\"deleted\":false,\"labels\":[\"inline\"]}\n",
        ),
    )
    .unwrap();

    encode_label_index_jsonl(&input, &output).unwrap();
    let index = EncodedLabelIndex::load(output).unwrap();

    assert_eq!(
        matching_ids(
            &compile(&index, &["solo"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![0]
    );
    assert_eq!(
        matching_ids(
            &compile(&index, &["left&right"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![1]
    );
    assert_eq!(
        matching_ids(
            &compile(
                &index,
                &["enabled&group=g&count=2&deleted=false&inline"],
                FilterExpressionType::DNF,
            ),
            index.num_vectors()
        ),
        vec![4]
    );
}

#[test]
fn bloom_supports_raw_and_object_jsonl_forms() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bloom.bin");
    std::fs::write(
        &input,
        concat!(
            "\"solo\"\n",
            "[\"left\",\"right\"]\n",
            "{\"doc_id\":4,\"enabled\":true,\"group\":\"g\",\"count\":2,\"deleted\":false,\"labels\":[\"inline\"]}\n",
        ),
    )
    .unwrap();

    encode_bloom_label_index_jsonl(&input, &output, BloomFilterConfig::new(1024, 8).unwrap())
        .unwrap();
    let index = EncodedLabelIndex::load(output).unwrap();

    for (label, expected_id) in [
        ("solo", 0),
        ("left", 1),
        ("right", 1),
        ("enabled", 4),
        ("group=g", 4),
        ("count=2", 4),
        ("deleted=false", 4),
        ("inline", 4),
    ] {
        let query = compile(&index, &[label], FilterExpressionType::DNF);
        assert!(
            query.is_match(expected_id),
            "Bloom query for {label} rejected true vector {expected_id}"
        );
    }
}

#[test]
fn blank_lines_do_not_shift_implicit_document_ids() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bin");
    std::fs::write(&input, "\n\"A\"\n\n\"B\"\n").unwrap();
    encode_label_index_jsonl(&input, &output).unwrap();
    let index = EncodedLabelIndex::load(output).unwrap();

    assert_eq!(
        matching_ids(
            &compile(&index, &["A"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![0]
    );
    assert_eq!(
        matching_ids(
            &compile(&index, &["B"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![1]
    );
}

#[test]
fn invalid_labels_and_duplicate_document_ids_are_rejected() {
    let cases = [
        "{\"doc_id\":0,\"labels\":[\"A&B\"]}\n",
        "{\"doc_id\":0,\"labels\":[\"A\\u0000B\"]}\n",
        "{\"doc_id\":0,\"labels\":[\" A\"]}\n",
        "{\"doc_id\":0,\"A\":true}\n{\"doc_id\":0,\"B\":true}\n",
    ];
    for contents in cases {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, contents).unwrap();
        assert!(encode_label_index_jsonl(&input, &output).is_err());
    }
}

#[test]
fn load_rejects_invalid_magic_and_unsupported_formats() {
    let dir = tempfile::tempdir().unwrap();
    let invalid_magic = dir.path().join("invalid-magic.bin");
    std::fs::write(&invalid_magic, b"not-an-index").unwrap();
    assert!(EncodedLabelIndex::load(invalid_magic).is_err());

    for format in [2, 3] {
        let path = dir.path().join(format!("format-{format}.bin"));
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, format).unwrap();
        writer.flush().unwrap();
        let error = EncodedLabelIndex::load(path).unwrap_err();
        assert!(error.to_string().contains("supported formats"));
    }
}

#[test]
fn load_rejects_invalid_bloom_configuration() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("labels.bloom.bin");
    let mut writer = BufWriter::new(File::create(&path).unwrap());
    writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
    write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
    write_u32(&mut writer, BLOOM_FORMAT).unwrap();
    write_u64(&mut writer, 1).unwrap();
    write_u64(&mut writer, 0).unwrap();
    write_u32(&mut writer, 0).unwrap();
    write_u32(&mut writer, 1).unwrap();
    writer.flush().unwrap();

    assert!(EncodedLabelIndex::load(path).is_err());
}

#[test]
fn load_rejects_invalid_bloom_row_length_and_padding() {
    let dir = tempfile::tempdir().unwrap();

    let invalid_length = dir.path().join("invalid-bloom-length.bin");
    let mut writer = BufWriter::new(File::create(&invalid_length).unwrap());
    writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
    write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
    write_u32(&mut writer, BLOOM_FORMAT).unwrap();
    write_u64(&mut writer, 1).unwrap();
    write_u64(&mut writer, 0).unwrap();
    write_u32(&mut writer, 1).unwrap();
    write_u32(&mut writer, 1).unwrap();
    write_u64(&mut writer, 2).unwrap();
    writer.flush().unwrap();
    assert!(EncodedLabelIndex::load(invalid_length).is_err());

    let invalid_padding = dir.path().join("invalid-bloom-padding.bin");
    let mut writer = BufWriter::new(File::create(&invalid_padding).unwrap());
    writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
    write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
    write_u32(&mut writer, BLOOM_FORMAT).unwrap();
    write_u64(&mut writer, 1).unwrap();
    write_u64(&mut writer, 0).unwrap();
    write_u32(&mut writer, 1).unwrap();
    write_u32(&mut writer, 1).unwrap();
    write_u64(&mut writer, 1).unwrap();
    write_u64(&mut writer, 1u64 << 1).unwrap();
    writer.flush().unwrap();
    assert!(EncodedLabelIndex::load(invalid_padding).is_err());
}

#[test]
fn load_rejects_excessive_label_count_before_allocation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("labels.bin");
    let mut writer = BufWriter::new(File::create(&path).unwrap());
    writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
    write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
    write_u32(&mut writer, BITSLICE_FORMAT).unwrap();
    write_u64(&mut writer, 1).unwrap();
    write_u64(&mut writer, (MAX_LABEL_COUNT as u64) + 1).unwrap();
    writer.flush().unwrap();
    assert!(EncodedLabelIndex::load(path).is_err());
}

#[test]
fn load_rejects_invalid_row_length_and_padding() {
    let dir = tempfile::tempdir().unwrap();

    let invalid_length = dir.path().join("invalid-length.bin");
    let mut writer = BufWriter::new(File::create(&invalid_length).unwrap());
    writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
    write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
    write_u32(&mut writer, BITSLICE_FORMAT).unwrap();
    write_u64(&mut writer, 1).unwrap();
    write_u64(&mut writer, 0).unwrap();
    write_u64(&mut writer, 2).unwrap();
    writer.flush().unwrap();
    assert!(EncodedLabelIndex::load(invalid_length).is_err());

    let invalid_padding = dir.path().join("invalid-padding.bin");
    let mut writer = BufWriter::new(File::create(&invalid_padding).unwrap());
    writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
    write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
    write_u32(&mut writer, BITSLICE_FORMAT).unwrap();
    write_u64(&mut writer, 1).unwrap();
    write_u64(&mut writer, 1).unwrap();
    write_u32(&mut writer, 1).unwrap();
    writer.write_all(b"A").unwrap();
    write_u64(&mut writer, 1).unwrap();
    write_u64(&mut writer, 1u64 << 1).unwrap();
    writer.flush().unwrap();
    assert!(EncodedLabelIndex::load(invalid_padding).is_err());
}

#[test]
fn load_rejects_trailing_bytes_and_zero_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bin");
    std::fs::write(&input, sample_jsonl()).unwrap();
    encode_label_index_jsonl(&input, &output).unwrap();
    let mut file = OpenOptions::new().append(true).open(&output).unwrap();
    file.write_all(&[0]).unwrap();
    drop(file);
    assert!(EncodedLabelIndex::load(output).is_err());

    let zero_vectors = dir.path().join("zero-vectors.bin");
    let mut writer = BufWriter::new(File::create(&zero_vectors).unwrap());
    writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
    write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
    write_u32(&mut writer, BITSLICE_FORMAT).unwrap();
    write_u64(&mut writer, 0).unwrap();
    writer.flush().unwrap();
    assert!(EncodedLabelIndex::load(zero_vectors).is_err());
}
