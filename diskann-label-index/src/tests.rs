/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    bloom::BLOOM_FORMAT,
    encode_bloom_label_index_jsonl, encode_label_index_jsonl,
    format::{
        write_u32, write_u64, BITSLICE_FORMAT, COUNTED_BLOOM_INDEX_VERSION, LABEL_INDEX_MAGIC,
        LABEL_INDEX_VERSION, MAX_LABEL_COUNT,
    },
    BloomFilterConfig, EncodedLabelIndex, EncodedLabelQuery, FilterExpressionType,
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
    assert_eq!(&bytes[8..12], &COUNTED_BLOOM_INDEX_VERSION.to_le_bytes());
    let index = EncodedLabelIndex::load(&output).unwrap();
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

    let old = dir.path().join("labels.legacy.bin");
    let mut old_bytes = bytes;
    old_bytes[8..12].copy_from_slice(&LABEL_INDEX_VERSION.to_le_bytes());
    old_bytes.truncate(old_bytes.len() - index.num_labels() * 4);
    std::fs::write(&old, old_bytes).unwrap();
    let old_index = EncodedLabelIndex::load(&old).unwrap();
    assert_eq!(
        compile(&old_index, &["A"], FilterExpressionType::DNF).match_upper_bound(),
        None
    );
    assert_eq!(
        matching_ids(&cnf, index.num_vectors()),
        matching_ids(
            &compile(&old_index, &["A|B", "C|D"], FilterExpressionType::CNF),
            old_index.num_vectors()
        )
    );
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

    assert!(BloomFilterConfig::new(0, 1).is_err());
    assert!(BloomFilterConfig::new(128, 0).is_err());
    assert!(BloomFilterConfig::new(4, 5).is_err());
    assert!(BloomFilterConfig::new(128, 65).is_err());
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
