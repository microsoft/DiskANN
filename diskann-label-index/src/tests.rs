/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    encode_label_index_jsonl,
    format::{
        write_u32, write_u64, BITSLICE_FORMAT, LABEL_INDEX_MAGIC, LABEL_INDEX_VERSION,
        MAX_LABEL_COUNT,
    },
    EncodedLabelIndex, EncodedLabelQuery, FilterExpressionType,
};
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

    for format in [1, 2, 3] {
        let path = dir.path().join(format!("format-{format}.bin"));
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, format).unwrap();
        writer.flush().unwrap();
        let error = EncodedLabelIndex::load(path).unwrap_err();
        assert!(error.to_string().contains("only dense bitslice"));
    }
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
