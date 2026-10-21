use crate::{
    encode_bloom_label_index_jsonl, encode_label_index_jsonl,
    format::{LABEL_INDEX_FLAGS, LABEL_INDEX_MAGIC, LABEL_INDEX_VERSION},
    BloomFilterConfig, BloomQueryConfig, EncodedLabelIndex, EncodedLabelPointEvaluator,
    EncodedLabelQuery, FilterExpressionType,
};
use diskann::graph::ext::labeled::{CandidateLabelProvider, QueryLabelProvider};
use std::{fs::OpenOptions, io::Write, sync::Arc};

fn compile(
    index: &EncodedLabelIndex,
    clauses: &[&str],
    expression_type: FilterExpressionType,
) -> EncodedLabelQuery<'static> {
    index.query(clauses, expression_type).unwrap()
}

fn matching_ids(query: &impl QueryLabelProvider<u32>, num_vectors: u32) -> Vec<u32> {
    (0..num_vectors)
        .filter(|&vector_id| query.is_match(vector_id))
        .collect()
}

fn sample_documents() -> &'static str {
    concat!(
        "[\"A\",\"B\"]\n",
        "\"A\"\n",
        "{\"labels\":[\"A\",\"C\"]}\n",
        "{\"B\":true,\"market\":\"US\"}\n",
    )
}

fn encode_exact(documents: &str) -> (tempfile::TempDir, EncodedLabelIndex) {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bin");
    std::fs::write(&input, documents).unwrap();
    encode_label_index_jsonl(&input, &output).unwrap();
    let index = EncodedLabelIndex::load(output).unwrap();
    (dir, index)
}

fn encode_bloom(documents: &str, exact_label_count: u32) -> (tempfile::TempDir, EncodedLabelIndex) {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bin");
    std::fs::write(&input, documents).unwrap();
    let config = BloomFilterConfig::new(128, 4)
        .unwrap()
        .with_exact_label_count(exact_label_count)
        .unwrap();
    encode_bloom_label_index_jsonl(&input, &output, config).unwrap();
    let index = EncodedLabelIndex::load(output).unwrap();
    (dir, index)
}

#[derive(Debug)]
struct Layout {
    exact_rows: usize,
    bloom_rows: usize,
    hash_count: usize,
    words_per_row: usize,
    summary_words_per_row: usize,
    metadata_start: usize,
    summary_start: usize,
    payload_start: usize,
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn layout(bytes: &[u8]) -> Layout {
    assert_eq!(&bytes[..8], &LABEL_INDEX_MAGIC);
    assert_eq!(read_u32(bytes, 8), LABEL_INDEX_VERSION);
    assert_eq!(read_u32(bytes, 12), LABEL_INDEX_FLAGS);
    let label_count = read_u64(bytes, 24) as usize;
    let exact_rows = read_u32(bytes, 32) as usize;
    let bloom_rows = read_u32(bytes, 36) as usize;
    let hash_count = read_u32(bytes, 40) as usize;
    let words_per_row = read_u64(bytes, 48) as usize;
    let summary_words_per_row = read_u64(bytes, 56) as usize;
    let mut cursor = 64usize;
    for _ in 0..label_count {
        let label_len = read_u32(bytes, cursor) as usize;
        cursor += 8 + label_len;
    }
    cursor += (label_count - exact_rows) * hash_count * 4;
    let metadata_start = cursor;
    let physical_rows = exact_rows + bloom_rows;
    let summary_start = metadata_start + physical_rows * 8;
    let payload_start = summary_start + physical_rows * summary_words_per_row * 8;
    Layout {
        exact_rows,
        bloom_rows,
        hash_count,
        words_per_row,
        summary_words_per_row,
        metadata_start,
        summary_start,
        payload_start,
    }
}

#[test]
fn exact_only_round_trip_supports_dnf_cnf_and_metadata() {
    let (_dir, index) = encode_exact(sample_documents());
    assert_eq!(index.num_vectors(), 4);
    assert_eq!(index.num_exact_labels(), index.num_labels());
    assert_eq!(
        matching_ids(
            &compile(&index, &["A&B", "C"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![0, 2]
    );
    assert_eq!(
        matching_ids(
            &compile(&index, &["A|B", "A|C"], FilterExpressionType::CNF),
            index.num_vectors()
        ),
        vec![0, 1, 2]
    );
    assert_eq!(index.physical_row_valid_counts().len(), index.num_labels());
    assert_eq!(
        index.physical_row_nonzero_word_counts().len(),
        index.num_labels()
    );
}

#[test]
fn bloom_only_round_trip_has_no_false_negatives() {
    let (_dir, index) = encode_bloom(sample_documents(), 0);
    assert_eq!(index.num_exact_labels(), 0);
    assert_eq!(index.physical_row_valid_counts().len(), 128);
    let query = compile(&index, &["A"], FilterExpressionType::DNF);
    let matches = matching_ids(&query, index.num_vectors());
    for expected in [0, 1, 2] {
        assert!(matches.contains(&expected));
    }
    assert!(QueryLabelProvider::cache_rejections(&query));
}

#[test]
fn hybrid_round_trip_uses_exact_prefix_and_bloom_suffix() {
    let documents = concat!(
        "[\"A\",\"B\"]\n",
        "\"A\"\n",
        "\"A\"\n",
        "\"B\"\n",
        "\"C\"\n",
    );
    let (_dir, index) = encode_bloom(documents, 2);
    assert_eq!(index.num_exact_labels(), 2);
    assert_eq!(index.physical_row_valid_counts().len(), 130);
    assert_eq!(
        matching_ids(
            &compile(&index, &["A"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![0, 1, 2]
    );
    assert_eq!(
        matching_ids(
            &compile(&index, &["B"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![0, 3]
    );
    assert!(QueryLabelProvider::cache_rejections(&compile(
        &index,
        &["C"],
        FilterExpressionType::DNF
    )));
}

#[test]
fn top_k_is_clamped_to_label_count() {
    let (_dir, index) = encode_bloom("\"A\"\n\"B\"\n", 1_000_000);
    assert_eq!(index.num_exact_labels(), 2);
    assert_eq!(index.physical_row_valid_counts(), &[1, 1]);
}

#[test]
fn dictionary_order_is_count_then_label_bytes() {
    let (_dir, index) = encode_bloom("\"B\"\n\"A\"\n\"A\"\n", 1);
    assert_eq!(index.num_exact_labels(), 1);
    assert_eq!(index.physical_row_valid_counts()[0], 2);
    assert_eq!(
        matching_ids(
            &compile(&index, &["A"], FilterExpressionType::DNF),
            index.num_vectors()
        ),
        vec![1, 2]
    );
}

#[test]
fn exact_pair_compiles_payload_and_summary_evaluators() {
    let (_dir, dense) = encode_bloom("[\"A\",\"B\"]\n\"A\"\n\"A\"\n\"C\"\n", 2);
    let dense_query = compile(&dense, &["A&B"], FilterExpressionType::DNF);
    assert!(dense_query.uses_exact_and2_point_plan());
    assert_eq!(dense_query.exact_and2_rows(), Some((1, 0)));
    assert!(!dense_query.exact_and2_uses_summary());
    let EncodedLabelPointEvaluator::ExactAnd2Payload(payload) = dense_query.point_evaluator()
    else {
        panic!("dense exact pair should use payload evaluator");
    };
    assert_eq!(matching_ids(&payload, dense.num_vectors()), vec![0]);
    assert_eq!(
        QueryLabelProvider::prefetch_lookahead(&payload),
        if cfg!(target_arch = "x86_64") { 32 } else { 0 }
    );

    let sparse_documents = std::iter::once("[\"A\",\"B\"]\n")
        .chain(std::iter::repeat_n("\"B\"\n", 4095))
        .collect::<String>();
    let (_dir, sparse) = encode_bloom(&sparse_documents, 2);
    let sparse_query = compile(&sparse, &["A&B"], FilterExpressionType::DNF);
    assert!(sparse_query.uses_exact_and2_point_plan());
    assert_eq!(sparse_query.exact_and2_rows(), Some((1, 0)));
    assert!(sparse_query.exact_and2_uses_summary());
    let EncodedLabelPointEvaluator::ExactAnd2Summary(summary) = sparse_query.point_evaluator()
    else {
        panic!("sparse exact pair should use summary evaluator");
    };
    assert_eq!(matching_ids(&summary, sparse.num_vectors()), vec![0]);
    assert_eq!(QueryLabelProvider::prefetch_lookahead(&summary), 0);
}

#[test]
fn candidate_provider_bound_and_visit_are_consistent() {
    let (_dir, index) = encode_bloom(sample_documents(), 2);
    let query = compile(&index, &["A&B"], FilterExpressionType::DNF);
    assert_eq!(CandidateLabelProvider::match_upper_bound(&query), Some(2));
    let mut visited = Vec::new();
    CandidateLabelProvider::visit_candidates(&query, |id| visited.push(id));
    assert_eq!(visited, matching_ids(&query, index.num_vectors()));
    assert_eq!(query.count_matches(), visited.len() as u64);
}

#[test]
fn query_remains_usable_after_index_drop() {
    let query = {
        let (_dir, index) = encode_bloom(sample_documents(), 2);
        compile(&index, &["A"], FilterExpressionType::DNF)
    };
    assert_eq!(matching_ids(&query, 4), vec![0, 1, 2]);
}

#[test]
fn unknown_labels_follow_normal_form_semantics() {
    let (_dir, index) = encode_exact("\"A\"\n\"B\"\n");
    assert!(matching_ids(
        &compile(&index, &["UNKNOWN"], FilterExpressionType::DNF),
        index.num_vectors()
    )
    .is_empty());
    assert_eq!(
        matching_ids(
            &compile(&index, &["A|UNKNOWN"], FilterExpressionType::CNF),
            index.num_vectors()
        ),
        vec![0]
    );
}

#[test]
fn raw_object_and_blank_jsonl_forms_are_supported() {
    let documents = concat!(
        "\n",
        "\"A\"\n",
        "[\"B\",\"C\"]\n",
        "{\"doc_id\":5,\"labels\":[\"D\"]}\n",
        "{\"market\":\"US\",\"enabled\":true,\"disabled\":false}\n",
    );
    let (_dir, index) = encode_exact(documents);
    assert_eq!(index.num_vectors(), 6);
    for label in ["A", "B", "C", "D", "market=US", "enabled", "disabled=false"] {
        assert!(index.contains_label(label));
    }
}

#[test]
fn invalid_input_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("labels.bin");
    for documents in [
        "",
        "{\"doc_id\":0,\"labels\":[\"A\"]}\n{\"doc_id\":0,\"labels\":[\"B\"]}\n",
        "\" leading\"\n",
        "\"A&B\"\n",
        "{\"nested\":{\"x\":1}}\n",
    ] {
        let input = dir.path().join("labels.jsonl");
        std::fs::write(&input, documents).unwrap();
        assert!(encode_label_index_jsonl(&input, &output).is_err());
    }
}

#[test]
fn bloom_configuration_is_validated() {
    assert!(BloomFilterConfig::new(0, 1).is_err());
    assert!(BloomFilterConfig::new(8, 0).is_err());
    assert!(BloomFilterConfig::new(8, 9).is_err());
    assert!(BloomFilterConfig::new(8, 2).is_ok());
    assert!(BloomFilterConfig::new(8, 2)
        .unwrap()
        .with_exact_label_count(1_000_001)
        .is_err());
}

#[test]
fn bloom_query_configuration_is_applied() {
    let default = BloomQueryConfig::default();
    assert_eq!(default.summary_cache_bound_percent(), 1.0);
    assert_eq!(default.summary_scan_max_word_percent(), 10.0);
    assert_eq!(default.exact_point_prefetch_lookahead(), 32);
    assert_eq!(
        default
            .with_exact_point_prefetch_lookahead(8)
            .exact_point_prefetch_lookahead(),
        8
    );
    assert!(BloomQueryConfig::new(0.99).is_err());
    assert!(default.with_summary_scan_max_word_percent(101.0).is_err());
}

#[test]
fn wire_layout_has_metadata_summaries_then_payload() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bin");
    std::fs::write(&input, sample_documents()).unwrap();
    let config = BloomFilterConfig::new(8, 2)
        .unwrap()
        .with_exact_label_count(2)
        .unwrap();
    encode_bloom_label_index_jsonl(&input, &output, config).unwrap();
    let bytes = std::fs::read(&output).unwrap();
    let layout = layout(&bytes);
    assert_eq!(layout.exact_rows, 2);
    assert_eq!(layout.bloom_rows, 8);
    assert_eq!(layout.hash_count, 2);
    assert_eq!(layout.words_per_row, 1);
    assert_eq!(layout.summary_words_per_row, 1);
    assert!(layout.metadata_start < layout.summary_start);
    assert!(layout.summary_start < layout.payload_start);
    assert_eq!(layout.payload_start + 10 * 8, bytes.len());
}

fn encoded_file() -> (tempfile::TempDir, std::path::PathBuf, Vec<u8>, Layout) {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("labels.jsonl");
    let output = dir.path().join("labels.bin");
    std::fs::write(&input, sample_documents()).unwrap();
    let config = BloomFilterConfig::new(8, 2)
        .unwrap()
        .with_exact_label_count(2)
        .unwrap();
    encode_bloom_label_index_jsonl(&input, &output, config).unwrap();
    let bytes = std::fs::read(&output).unwrap();
    let layout = layout(&bytes);
    (dir, output, bytes, layout)
}

#[test]
fn loader_rejects_header_corruption() {
    let (_dir, output, bytes, _layout) = encoded_file();
    for (offset, value) in [(0usize, b'X'), (8, 9), (12, 1), (44, 1)] {
        let mut corrupt = bytes.clone();
        if offset == 0 {
            corrupt[offset] = value;
        } else {
            corrupt[offset..offset + 4].copy_from_slice(&(value as u32).to_le_bytes());
        }
        std::fs::write(&output, corrupt).unwrap();
        assert!(EncodedLabelIndex::load(&output).is_err());
    }
}

#[test]
fn loader_rejects_metadata_summary_padding_and_trailing_corruption() {
    let (_dir, output, bytes, layout) = encoded_file();

    let mut metadata = bytes.clone();
    metadata[layout.metadata_start + 4..layout.metadata_start + 8]
        .copy_from_slice(&u32::MAX.to_le_bytes());
    std::fs::write(&output, metadata).unwrap();
    assert!(EncodedLabelIndex::load(&output).is_err());

    let mut summary = bytes.clone();
    summary[layout.summary_start] ^= 1;
    std::fs::write(&output, summary).unwrap();
    assert!(EncodedLabelIndex::load(&output).is_err());

    let mut padding = bytes.clone();
    let last_payload_word = layout.payload_start + (layout.exact_rows + layout.bloom_rows) * 8 - 8;
    padding[last_payload_word + 7] |= 0x80;
    std::fs::write(&output, padding).unwrap();
    assert!(EncodedLabelIndex::load(&output).is_err());

    let mut trailing = bytes;
    trailing.push(0);
    std::fs::write(&output, trailing).unwrap();
    assert!(EncodedLabelIndex::load(&output).is_err());
}

#[test]
fn loader_rejects_truncated_sections() {
    let (_dir, output, bytes, layout) = encoded_file();
    for length in [
        0usize,
        63,
        layout.metadata_start,
        layout.summary_start,
        layout.payload_start,
        bytes.len() - 1,
    ] {
        std::fs::write(&output, &bytes[..length]).unwrap();
        assert!(EncodedLabelIndex::load(&output).is_err());
    }
}

#[test]
fn compiled_types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>(_: &T) {}
    let (_dir, index) = encode_bloom(sample_documents(), 2);
    let query = compile(&index, &["A"], FilterExpressionType::DNF);
    assert_send_sync(&index);
    assert_send_sync(&query);
    assert_send_sync(&Arc::new(query));
}

#[test]
fn appending_to_an_encoded_file_is_detected() {
    let (_dir, output, _bytes, _layout) = encoded_file();
    let mut file = OpenOptions::new().append(true).open(&output).unwrap();
    file.write_all(&[0]).unwrap();
    drop(file);
    assert!(EncodedLabelIndex::load(output).is_err());
}
