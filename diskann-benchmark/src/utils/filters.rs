/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use bit_set::BitSet;
use std::{
    fmt::{self, Debug},
    sync::{Arc, OnceLock},
};

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use diskann::{
    graph::ext::labeled::QueryLabelProvider,
    utils::{IntoUsize, VectorId},
};
use diskann_benchmark_runner::files::InputFile;
use diskann_label_filter::{
    kv_index::GenericIndex,
    stores::bftree_store::BfTreeStore,
    traits::{
        posting_list_trait::{PostingList, RoaringPostingList},
        query_evaluator::QueryEvaluator,
    },
    ASTExpr, CompareOp, DefaultKeyCodec,
};
use diskann_label_index::{EncodedLabelIndex, EncodedLabelQuery, FilterExpressionType};
use diskann_providers::model::graph::provider::layers::BetaFilter;

use diskann_tools::utils::ground_truth::read_labels_and_compute_bitmap;

use diskann_label_filter::read_and_parse_queries;
use serde_json::Value;

pub struct QueryBitmapEvaluator {
    pub ast_expr: ASTExpr,
    evaluated_bitmap: RoaringPostingList,
}

impl QueryBitmapEvaluator {
    /// Create a new filter and evaluate the bitmap immediately (existing behavior).
    pub fn new(
        ast_expr: ASTExpr,
        inverted_index: &GenericIndex<BfTreeStore, RoaringPostingList, DefaultKeyCodec>,
    ) -> Self {
        let evaluated_bitmap = inverted_index.evaluate_query(&ast_expr).unwrap();
        Self {
            ast_expr,
            evaluated_bitmap,
        }
    }

    /// Ensure evaluated and return a reference to the bitmap (convenience).
    fn get_bitmap(&self) -> &RoaringPostingList {
        &self.evaluated_bitmap
    }

    /// Number of matching labels in this filter's evaluated bitmap.
    pub fn count(&self) -> usize {
        self.get_bitmap().len()
    }
}

impl Debug for QueryBitmapEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitmapFilter")
            .field("ast_expr", &self.ast_expr)
            .field("evaluated_bitmap", &self.evaluated_bitmap)
            .finish()
    }
}

impl<T> QueryLabelProvider<T> for QueryBitmapEvaluator
where
    T: VectorId + IntoUsize,
{
    fn is_match(&self, vec_id: T) -> bool {
        self.get_bitmap().contains(vec_id.into_usize())
    }
}

#[derive(Debug)]
pub struct BitmapFilter(pub BitSet);

impl<T> QueryLabelProvider<T> for BitmapFilter
where
    T: VectorId + IntoUsize,
{
    fn is_match(&self, vec_id: T) -> bool {
        self.0.contains(vec_id.into_usize())
    }
}

pub(crate) type ValidatedEncodedQuerySource = Box<[String]>;

/// Per-query lazy encoded-label provider used by the encoded multihop benchmarks.
///
/// Outside timed search we parse query JSONL, validate its DNF shape, and store its flat clauses.
/// The first `is_match` call inside timed ANN search compiles label IDs; later probes reuse the
/// cached [`EncodedLabelQuery`] for that one query row only.
pub(crate) struct LazyEncodedQueryProvider {
    index: Arc<EncodedLabelIndex>,
    source: ValidatedEncodedQuerySource,
    compiled: OnceLock<EncodedLabelQuery<'static>>,
    #[cfg(test)]
    compile_count: AtomicUsize,
}

impl LazyEncodedQueryProvider {
    fn new(index: Arc<EncodedLabelIndex>, source: ValidatedEncodedQuerySource) -> Self {
        Self {
            index,
            source,
            compiled: OnceLock::new(),
            #[cfg(test)]
            compile_count: AtomicUsize::new(0),
        }
    }

    fn compile(&self) -> EncodedLabelQuery<'static> {
        #[cfg(test)]
        self.compile_count.fetch_add(1, Ordering::Relaxed);

        compile_encoded_query(self.index.as_ref(), &self.source).unwrap_or_else(|e| {
            panic!(
                "validated encoded benchmark query failed lazy compilation for {:?}: {e}",
                self.source
            )
        })
    }

    #[cfg(test)]
    fn compile_count(&self) -> usize {
        self.compile_count.load(Ordering::Relaxed)
    }
}

impl fmt::Debug for LazyEncodedQueryProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyEncodedQueryProvider")
            .field("source", &self.source)
            .field("compiled", &self.compiled.get().is_some())
            .finish()
    }
}

impl QueryLabelProvider<u32> for LazyEncodedQueryProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        self.compiled
            .get_or_init(|| self.compile())
            .is_match(vec_id)
    }
}

pub(crate) fn generate_bitmaps(
    query_predicates: &InputFile,
    data_labels: &InputFile,
) -> anyhow::Result<Vec<BitSet>> {
    let bit_maps = match read_labels_and_compute_bitmap(
        data_labels.to_str().unwrap(),
        query_predicates.to_str().unwrap(),
    ) {
        Ok(bit_maps) => bit_maps,
        Err(e) => {
            return Err(e.into());
        }
    };
    Ok(bit_maps)
}

pub(crate) fn setup_filter_strategies<I, S>(
    beta: f32,
    bit_maps: I,
    search_strategy: S,
) -> Vec<BetaFilter<S, u32>>
where
    I: IntoIterator<Item = Arc<dyn QueryLabelProvider<u32>>>,
    S: Clone,
{
    bit_maps
        .into_iter()
        .map(|bit_map| BetaFilter::<S, u32>::new(search_strategy.clone(), bit_map, beta))
        .collect::<Vec<_>>()
}

pub(crate) fn as_query_label_provider(set: BitSet) -> Arc<dyn QueryLabelProvider<u32>> {
    Arc::new(BitmapFilter(set))
}

pub(crate) fn load_encoded_label_index(
    data_labels: &InputFile,
) -> anyhow::Result<Arc<EncodedLabelIndex>> {
    EncodedLabelIndex::load(&**data_labels)
        .map(Arc::new)
        .map_err(|e| {
            anyhow::anyhow!(
                "failed to load encoded label index {}: {e}",
                data_labels.display()
            )
        })
}

/// Parse/validate the benchmark predicate JSONL outside timed ANN search.
///
/// Setup includes JSONL parsing, supported-operator checks, null/unknown-label rejection, and
/// DNF-shape validation. Timed search begins when the lazy provider compiles label IDs.
pub(crate) fn prepare_encoded_query_sources(
    index: &EncodedLabelIndex,
    query_predicates: &InputFile,
) -> anyhow::Result<Vec<ValidatedEncodedQuerySource>> {
    let parsed = read_and_parse_queries(query_predicates.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to parse query predicates: {e}"))?;
    parsed
        .into_iter()
        .map(|(_query_id, ast)| {
            let clauses = flatten_dnf_clauses(&ast)?;
            validate_encoded_benchmark_labels(index, &clauses)?;
            Ok(clauses.into_boxed_slice())
        })
        .collect()
}

/// Instantiate one independent lazy provider per query row.
///
/// Each provider receives its own [`OnceLock`], so filter compilation is never shared across query
/// rows or across repeated benchmark executions.
pub(crate) fn make_encoded_query_providers(
    index: Arc<EncodedLabelIndex>,
    query_sources: &[ValidatedEncodedQuerySource],
) -> Vec<Arc<LazyEncodedQueryProvider>> {
    query_sources
        .iter()
        .cloned()
        .map(|source| Arc::new(LazyEncodedQueryProvider::new(index.clone(), source)))
        .collect()
}

fn compile_encoded_query(
    index: &EncodedLabelIndex,
    source: &ValidatedEncodedQuerySource,
) -> anyhow::Result<EncodedLabelQuery<'static>> {
    index
        .query(source.as_ref(), FilterExpressionType::DNF)
        .map_err(|e| anyhow::anyhow!("failed to compile encoded DNF query: {e}"))
}

fn validate_encoded_benchmark_labels(
    index: &EncodedLabelIndex,
    clauses: &[String],
) -> anyhow::Result<()> {
    for clause in clauses {
        for label in clause.split('&') {
            if !index.contains_label(label) {
                anyhow::bail!(
                    "encoded benchmark query references label '{label}' absent from the label index"
                );
            }
        }
    }
    Ok(())
}

fn compare_to_label(field: &str, op: &CompareOp) -> anyhow::Result<String> {
    match op {
        CompareOp::Eq(value) => eq_value_to_label(field, value),
        CompareOp::Ne(_)
        | CompareOp::Lt(_)
        | CompareOp::Lte(_)
        | CompareOp::Gt(_)
        | CompareOp::Gte(_) => Err(anyhow::anyhow!(
            "encoded label-index queries only support equality/set-membership predicates; field '{field}' used unsupported operator {op}"
        )),
    }
}

fn eq_value_to_label(field: &str, value: &Value) -> anyhow::Result<String> {
    match value {
        Value::Bool(true) => Ok(field.to_string()),
        Value::Bool(false) | Value::Number(_) | Value::String(_) => {
            Ok(format!("{field}={}", value_repr(value)))
        }
        Value::Null => Err(anyhow::anyhow!(
            "encoded benchmark equality predicates do not support null for field '{field}'"
        )),
        Value::Array(_) => Err(anyhow::anyhow!(
            "encoded label-index equality predicates do not support array values for field '{field}'"
        )),
        Value::Object(_) => Err(anyhow::anyhow!(
            "encoded label-index equality predicates require scalar or array values; field '{field}' used an object"
        )),
    }
}

fn value_repr(value: &Value) -> String {
    match value {
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Null => "null".to_string(),
        Value::Array(_) | Value::Object(_) => {
            unreachable!("handled by eq_value_to_label")
        }
    }
}

fn flatten_dnf_clauses(expression: &ASTExpr) -> anyhow::Result<Vec<String>> {
    let mut clauses = Vec::new();
    collect_dnf_clauses(expression, &mut clauses)?;
    Ok(clauses)
}

fn collect_dnf_clauses(expression: &ASTExpr, clauses: &mut Vec<String>) -> anyhow::Result<()> {
    match expression {
        ASTExpr::Compare { field, op } => clauses.push(compare_to_label(field, op)?),
        ASTExpr::And(children) => {
            let mut terminals = Vec::new();
            for child in children {
                collect_dnf_conjunction(child, &mut terminals)?;
            }
            clauses.push(terminals.join("&"));
        }
        ASTExpr::Or(children) => {
            for child in children {
                collect_dnf_clauses(child, clauses)?;
            }
        }
        ASTExpr::Not(_) => {
            anyhow::bail!("encoded DNF queries require an OR-of-AND predicate shape; NOT expressions are unsupported")
        }
    }
    Ok(())
}

fn collect_dnf_conjunction(
    expression: &ASTExpr,
    terminals: &mut Vec<String>,
) -> anyhow::Result<()> {
    match expression {
        ASTExpr::Compare { field, op } => terminals.push(compare_to_label(field, op)?),
        ASTExpr::And(children) => {
            for child in children {
                collect_dnf_conjunction(child, terminals)?;
            }
        }
        ASTExpr::Or(_) => {
            anyhow::bail!("encoded DNF queries require an OR-of-AND predicate shape; OR expressions cannot appear inside a conjunction")
        }
        ASTExpr::Not(_) => {
            anyhow::bail!("encoded DNF queries require an OR-of-AND predicate shape; NOT expressions are unsupported")
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs::File, io::Write};

    use diskann_label_index::encode_label_index_jsonl;
    use serde_json::json;
    use tempfile::tempdir;

    fn write_lines(path: &std::path::Path, lines: &[&str]) {
        let mut file = File::create(path).unwrap();
        for line in lines {
            writeln!(file, "{line}").unwrap();
        }
    }

    fn test_query_matches<P>(provider: &Arc<P>, num_vectors: u32) -> Vec<u32>
    where
        P: QueryLabelProvider<u32> + ?Sized,
    {
        (0..num_vectors)
            .filter(|&vec_id| provider.is_match(vec_id))
            .collect()
    }

    fn test_compiled_query_matches(query: &EncodedLabelQuery<'_>, num_vectors: u32) -> Vec<u32> {
        (0..num_vectors)
            .filter(|&vec_id| query.is_match(vec_id))
            .collect()
    }

    fn encoded_fixture() -> (tempfile::TempDir, Arc<EncodedLabelIndex>, InputFile) {
        let dir = tempdir().unwrap();
        let labels_jsonl = dir.path().join("labels.jsonl");
        let bitslice_index = dir.path().join("labels.bitslice");
        let queries_jsonl = dir.path().join("queries.jsonl");

        write_lines(
            &labels_jsonl,
            &[
                r#"{"doc_id":0,"brand":"A","color":"red","promo":true,"active":false,"score":2,"missing":null}"#,
                r#"{"doc_id":1,"brand":"A","color":"blue","promo":false,"active":false,"score":3,"missing":"value"}"#,
                r#"{"doc_id":2,"brand":"B","color":"red","promo":false,"active":true,"score":2,"missing":null}"#,
                r#"{"doc_id":3,"brand":"C","color":"green","promo":true,"active":false,"score":5,"missing":null}"#,
            ],
        );
        write_lines(
            &queries_jsonl,
            &[
                r#"{"query_id":0,"filter":{"$or":[{"$and":[{"brand":{"$eq":"A"}},{"color":{"$eq":"red"}}]},{"promo":{"$eq":true}}]}}"#,
                r#"{"query_id":1,"filter":{"active":{"$eq":false},"score":{"$eq":2}}}"#,
                r#"{"query_id":2,"filter":{"$or":[{"brand":{"$eq":"B"}},{"color":{"$eq":"green"}}]}}"#,
            ],
        );

        encode_label_index_jsonl(&labels_jsonl, &bitslice_index).unwrap();

        (
            dir,
            load_encoded_label_index(&InputFile::new(bitslice_index)).unwrap(),
            InputFile::new(queries_jsonl),
        )
    }

    fn eager_encoded_matches(
        index: &EncodedLabelIndex,
        source: &ValidatedEncodedQuerySource,
        num_vectors: u32,
    ) -> Vec<u32> {
        let query = compile_encoded_query(index, source).unwrap();
        test_compiled_query_matches(&query, num_vectors)
    }

    #[test]
    fn test_bitmap_filter_match() {
        let mut bitset = BitSet::new();
        bitset.insert(1);
        bitset.insert(3);
        let filter = BitmapFilter(bitset);

        assert!(filter.is_match(1u32));
        assert!(filter.is_match(3u32));
        assert!(!filter.is_match(2u32));
        assert!(!filter.is_match(0u32));
    }

    #[test]
    fn test_bitmap_filter_empty() {
        let bitset = BitSet::new();
        let filter = BitmapFilter(bitset);

        assert!(!filter.is_match(0u32));
        assert!(!filter.is_match(10u32));
    }

    #[test]
    fn test_bitmap_filter_large_id() {
        let mut bitset = BitSet::new();
        bitset.insert(1000);
        let filter = BitmapFilter(bitset);

        assert!(filter.is_match(1000u32));
        assert!(!filter.is_match(999u32));
    }

    #[test]
    fn test_lazy_encoded_dnf_provider_matches_eager_query() {
        let (_dir, bitslice_index, query_file) = encoded_fixture();
        let sources = prepare_encoded_query_sources(bitslice_index.as_ref(), &query_file).unwrap();
        let providers = make_encoded_query_providers(bitslice_index.clone(), &sources);

        assert_eq!(providers.len(), sources.len());
        for (provider, source) in providers.iter().zip(&sources) {
            assert_eq!(
                test_query_matches(provider, 4),
                eager_encoded_matches(bitslice_index.as_ref(), source, 4)
            );
        }
    }

    #[test]
    fn test_lazy_encoded_provider_reuses_compilation_per_instance() {
        let dir = tempdir().unwrap();
        let labels_jsonl = dir.path().join("labels.jsonl");
        let bitslice_index = dir.path().join("labels.bitslice");
        write_lines(
            &labels_jsonl,
            &[
                r#"{"doc_id":0,"A":true,"B":true}"#,
                r#"{"doc_id":1,"A":false,"B":true}"#,
            ],
        );
        encode_label_index_jsonl(&labels_jsonl, &bitslice_index).unwrap();
        let index = load_encoded_label_index(&InputFile::new(bitslice_index)).unwrap();
        let source = vec!["A&B".to_string()].into_boxed_slice();
        let provider_a = Arc::new(LazyEncodedQueryProvider::new(index.clone(), source.clone()));
        let provider_b = Arc::new(LazyEncodedQueryProvider::new(index, source));

        assert_eq!(provider_a.compile_count(), 0);
        assert_eq!(provider_b.compile_count(), 0);
        assert!(provider_a.is_match(0));
        assert!(!provider_a.is_match(1));
        assert!(provider_a.is_match(0));
        assert_eq!(provider_a.compile_count(), 1);
        assert_eq!(provider_b.compile_count(), 0);

        assert!(provider_b.is_match(0));
        assert_eq!(provider_b.compile_count(), 1);
        assert_eq!(provider_a.compile_count(), 1);
    }

    #[test]
    fn test_encoded_query_rejects_unsupported_relational_operator() {
        let error = compare_to_label("score", &CompareOp::Gt(2.0)).unwrap_err();
        assert!(error.to_string().contains("unsupported operator >"));
    }

    #[test]
    fn test_encoded_dnf_rejects_non_dnf_shape() {
        let expression = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "brand".to_string(),
                op: CompareOp::Eq(json!("A")),
            },
            ASTExpr::Or(vec![
                ASTExpr::Compare {
                    field: "color".to_string(),
                    op: CompareOp::Eq(json!("red")),
                },
                ASTExpr::Compare {
                    field: "color".to_string(),
                    op: CompareOp::Eq(json!("blue")),
                },
            ]),
        ]);

        let error = flatten_dnf_clauses(&expression).unwrap_err();
        assert!(error
            .to_string()
            .contains("OR expressions cannot appear inside a conjunction"));
    }

    #[test]
    fn test_encoded_dnf_rejects_array_equality_explicitly() {
        let error = compare_to_label("brand", &CompareOp::Eq(json!(["A", "B"]))).unwrap_err();
        assert!(error.to_string().contains("do not support array values"));
    }

    #[test]
    fn test_encoded_query_rejects_null_equality() {
        let error = compare_to_label("missing", &CompareOp::Eq(Value::Null)).unwrap_err();
        assert!(error.to_string().contains("do not support null"));
    }

    #[test]
    fn test_encoded_benchmark_rejects_unknown_labels() {
        let dir = tempdir().unwrap();
        let labels_jsonl = dir.path().join("labels.jsonl");
        let bitslice_index = dir.path().join("labels.bitslice");
        write_lines(&labels_jsonl, &[r#"{"doc_id":0,"A":true}"#]);
        encode_label_index_jsonl(&labels_jsonl, &bitslice_index).unwrap();
        let index = load_encoded_label_index(&InputFile::new(bitslice_index)).unwrap();
        let error =
            validate_encoded_benchmark_labels(&index, &["missing".to_string()]).unwrap_err();
        assert!(error.to_string().contains("absent from the label index"));
    }
}
