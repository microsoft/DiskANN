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
use diskann_label_index::{
    parse_label_expression_json, EncodedLabelIndex, EncodedLabelQuery, FilterExpressionType,
    LabelExpression,
};
use diskann_providers::model::graph::provider::layers::BetaFilter;

use diskann_tools::utils::ground_truth::read_labels_and_compute_bitmap;

use diskann_label_filter::attribute::Attribute;
use diskann_label_filter::{
    read_and_parse_queries, read_baselabels, FrozenAttributeIndex, FrozenAttributeIndexAuto,
    FrozenAttributeIndexBitslice, FrozenAttributeIndexCsr, FrozenAttributeIndexPosting,
    InlineAttributeIndex, InlineAttributeIndexAuto, InlineAttributeIndexBitslice,
    InlineAttributeIndexCsr, InlineAttributeIndexPosting,
};
use serde_json::{json, Value};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EncodedQueryMode {
    Dnf,
    Ast,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ValidatedEncodedQuerySource {
    Dnf(Box<[String]>),
    AstJson(String),
}

/// Per-query lazy encoded-label provider used by the encoded multihop benchmarks.
///
/// Outside timed search we only parse query JSONL, convert/validate the predicate shape, and store
/// either validated DNF clauses or the recursive AST JSON string. The first `is_match` call inside
/// timed ANN search performs the actual `EncodedLabelIndex::{query, query_ast_json}` compilation
/// (plus bitmap AST materialization for bitmap-backed indexes); later probes reuse the cached
/// [`EncodedLabelQuery`] for that one query row only.
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
            .field("format", &self.index.format())
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
/// Setup still includes JSONL parsing, AST-to-`LabelExpression` conversion, supported-operator
/// checks, null/unknown-label rejection, DNF-shape validation, and benchmark-local AST JSON
/// serialization. Timed search begins only when the lazy provider first calls back into
/// `EncodedLabelIndex::{query, query_ast_json}`.
pub(crate) fn prepare_encoded_query_sources(
    index: &EncodedLabelIndex,
    query_predicates: &InputFile,
    mode: EncodedQueryMode,
) -> anyhow::Result<Vec<ValidatedEncodedQuerySource>> {
    let parsed = read_and_parse_queries(query_predicates.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to parse query predicates: {e}"))?;
    parsed
        .into_iter()
        .map(|(_query_id, ast)| {
            let label_expression = ast_to_label_expression(&ast)?;
            validate_encoded_benchmark_labels(index, &label_expression)?;
            encoded_query_source_from_expression(&label_expression, mode)
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

fn encoded_query_source_from_expression(
    expression: &LabelExpression,
    mode: EncodedQueryMode,
) -> anyhow::Result<ValidatedEncodedQuerySource> {
    match mode {
        EncodedQueryMode::Dnf => Ok(ValidatedEncodedQuerySource::Dnf(
            flatten_dnf_clauses(expression)?.into_boxed_slice(),
        )),
        EncodedQueryMode::Ast => {
            let expression_json = label_expression_to_ast_json(expression)?;
            parse_label_expression_json(&expression_json)
                .map_err(|e| anyhow::anyhow!("encoded AST source failed validation: {e}"))?;
            Ok(ValidatedEncodedQuerySource::AstJson(expression_json))
        }
    }
}

fn compile_encoded_query(
    index: &EncodedLabelIndex,
    source: &ValidatedEncodedQuerySource,
) -> anyhow::Result<EncodedLabelQuery<'static>> {
    match source {
        ValidatedEncodedQuerySource::Dnf(clauses) => index
            .query(clauses.as_ref(), FilterExpressionType::DNF)
            .map_err(|e| anyhow::anyhow!("failed to compile encoded DNF query: {e}")),
        ValidatedEncodedQuerySource::AstJson(expression_json) => index
            .query_ast_json(expression_json)
            .map_err(|e| anyhow::anyhow!("failed to compile encoded AST query: {e}")),
    }
}

fn label_expression_to_ast_json(expression: &LabelExpression) -> anyhow::Result<String> {
    serde_json::to_string(&label_expression_to_ast_value(expression))
        .map_err(|e| anyhow::anyhow!("failed to serialize encoded AST query: {e}"))
}

fn label_expression_to_ast_value(expression: &LabelExpression) -> Value {
    match expression {
        LabelExpression::Label(label) => Value::String(label.clone()),
        LabelExpression::And(children) => json!({
            "and": children
                .iter()
                .map(label_expression_to_ast_value)
                .collect::<Vec<_>>()
        }),
        LabelExpression::Or(children) => json!({
            "or": children
                .iter()
                .map(label_expression_to_ast_value)
                .collect::<Vec<_>>()
        }),
        LabelExpression::Not(child) => json!({
            "not": label_expression_to_ast_value(child)
        }),
    }
}

fn checked_doc_id(doc_id: usize) -> anyhow::Result<u32> {
    u32::try_from(doc_id)
        .map_err(|_| anyhow::anyhow!("document id {doc_id} exceeds the u32 graph-id range"))
}

fn validate_encoded_benchmark_labels(
    index: &EncodedLabelIndex,
    expression: &LabelExpression,
) -> anyhow::Result<()> {
    let mut stack = vec![expression];
    while let Some(expression) = stack.pop() {
        match expression {
            LabelExpression::Label(label) => {
                if !index.contains_label(label) {
                    anyhow::bail!(
                        "encoded benchmark query references label '{label}' absent from the label index"
                    );
                }
            }
            LabelExpression::And(children) | LabelExpression::Or(children) => {
                stack.extend(children);
            }
            LabelExpression::Not(child) => stack.push(child),
        }
    }
    Ok(())
}

fn ast_to_label_expression(ast: &ASTExpr) -> anyhow::Result<LabelExpression> {
    match ast {
        ASTExpr::And(exprs) => Ok(LabelExpression::And(
            exprs
                .iter()
                .map(ast_to_label_expression)
                .collect::<anyhow::Result<Vec<_>>>()?,
        )),
        ASTExpr::Or(exprs) => Ok(LabelExpression::Or(
            exprs
                .iter()
                .map(ast_to_label_expression)
                .collect::<anyhow::Result<Vec<_>>>()?,
        )),
        ASTExpr::Not(expr) => Ok(LabelExpression::Not(Box::new(ast_to_label_expression(
            expr,
        )?))),
        ASTExpr::Compare { field, op } => compare_to_label_expression(field, op),
    }
}

fn compare_to_label_expression(field: &str, op: &CompareOp) -> anyhow::Result<LabelExpression> {
    match op {
        CompareOp::Eq(value) => eq_value_to_label_expression(field, value),
        CompareOp::Ne(_)
        | CompareOp::Lt(_)
        | CompareOp::Lte(_)
        | CompareOp::Gt(_)
        | CompareOp::Gte(_) => Err(anyhow::anyhow!(
            "encoded label-index queries only support equality/set-membership predicates; field '{field}' used unsupported operator {op}"
        )),
    }
}

fn eq_value_to_label_expression(field: &str, value: &Value) -> anyhow::Result<LabelExpression> {
    match value {
        Value::Bool(true) => Ok(LabelExpression::Label(field.to_string())),
        Value::Bool(false) | Value::Number(_) | Value::String(_) => {
            Ok(LabelExpression::Label(format!("{field}={}", value_repr(value))))
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
            unreachable!("handled by eq_value_to_label_expression")
        }
    }
}

fn flatten_dnf_clauses(expression: &LabelExpression) -> anyhow::Result<Vec<String>> {
    let mut clauses = Vec::new();
    collect_dnf_clauses(expression, &mut clauses)?;
    Ok(clauses)
}

fn collect_dnf_clauses(
    expression: &LabelExpression,
    clauses: &mut Vec<String>,
) -> anyhow::Result<()> {
    match expression {
        LabelExpression::Label(label) => clauses.push(label.clone()),
        LabelExpression::And(children) => {
            let mut terminals = Vec::new();
            for child in children {
                collect_dnf_conjunction(child, &mut terminals)?;
            }
            clauses.push(terminals.join("&"));
        }
        LabelExpression::Or(children) => {
            for child in children {
                collect_dnf_clauses(child, clauses)?;
            }
        }
        LabelExpression::Not(_) => {
            anyhow::bail!("encoded DNF queries require an OR-of-AND predicate shape; NOT expressions are unsupported")
        }
    }
    Ok(())
}

fn collect_dnf_conjunction(
    expression: &LabelExpression,
    terminals: &mut Vec<String>,
) -> anyhow::Result<()> {
    match expression {
        LabelExpression::Label(label) => terminals.push(label.clone()),
        LabelExpression::And(children) => {
            for child in children {
                collect_dnf_conjunction(child, terminals)?;
            }
        }
        LabelExpression::Or(_) => {
            anyhow::bail!("encoded DNF queries require an OR-of-AND predicate shape; OR expressions cannot appear inside a conjunction")
        }
        LabelExpression::Not(_) => {
            anyhow::bail!("encoded DNF queries require an OR-of-AND predicate shape; NOT expressions are unsupported")
        }
    }
    Ok(())
}

/// Build an in-memory inline attribute index from a jsonl label file (one document per line).
///
/// Each document's flattened `(field, value)` pairs are encoded to integer attribute-ids and
/// stored as a roaring set keyed by `doc_id`. This is a one-time index build, reused across
/// all queries; the per-node match decision itself is computed live during search.
pub(crate) fn build_inline_attribute_index(
    data_labels: &InputFile,
) -> anyhow::Result<Arc<FrozenAttributeIndex>> {
    let docs = read_baselabels(data_labels.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to read base labels: {e}"))?;
    let mut index = InlineAttributeIndex::new();
    let mut attrs: Vec<Attribute> = Vec::new();
    for doc in &docs {
        attrs.clear();
        if let Some(obj) = doc.label.as_object() {
            for (field, value) in obj {
                attrs.push(Attribute::from_json_value(field, value).map_err(|e| {
                    anyhow::anyhow!("attribute conversion failed for field '{field}': {e:?}")
                })?);
            }
        }
        let doc_id = checked_doc_id(doc.doc_id)?;
        index
            .insert_document(doc_id, &attrs)
            .map_err(|e| anyhow::anyhow!("failed to insert document {}: {e:?}", doc.doc_id))?;
    }
    Ok(Arc::new(index.freeze()))
}

/// Parse per-query predicates and build one live [`QueryLabelProvider`] per query, all sharing
/// the same attribute `index`. The predicate is encoded once here; matching happens per node
/// during search.
pub(crate) fn make_live_providers(
    index: &FrozenAttributeIndex,
    query_predicates: &InputFile,
) -> anyhow::Result<Vec<Arc<dyn QueryLabelProvider<u32>>>> {
    let parsed = read_and_parse_queries(query_predicates.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to parse query predicates: {e}"))?;
    let mut providers = Vec::with_capacity(parsed.len());
    for (_query_id, ast) in parsed {
        providers.push(
            index
                .make_provider(&ast)
                .map_err(|e| anyhow::anyhow!("failed to build live provider: {e:?}"))?,
        );
    }
    Ok(providers)
}

/// Build an in-memory flat CSR attribute index from a jsonl label file (one document per line).
///
/// Identical inputs and integer encoding to [`build_inline_attribute_index`], but stores each
/// document's attribute ids in a contiguous CSR layout (`offsets` + sorted `values`) instead of a
/// roaring treemap, so the per-node live match reads a single contiguous row.
pub(crate) fn build_inline_attribute_index_csr(
    data_labels: &InputFile,
) -> anyhow::Result<Arc<FrozenAttributeIndexCsr>> {
    let docs = read_baselabels(data_labels.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to read base labels: {e}"))?;
    let mut index = InlineAttributeIndexCsr::new();
    let mut attrs: Vec<Attribute> = Vec::new();
    for doc in &docs {
        attrs.clear();
        if let Some(obj) = doc.label.as_object() {
            for (field, value) in obj {
                attrs.push(Attribute::from_json_value(field, value).map_err(|e| {
                    anyhow::anyhow!("attribute conversion failed for field '{field}': {e:?}")
                })?);
            }
        }
        let doc_id = checked_doc_id(doc.doc_id)?;
        index
            .insert_document(doc_id, &attrs)
            .map_err(|e| anyhow::anyhow!("failed to insert document {}: {e:?}", doc.doc_id))?;
    }
    Ok(Arc::new(index.freeze()))
}

/// Parse per-query predicates and build one live CSR [`QueryLabelProvider`] per query, all sharing
/// the same CSR attribute `index`. Mirrors [`make_live_providers`].
pub(crate) fn make_live_providers_csr(
    index: &FrozenAttributeIndexCsr,
    query_predicates: &InputFile,
) -> anyhow::Result<Vec<Arc<dyn QueryLabelProvider<u32>>>> {
    let parsed = read_and_parse_queries(query_predicates.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to parse query predicates: {e}"))?;
    let mut providers = Vec::with_capacity(parsed.len());
    for (_query_id, ast) in parsed {
        providers.push(
            index
                .make_provider(&ast)
                .map_err(|e| anyhow::anyhow!("failed to build live provider: {e:?}"))?,
        );
    }
    Ok(providers)
}

/// Build an in-memory posting-list attribute index (one `RoaringBitmap` of vector-ids per
/// attribute) from a jsonl label file. Same inputs/encoding as [`build_inline_attribute_index`];
/// used by the materialized-bitmap live provider.
pub(crate) fn build_inline_attribute_index_posting(
    data_labels: &InputFile,
) -> anyhow::Result<Arc<FrozenAttributeIndexPosting>> {
    let docs = read_baselabels(data_labels.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to read base labels: {e}"))?;
    let mut index = InlineAttributeIndexPosting::new();
    let mut attrs: Vec<Attribute> = Vec::new();
    for doc in &docs {
        attrs.clear();
        if let Some(obj) = doc.label.as_object() {
            for (field, value) in obj {
                attrs.push(Attribute::from_json_value(field, value).map_err(|e| {
                    anyhow::anyhow!("attribute conversion failed for field '{field}': {e:?}")
                })?);
            }
        }
        let doc_id = checked_doc_id(doc.doc_id)?;
        index
            .insert_document(doc_id, &attrs)
            .map_err(|e| anyhow::anyhow!("failed to insert document {}: {e:?}", doc.doc_id))?;
    }
    Ok(Arc::new(index.freeze()))
}

/// Parse per-query predicates and build one materialized-bitmap [`QueryLabelProvider`] per query,
/// all sharing the same posting `index`. Mirrors [`make_live_providers`].
pub(crate) fn make_live_providers_posting(
    index: &FrozenAttributeIndexPosting,
    query_predicates: &InputFile,
) -> anyhow::Result<Vec<Arc<dyn QueryLabelProvider<u32>>>> {
    let parsed = read_and_parse_queries(query_predicates.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to parse query predicates: {e}"))?;
    let mut providers = Vec::with_capacity(parsed.len());
    for (_query_id, ast) in parsed {
        providers.push(
            index
                .make_provider(&ast)
                .map_err(|e| anyhow::anyhow!("failed to build live provider: {e:?}"))?,
        );
    }
    Ok(providers)
}

/// Build an in-memory adaptive (auto) attribute index (CSR + posting lists) from a jsonl label file.
pub(crate) fn build_inline_attribute_index_auto(
    data_labels: &InputFile,
) -> anyhow::Result<Arc<FrozenAttributeIndexAuto>> {
    let docs = read_baselabels(data_labels.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to read base labels: {e}"))?;
    let mut index = InlineAttributeIndexAuto::new();
    let mut attrs: Vec<Attribute> = Vec::new();
    for doc in &docs {
        attrs.clear();
        if let Some(obj) = doc.label.as_object() {
            for (field, value) in obj {
                attrs.push(Attribute::from_json_value(field, value).map_err(|e| {
                    anyhow::anyhow!("attribute conversion failed for field '{field}': {e:?}")
                })?);
            }
        }
        let doc_id = checked_doc_id(doc.doc_id)?;
        index
            .insert_document(doc_id, &attrs)
            .map_err(|e| anyhow::anyhow!("failed to insert document {}: {e:?}", doc.doc_id))?;
    }
    Ok(Arc::new(index.freeze()))
}

/// Build one adaptive (auto) [`QueryLabelProvider`] per query. Mirrors [`make_live_providers`].
pub(crate) fn make_live_providers_auto(
    index: &FrozenAttributeIndexAuto,
    query_predicates: &InputFile,
) -> anyhow::Result<Vec<Arc<dyn QueryLabelProvider<u32>>>> {
    let parsed = read_and_parse_queries(query_predicates.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to parse query predicates: {e}"))?;
    let mut providers = Vec::with_capacity(parsed.len());
    for (_query_id, ast) in parsed {
        providers.push(
            index
                .make_provider(&ast)
                .map_err(|e| anyhow::anyhow!("failed to build live provider: {e:?}"))?,
        );
    }
    Ok(providers)
}

/// Build an in-memory bit-sliced attribute index (one dense bitset per attribute) from a jsonl file.
pub(crate) fn build_inline_attribute_index_bitslice(
    data_labels: &InputFile,
) -> anyhow::Result<Arc<FrozenAttributeIndexBitslice>> {
    let docs = read_baselabels(data_labels.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to read base labels: {e}"))?;
    let mut index = InlineAttributeIndexBitslice::new();
    let mut attrs: Vec<Attribute> = Vec::new();
    for doc in &docs {
        attrs.clear();
        if let Some(obj) = doc.label.as_object() {
            for (field, value) in obj {
                attrs.push(Attribute::from_json_value(field, value).map_err(|e| {
                    anyhow::anyhow!("attribute conversion failed for field '{field}': {e:?}")
                })?);
            }
        }
        let doc_id = checked_doc_id(doc.doc_id)?;
        index
            .insert_document(doc_id, &attrs)
            .map_err(|e| anyhow::anyhow!("failed to insert document {}: {e:?}", doc.doc_id))?;
    }
    drop(docs);
    Ok(Arc::new(index.freeze()))
}

/// Build one bit-sliced [`QueryLabelProvider`] per query. Mirrors [`make_live_providers`].
pub(crate) fn make_live_providers_bitslice(
    index: &FrozenAttributeIndexBitslice,
    query_predicates: &InputFile,
) -> anyhow::Result<Vec<Arc<dyn QueryLabelProvider<u32>>>> {
    let parsed = read_and_parse_queries(query_predicates.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to parse query predicates: {e}"))?;
    let mut providers = Vec::with_capacity(parsed.len());
    for (_query_id, ast) in parsed {
        providers.push(
            index
                .make_provider(&ast)
                .map_err(|e| anyhow::anyhow!("failed to build live provider: {e:?}"))?,
        );
    }
    Ok(providers)
}

/// Build one flat-DNF bit-sliced [`QueryLabelProvider`] per query.
pub(crate) fn make_live_providers_bitslice_dnf(
    index: &FrozenAttributeIndexBitslice,
    query_predicates: &InputFile,
) -> anyhow::Result<Vec<Arc<dyn QueryLabelProvider<u32>>>> {
    let parsed = read_and_parse_queries(query_predicates.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed to parse query predicates: {e}"))?;
    let mut providers = Vec::with_capacity(parsed.len());
    for (_query_id, ast) in parsed {
        providers.push(
            index
                .make_dnf_provider(&ast)
                .map_err(|e| anyhow::anyhow!("failed to build DNF live provider: {e:?}"))?,
        );
    }
    Ok(providers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs::File, io::Write};

    use diskann_label_index::{encode_label_index_jsonl, LabelIndexFormat};
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

    fn encoded_fixture() -> (
        tempfile::TempDir,
        Arc<EncodedLabelIndex>,
        Arc<EncodedLabelIndex>,
        InputFile,
    ) {
        let dir = tempdir().unwrap();
        let labels_jsonl = dir.path().join("labels.jsonl");
        let bitslice_index = dir.path().join("labels.bitslice");
        let bitmap_index = dir.path().join("labels.bitmap");
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

        encode_label_index_jsonl(&labels_jsonl, &bitslice_index, LabelIndexFormat::Bitslice)
            .unwrap();
        encode_label_index_jsonl(&labels_jsonl, &bitmap_index, LabelIndexFormat::Bitmap).unwrap();

        (
            dir,
            load_encoded_label_index(&InputFile::new(bitslice_index)).unwrap(),
            load_encoded_label_index(&InputFile::new(bitmap_index)).unwrap(),
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
    fn test_checked_doc_id() {
        assert_eq!(checked_doc_id(42).unwrap(), 42);
        assert_eq!(checked_doc_id(u32::MAX as usize).unwrap(), u32::MAX);
        if let Some(too_large) = (u32::MAX as usize).checked_add(1) {
            assert!(checked_doc_id(too_large).is_err());
        }
    }

    #[test]
    fn test_lazy_encoded_dnf_provider_matches_eager_query() {
        let (_dir, bitslice_index, _bitmap_index, query_file) = encoded_fixture();
        let sources = prepare_encoded_query_sources(
            bitslice_index.as_ref(),
            &query_file,
            EncodedQueryMode::Dnf,
        )
        .unwrap();
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
    fn test_lazy_encoded_ast_provider_matches_eager_query_for_bitslice_and_bitmap() {
        let (_dir, bitslice_index, bitmap_index, query_file) = encoded_fixture();
        let bitslice_sources = prepare_encoded_query_sources(
            bitslice_index.as_ref(),
            &query_file,
            EncodedQueryMode::Ast,
        )
        .unwrap();
        let bitmap_sources = prepare_encoded_query_sources(
            bitmap_index.as_ref(),
            &query_file,
            EncodedQueryMode::Ast,
        )
        .unwrap();
        let bitslice_providers =
            make_encoded_query_providers(bitslice_index.clone(), &bitslice_sources);
        let bitmap_providers = make_encoded_query_providers(bitmap_index.clone(), &bitmap_sources);

        assert_eq!(bitslice_providers.len(), bitslice_sources.len());
        assert_eq!(bitmap_providers.len(), bitmap_sources.len());
        for ((bitslice_provider, bitslice_source), (bitmap_provider, bitmap_source)) in
            bitslice_providers
                .iter()
                .zip(&bitslice_sources)
                .zip(bitmap_providers.iter().zip(&bitmap_sources))
        {
            assert_eq!(
                test_query_matches(bitslice_provider, 4),
                eager_encoded_matches(bitslice_index.as_ref(), bitslice_source, 4)
            );
            assert_eq!(
                test_query_matches(bitmap_provider, 4),
                eager_encoded_matches(bitmap_index.as_ref(), bitmap_source, 4)
            );
        }
    }

    #[test]
    fn test_encoded_ast_source_json_shape() {
        let expression = LabelExpression::Or(vec![
            LabelExpression::And(vec![
                LabelExpression::Label("brand=A".into()),
                LabelExpression::Not(Box::new(LabelExpression::Label("promo".into()))),
            ]),
            LabelExpression::Label("color=red".into()),
        ]);
        let source =
            encoded_query_source_from_expression(&expression, EncodedQueryMode::Ast).unwrap();
        let ValidatedEncodedQuerySource::AstJson(expression_json) = source else {
            panic!("expected AST JSON source");
        };

        assert_eq!(
            serde_json::from_str::<Value>(&expression_json).unwrap(),
            json!({
                "or": [
                    {
                        "and": [
                            "brand=A",
                            { "not": "promo" }
                        ]
                    },
                    "color=red"
                ]
            })
        );
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
        encode_label_index_jsonl(&labels_jsonl, &bitslice_index, LabelIndexFormat::Bitslice)
            .unwrap();
        let index = load_encoded_label_index(&InputFile::new(bitslice_index)).unwrap();
        let source = ValidatedEncodedQuerySource::AstJson(r#"{"and":["A","B"]}"#.to_string());
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
        let error = ast_to_label_expression(&ASTExpr::Compare {
            field: "score".to_string(),
            op: CompareOp::Gt(2.0),
        })
        .unwrap_err();
        assert!(error.to_string().contains("unsupported operator >"));
    }

    #[test]
    fn test_encoded_dnf_rejects_non_dnf_shape() {
        let expression = ast_to_label_expression(&ASTExpr::And(vec![
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
        ]))
        .unwrap();

        let error = flatten_dnf_clauses(&expression).unwrap_err();
        assert!(error
            .to_string()
            .contains("OR expressions cannot appear inside a conjunction"));
    }

    #[test]
    fn test_encoded_dnf_rejects_array_equality_explicitly() {
        let error = ast_to_label_expression(&ASTExpr::Compare {
            field: "brand".to_string(),
            op: CompareOp::Eq(json!(["A", "B"])),
        })
        .unwrap_err();
        assert!(error.to_string().contains("do not support array values"));
    }

    #[test]
    fn test_encoded_query_rejects_null_equality() {
        let error = ast_to_label_expression(&ASTExpr::Compare {
            field: "missing".to_string(),
            op: CompareOp::Eq(Value::Null),
        })
        .unwrap_err();
        assert!(error.to_string().contains("do not support null"));
    }

    #[test]
    fn test_encoded_benchmark_rejects_unknown_labels() {
        let dir = tempdir().unwrap();
        let labels_jsonl = dir.path().join("labels.jsonl");
        let bitslice_index = dir.path().join("labels.bitslice");
        write_lines(&labels_jsonl, &[r#"{"doc_id":0,"A":true}"#]);
        encode_label_index_jsonl(&labels_jsonl, &bitslice_index, LabelIndexFormat::Bitslice)
            .unwrap();
        let index = load_encoded_label_index(&InputFile::new(bitslice_index)).unwrap();
        let error =
            validate_encoded_benchmark_labels(&index, &LabelExpression::Label("missing".into()))
                .unwrap_err();
        assert!(error.to_string().contains("absent from the label index"));
    }
}
