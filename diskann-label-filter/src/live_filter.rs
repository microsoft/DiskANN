/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Live (query-time, per-node) filter matching for graph search.
//!
//! This module provides a [`QueryLabelProvider`] implementation whose `is_match` evaluates
//! a filter predicate **against a single node's attributes at search time**, rather than
//! consulting a precomputed whole-corpus bitmap.
//!
//! Design:
//! * Each vector's labels are stored as a roaring set of integer attribute-ids in an
//!   in-memory [`RoaringTreemapSetProvider`] ([`InlineAttributeIndex`]). This is built once
//!   from the label data (analogous to building the vector index).
//! * A query's filter expression is encoded **once** into an [`EncodedFilterExpr`]
//!   (`ASTIdExpr<u64>`), turning field/value comparisons into integer terminals.
//! * At each visited node, [`PredicateEvaluator`] walks the encoded expression, resolving
//!   each terminal by an `O(1)`-ish roaring `contains` against the node's own attribute set.
//!   No FFI and no global posting-list materialization.
//!
//! Only the `AND`/`OR` + equality (set-membership) subset is supported; `NOT` and relational
//! operators are rejected at construction time.

use std::borrow::Cow;
use std::sync::{Arc, OnceLock, RwLock};

use diskann::graph::ext::labeled::QueryLabelProvider;
use diskann::{ANNError, ANNResult};
use roaring::{RoaringBitmap, RoaringTreemap};

use crate::attribute::Attribute;
use crate::encoded_attribute_provider::ast_id_expr::{ASTIdExpr, ASTIdExprVisitor};
use crate::encoded_attribute_provider::attribute_encoder::AttributeEncoder;
use crate::encoded_attribute_provider::encoded_filter_expr::EncodedFilterExpr;
use crate::inline_beta_search::predicate_evaluator::PredicateEvaluator;
use crate::set::roaring_set_provider::RoaringTreemapSetProvider;
use crate::set::SetProvider;
use crate::{ASTExpr, CompareOp};

/// An in-memory attribute index mapping vector ids to their encoded attribute-id sets.
///
/// This is the **builder**: attributes are inserted while building. Call [`Self::freeze`] to
/// obtain a read-only, lock-free [`FrozenAttributeIndex`] for use during search.
/// Vector ids are `u32` (the graph internal id) and attribute ids are `u64`.
pub struct InlineAttributeIndex {
    attribute_map: AttributeEncoder,
    index: RoaringTreemapSetProvider<u32>,
}

impl Default for InlineAttributeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl InlineAttributeIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            attribute_map: AttributeEncoder::new(),
            index: RoaringTreemapSetProvider::<u32>::new(),
        }
    }

    /// Register the attributes of a single vector (build-time).
    ///
    /// Each attribute is encoded to a stable integer id and inserted into the vector's set.
    /// A vector with no attributes is allowed (it simply has no entry, and will fail every
    /// equality predicate).
    pub fn insert_document(&mut self, vec_id: u32, attributes: &[Attribute]) -> ANNResult<()> {
        for attr in attributes {
            let attr_id = self.attribute_map.insert(attr);
            self.index.insert(&vec_id, &attr_id)?;
        }
        Ok(())
    }

    /// Freeze the builder into a read-only, shareable index for search.
    ///
    /// The attribute set store is shared lock-free (`Arc<..>`); the attribute encoder is placed
    /// behind an `RwLock` since it is only consulted once per query (at predicate encoding
    /// time), never per node.
    pub fn freeze(self) -> FrozenAttributeIndex {
        FrozenAttributeIndex {
            attribute_map: Arc::new(RwLock::new(self.attribute_map)),
            index: Arc::new(self.index),
        }
    }
}

/// A read-only, lock-free (for per-node reads) attribute index shared across queries.
pub struct FrozenAttributeIndex {
    attribute_map: Arc<RwLock<AttributeEncoder>>,
    index: Arc<RoaringTreemapSetProvider<u32>>,
}

impl FrozenAttributeIndex {
    /// Build a per-query [`QueryLabelProvider`] for `ast`.
    ///
    /// The expression is encoded once against the attribute map. Only `AND`/`OR` combined with
    /// equality terminals are permitted; `NOT` and relational operators (`!=`, `<`, `<=`, `>`,
    /// `>=`) are rejected.
    ///
    /// # Errors
    /// Returns an error if the expression uses an unsupported operator, or references a
    /// field/value that does not exist in the dataset.
    pub fn make_provider(&self, ast: &ASTExpr) -> ANNResult<Arc<dyn QueryLabelProvider<u32>>> {
        ensure_and_or_only(ast)?;
        let encoded = EncodedFilterExpr::new(ast, self.attribute_map.clone())?;
        Ok(Arc::new(InlineAttributeLabelProvider {
            encoded,
            index: self.index.clone(),
        }))
    }
}

/// Reject anything outside the supported `AND`/`OR` + equality subset.
fn ensure_and_or_only(ast: &ASTExpr) -> ANNResult<()> {
    match ast {
        ASTExpr::And(exprs) | ASTExpr::Or(exprs) => {
            for e in exprs {
                ensure_and_or_only(e)?;
            }
            Ok(())
        }
        ASTExpr::Compare {
            op: CompareOp::Eq(_),
            ..
        } => Ok(()),
        ASTExpr::Not(_) => Err(ANNError::message(
            "NOT is not supported by the live AND/OR filter",
        )),
        ASTExpr::Compare { op, .. } => Err(ANNError::message(format!(
            "operator {} is not supported by the live AND/OR filter (only $eq)",
            op
        ))),
    }
}

fn checked_vector_count(vec_id: u32) -> ANNResult<u32> {
    vec_id.checked_add(1).ok_or_else(|| {
        ANNError::message("dense live-filter indexes do not support vector id u32::MAX")
    })
}

/// A [`QueryLabelProvider`] that evaluates the encoded predicate against each node's own
/// attribute set at search time, reading the shared index lock-free.
struct InlineAttributeLabelProvider {
    encoded: EncodedFilterExpr,
    index: Arc<RoaringTreemapSetProvider<u32>>,
}

impl std::fmt::Debug for InlineAttributeLabelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InlineAttributeLabelProvider").finish()
    }
}

impl QueryLabelProvider<u32> for InlineAttributeLabelProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        // Fetch the node's attribute set, or use an empty set when the node has no
        // attributes (equality terminals then evaluate to false, i.e. non-match).
        let empty;
        let labels: &RoaringTreemap = match self.index.get(&vec_id) {
            Ok(Some(Cow::Borrowed(set))) => set,
            Ok(Some(Cow::Owned(ref set))) => {
                empty = set.clone();
                &empty
            }
            Ok(None) => {
                empty = RoaringTreemap::new();
                &empty
            }
            Err(_) => {
                tracing::warn!("live filter: failed to read node attributes; non-match");
                return false;
            }
        };

        let evaluator = PredicateEvaluator::new(labels);
        match evaluator.visit(self.encoded.encoded_filter_expr()) {
            Ok(matched) => matched,
            Err(_) => {
                tracing::warn!("live filter: predicate evaluation failed; treating as non-match");
                false
            }
        }
    }
}

/// A flat CSR (compressed-sparse-row) attribute index: an alternative to
/// [`InlineAttributeIndex`] that trades the per-node `HashMap<u32, RoaringTreemap>` for two
/// contiguous arrays.
///
/// * `offsets[id]..offsets[id + 1]` slices `values` to yield node `id`'s sorted attribute-id row.
/// * `is_match` reads that one contiguous slice (≈1 cache line for typical small rows) and answers
///   each equality terminal with a `binary_search`, avoiding the hash probe, the heap pointer chase
///   to a `RoaringTreemap`, and the `BTreeMap`/`RoaringBitmap` container lookups of the roaring path.
///
/// This is the **builder**: attributes are inserted while building. Call [`Self::freeze`] to obtain
/// a read-only, lock-free [`FrozenAttributeIndexCsr`] for use during search. Vector ids are `u32`
/// and are expected to be dense (`0..n`); attribute ids are `u32` (supports up to `u32::MAX`
/// distinct attributes and `u32::MAX` total attribute occurrences).
pub struct InlineAttributeIndexCsr {
    attribute_map: AttributeEncoder,
    rows: Vec<Vec<u32>>,
}

impl Default for InlineAttributeIndexCsr {
    fn default() -> Self {
        Self::new()
    }
}

impl InlineAttributeIndexCsr {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            attribute_map: AttributeEncoder::new(),
            rows: Vec::new(),
        }
    }

    /// Register the attributes of a single vector (build-time).
    ///
    /// Each attribute is encoded to a stable integer id (shared with the query encoder, so query
    /// terminals resolve to the same ids) and appended to the vector's row. A vector with no
    /// attributes is allowed. Vector ids need not be inserted in order; gaps are filled with empty
    /// rows.
    ///
    /// # Errors
    /// Returns an error if more than `u32::MAX` distinct attributes are encountered.
    pub fn insert_document(&mut self, vec_id: u32, attributes: &[Attribute]) -> ANNResult<()> {
        checked_vector_count(vec_id)?;
        let idx = vec_id as usize;
        if idx >= self.rows.len() {
            self.rows.resize_with(idx + 1, Vec::new);
        }
        for attr in attributes {
            let attr_id = self.attribute_map.insert(attr);
            let attr_id = u32::try_from(attr_id).map_err(|_| {
                ANNError::message("live CSR filter supports at most u32::MAX distinct attributes")
            })?;
            self.rows[idx].push(attr_id);
        }
        Ok(())
    }

    /// Freeze the builder into a read-only, shareable index for search.
    ///
    /// Each row is sorted and deduped (so `is_match` can `binary_search`), then flattened into the
    /// contiguous `values` array with a parallel `offsets` array. The attribute encoder is placed
    /// behind an `RwLock` since it is only consulted once per query (at predicate encoding time),
    /// never per node.
    pub fn freeze(mut self) -> FrozenAttributeIndexCsr {
        let mut offsets: Vec<u32> = Vec::with_capacity(self.rows.len() + 1);
        offsets.push(0);
        let total: usize = self.rows.iter().map(Vec::len).sum();
        let mut values: Vec<u32> = Vec::with_capacity(total);
        for row in &mut self.rows {
            row.sort_unstable();
            row.dedup();
            values.extend_from_slice(row);
            let end = u32::try_from(values.len())
                .expect("live CSR filter supports at most u32::MAX total attribute occurrences");
            offsets.push(end);
        }
        FrozenAttributeIndexCsr {
            attribute_map: Arc::new(RwLock::new(self.attribute_map)),
            offsets: offsets.into(),
            values: values.into(),
        }
    }
}

/// A read-only, lock-free (for per-node reads) CSR attribute index shared across queries.
pub struct FrozenAttributeIndexCsr {
    attribute_map: Arc<RwLock<AttributeEncoder>>,
    offsets: Arc<[u32]>,
    values: Arc<[u32]>,
}

impl FrozenAttributeIndexCsr {
    /// Build a per-query [`QueryLabelProvider`] for `ast`.
    ///
    /// The expression is encoded once against the shared attribute map (identical semantics to
    /// [`FrozenAttributeIndex::make_provider`]): only `AND`/`OR` combined with equality terminals
    /// are permitted; `NOT` and relational operators are rejected.
    ///
    /// # Errors
    /// Returns an error if the expression uses an unsupported operator, or references a field/value
    /// absent from the dataset.
    pub fn make_provider(&self, ast: &ASTExpr) -> ANNResult<Arc<dyn QueryLabelProvider<u32>>> {
        ensure_and_or_only(ast)?;
        let encoded = EncodedFilterExpr::new(ast, self.attribute_map.clone())?;
        Ok(Arc::new(CsrAttributeLabelProvider {
            encoded,
            offsets: self.offsets.clone(),
            values: self.values.clone(),
        }))
    }
}

/// A [`QueryLabelProvider`] that evaluates the encoded predicate against each node's contiguous
/// CSR attribute row at search time, reading the shared arrays lock-free.
struct CsrAttributeLabelProvider {
    encoded: EncodedFilterExpr,
    offsets: Arc<[u32]>,
    values: Arc<[u32]>,
}

impl std::fmt::Debug for CsrAttributeLabelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsrAttributeLabelProvider").finish()
    }
}

impl QueryLabelProvider<u32> for CsrAttributeLabelProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        let idx = vec_id as usize;
        // A node id outside the built range has no attributes: every equality terminal is a
        // non-match (consistent with the roaring path treating a missing node as an empty set).
        let (Some(&start), Some(&end)) = (self.offsets.get(idx), self.offsets.get(idx + 1)) else {
            return false;
        };
        let row = &self.values[start as usize..end as usize];
        let evaluator = CsrRowEvaluator { row };
        evaluator.visit(self.encoded.encoded_filter_expr())
    }
}

/// Evaluates an encoded `AND`/`OR` + equality predicate against a single node's sorted CSR row.
struct CsrRowEvaluator<'a> {
    row: &'a [u32],
}

impl ASTIdExprVisitor<u64> for CsrRowEvaluator<'_> {
    type Output = bool;

    fn visit_and(&self, exprs: &[ASTIdExpr<u64>]) -> bool {
        exprs.iter().all(|e| self.visit(e))
    }

    fn visit_or(&self, exprs: &[ASTIdExpr<u64>]) -> bool {
        exprs.iter().any(|e| self.visit(e))
    }

    fn visit_not(&self, expr: &ASTIdExpr<u64>) -> bool {
        // `NOT` is rejected at construction, so this is unreachable in practice; handle it
        // correctly anyway for completeness.
        !self.visit(expr)
    }

    fn visit_terminal(&self, id: &u64) -> bool {
        match u32::try_from(*id) {
            Ok(id) => self.row.binary_search(&id).is_ok(),
            // Attribute ids are assigned densely from 0, so this never triggers; a terminal beyond
            // the `u32` range cannot be present in any row.
            Err(_) => false,
        }
    }
}

/// A posting-list attribute index: an in-memory inverted index mapping each attribute-id to a
/// [`RoaringBitmap`] of the vector-ids that carry it. This is the representation used by
/// Lucene/Milvus/FAISS-style filtered search.
///
/// Unlike the roaring/CSR providers (which evaluate the predicate *per visited node*), this
/// builder feeds a provider that evaluates the predicate **once per query** via roaring set-algebra
/// (`AND`=intersect, `OR`=union) over the posting lists, materializes the resulting match set into a
/// dense bitset, and then answers each per-node `is_match` with an `O(1)` bit test. The match-set
/// materialization is done lazily at the first `is_match` so its cost is counted as live query time
/// (not amortized into an offline pass).
///
/// This is the **builder**; call [`Self::freeze`] for the shareable read-only index.
pub struct InlineAttributeIndexPosting {
    attribute_map: AttributeEncoder,
    posting: Vec<RoaringBitmap>,
    num_vectors: u32,
}

impl Default for InlineAttributeIndexPosting {
    fn default() -> Self {
        Self::new()
    }
}

impl InlineAttributeIndexPosting {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            attribute_map: AttributeEncoder::new(),
            posting: Vec::new(),
            num_vectors: 0,
        }
    }

    /// Register the attributes of a single vector (build-time): insert `vec_id` into the posting
    /// list of each of its (encoded) attribute-ids.
    pub fn insert_document(&mut self, vec_id: u32, attributes: &[Attribute]) -> ANNResult<()> {
        self.num_vectors = self.num_vectors.max(checked_vector_count(vec_id)?);
        for attr in attributes {
            let attr_id = self.attribute_map.insert(attr);
            let attr_id = usize::try_from(attr_id).map_err(|_| {
                ANNError::message(
                    "posting-list filter supports at most usize::MAX distinct attributes",
                )
            })?;
            if attr_id >= self.posting.len() {
                self.posting.resize_with(attr_id + 1, RoaringBitmap::new);
            }
            self.posting[attr_id].insert(vec_id);
        }
        Ok(())
    }

    /// Freeze the builder into a read-only, shareable index for search.
    pub fn freeze(self) -> FrozenAttributeIndexPosting {
        FrozenAttributeIndexPosting {
            attribute_map: Arc::new(RwLock::new(self.attribute_map)),
            posting: Arc::from(self.posting),
            num_vectors: self.num_vectors,
        }
    }
}

/// A read-only, shareable posting-list index.
pub struct FrozenAttributeIndexPosting {
    attribute_map: Arc<RwLock<AttributeEncoder>>,
    posting: Arc<[RoaringBitmap]>,
    num_vectors: u32,
}

impl FrozenAttributeIndexPosting {
    /// Build a per-query [`QueryLabelProvider`] for `ast`.
    ///
    /// The predicate is encoded once (same `AND`/`OR` + equality subset as the other live
    /// providers); the actual match-set bitmap is materialized lazily on first use. See
    /// [`InlineAttributeIndexPosting`] for the rationale.
    ///
    /// # Errors
    /// Returns an error if the expression uses an unsupported operator, or references a field/value
    /// absent from the dataset.
    pub fn make_provider(&self, ast: &ASTExpr) -> ANNResult<Arc<dyn QueryLabelProvider<u32>>> {
        ensure_and_or_only(ast)?;
        let encoded = EncodedFilterExpr::new(ast, self.attribute_map.clone())?;
        Ok(Arc::new(MaterializedBitmapProvider {
            encoded,
            posting: self.posting.clone(),
            num_vectors: self.num_vectors,
            dense: OnceLock::new(),
        }))
    }
}

/// A [`QueryLabelProvider`] that materializes the query's whole match set once (via roaring
/// set-algebra over posting lists) into a dense bitset, then answers each node with an `O(1)` test.
struct MaterializedBitmapProvider {
    encoded: EncodedFilterExpr,
    posting: Arc<[RoaringBitmap]>,
    num_vectors: u32,
    /// Dense match bitset, lazily built on first `is_match` so its cost is timed as query latency.
    dense: OnceLock<Vec<u64>>,
}

impl std::fmt::Debug for MaterializedBitmapProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaterializedBitmapProvider").finish()
    }
}

/// Evaluate an encoded `AND`/`OR` + equality predicate into the roaring set of matching vector-ids,
/// using per-attribute posting lists. `num_vectors` is only used for the (unreachable) empty-`AND`
/// case (vacuously true).
fn eval_posting_to_roaring(
    expr: &ASTIdExpr<u64>,
    posting: &[RoaringBitmap],
    num_vectors: u32,
) -> RoaringBitmap {
    match expr {
        ASTIdExpr::Terminal(id) => usize::try_from(*id)
            .ok()
            .and_then(|i| posting.get(i))
            .cloned()
            .unwrap_or_default(),
        ASTIdExpr::And(children) => {
            let mut iter = children.iter();
            let Some(first) = iter.next() else {
                let mut all = RoaringBitmap::new();
                all.insert_range(0..num_vectors);
                return all;
            };
            let mut acc = eval_posting_to_roaring(first, posting, num_vectors);
            for child in iter {
                if acc.is_empty() {
                    break;
                }
                acc &= eval_posting_to_roaring(child, posting, num_vectors);
            }
            acc
        }
        ASTIdExpr::Or(children) => {
            let mut acc = RoaringBitmap::new();
            for child in children {
                acc |= eval_posting_to_roaring(child, posting, num_vectors);
            }
            acc
        }
        // `NOT` is rejected at construction, so this is unreachable in practice.
        ASTIdExpr::Not(_) => RoaringBitmap::new(),
    }
}

/// Materialize a roaring match set into a dense bitset indexed by vector-id.
fn densify(result: &RoaringBitmap, num_vectors: u32) -> Vec<u64> {
    let mut bits = vec![0u64; (num_vectors as usize).div_ceil(64)];
    for id in result.iter() {
        let i = id as usize;
        bits[i / 64] |= 1u64 << (i % 64);
    }
    bits
}

/// Test a bit in a dense bitset produced by [`densify`].
#[inline]
fn dense_contains(bits: &[u64], vec_id: u32) -> bool {
    let i = vec_id as usize;
    (bits[i / 64] >> (i % 64)) & 1 == 1
}

impl QueryLabelProvider<u32> for MaterializedBitmapProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        if vec_id >= self.num_vectors {
            return false;
        }
        let bits = self.dense.get_or_init(|| {
            let result = eval_posting_to_roaring(
                self.encoded.encoded_filter_expr(),
                &self.posting,
                self.num_vectors,
            );
            densify(&result, self.num_vectors)
        });
        dense_contains(bits, vec_id)
    }
}

/// An adaptive live index that stores *both* a flat CSR layout and per-attribute posting lists
/// (sharing one encoder). At query time it materializes the match set's roaring bitmap once (via the
/// posting lists), reads its cardinality for free, and then picks the regime that is fastest for
/// that selectivity:
///
/// * **selective** match set (`<=` threshold) -> densify to a bitset, answer each node with an
///   `O(1)` bit test (best when few match and traversal is heavy);
/// * **broad** match set -> per-node **CSR** row scan (avoids the expensive densify of a large set).
///
/// The decision is made lazily on first `is_match`, so its (cheap) cost is counted as query latency.
pub struct InlineAttributeIndexAuto {
    attribute_map: AttributeEncoder,
    rows: Vec<Vec<u32>>,
    posting: Vec<RoaringBitmap>,
    num_vectors: u32,
}

impl Default for InlineAttributeIndexAuto {
    fn default() -> Self {
        Self::new()
    }
}

impl InlineAttributeIndexAuto {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            attribute_map: AttributeEncoder::new(),
            rows: Vec::new(),
            posting: Vec::new(),
            num_vectors: 0,
        }
    }

    /// Register a vector's attributes into both the CSR rows and the posting lists.
    pub fn insert_document(&mut self, vec_id: u32, attributes: &[Attribute]) -> ANNResult<()> {
        self.num_vectors = self.num_vectors.max(checked_vector_count(vec_id)?);
        let idx = vec_id as usize;
        if idx >= self.rows.len() {
            self.rows.resize_with(idx + 1, Vec::new);
        }
        for attr in attributes {
            let attr_id = self.attribute_map.insert(attr);
            let attr_idx = usize::try_from(attr_id).map_err(|_| {
                ANNError::message("auto filter supports at most usize::MAX distinct attributes")
            })?;
            let attr_id_u32 = u32::try_from(attr_id).map_err(|_| {
                ANNError::message("auto filter (CSR) supports at most u32::MAX distinct attributes")
            })?;
            self.rows[idx].push(attr_id_u32);
            if attr_idx >= self.posting.len() {
                self.posting.resize_with(attr_idx + 1, RoaringBitmap::new);
            }
            self.posting[attr_idx].insert(vec_id);
        }
        Ok(())
    }

    /// Freeze the builder into a read-only, shareable index for search.
    pub fn freeze(mut self) -> FrozenAttributeIndexAuto {
        let mut offsets: Vec<u32> = Vec::with_capacity(self.rows.len() + 1);
        offsets.push(0);
        let total: usize = self.rows.iter().map(Vec::len).sum();
        let mut values: Vec<u32> = Vec::with_capacity(total);
        for row in &mut self.rows {
            row.sort_unstable();
            row.dedup();
            values.extend_from_slice(row);
            let end = u32::try_from(values.len())
                .expect("auto filter supports at most u32::MAX total attribute occurrences");
            offsets.push(end);
        }
        FrozenAttributeIndexAuto {
            attribute_map: Arc::new(RwLock::new(self.attribute_map)),
            offsets: offsets.into(),
            values: values.into(),
            posting: Arc::from(self.posting),
            num_vectors: self.num_vectors,
        }
    }
}

/// A read-only, shareable adaptive index. See [`InlineAttributeIndexAuto`].
pub struct FrozenAttributeIndexAuto {
    attribute_map: Arc<RwLock<AttributeEncoder>>,
    offsets: Arc<[u32]>,
    values: Arc<[u32]>,
    posting: Arc<[RoaringBitmap]>,
    num_vectors: u32,
}

impl FrozenAttributeIndexAuto {
    /// Build a per-query [`QueryLabelProvider`] for `ast`.
    ///
    /// # Errors
    /// Returns an error if the expression uses an unsupported operator, or references a field/value
    /// absent from the dataset.
    pub fn make_provider(&self, ast: &ASTExpr) -> ANNResult<Arc<dyn QueryLabelProvider<u32>>> {
        ensure_and_or_only(ast)?;
        let encoded = EncodedFilterExpr::new(ast, self.attribute_map.clone())?;
        // Treat a match set below 1/8 of the corpus as "selective" -> bitmap; else CSR. This sits
        // below the measured bitmap<->CSR crossover (~10-15% selectivity).
        let selectivity_threshold = u64::from(self.num_vectors) / 8;
        Ok(Arc::new(AutoProvider {
            encoded,
            offsets: self.offsets.clone(),
            values: self.values.clone(),
            posting: self.posting.clone(),
            num_vectors: self.num_vectors,
            selectivity_threshold,
            decision: OnceLock::new(),
        }))
    }
}

/// Lazily-chosen per-node evaluation strategy for [`AutoProvider`].
enum AutoDecision {
    /// Selective filter: dense match bitset, `O(1)` lookups.
    Dense(Vec<u64>),
    /// Broad filter: evaluate the predicate against the node's CSR row.
    Csr,
}

/// A [`QueryLabelProvider`] that picks bitmap (selective) or CSR (broad) per query.
struct AutoProvider {
    encoded: EncodedFilterExpr,
    offsets: Arc<[u32]>,
    values: Arc<[u32]>,
    posting: Arc<[RoaringBitmap]>,
    num_vectors: u32,
    selectivity_threshold: u64,
    decision: OnceLock<AutoDecision>,
}

impl std::fmt::Debug for AutoProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoProvider").finish()
    }
}

impl AutoProvider {
    fn decide(&self) -> AutoDecision {
        let result = eval_posting_to_roaring(
            self.encoded.encoded_filter_expr(),
            &self.posting,
            self.num_vectors,
        );
        if result.len() <= self.selectivity_threshold {
            AutoDecision::Dense(densify(&result, self.num_vectors))
        } else {
            AutoDecision::Csr
        }
    }
}

impl QueryLabelProvider<u32> for AutoProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        if vec_id >= self.num_vectors {
            return false;
        }
        match self.decision.get_or_init(|| self.decide()) {
            AutoDecision::Dense(bits) => dense_contains(bits, vec_id),
            AutoDecision::Csr => {
                let idx = vec_id as usize;
                let start = self.offsets[idx] as usize;
                let end = self.offsets[idx + 1] as usize;
                let evaluator = CsrRowEvaluator {
                    row: &self.values[start..end],
                };
                evaluator.visit(self.encoded.encoded_filter_expr())
            }
        }
    }
}

/// A bit-sliced live index: one dense bitset per attribute, built at [`Self::freeze`] (index time).
/// `is_match` is one `O(1)` bit test per equality terminal, with **no** per-query build. Memory is
/// `num_attributes * ceil(num_vectors / 64) * 8` bytes, so it is best when the label vocabulary is
/// modest.
pub struct InlineAttributeIndexBitslice {
    attribute_map: AttributeEncoder,
    posting: Vec<RoaringBitmap>,
    num_vectors: u32,
}

impl Default for InlineAttributeIndexBitslice {
    fn default() -> Self {
        Self::new()
    }
}

impl InlineAttributeIndexBitslice {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            attribute_map: AttributeEncoder::new(),
            posting: Vec::new(),
            num_vectors: 0,
        }
    }

    /// Register a vector's attributes (same as the posting-list builder).
    pub fn insert_document(&mut self, vec_id: u32, attributes: &[Attribute]) -> ANNResult<()> {
        self.num_vectors = self.num_vectors.max(checked_vector_count(vec_id)?);
        for attr in attributes {
            let attr_id = self.attribute_map.insert(attr);
            let attr_idx = usize::try_from(attr_id).map_err(|_| {
                ANNError::message(
                    "bit-slice filter supports at most usize::MAX distinct attributes",
                )
            })?;
            if attr_idx >= self.posting.len() {
                self.posting.resize_with(attr_idx + 1, RoaringBitmap::new);
            }
            self.posting[attr_idx].insert(vec_id);
        }
        Ok(())
    }

    /// Freeze into per-attribute dense bitsets.
    pub fn freeze(self) -> FrozenAttributeIndexBitslice {
        let bitsets: Vec<Box<[u64]>> = self
            .posting
            .iter()
            .map(|rb| densify(rb, self.num_vectors).into_boxed_slice())
            .collect();
        FrozenAttributeIndexBitslice {
            attribute_map: Arc::new(RwLock::new(self.attribute_map)),
            bitsets: Arc::from(bitsets),
            num_vectors: self.num_vectors,
        }
    }
}

/// A read-only, shareable bit-sliced index. See [`InlineAttributeIndexBitslice`].
pub struct FrozenAttributeIndexBitslice {
    attribute_map: Arc<RwLock<AttributeEncoder>>,
    bitsets: Arc<[Box<[u64]>]>,
    num_vectors: u32,
}

impl FrozenAttributeIndexBitslice {
    /// Build a per-query [`QueryLabelProvider`] for `ast`.
    ///
    /// # Errors
    /// Returns an error if the expression uses an unsupported operator, or references a field/value
    /// absent from the dataset.
    pub fn make_provider(&self, ast: &ASTExpr) -> ANNResult<Arc<dyn QueryLabelProvider<u32>>> {
        ensure_and_or_only(ast)?;
        let encoded = EncodedFilterExpr::new(ast, self.attribute_map.clone())?;
        Ok(Arc::new(BitsliceProvider {
            encoded,
            bitsets: self.bitsets.clone(),
            num_vectors: self.num_vectors,
        }))
    }

    /// Build a provider using a flat disjunctive-normal-form representation.
    ///
    /// Accepted expressions are a terminal, an AND of terminals, or an OR whose children are
    /// terminals or ANDs of terminals. Associatively nested AND and OR groups are flattened.
    /// Expressions such as `(A OR B) AND (C OR D)` must be normalized before provider
    /// construction.
    ///
    /// # Errors
    /// Returns an error if the expression is unsupported, references an unknown attribute, or is
    /// not already in the accepted DNF shape.
    pub fn make_dnf_provider(&self, ast: &ASTExpr) -> ANNResult<Arc<dyn QueryLabelProvider<u32>>> {
        ensure_and_or_only(ast)?;
        let encoded = EncodedFilterExpr::new(ast, self.attribute_map.clone())?;
        let dnf = EncodedDnf::new(encoded.encoded_filter_expr(), self.bitsets.len())?;

        if let Some(attribute_id) = dnf.single_attribute() {
            Ok(Arc::new(BitsliceSingleProvider {
                attribute_id,
                bitsets: self.bitsets.clone(),
                num_vectors: self.num_vectors,
            }))
        } else {
            Ok(Arc::new(BitsliceDnfProvider {
                dnf,
                bitsets: self.bitsets.clone(),
                num_vectors: self.num_vectors,
            }))
        }
    }
}

/// A [`QueryLabelProvider`] that evaluates the predicate as one `O(1)` bit test per terminal against
/// the per-attribute dense bitsets. No per-query build.
struct BitsliceProvider {
    encoded: EncodedFilterExpr,
    bitsets: Arc<[Box<[u64]>]>,
    num_vectors: u32,
}

impl std::fmt::Debug for BitsliceProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitsliceProvider").finish()
    }
}

impl QueryLabelProvider<u32> for BitsliceProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        if vec_id >= self.num_vectors {
            return false;
        }
        let evaluator = BitsliceEvaluator {
            vec_id,
            bitsets: &self.bitsets,
        };
        evaluator.visit(self.encoded.encoded_filter_expr())
    }
}

/// Evaluates an encoded `AND`/`OR` + equality predicate for one node via per-attribute bitsets.
struct BitsliceEvaluator<'a> {
    vec_id: u32,
    bitsets: &'a [Box<[u64]>],
}

impl ASTIdExprVisitor<u64> for BitsliceEvaluator<'_> {
    type Output = bool;

    fn visit_and(&self, exprs: &[ASTIdExpr<u64>]) -> bool {
        exprs.iter().all(|e| self.visit(e))
    }

    fn visit_or(&self, exprs: &[ASTIdExpr<u64>]) -> bool {
        exprs.iter().any(|e| self.visit(e))
    }

    fn visit_not(&self, expr: &ASTIdExpr<u64>) -> bool {
        // `NOT` is rejected at construction; handled for completeness.
        !self.visit(expr)
    }

    fn visit_terminal(&self, id: &u64) -> bool {
        usize::try_from(*id)
            .ok()
            .and_then(|i| self.bitsets.get(i))
            .is_some_and(|bits| dense_contains(bits, self.vec_id))
    }
}

/// A flat OR-of-ANDs expression.
///
/// Clause `i` occupies `attributes[clause_offsets[i]..clause_offsets[i + 1]]`.
struct EncodedDnf {
    clause_offsets: Box<[usize]>,
    attributes: Box<[usize]>,
}

impl EncodedDnf {
    fn new(expr: &ASTIdExpr<u64>, num_attributes: usize) -> ANNResult<Self> {
        let mut clause_offsets = vec![0usize];
        let mut attributes = Vec::new();

        Self::append_disjunction(expr, num_attributes, &mut clause_offsets, &mut attributes)?;

        Ok(Self {
            clause_offsets: clause_offsets.into_boxed_slice(),
            attributes: attributes.into_boxed_slice(),
        })
    }

    fn append_disjunction(
        expr: &ASTIdExpr<u64>,
        num_attributes: usize,
        clause_offsets: &mut Vec<usize>,
        attributes: &mut Vec<usize>,
    ) -> ANNResult<()> {
        match expr {
            ASTIdExpr::Terminal(_) | ASTIdExpr::And(_) => {
                let start = attributes.len();
                Self::append_conjunction(expr, num_attributes, attributes)?;
                if attributes.len() == start {
                    return Err(Self::shape_error());
                }
                clause_offsets.push(attributes.len());
            }
            ASTIdExpr::Or(clauses) => {
                if clauses.is_empty() {
                    return Err(Self::shape_error());
                }
                for clause in clauses {
                    Self::append_disjunction(clause, num_attributes, clause_offsets, attributes)?;
                }
            }
            ASTIdExpr::Not(_) => return Err(Self::shape_error()),
        }

        Ok(())
    }

    fn append_conjunction(
        expr: &ASTIdExpr<u64>,
        num_attributes: usize,
        attributes: &mut Vec<usize>,
    ) -> ANNResult<()> {
        match expr {
            ASTIdExpr::Terminal(id) => {
                attributes.push(Self::attribute_index(*id, num_attributes)?);
            }
            ASTIdExpr::And(terms) => {
                if terms.is_empty() {
                    return Err(Self::shape_error());
                }
                for term in terms {
                    Self::append_conjunction(term, num_attributes, attributes)?;
                }
            }
            ASTIdExpr::Or(_) | ASTIdExpr::Not(_) => return Err(Self::shape_error()),
        }

        Ok(())
    }

    fn attribute_index(id: u64, num_attributes: usize) -> ANNResult<usize> {
        let index = usize::try_from(id).map_err(|_| Self::shape_error())?;
        if index >= num_attributes {
            return Err(ANNError::message(format!(
                "encoded attribute id {id} is outside the bit-slice index"
            )));
        }
        Ok(index)
    }

    fn single_attribute(&self) -> Option<usize> {
        (self.clause_offsets.len() == 2 && self.attributes.len() == 1).then_some(self.attributes[0])
    }

    fn shape_error() -> ANNError {
        ANNError::message(
            "bit-slice DNF requires a terminal, an AND of terminals, or an OR of terminal/AND clauses",
        )
    }
}

/// Specialized query provider for the common single-terminal predicate.
struct BitsliceSingleProvider {
    attribute_id: usize,
    bitsets: Arc<[Box<[u64]>]>,
    num_vectors: u32,
}

impl std::fmt::Debug for BitsliceSingleProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitsliceSingleProvider").finish()
    }
}

impl QueryLabelProvider<u32> for BitsliceSingleProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        vec_id < self.num_vectors && dense_contains(&self.bitsets[self.attribute_id], vec_id)
    }
}

/// Query provider that evaluates a flat OR of AND clauses without recursive AST traversal.
struct BitsliceDnfProvider {
    dnf: EncodedDnf,
    bitsets: Arc<[Box<[u64]>]>,
    num_vectors: u32,
}

impl std::fmt::Debug for BitsliceDnfProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitsliceDnfProvider").finish()
    }
}

impl QueryLabelProvider<u32> for BitsliceDnfProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        if vec_id >= self.num_vectors {
            return false;
        }

        let bit_index = vec_id as usize;
        let word_index = bit_index / 64;
        let mask = 1u64 << (bit_index % 64);

        for clause in self.dnf.clause_offsets.windows(2) {
            let start = clause[0];
            let end = clause[1];
            let matches = self.dnf.attributes[start..end]
                .iter()
                .all(|&attribute_id| self.bitsets[attribute_id][word_index] & mask != 0);
            if matches {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn attr(field: &str, value: bool) -> Attribute {
        Attribute::from_json_value(field, &json!(value)).unwrap()
    }

    fn eq_true(field: &str) -> ASTExpr {
        ASTExpr::Compare {
            field: field.to_string(),
            op: CompareOp::Eq(json!(true)),
        }
    }

    #[test]
    fn matches_and_or_predicates() {
        let mut index = InlineAttributeIndex::new();
        // doc 0: {A, B}, doc 1: {A}, doc 2: {C}, doc 3: {} (no attributes)
        index
            .insert_document(0, &[attr("A", true), attr("B", true)])
            .unwrap();
        index.insert_document(1, &[attr("A", true)]).unwrap();
        index.insert_document(2, &[attr("C", true)]).unwrap();
        index.insert_document(3, &[]).unwrap();
        let index = index.freeze();

        // A AND B
        let p = index
            .make_provider(&ASTExpr::And(vec![eq_true("A"), eq_true("B")]))
            .unwrap();
        assert!(p.is_match(0));
        assert!(!p.is_match(1));
        assert!(!p.is_match(2));
        assert!(!p.is_match(3));

        // A OR C
        let p = index
            .make_provider(&ASTExpr::Or(vec![eq_true("A"), eq_true("C")]))
            .unwrap();
        assert!(p.is_match(0));
        assert!(p.is_match(1));
        assert!(p.is_match(2));
        assert!(!p.is_match(3));
    }

    #[test]
    fn rejects_not_and_relational() {
        let mut index = InlineAttributeIndex::new();
        index.insert_document(0, &[attr("A", true)]).unwrap();
        let index = index.freeze();

        // NOT is rejected
        assert!(index
            .make_provider(&ASTExpr::Not(Box::new(eq_true("A"))))
            .is_err());

        // relational is rejected
        let rel = ASTExpr::Compare {
            field: "n".to_string(),
            op: CompareOp::Gt(1.0),
        };
        assert!(index.make_provider(&rel).is_err());
    }

    #[test]
    fn csr_matches_and_or_predicates() {
        let mut index = InlineAttributeIndexCsr::new();
        // doc 0: {A, B}, doc 1: {A}, doc 2: {C}, doc 3: {} (no attributes)
        index
            .insert_document(0, &[attr("A", true), attr("B", true)])
            .unwrap();
        index.insert_document(1, &[attr("A", true)]).unwrap();
        index.insert_document(2, &[attr("C", true)]).unwrap();
        index.insert_document(3, &[]).unwrap();
        let index = index.freeze();

        // A AND B
        let p = index
            .make_provider(&ASTExpr::And(vec![eq_true("A"), eq_true("B")]))
            .unwrap();
        assert!(p.is_match(0));
        assert!(!p.is_match(1));
        assert!(!p.is_match(2));
        assert!(!p.is_match(3));
        // A node id beyond the built range is a non-match.
        assert!(!p.is_match(99));

        // A OR C
        let p = index
            .make_provider(&ASTExpr::Or(vec![eq_true("A"), eq_true("C")]))
            .unwrap();
        assert!(p.is_match(0));
        assert!(p.is_match(1));
        assert!(p.is_match(2));
        assert!(!p.is_match(3));
    }

    #[test]
    fn csr_rejects_not_and_relational() {
        let mut index = InlineAttributeIndexCsr::new();
        index.insert_document(0, &[attr("A", true)]).unwrap();
        let index = index.freeze();

        // NOT is rejected
        assert!(index
            .make_provider(&ASTExpr::Not(Box::new(eq_true("A"))))
            .is_err());

        // relational is rejected
        let rel = ASTExpr::Compare {
            field: "n".to_string(),
            op: CompareOp::Gt(1.0),
        };
        assert!(index.make_provider(&rel).is_err());
    }

    #[test]
    fn csr_out_of_order_inserts_match_roaring() {
        // Insert documents out of vec_id order to exercise gap-filling in the CSR builder, and
        // assert the CSR provider agrees with the roaring provider on the same data + predicate.
        let docs: [(u32, &[(&str, bool)]); 4] = [
            (2, &[("A", true), ("C", true)]),
            (0, &[("A", true), ("B", true)]),
            (3, &[("B", true)]),
            (1, &[]),
        ];

        let mut roaring = InlineAttributeIndex::new();
        let mut csr = InlineAttributeIndexCsr::new();
        for (id, labels) in docs {
            let attrs: Vec<Attribute> = labels.iter().map(|(f, v)| attr(f, *v)).collect();
            roaring.insert_document(id, &attrs).unwrap();
            csr.insert_document(id, &attrs).unwrap();
        }
        let roaring = roaring.freeze();
        let csr = csr.freeze();

        let ast = ASTExpr::Or(vec![
            ASTExpr::And(vec![eq_true("A"), eq_true("B")]),
            eq_true("C"),
        ]);
        let rp = roaring.make_provider(&ast).unwrap();
        let cp = csr.make_provider(&ast).unwrap();
        for id in 0..4u32 {
            assert_eq!(rp.is_match(id), cp.is_match(id), "mismatch at node {id}");
        }
    }

    #[test]
    fn posting_matches_roaring() {
        // The materialized-bitmap provider must agree with the roaring provider on the same data
        // and predicate, including an empty node and out-of-range ids.
        let docs: [(u32, &[(&str, bool)]); 4] = [
            (0, &[("A", true), ("B", true)]),
            (1, &[("A", true)]),
            (2, &[("C", true)]),
            (3, &[]),
        ];

        let mut roaring = InlineAttributeIndex::new();
        let mut posting = InlineAttributeIndexPosting::new();
        for (id, labels) in docs {
            let attrs: Vec<Attribute> = labels.iter().map(|(f, v)| attr(f, *v)).collect();
            roaring.insert_document(id, &attrs).unwrap();
            posting.insert_document(id, &attrs).unwrap();
        }
        let roaring = roaring.freeze();
        let posting = posting.freeze();

        for ast in [
            ASTExpr::And(vec![eq_true("A"), eq_true("B")]),
            ASTExpr::Or(vec![eq_true("A"), eq_true("C")]),
        ] {
            let rp = roaring.make_provider(&ast).unwrap();
            let pp = posting.make_provider(&ast).unwrap();
            for id in 0..6u32 {
                assert_eq!(rp.is_match(id), pp.is_match(id), "mismatch at node {id}");
            }
        }
    }

    #[test]
    fn posting_rejects_not_and_relational() {
        let mut index = InlineAttributeIndexPosting::new();
        index.insert_document(0, &[attr("A", true)]).unwrap();
        let index = index.freeze();

        assert!(index
            .make_provider(&ASTExpr::Not(Box::new(eq_true("A"))))
            .is_err());
        let rel = ASTExpr::Compare {
            field: "n".to_string(),
            op: CompareOp::Gt(1.0),
        };
        assert!(index.make_provider(&rel).is_err());
    }

    #[test]
    fn auto_dense_and_csr_branches_match_roaring() {
        // 40 vectors -> auto selectivity threshold = 40/8 = 5. "COMMON" (on all) is broad -> CSR
        // branch; "RARE" (on one) is selective -> Dense branch. Both must match the roaring result.
        let n = 40u32;
        let mut roaring = InlineAttributeIndex::new();
        let mut auto = InlineAttributeIndexAuto::new();
        for id in 0..n {
            let mut labels = vec![attr("COMMON", true)];
            if id == 7 {
                labels.push(attr("RARE", true));
            }
            roaring.insert_document(id, &labels).unwrap();
            auto.insert_document(id, &labels).unwrap();
        }
        let roaring = roaring.freeze();
        let auto = auto.freeze();
        for ast in [eq_true("COMMON"), eq_true("RARE")] {
            let rp = roaring.make_provider(&ast).unwrap();
            let ap = auto.make_provider(&ast).unwrap();
            for id in 0..n {
                assert_eq!(
                    rp.is_match(id),
                    ap.is_match(id),
                    "auto mismatch at node {id}"
                );
            }
        }
    }

    #[test]
    fn all_live_providers_agree() {
        let docs: [(u32, &[(&str, bool)]); 5] = [
            (0, &[("A", true), ("B", true)]),
            (1, &[("A", true)]),
            (2, &[("C", true)]),
            (3, &[]),
            (4, &[("B", true), ("C", true)]),
        ];
        let mut roaring = InlineAttributeIndex::new();
        let mut csr = InlineAttributeIndexCsr::new();
        let mut posting = InlineAttributeIndexPosting::new();
        let mut auto = InlineAttributeIndexAuto::new();
        let mut bitslice = InlineAttributeIndexBitslice::new();
        for (id, labels) in docs {
            let attrs: Vec<Attribute> = labels.iter().map(|(f, v)| attr(f, *v)).collect();
            roaring.insert_document(id, &attrs).unwrap();
            csr.insert_document(id, &attrs).unwrap();
            posting.insert_document(id, &attrs).unwrap();
            auto.insert_document(id, &attrs).unwrap();
            bitslice.insert_document(id, &attrs).unwrap();
        }
        let roaring = roaring.freeze();
        let csr = csr.freeze();
        let posting = posting.freeze();
        let auto = auto.freeze();
        let bitslice = bitslice.freeze();

        let asts = [
            eq_true("A"),
            ASTExpr::And(vec![eq_true("A"), eq_true("B")]),
            ASTExpr::Or(vec![eq_true("A"), eq_true("C")]),
            ASTExpr::Or(vec![
                ASTExpr::And(vec![eq_true("A"), eq_true("B")]),
                eq_true("C"),
            ]),
            ASTExpr::Or(vec![
                ASTExpr::And(vec![eq_true("A"), ASTExpr::And(vec![eq_true("B")])]),
                ASTExpr::Or(vec![eq_true("C")]),
            ]),
        ];
        for ast in &asts {
            let rp = roaring.make_provider(ast).unwrap();
            let others = [
                csr.make_provider(ast).unwrap(),
                posting.make_provider(ast).unwrap(),
                auto.make_provider(ast).unwrap(),
                bitslice.make_provider(ast).unwrap(),
                bitslice.make_dnf_provider(ast).unwrap(),
            ];
            for id in 0..7u32 {
                let expected = rp.is_match(id);
                for p in &others {
                    assert_eq!(p.is_match(id), expected, "disagreement at node {id}");
                }
            }
        }
    }

    #[test]
    fn dense_live_providers_reject_max_vector_id() {
        let attrs = [attr("A", true)];

        let mut csr = InlineAttributeIndexCsr::new();
        assert!(csr.insert_document(u32::MAX, &attrs).is_err());

        let mut posting = InlineAttributeIndexPosting::new();
        assert!(posting.insert_document(u32::MAX, &attrs).is_err());

        let mut auto = InlineAttributeIndexAuto::new();
        assert!(auto.insert_document(u32::MAX, &attrs).is_err());

        let mut bitslice = InlineAttributeIndexBitslice::new();
        assert!(bitslice.insert_document(u32::MAX, &attrs).is_err());
    }

    #[test]
    fn bitslice_dnf_rejects_and_of_or_clauses() {
        let mut index = InlineAttributeIndexBitslice::new();
        index
            .insert_document(
                0,
                &[
                    attr("A", true),
                    attr("B", true),
                    attr("C", true),
                    attr("D", true),
                ],
            )
            .unwrap();
        let index = index.freeze();
        let ast = ASTExpr::And(vec![
            ASTExpr::Or(vec![eq_true("A"), eq_true("B")]),
            ASTExpr::Or(vec![eq_true("C"), eq_true("D")]),
        ]);

        assert!(index.make_provider(&ast).is_ok());
        assert!(index.make_dnf_provider(&ast).is_err());
    }
}
