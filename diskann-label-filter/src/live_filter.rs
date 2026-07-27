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
use std::sync::{Arc, RwLock};

use diskann::graph::index::QueryLabelProvider;
use diskann::{ANNError, ANNErrorKind, ANNResult};
use roaring::RoaringTreemap;

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
            ANNErrorKind::Opaque,
            "NOT is not supported by the live AND/OR filter",
        )),
        ASTExpr::Compare { op, .. } => Err(ANNError::message(
            ANNErrorKind::Opaque,
            format!(
                "operator {} is not supported by the live AND/OR filter (only $eq)",
                op
            ),
        )),
    }
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
        let idx = vec_id as usize;
        if idx >= self.rows.len() {
            self.rows.resize_with(idx + 1, Vec::new);
        }
        for attr in attributes {
            let attr_id = self.attribute_map.insert(attr);
            let attr_id = u32::try_from(attr_id).map_err(|_| {
                ANNError::message(
                    ANNErrorKind::Opaque,
                    "live CSR filter supports at most u32::MAX distinct attributes",
                )
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
}
