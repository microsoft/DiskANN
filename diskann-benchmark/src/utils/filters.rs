/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use bit_set::BitSet;
use std::fmt::Debug;

use diskann::{graph::index::QueryLabelProvider, utils::VectorId};
use diskann_benchmark_runner::files::InputFile;
use diskann_label_filter::{
    kv_index::GenericIndex,
    stores::bftree_store::BfTreeStore,
    traits::{
        posting_list_trait::{PostingList, RoaringPostingList},
        query_evaluator::QueryEvaluator,
    },
    ASTExpr, DefaultKeyCodec,
};
use diskann_providers::model::graph::provider::layers::BetaFilter;

use diskann_tools::utils::ground_truth::read_labels_and_compute_bitmap;
use std::sync::Arc;

use diskann_label_filter::attribute::Attribute;
use diskann_label_filter::{
    read_and_parse_queries, read_baselabels, FrozenAttributeIndex, FrozenAttributeIndexAuto,
    FrozenAttributeIndexBitslice, FrozenAttributeIndexCsr, FrozenAttributeIndexPosting,
    InlineAttributeIndex, InlineAttributeIndexAuto, InlineAttributeIndexBitslice,
    InlineAttributeIndexCsr, InlineAttributeIndexPosting,
};

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
    T: VectorId,
{
    fn is_match(&self, vec_id: T) -> bool {
        self.get_bitmap().contains(vec_id.into_usize())
    }
}

#[derive(Debug)]
pub struct BitmapFilter(pub BitSet);

impl<T> QueryLabelProvider<T> for BitmapFilter
where
    T: VectorId,
{
    fn is_match(&self, vec_id: T) -> bool {
        self.0.contains(vec_id.into_usize())
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

fn checked_doc_id(doc_id: usize) -> anyhow::Result<u32> {
    u32::try_from(doc_id)
        .map_err(|_| anyhow::anyhow!("document id {doc_id} exceeds the u32 graph-id range"))
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
