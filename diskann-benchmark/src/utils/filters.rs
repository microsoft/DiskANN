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
    read_and_parse_queries, read_baselabels, FrozenAttributeIndex, InlineAttributeIndex,
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
        index
            .insert_document(doc.doc_id as u32, &attrs)
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
}
