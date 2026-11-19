/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use bit_set::BitSet;
use std::{
    fmt::Debug,
    fs::File,
    io::{BufReader, Read},
};

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
    ASTExpr, DefaultKeyCodec,
};
use diskann_providers::model::graph::provider::layers::BetaFilter;

use diskann_tools::utils::ground_truth::read_labels_and_compute_bitmap;
use std::sync::Arc;

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

    fn match_count(&self) -> Option<usize> {
        Some(self.count())
    }
}

#[derive(Debug)]
pub struct BitmapFilter {
    bits: BitSet,
    count: usize,
}

impl BitmapFilter {
    pub fn new(bits: BitSet) -> Self {
        let count = bits.len();
        Self { bits, count }
    }
}

impl<T> QueryLabelProvider<T> for BitmapFilter
where
    T: VectorId + IntoUsize,
{
    fn is_match(&self, vec_id: T) -> bool {
        self.bits.contains(vec_id.into_usize())
    }

    fn match_count(&self) -> Option<usize> {
        Some(self.count)
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

/// Per-query allow lists. Format: b"DANFILT1", little-endian u32 query count,
/// little-endian u32 base count, then for each query a u8 mode and u32 count.
/// Mode 0 is `count` sorted u32 IDs; mode 1 is `count` packed bytes, with
/// vector ID zero in the high bit of the first byte. A prefix of query rows may
/// be read when the benchmark uses the first N queries of a dataset.
pub(crate) fn load_filter_bitmaps(
    path: &InputFile,
    query_count: usize,
) -> anyhow::Result<Vec<BitSet>> {
    let mut file = BufReader::new(File::open(path.to_str().unwrap())?);
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    anyhow::ensure!(&magic == b"DANFILT1", "invalid filter bitmap magic");
    let mut word = [0u8; 4];
    file.read_exact(&mut word)?;
    let rows = u32::from_le_bytes(word) as usize;
    file.read_exact(&mut word)?;
    let points = u32::from_le_bytes(word) as usize;
    anyhow::ensure!(
        rows >= query_count,
        "filter bitmap count {} is smaller than query count {}",
        rows,
        query_count
    );
    anyhow::ensure!(points > 0, "filter bitmap base count must be positive");
    let mut bitmaps = Vec::with_capacity(query_count);
    for row in 0..query_count {
        let mut mode = [0u8; 1];
        file.read_exact(&mut mode)?;
        file.read_exact(&mut word)?;
        let count = u32::from_le_bytes(word) as usize;
        let bitmap = match mode[0] {
            0 => {
                anyhow::ensure!(count <= points, "too many IDs in filter row {}", row);
                let mut bitmap = BitSet::new();
                for _ in 0..count {
                    file.read_exact(&mut word)?;
                    let id = u32::from_le_bytes(word) as usize;
                    anyhow::ensure!(id < points, "out-of-range ID {} in filter row {}", id, row);
                    bitmap.insert(id);
                }
                bitmap
            }
            1 => {
                anyhow::ensure!(
                    count == points.div_ceil(8),
                    "invalid packed filter length in row {}",
                    row
                );
                let mut bytes = vec![0u8; count];
                file.read_exact(&mut bytes)?;
                BitSet::from_bytes(&bytes)
            }
            _ => anyhow::bail!("invalid filter mode {} in row {}", mode[0], row),
        };
        bitmaps.push(bitmap);
    }
    if rows == query_count {
        let mut extra = [0u8; 1];
        anyhow::ensure!(
            file.read(&mut extra)? == 0,
            "trailing bytes in filter bitmap file"
        );
    }
    Ok(bitmaps)
}

pub(crate) fn get_filter_bitmaps(
    query_predicates: &Option<InputFile>,
    data_labels: &Option<InputFile>,
    filter_bitmaps: &Option<InputFile>,
    query_count: usize,
) -> anyhow::Result<Vec<BitSet>> {
    if let Some(path) = filter_bitmaps {
        load_filter_bitmaps(path, query_count)
    } else {
        generate_bitmaps(
            query_predicates.as_ref().unwrap(),
            data_labels.as_ref().unwrap(),
        )
    }
}

#[cfg(test)]
mod filter_file_tests {
    use super::*;

    #[test]
    fn reads_sparse_and_packed_allow_lists() {
        let path = std::env::temp_dir().join(format!("diskann-filter-{}.bin", std::process::id()));
        let mut bytes = b"DANFILT1".to_vec();
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.push(1);
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&[0b1000_0001, 0b0100_0000]);
        std::fs::write(&path, bytes).unwrap();
        let result = load_filter_bitmaps(&InputFile::new(&path), 2);
        let prefix = load_filter_bitmaps(&InputFile::new(&path), 1).unwrap();
        std::fs::remove_file(&path).unwrap();
        let bitmaps = result.unwrap();
        assert_eq!(prefix.len(), 1);
        assert_eq!(prefix[0].iter().collect::<Vec<_>>(), vec![1, 9]);
        assert_eq!(bitmaps[0].iter().collect::<Vec<_>>(), vec![1, 9]);
        assert_eq!(bitmaps[1].iter().collect::<Vec<_>>(), vec![0, 7, 9]);
    }
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
    Arc::new(BitmapFilter::new(set))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmap_filter_match() {
        let mut bitset = BitSet::new();
        bitset.insert(1);
        bitset.insert(3);
        let filter = BitmapFilter::new(bitset);

        assert!(filter.is_match(1u32));
        assert!(filter.is_match(3u32));
        assert!(!filter.is_match(2u32));
        assert!(!filter.is_match(0u32));
        assert_eq!(
            <BitmapFilter as QueryLabelProvider<u32>>::match_count(&filter),
            Some(2)
        );
    }

    #[test]
    fn test_bitmap_filter_empty() {
        let bitset = BitSet::new();
        let filter = BitmapFilter::new(bitset);

        assert!(!filter.is_match(0u32));
        assert!(!filter.is_match(10u32));
    }

    #[test]
    fn test_bitmap_filter_large_id() {
        let mut bitset = BitSet::new();
        bitset.insert(1000);
        let filter = BitmapFilter::new(bitset);

        assert!(filter.is_match(1000u32));
        assert!(!filter.is_match(999u32));
    }
}
