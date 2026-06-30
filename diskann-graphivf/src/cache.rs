/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! An in-memory cache of the largest inverted lists.
//!
//! The list file is read with direct I/O (`FILE_FLAG_NO_BUFFERING` /
//! `O_DIRECT`), which bypasses the OS page cache. Without an application cache,
//! every probe of a hot (dense) cluster re-reads its list from disk. Dense lists
//! are both *larger* (they dominate the size-weighted mean list size, which
//! predicts disk-read cost) and *more likely to be probed* (queries land in
//! dense regions). Caching the largest lists up to a byte budget therefore
//! removes a disproportionate share of the bytes read per query.

use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

use diskann::utils::VectorRepr;

use crate::{storage::Layout, Result};

/// One cluster's inverted list held in RAM: parsed ids and flattened vectors.
struct CachedList<T> {
    ids: Box<[u32]>,
    /// `count * dim` components in row-major order.
    vectors: Box<[T]>,
}

/// A read-only cache mapping a subset of cluster ids to their in-RAM lists.
///
/// Shared across searchers behind an `Arc`; immutable after construction.
pub(crate) struct ListCache<T> {
    /// Indexed by cluster id; `Some` for cached clusters.
    entries: Vec<Option<CachedList<T>>>,
    /// Total list payload bytes held in the cache.
    bytes: u64,
    /// Number of cached clusters.
    len: usize,
}

impl<T: VectorRepr> ListCache<T> {
    /// An empty cache covering `num_clusters` clusters (none cached).
    pub(crate) fn empty(num_clusters: usize) -> Self {
        Self {
            entries: (0..num_clusters).map(|_| None).collect(),
            bytes: 0,
            len: 0,
        }
    }

    /// Build a cache holding the largest lists whose cumulative payload fits in
    /// `budget_bytes`, reading them from the list file at `lists_path`.
    ///
    /// Selection is greedy by descending list size: dense lists maximize both
    /// the bytes served per cache byte and the probe-hit probability. A
    /// `budget_bytes` of zero yields an empty cache.
    pub(crate) fn build(lists_path: &Path, layout: &Layout, budget_bytes: u64) -> Result<Self> {
        let num_clusters = layout.num_clusters();
        let mut cache = Self::empty(num_clusters);
        if budget_bytes == 0 {
            return Ok(cache);
        }
        let dim = layout.dim;
        let elem = layout.element_size;
        let payload_of = |count: usize| count * (4 + dim * elem);

        // Order non-empty clusters by descending payload, then greedily take the
        // largest until the budget is exhausted.
        let mut order: Vec<usize> = (0..num_clusters)
            .filter(|&c| layout.counts[c] > 0)
            .collect();
        order.sort_unstable_by_key(|&c| std::cmp::Reverse(layout.counts[c]));

        let mut selected: Vec<usize> = Vec::new();
        let mut used: u64 = 0;
        for &c in &order {
            let payload = payload_of(layout.counts[c] as usize) as u64;
            if used + payload > budget_bytes {
                break;
            }
            used += payload;
            selected.push(c);
        }

        // Read the selected clusters in on-disk order for sequential locality.
        // A normal buffered read suffices here (the cache fill is one-off and
        // off the query path), so this does not need the direct-I/O reader.
        selected.sort_unstable();
        let mut reader = BufReader::new(File::open(lists_path)?);
        for &c in &selected {
            let count = layout.counts[c] as usize;
            reader.seek(SeekFrom::Start(layout.offsets[c]))?;
            // ids and vectors are stored back-to-back; read each into a
            // correctly aligned typed buffer to avoid byte-slice realignment.
            let mut ids = vec![0u32; count];
            reader.read_exact(bytemuck::cast_slice_mut(ids.as_mut_slice()))?;
            let mut vectors = vec![T::zeroed(); count * dim];
            reader.read_exact(bytemuck::cast_slice_mut(vectors.as_mut_slice()))?;
            cache.entries[c] = Some(CachedList {
                ids: ids.into_boxed_slice(),
                vectors: vectors.into_boxed_slice(),
            });
        }
        cache.bytes = used;
        cache.len = selected.len();
        Ok(cache)
    }

    /// Number of cached clusters.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Total list payload bytes held in the cache.
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The cached ids and flattened vectors for cluster `c`, or `None` if `c` is
    /// not cached.
    pub(crate) fn get(&self, c: usize) -> Option<(&[u32], &[T])> {
        self.entries[c].as_ref().map(|e| (&*e.ids, &*e.vectors))
    }
}
