/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A file-backed [`AttributeValueProvider`] for diversity-aware benchmarks.

use std::path::Path;

use anyhow::Context;
use diskann::{neighbor::AttributeValueProvider, provider::HasId};

/// The reserved attribute bucket assigned to graph navigation nodes (frozen start points)
/// that fall outside the range of loaded per-vector attributes.
///
/// Greedy graph search seeds traversal from the index's start points. Those points must
/// therefore carry an attribute, otherwise the diverse queue would drop them and search
/// would never expand beyond the entry node. Assigning them a dedicated bucket keeps them
/// traversable without merging them into any real attribute group.
const NAVIGATION_BUCKET: u32 = u32::MAX;

/// An attribute value provider backed by a plaintext attribute file.
///
/// The file is expected to contain one unsigned integer per line, where the value on the
/// `N`-th line (0-indexed) is the diversity attribute of the `N`-th vector. This matches the
/// on-disk layout produced by the labelling tools used elsewhere in the pipeline.
///
/// Ids outside the range of loaded attributes (for example the graph's frozen start points)
/// are mapped to a reserved [`NAVIGATION_BUCKET`] so that greedy search can still traverse
/// the graph through them.
#[derive(Debug, Clone)]
pub(crate) struct FileAttributeProvider {
    attributes: Vec<u32>,
    num_buckets: usize,
}

impl FileAttributeProvider {
    /// Load attributes from the plaintext file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if any line fails to parse as a `u32`.
    pub(crate) fn load(path: &Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("while reading attribute file {}", path.display()))?;

        let attributes = contents
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .enumerate()
            .map(|(line, value)| {
                value.parse::<u32>().with_context(|| {
                    format!("invalid attribute value {value:?} on line {}", line + 1)
                })
            })
            .collect::<anyhow::Result<Vec<u32>>>()?;

        let num_buckets = attributes
            .iter()
            .copied()
            .collect::<std::collections::HashSet<u32>>()
            .len();

        Ok(Self {
            attributes,
            num_buckets,
        })
    }
}

impl HasId for FileAttributeProvider {
    type Id = u32;
}

impl AttributeValueProvider for FileAttributeProvider {
    type Value = u32;

    fn get(&self, id: Self::Id) -> Option<Self::Value> {
        Some(
            self.attributes
                .get(id as usize)
                .copied()
                .unwrap_or(NAVIGATION_BUCKET),
        )
    }

    fn num_buckets(&self) -> Option<usize> {
        Some(self.num_buckets)
    }
}
