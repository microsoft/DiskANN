/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Configuration and deterministic row hashing for transposed Bloom label indexes.

use crate::{error::EncodedLabelIndexError, format::MAX_LABEL_COUNT};
use xxhash_rust::xxh3::xxh3_128;

const MAX_BLOOM_BITS: u32 = 65_536;
const MAX_BLOOM_HASHES: u32 = 64;

/// Configuration for a transposed Bloom label index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BloomFilterConfig {
    bit_count: u32,
    hash_count: u32,
    exact_label_count: u32,
}

impl BloomFilterConfig {
    /// Create a Bloom configuration with `bit_count` transposed rows and `hash_count` row hashes.
    pub fn new(bit_count: u32, hash_count: u32) -> Result<Self, EncodedLabelIndexError> {
        validate_bloom_config(bit_count, hash_count)?;
        Ok(Self {
            bit_count,
            hash_count,
            exact_label_count: 0,
        })
    }

    /// Create a configuration that stores every possible dictionary label as an exact row.
    pub fn exact_only() -> Self {
        Self {
            exact_label_count: MAX_LABEL_COUNT as u32,
            ..Self::default()
        }
    }

    /// Store the most frequent `exact_label_count` labels as exact rows instead of Bloom.
    pub fn with_exact_label_count(
        mut self,
        exact_label_count: u32,
    ) -> Result<Self, EncodedLabelIndexError> {
        if exact_label_count as usize > MAX_LABEL_COUNT {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "exact Bloom label count must be at most {MAX_LABEL_COUNT}, got {exact_label_count}"
            )));
        }
        self.exact_label_count = exact_label_count;
        Ok(self)
    }

    /// Return the fixed Bloom signature length used for every vector.
    pub fn bit_count(self) -> u32 {
        self.bit_count
    }

    /// Return the number of distinct Bloom rows selected for each label.
    pub fn hash_count(self) -> u32 {
        self.hash_count
    }

    /// Return how many highest-frequency labels are also stored as exact rows.
    pub fn exact_label_count(self) -> u32 {
        self.exact_label_count
    }
}

impl Default for BloomFilterConfig {
    fn default() -> Self {
        Self {
            bit_count: 128,
            hash_count: 4,
            exact_label_count: 0,
        }
    }
}

pub(crate) fn validate_bloom_config(
    bit_count: u32,
    hash_count: u32,
) -> Result<(), EncodedLabelIndexError> {
    if bit_count == 0 || bit_count > MAX_BLOOM_BITS {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "Bloom bit count must be in 1..={MAX_BLOOM_BITS}, got {bit_count}"
        )));
    }
    if hash_count == 0 || hash_count > MAX_BLOOM_HASHES || hash_count > bit_count {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "Bloom hash count must be in 1..={}, got {hash_count}",
            bit_count.min(MAX_BLOOM_HASHES)
        )));
    }
    Ok(())
}

pub(crate) fn append_label_rows(label: &str, config: BloomFilterConfig, rows: &mut Vec<u32>) {
    let start = rows.len();
    // Kirsch-Mitzenmacher double hashing derives all Bloom positions from one 128-bit hash.
    let (first_hash, step_hash) = label_hash_pair(label.as_bytes());
    for hash_index in 0..config.hash_count {
        let mut row = first_hash.wrapping_add(u64::from(hash_index).wrapping_mul(step_hash))
            % u64::from(config.bit_count);
        while rows[start..].contains(&(row as u32)) {
            row = (row + 1) % u64::from(config.bit_count);
        }
        rows.push(row as u32);
    }
}

fn label_hash_pair(bytes: &[u8]) -> (u64, u64) {
    let hash = xxh3_128(bytes);
    let first_hash = hash as u64;
    // An odd step traverses every row when the configured row count is a power of two.
    let step_hash = ((hash >> 64) as u64) | 1;
    (first_hash, step_hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_hashing_is_stable() {
        assert_eq!(
            label_hash_pair(b""),
            (6_918_025_063_187_695_999, 11_072_670_137_173_121_241)
        );
        assert_eq!(
            label_hash_pair(b"MARKET=EN-US"),
            (6_522_125_905_851_381_566, 1_379_806_965_550_470_695)
        );
        let mut rows = Vec::new();
        append_label_rows(
            "MARKET=EN-US",
            BloomFilterConfig::new(256, 4).unwrap(),
            &mut rows,
        );
        assert_eq!(rows, [62, 101, 140, 179]);
    }

    #[test]
    fn double_hashing_produces_distinct_rows() {
        let config = BloomFilterConfig::new(8, 8).unwrap();
        for label in ["A", "B", "MARKET=EN-US", "WWW.EXAMPLE.COM"] {
            let mut rows = Vec::new();
            append_label_rows(label, config, &mut rows);
            rows.sort_unstable();
            rows.dedup();
            assert_eq!(rows.len(), 8);
        }
    }

    #[test]
    fn double_hashing_distributes_rows_evenly() {
        let config = BloomFilterConfig::new(256, 4).unwrap();
        let mut counts = [0u32; 256];
        for index in 0..10_000 {
            let mut rows = Vec::new();
            append_label_rows(&format!("label-{index}"), config, &mut rows);
            for row in rows {
                counts[row as usize] += 1;
            }
        }
        assert!(counts.iter().all(|&count| (120..=195).contains(&count)));
    }
}
