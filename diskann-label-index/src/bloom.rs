/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Configuration and deterministic row hashing for transposed Bloom label indexes.

use crate::{error::EncodedLabelIndexError, format::MAX_LABEL_COUNT};
use xxhash_rust::xxh3::xxh3_64_with_seed;

const MAX_BLOOM_BITS: u32 = 65_536;
const MAX_BLOOM_HASHES: u32 = 64;
// This Weyl sequence is part of the persisted Bloom-row selection contract. Each hash index gets
// a stable XXH3 seed without deriving its row from another hash result.
const HASH_SEED_BASE: u64 = 0x243f_6a88_85a3_08d3;
const HASH_SEED_STEP: u64 = 0x9e37_79b9_7f4a_7c15;

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
    let label = label.as_bytes();
    let bit_count = u64::from(config.bit_count);
    for hash_index in 0..config.hash_count {
        let mut row = label_hash(label, hash_index) % bit_count;
        while rows[start..].contains(&(row as u32)) {
            row = (row + 1) % bit_count;
        }
        rows.push(row as u32);
    }
}

#[inline]
fn label_hash(bytes: &[u8], hash_index: u32) -> u64 {
    let seed = HASH_SEED_BASE.wrapping_add(u64::from(hash_index).wrapping_mul(HASH_SEED_STEP));
    xxh3_64_with_seed(bytes, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use xxhash_rust::xxh3::xxh3_128;

    #[test]
    fn seeded_hashing_is_stable() {
        let empty_hash = label_hash(b"", 0);
        let synthetic_hash = label_hash(b"synthetic-label-42", 2);
        let mut rows = Vec::new();
        append_label_rows(
            "synthetic-label-42",
            BloomFilterConfig::new(256, 4).unwrap(),
            &mut rows,
        );
        assert_eq!(
            (empty_hash, synthetic_hash, rows),
            (
                3_305_381_855_338_814_158,
                11_187_711_007_975_435_878,
                vec![6, 216, 102, 20],
            )
        );
        assert_eq!(
            seeded_signature(
                "synthetic-label-42",
                BloomFilterConfig::new(256, 3).unwrap()
            ),
            [6, 102, 216]
        );
    }

    #[test]
    fn seeded_hashing_produces_distinct_rows() {
        let config = BloomFilterConfig::new(8, 8).unwrap();
        for label in [
            "synthetic-label-0",
            "synthetic-label-1",
            "synthetic-label-42",
            "synthetic-label-9999",
        ] {
            let mut rows = Vec::new();
            append_label_rows(label, config, &mut rows);
            rows.sort_unstable();
            rows.dedup();
            assert_eq!(rows.len(), 8);
        }
    }

    #[test]
    fn seeded_hashing_distributes_rows_evenly() {
        let config = BloomFilterConfig::new(256, 4).unwrap();
        let mut counts = [0u32; 256];
        for index in 0..10_000 {
            let mut rows = Vec::with_capacity(config.hash_count() as usize);
            append_label_rows(&format!("synthetic-label-{index}"), config, &mut rows);
            for row in rows {
                counts[row as usize] += 1;
            }
        }
        let min = counts.iter().copied().min().unwrap();
        let max = counts.iter().copied().max().unwrap();
        assert!(
            min >= 110 && max <= 220,
            "row counts range from {min} to {max}"
        );
    }

    #[derive(Debug, PartialEq, Eq)]
    struct CollisionStats {
        unique_signatures: usize,
        collision_groups: usize,
        aliased_labels: usize,
        largest_group: usize,
    }

    #[test]
    fn seeded_hashing_avoids_double_hash_signature_ceiling() {
        const LABEL_COUNT: usize = 22_186;
        let config = BloomFilterConfig::new(256, 3).unwrap();
        let seeded = collision_stats(
            (0..LABEL_COUNT)
                .map(|index| seeded_signature(&format!("synthetic-label-{index}"), config)),
        );
        let double = collision_stats((0..LABEL_COUNT).map(|index| {
            double_hash_signature(
                format!("synthetic-label-{index}").as_bytes(),
                config.bit_count(),
            )
        }));

        eprintln!("seeded XXH3: {seeded:?}");
        eprintln!("double hash: {double:?}");
        assert_eq!(
            seeded,
            CollisionStats {
                unique_signatures: 22_084,
                collision_groups: 102,
                aliased_labels: 204,
                largest_group: 2,
            }
        );
        assert_eq!(
            double,
            CollisionStats {
                unique_signatures: 12_171,
                collision_groups: 6_452,
                aliased_labels: 16_467,
                largest_group: 9,
            }
        );
        assert!(seeded.unique_signatures > 22_000);
        assert!(seeded.largest_group <= 3);
        assert!(double.unique_signatures <= 16_384);
    }

    fn seeded_signature(label: &str, config: BloomFilterConfig) -> [u32; 3] {
        let mut rows = Vec::with_capacity(3);
        append_label_rows(label, config, &mut rows);
        let mut signature = [rows[0], rows[1], rows[2]];
        signature.sort_unstable();
        signature
    }

    fn double_hash_signature(bytes: &[u8], bit_count: u32) -> [u32; 3] {
        let hash = xxh3_128(bytes);
        let first_hash = hash as u64;
        let step_hash = ((hash >> 64) as u64) | 1;
        let bit_count = u64::from(bit_count);
        let mut signature = [0u32; 3];
        for hash_index in 0..3 {
            let mut row =
                first_hash.wrapping_add((hash_index as u64).wrapping_mul(step_hash)) % bit_count;
            while signature[..hash_index].contains(&(row as u32)) {
                row = (row + 1) % bit_count;
            }
            signature[hash_index] = row as u32;
        }
        signature.sort_unstable();
        signature
    }

    fn collision_stats(signatures: impl IntoIterator<Item = [u32; 3]>) -> CollisionStats {
        let mut groups = HashMap::new();
        for signature in signatures {
            *groups.entry(signature).or_insert(0usize) += 1;
        }
        CollisionStats {
            unique_signatures: groups.len(),
            collision_groups: groups.values().filter(|&&count| count > 1).count(),
            aliased_labels: groups.values().filter(|&&count| count > 1).copied().sum(),
            largest_group: groups.values().copied().max().unwrap_or(0),
        }
    }
}
