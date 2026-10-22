/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Configuration and deterministic row hashing for transposed Bloom label indexes.

use crate::{error::EncodedLabelIndexError, format::MAX_LABEL_COUNT};

const MAX_BLOOM_BITS: u32 = 65_536;
const MAX_BLOOM_HASHES: u32 = 64;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
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
    for hash_index in 0..config.hash_count {
        let mut row = stable_hash(label.as_bytes(), hash_index) % u64::from(config.bit_count);
        while rows[start..].contains(&(row as u32)) {
            row = (row + 1) % u64::from(config.bit_count);
        }
        rows.push(row as u32);
    }
}

fn stable_hash(bytes: &[u8], hash_index: u32) -> u64 {
    let mut hash = FNV_OFFSET_BASIS ^ u64::from(hash_index).wrapping_mul(HASH_SEED_STEP);
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash ^= hash >> 32;
    hash = hash.wrapping_mul(0xd6e8_feb8_6659_fd93);
    hash ^ (hash >> 32)
}
