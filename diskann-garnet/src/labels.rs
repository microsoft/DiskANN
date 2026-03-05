/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Garnet label filtering for DiskANN vector search.
//!
//! This module provides [`GarnetQueryLabelProvider`], an implementation of DiskANN's
//! [`QueryLabelProvider`] trait that checks vector IDs against a pre-computed
//! bitmap received from Garnet's C# `FilterBitmap`.
//!
//! ## Design
//!
//! Garnet sends a **pre-computed dense bitmap** where each bit position
//! corresponds to a DiskANN internal ID (`u32`). This makes `is_match()`
//! a simple bit lookup — no cross-FFI callback is needed.
//!
//! ## Bitmap Layout
//!
//! The bitmap is a `&[u8]` interpreted as little-endian bytes:
//! - Bit position `id` maps to byte `id / 8`, bit `id % 8`
//! - A set bit means the vector at that internal ID passes the filter
//! - This matches C#'s `FilterBitmap` (`ulong[]`) byte layout via `GetBytes()`
//! - TODO Replace with roaring bitmap for compression

use diskann::graph::index::QueryLabelProvider;

/// A zero-copy bitmap-based label provider for Garnet filtered vector search.
///
/// Holds a raw pointer to the bitmap data owned by the FFI caller.
/// No allocation — bit lookups are performed directly on the caller's memory.
///
/// # Safety
///
/// The bitmap pointer must remain valid and unmodified for the lifetime of
/// this struct. In practice, the struct is created and dropped within a
/// single FFI search call.
pub struct GarnetQueryLabelProvider {
    data: *const u8,
    len: usize,
}

// Safety: the bitmap data is read-only and the caller guarantees the pointer
// is valid for the duration of the search call.
unsafe impl Send for GarnetQueryLabelProvider {}
unsafe impl Sync for GarnetQueryLabelProvider {}

impl std::fmt::Debug for GarnetQueryLabelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GarnetQueryLabelProvider")
            .field("len", &self.len)
            .finish()
    }
}

impl Clone for GarnetQueryLabelProvider {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            len: self.len,
        }
    }
}

impl GarnetQueryLabelProvider {
    /// Construct a `GarnetQueryLabelProvider` from raw bitmap bytes.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw bytes from C#'s `FilterBitmap.GetBytes()`.
    /// * `len` - Length of the data in bytes.
    ///
    /// # Safety
    ///
    /// The caller must ensure `data` is valid for reads of `len` bytes
    /// and that the memory remains valid and unmodified for the lifetime
    /// of this struct.
    pub unsafe fn from_raw(data: *const u8, len: usize) -> Self {
        if data.is_null() || len == 0 {
            return Self {
                data: std::ptr::null(),
                len: 0,
            };
        }
        Self { data, len }
    }

    /// Construct a `GarnetQueryLabelProvider` from a byte slice.
    #[allow(dead_code)]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            Self {
                data: std::ptr::null(),
                len: 0,
            }
        } else {
            Self {
                data: bytes.as_ptr(),
                len: bytes.len(),
            }
        }
    }

    /// Check if the given internal ID has its bit set in the bitmap.
    /// Empty bitmap (len == 0) matches all IDs (no filter).
    /// If the bitmap is smaller than the ID being checked, it's considered a partial bitmap and out-of-range IDs are treated as matching (not filtered out)
    #[inline(always)]
    fn is_set(&self, id: u32) -> bool {
        if self.len == 0 {
            return true;
        }
        let byte_idx = (id / 8) as usize;
        if byte_idx >= self.len {
            return true;
        }
        let bit_idx = id % 8;
        let byte = unsafe { *self.data.add(byte_idx) };
        (byte >> bit_idx) & 1 == 1
    }
}

impl QueryLabelProvider<u32> for GarnetQueryLabelProvider {
    /// Check if the vector at `internal_id` passes the filter.
    ///
    /// Returns `true` if the corresponding bit is set in the bitmap,
    /// meaning the vector's attributes match the query filter predicate.
    #[inline(always)]
    fn is_match(&self, internal_id: u32) -> bool {
        self.is_set(internal_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_query_label_provider_match_all() {
        let query_label_provider = GarnetQueryLabelProvider::from_bytes(&[]);
        assert!(query_label_provider.is_match(0));
        assert!(query_label_provider.is_match(100));
    }

    #[test]
    fn single_byte_bitmap() {
        // Byte 0b00000101 = bits 0 and 2 set
        let query_label_provider = GarnetQueryLabelProvider::from_bytes(&[0b00000101]);
        assert!(query_label_provider.is_match(0));
        assert!(!query_label_provider.is_match(1));
        assert!(query_label_provider.is_match(2));
        assert!(!query_label_provider.is_match(3));
        assert!(query_label_provider.is_match(64)); // out of range, partial bitmap matches
    }

    #[test]
    fn full_u64_word() {
        // First word: all bits set (0..63 match)
        let bytes = 0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes();
        let query_label_provider = GarnetQueryLabelProvider::from_bytes(&bytes);
        for i in 0..64u32 {
            assert!(query_label_provider.is_match(i), "bit {i} should be set");
        }
        assert!(query_label_provider.is_match(64)); // partial bitmap, matches
    }

    #[test]
    fn multi_word_bitmap() {
        // Two words: first all zeros, second has bit 0 set (= global bit 64)
        let mut bytes = vec![0u8; 16];
        bytes[8] = 0x01; // bit 64
        let query_label_provider = GarnetQueryLabelProvider::from_bytes(&bytes);
        assert!(!query_label_provider.is_match(0));
        assert!(!query_label_provider.is_match(63));
        assert!(query_label_provider.is_match(64));
        assert!(!query_label_provider.is_match(65));
    }

    #[test]
    fn partial_trailing_bytes() {
        // 10 bytes = 1 full word + 2 trailing bytes
        let mut bytes = vec![0u8; 10];
        bytes[9] = 0x80; // bit 79 (word 1, byte 1, bit 7)
        let query_label_provider = GarnetQueryLabelProvider::from_bytes(&bytes);
        assert!(!query_label_provider.is_match(0));
        assert!(query_label_provider.is_match(79));
        assert!(!query_label_provider.is_match(78));
    }

    #[test]
    fn matches_csharp_filterbmap_layout() {
        // Simulate C# FilterBitmap with IDs {1, 5, 64, 100} set
        // Word 0: bits 1 and 5 → 0b00100010 = 0x22
        // Word 1: bit 0 (=64) and bit 36 (=100) → (1 << 0) | (1 << 36)
        let word0: u64 = (1 << 1) | (1 << 5);
        let word1: u64 = (1 << 0) | (1 << 36);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&word0.to_le_bytes());
        bytes.extend_from_slice(&word1.to_le_bytes());

        let query_label_provider = GarnetQueryLabelProvider::from_bytes(&bytes);
        assert!(!query_label_provider.is_match(0));
        assert!(query_label_provider.is_match(1));
        assert!(query_label_provider.is_match(5));
        assert!(!query_label_provider.is_match(6));
        assert!(query_label_provider.is_match(64));
        assert!(!query_label_provider.is_match(99));
        assert!(query_label_provider.is_match(100));
        assert!(!query_label_provider.is_match(101));
    }

    #[test]
    fn from_raw_null_pointer() {
        let query_label_provider =
            unsafe { GarnetQueryLabelProvider::from_raw(std::ptr::null(), 0) };
        assert!(query_label_provider.is_match(0));
    }

    #[test]
    fn from_raw_valid_data() {
        let data: [u8; 8] = [0b00000001, 0, 0, 0, 0, 0, 0, 0]; // bit 0 set
        let query_label_provider =
            unsafe { GarnetQueryLabelProvider::from_raw(data.as_ptr(), data.len()) };
        assert!(query_label_provider.is_match(0));
        assert!(!query_label_provider.is_match(1));
    }

    #[test]
    fn out_of_bounds_partial_bitmap_matches() {
        let query_label_provider = GarnetQueryLabelProvider::from_bytes(&[0xFF]); // bits 0-7 set
        assert!(query_label_provider.is_match(7));
        assert!(query_label_provider.is_match(64)); // partial bitmap, matches
        assert!(query_label_provider.is_match(u32::MAX)); // partial bitmap, matches
    }
}
