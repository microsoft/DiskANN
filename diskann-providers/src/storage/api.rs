/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{future::Future, num::NonZeroUsize};

use super::{StorageReadProvider, StorageWriteProvider};
use diskann_vector::distance::Metric;

use super::get_mem_index_data_file;
use crate::model::graph::provider::async_::PrefetchCacheLineLevel;

/// A general trait for saving `self` to disk.
///
/// The generic parameter `T` is used to allow types to require arbitrary associated
/// metadata in order to successfully save themselves.
///
/// Additionally, types may overload the auxiliary state to customize semantics.
///
/// See also: [`LoadWith`].
pub trait SaveWith<T> {
    /// The return type upon successful saving. This is often (but is not required to be)
    /// the number of bytes written to disk.
    type Ok: Send;

    /// The error type if serialization is unsuccessful.
    type Error: std::error::Error + Send;

    /// Safe `self` to disk using `provider` for IO-related needs. Argument `auxiliary`
    /// can be arbitrary metadata required for a successful operation, such as file paths.
    fn save_with<P>(
        &self,
        provider: &P,
        auxiliary: &T,
    ) -> impl Future<Output = Result<Self::Ok, Self::Error>> + Send
    where
        P: StorageWriteProvider;
}

/// A general trait for loading `self` from disk.
///
/// The generic parameter `T` is used to allow types to require arbitrary associated
/// metadata in order to successfully load themselves.
///
/// Additionally, types may overload the auxiliary state to customize semantics.
///
/// See also: [`SaveWith`].
pub trait LoadWith<T>: Sized {
    /// The error type if deserialization is unsuccessful.
    type Error: std::error::Error + Send;

    /// Load `self` form disk using `provider` for IO-related needs. Argument `auxiliary`
    /// can be arbitrary metadata required for a successful operation, such as file paths.
    fn load_with<P>(
        provider: &P,
        auxiliary: &T,
    ) -> impl Future<Output = Result<Self, Self::Error>> + Send
    where
        P: StorageReadProvider;
}

/// The file-path prefix for saving and loading an async index.
///
/// An auxiliary type for [`SaveWith`] and [`LoadWith`] to indicate that the object being
/// saved or loaded is part of an async in-memory index.
///
/// This mainly controls how file-paths are generated.
///
/// For example, graph data is located at the raw-prefix, while the full-precision data
/// is saved using the `.data` suffix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsyncIndexMetadata {
    prefix: String,
}

impl AsyncIndexMetadata {
    /// Construct a new `AsyncIndexPrefix` from `pathlike`.
    pub fn new<T>(pathlike: T) -> Self
    where
        String: From<T>,
    {
        Self {
            prefix: pathlike.into(),
        }
    }

    /// Return the file path contained in `self` as a `&str`.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Obtain the file path for full-precision data using `self` as the file path prefix.
    pub fn data_path(&self) -> String {
        get_mem_index_data_file(&self.prefix)
    }

    /// Obtain the file path for additional points file
    pub fn additional_points_id_path(&self) -> String {
        format!("{}.additional_points_id", self.prefix)
    }
}

/// The file-path prefix for saving only graph data during disk index construction.
///
/// An auxiliary type for [`SaveWith`] to specify graph-only serialization.
#[derive(Debug, Clone)]
pub struct DiskGraphOnly {
    prefix: String,
}

impl DiskGraphOnly {
    /// Constructs a new `DiskGraphOnly` from a path-like object.
    pub fn new<T>(pathlike: T) -> Self
    where
        String: From<T>,
    {
        Self {
            prefix: pathlike.into(),
        }
    }

    /// Return the file path contained in `self` as a `&str`.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

/// Indicates that the canonical layout for a Quant index is expected for deserialization.
///
/// For a file-path prefix `prefix`, this layout includes the following files:
///
/// **Common:**
/// * `prefix`: The file that contains the saved graph.
/// * `prefix.data`: The serialized full-precision data in `.bin` form.
///
/// Depending on the quantization method used, one of the following sets of files will also be present:
///
/// **If Product Quantization (PQ) is used:**
/// * `prefix_build_pq_pivots.bin`: The saved PQ pivot table.
/// * `prefix_build_pq_compressed.bin`: The saved PQ codes.
///
/// **If Scalar Quantization (SQ) is used:**
/// * `prefix_sq_compressed.bin`: The saved scalar quantized codes.
/// * `prefix_scalar_quantizer_proto.bin`: The saved scalar quantizer metadata.
pub struct AsyncQuantLoadContext {
    /// The file path prefix of the index.
    pub metadata: AsyncIndexMetadata,
    /// The number of frozen points stored in the datasets.
    pub num_frozen_points: NonZeroUsize,
    /// The metric to use for this index.
    pub metric: Metric,
    /// The number of iterations to prefetch when performing bulk-retrievals.
    pub prefetch_lookahead: Option<usize>,
    /// Temporary parameter to indicate if the index is a disk index to load right file names.
    /// This can be removed once disk index uses same file names as async index.
    pub is_disk_index: bool,
    /// controls the prefetch cache line level for the index.
    pub prefetch_cache_line_level: Option<PrefetchCacheLineLevel>,
}
