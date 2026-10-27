/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Read, sync::Arc};

use diskann::{ANNError, ANNResult};
use diskann_providers::storage::{get_compressed_pq_file, get_pq_pivot_file, StorageReadProvider};
use diskann_quantization::{
    alloc::GlobalAllocator,
    spherical::iface::{self, Quantizer},
};
use diskann_utils::{
    io::{read_bin, Metadata},
    views::rowmajor::{self, Matrix},
};
use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};

use super::PQData;
use crate::{
    error::{diskann_error, ErrorKind},
    storage::disk_index_reader::DiskIndexReader,
};

/// In-memory distance approximation used while traversing the disk graph.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchQuantization {
    /// Product quantization using the existing PQ sidecar format.
    #[default]
    PQ,
    /// Three-bit spherical data with compensation metadata and full-precision queries.
    SphericalQ3,
}

impl SearchQuantization {
    /// Sidecar files required by this backend. The disk graph is shared by both backends.
    pub fn paths(self, prefix: &str) -> [String; 2] {
        match self {
            Self::PQ => [get_pq_pivot_file(prefix), get_compressed_pq_file(prefix)],
            Self::SphericalQ3 => [
                format!("{prefix}_spherical_q3_quantizer.bin"),
                format!("{prefix}_spherical_q3_compressed.bin"),
            ],
        }
    }
}

pub(crate) enum SearchDataInner {
    PQ(Arc<PQData>),
    SphericalQ3 {
        plan: Box<dyn Quantizer>,
        codes: rowmajor::Owned<u8>,
        source_dim: usize,
    },
}

/// Immutable quantizer and codes, shared across disk-search queries.
pub struct SearchData(pub(crate) SearchDataInner);

impl SearchData {
    /// Share an existing PQ codebook and compressed dataset without copying codes.
    pub fn from_pq(data: Arc<PQData>) -> Self {
        Self(SearchDataInner::PQ(data))
    }

    /// Load only the explicitly selected backend; never fall back to another format.
    ///
    /// # Errors
    /// Returns an error for missing, truncated, or incompatible sidecars.
    pub fn load<S: StorageReadProvider>(
        prefix: &str,
        quantization: SearchQuantization,
        storage: &S,
    ) -> ANNResult<Self> {
        let [quantizer_path, codes_path] = quantization.paths(prefix);
        match quantization {
            SearchQuantization::PQ => Ok(Self::from_pq(
                DiskIndexReader::new(quantizer_path, codes_path, storage)?.get_pq_data(),
            )),
            SearchQuantization::SphericalQ3 => {
                let mut serialized = Vec::new();
                storage
                    .open_reader(&quantizer_path)?
                    .read_to_end(&mut serialized)?;
                if serialized.len() < 16 || &serialized[..4] != b"DSQ3" {
                    return Err(ANNError::message(
                        "Invalid spherical search quantizer header",
                    ));
                }
                let source_dim = u32::from_le_bytes([
                    serialized[4],
                    serialized[5],
                    serialized[6],
                    serialized[7],
                ]) as usize;
                if source_dim == 0 {
                    return Err(ANNError::message(
                        "Spherical source dimension must be positive",
                    ));
                }
                let plan: Box<dyn Quantizer> = Box::new(
                    iface::Impl::<3>::try_deserialize(&serialized[8..], GlobalAllocator)
                        .map_err(ANNError::new)?,
                );
                let mut reader = storage.open_reader(&codes_path)?;
                let metadata = Metadata::read(&mut reader)?;
                let expected = (metadata.npoints() as u64)
                    .checked_mul(plan.bytes() as u64)
                    .and_then(|bytes| bytes.checked_add(8));
                if metadata.npoints() == 0
                    || metadata.ndims() != plan.bytes()
                    || expected != Some(storage.get_length(&codes_path)?)
                {
                    return Err(ANNError::message(
                        "Invalid spherical compressed data shape or length",
                    ));
                }
                let codes =
                    read_bin(&mut storage.open_reader(&codes_path)?).map_err(ANNError::new)?;
                Ok(Self(SearchDataInner::SphericalQ3 {
                    plan,
                    codes,
                    source_dim,
                }))
            }
        }
    }

    /// Report the backend actually loaded.
    pub fn quantization(&self) -> SearchQuantization {
        match &self.0 {
            SearchDataInner::PQ(_) => SearchQuantization::PQ,
            SearchDataInner::SphericalQ3 { .. } => SearchQuantization::SphericalQ3,
        }
    }

    /// Number of compressed records indexed by the disk graph's internal IDs.
    pub fn num_points(&self) -> usize {
        match &self.0 {
            SearchDataInner::PQ(data) => data.pq_compressed_data().nrows(),
            SearchDataInner::SphericalQ3 { codes, .. } => codes.nrows(),
        }
    }

    pub(crate) fn validate(&self, dim: usize, num_points: usize, metric: Metric) -> ANNResult<()> {
        if self.num_points() != num_points {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "Quantized data contains {} points, expected {num_points}",
                self.num_points()
            ));
        }
        if let SearchDataInner::SphericalQ3 {
            plan, source_dim, ..
        } = &self.0
        {
            if *source_dim != dim || Some(plan.metric()) != metric.try_into().ok() {
                return Err(ANNError::message(
                    "Spherical quantizer dimension or metric does not match disk graph",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn validate_full_dimension(&self, full_dim: usize) -> ANNResult<()> {
        if let SearchDataInner::SphericalQ3 { plan, .. } = &self.0 {
            if plan.full_dim() != full_dim {
                return Err(diskann_error!(
                    ErrorKind::DimensionMismatchError,
                    "Spherical quantizer input dimension {} does not match disk vector dimension {full_dim}",
                    plan.full_dim()
                ));
            }
        }
        Ok(())
    }
}
