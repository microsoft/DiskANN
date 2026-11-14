/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Mini-batch e-means for squared Euclidean clustering.
//!
//! Implements the EMA updates, scaling laws, cosine schedule, cluster splitting and
//! last-epoch finalization described by [Bessa et al.](https://arxiv.org/abs/2611.09501).
//! Recipe constants follow the [Faiss implementation](https://github.com/facebookresearch/faiss/tree/65a4e3880ece1494c9d79e06f886c89810a1f85a/contrib/e_means).
//! Assignment reuses the SIMD kernel used by [super::kmeans::lloyds]; sums and masses use f64.
//! Runs are sequential and use only the caller's RNG.
//!
//! # Example
//!
//! ```rust
//! use diskann_quantization::{
//!     algorithms::emeans::{emeans, EmeansConfig},
//!     cancel::DontCancel,
//!     random::{RngBuilder, StdRngBuilder},
//! };
//! use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let data = rowmajor::Owned::from_fn(100, 4, |rc| (rc.row % 2 + rc.col) as f32);
//! let mut centers = rowmajor::Owned::from_element(2, 4, 0.0);
//! let mut rng = StdRngBuilder::new(1234).build_rng(0_u64);
//! let result = emeans(data.as_view(), centers.as_view_mut(),
//!     &EmeansConfig::default(), &mut rng, &DontCancel)?;
//! assert_eq!(result.assignments.len(), 100);
//! # Ok(())
//! # }
//! ```

mod train;

pub use train::{
    BatchSizePolicy, EmeansConfig, EmeansError, EmeansResult, emeans, emeans_from_centers,
};
