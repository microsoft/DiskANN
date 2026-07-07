// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Projected-eigen distance type for multi-vector representations.

/////////////////////
// ProjectedEigen //
/////////////////////

/// Projected-eigen distance for multi-vector similarity.
///
/// Computes the negated sum of squared inner products over *all*
/// query/document vector pairs:
///
/// ```text
/// ProjectedEigen(Q, D) = \sum_{i} \sum_{j} -IP(q_i, d_j)²
/// ```
///
/// Unlike [`Chamfer`](super::Chamfer), which keeps only the best-matching
/// document vector per query vector, this accumulates a contribution from
/// every pair, so the score reflects the full query–document interaction. As
/// with the other multi-vector distances, lower is better.
///
/// Implements [`PureDistanceFunction`](diskann_vector::PureDistanceFunction)
/// for matrix view types.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedEigen;
