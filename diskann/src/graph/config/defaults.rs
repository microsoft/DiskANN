/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroU32;

use super::IntraBatchCandidates;

/// A relatively large value to allow a good collection of candidates to be passed to
/// pruning. Since pruning has quadratic complexity in the number of candidates, a default
/// value of 750 provides plenty of room for new candidates without leading to excessive
/// prune times.
pub const MAX_OCCLUSION_SIZE: NonZeroU32 = NonZeroU32::new(750).unwrap();

/// "Adjacency list saturation" after pruning involves packing the adjcancy list until it
/// reaches exactly the configured target degree.
///
/// This can be useful in situations where the length of adjacency lists is fixed by the
/// backend data provider.
///
/// The `false` default is because this generally slows down index construction.
pub const SATURATE_AFTER_PRUNE: bool = false;

/// Allow the maximum graph degree to exceed the target degree by up to 1.3x. This provides
/// an empirically good balance between minimal backedge prunes and memory overhead.
pub const GRAPH_SLACK_FACTOR: f32 = 1.3;

/// The default occlusion factor for pruning. This is an empirically good value across a
/// range of datasets and metrics.
pub const ALPHA: f32 = 1.2;

/// Conservatively defaults to sequential execution.
pub const MAX_MINIBATCH_PARALLELISM: NonZeroU32 = NonZeroU32::new(1).unwrap();

/// Conservatively consider all candidates within a batch.
pub const INTRA_BATCH_CANDIDATES: IntraBatchCandidates = IntraBatchCandidates::All;
