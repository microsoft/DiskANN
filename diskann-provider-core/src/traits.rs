/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// A helper trait to select the delete provider.
///
/// This is also implemented for [`NoDeletes`], which explicitly disables deletion
/// related functionality.
pub trait CreateDeleteProvider {
    /// The type of the created delete provider.
    type Target;

    /// Create a delete provider capable of tracking `total_points` number of deletes
    /// (or disabling deletion check all together).
    ///
    /// NOTE: The value `total_points` consists of the sum of `max_points` and
    /// `frozen_points`.
    fn create(self, total_points: usize) -> Self::Target;
}
