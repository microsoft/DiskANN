/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Properties of the input to a single training or compression operation.
///
/// Defaults make no assumptions about normalization or distribution. These hints are not
/// part of the quantizer's serialized state and do not change its distance metric.
#[derive(Debug, Default, Clone, Copy)]
pub struct InputHints {
    normalized: bool,
    maybe_out_of_distribution: bool,
}

impl InputHints {
    /// Make no assumptions about the input.
    pub const fn new() -> Self {
        Self {
            normalized: false,
            maybe_out_of_distribution: false,
        }
    }

    /// Declare that the input has unit L2 norm (zero vectors may remain zero).
    ///
    /// Cosine can skip input normalization. Incorrectly declaring non-unit inputs can
    /// produce incorrect distances. Converting integer values to floats does not normalize
    /// them. Training hints describe every row; compression hints describe only that input.
    pub const fn with_normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    /// Allow an inner-product query to be rescaled to the training distribution's mean norm.
    ///
    /// This preserves ideal rankings for a fixed query, but changes the distance scale and
    /// may change quantization error. Training and stored-data compression ignore this hint:
    /// independently rescaling database vectors would change inner-product rankings.
    pub const fn with_maybe_out_of_distribution(mut self, value: bool) -> Self {
        self.maybe_out_of_distribution = value;
        self
    }

    pub(super) const fn normalized(self) -> bool {
        self.normalized
    }
    pub(super) const fn maybe_out_of_distribution(self) -> bool {
        self.maybe_out_of_distribution
    }
}
