/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;
use thiserror::Error;
use crate::{ANNError, ANNErrorKind};

// enum used to return the status of the vector that `consolidate_vector`
// was called on: Deleted if the vector was already deleted, and Complete
// if the vector was not deleted (and thus is now consolidated)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsolidateKind {
    /// Consolidate was called on a deleted vector.
    Deleted,

    /// Consolidate was called on valid vector, but retrieving the data for that vector
    /// failed with a transient error.
    FailedVectorRetrieval,

    /// Consolidate completed successfully.
    Complete,
}

// enum used to encode the algorithmic choices for inplace delete
// the first term indicates what is used to approximate the in-neighbors
// the second term indicates what is used to approximate the replace
// candidates
// also includes any params specific to that choice
#[derive(Copy, Clone, Debug)]
pub enum InplaceDeleteMethod {
    VisitedAndTopK { k_value: usize, l_value: usize },
    TwoHopAndOneHop,
    OneHop,
}

/// Error type for [`DiverseSearchParams`] parameter validation.
#[cfg(feature = "experimental_diversity_search")]
#[derive(Debug, Error)]
pub enum DiverseSearchError {
    #[error("original k_value cannot be zero")]
    OriginalKZero,
    #[error("diverse k_value cannot be zero")]
    DiverseKZero,
}

#[cfg(feature = "experimental_diversity_search")]
impl From<DiverseSearchError> for ANNError {
    #[track_caller]
    fn from(err: DiverseSearchError) -> Self {
        Self::new(ANNErrorKind::IndexError, err)
    }
}

// Parameters for diverse search
#[cfg(feature = "experimental_diversity_search")]
#[derive(Clone, Debug)]
pub struct DiverseSearchParams<P>
where
    P: crate::neighbor::AttributeValueProvider,
{
    pub diverse_attribute_id: usize,
    pub diverse_results_k: NonZeroUsize,
    pub original_k_value: NonZeroUsize,
    pub attribute_provider: std::sync::Arc<P>,
}

#[cfg(feature = "experimental_diversity_search")]
impl<P> DiverseSearchParams<P>
where
    P: crate::neighbor::AttributeValueProvider,
{
    pub fn new(
        diverse_attribute_id: usize,
        diverse_results_k: usize,
        original_k_value: usize,
        attribute_provider: std::sync::Arc<P>,
    ) -> Result<Self, DiverseSearchError> {
        let diverse_results_k = NonZeroUsize::new(diverse_results_k).ok_or(DiverseSearchError::DiverseKZero)?;
        let original_k_value = NonZeroUsize::new(original_k_value).ok_or(DiverseSearchError::OriginalKZero)?;

        Ok(Self {
            diverse_attribute_id,
            diverse_results_k,
            original_k_value,
            attribute_provider,
        })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consolidate_enum() {
        // test the already deleted variant
        let delete_res_already_deleted = ConsolidateKind::Deleted;
        match delete_res_already_deleted {
            ConsolidateKind::Deleted => {}
            _ => panic!("Expected already deleted variant"),
        }

        // test the not deleted variant
        let delete_res_not_deleted = ConsolidateKind::Complete;
        match delete_res_not_deleted {
            ConsolidateKind::Complete => {}
            _ => panic!("Expected not deleted variant"),
        }
    }
}
