/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

use crate::{ANNError, ANNErrorKind, error::ensure_positive};

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

// Parameters for the search algorithm
#[derive(Copy, Clone, Debug)]
pub struct SearchParams {
    pub k_value: usize,
    pub l_value: usize,
    pub beam_width: Option<usize>,
}

#[derive(Debug, Error)]
pub enum SearchParamsError {
    #[error("l_value ({l_value}) cannot be less than k_value ({k_value})")]
    LLessThanK { l_value: usize, k_value: usize },
    #[error("beam width cannot be zero")]
    BeamWidthZero,
    #[error("l_value cannot be zero")]
    LZero,
    #[error("k_value cannot be zero")]
    KZero,
}

impl From<SearchParamsError> for ANNError {
    fn from(err: SearchParamsError) -> Self {
        Self::new(ANNErrorKind::IndexError, err)
    }
}

impl SearchParams {
    pub fn new(
        k_value: usize,
        l_value: usize,
        beam_width: Option<usize>,
    ) -> Result<Self, SearchParamsError> {
        if k_value > l_value {
            return Err(SearchParamsError::LLessThanK { l_value, k_value });
        }
        if let Some(beam_width) = beam_width {
            ensure_positive(beam_width, SearchParamsError::BeamWidthZero)?;
        }
        ensure_positive(k_value, SearchParamsError::KZero)?;
        ensure_positive(l_value, SearchParamsError::LZero)?;

        Ok(Self {
            k_value,
            l_value,
            beam_width,
        })
    }

    pub fn new_default(k_value: usize, l_value: usize) -> Result<Self, SearchParamsError> {
        SearchParams::new(k_value, l_value, None)
    }
}

// Parameters for the search algorithm
#[derive(Copy, Clone, Debug)]
pub struct RangeSearchParams {
    pub max_returned: Option<usize>,
    pub starting_l_value: usize,
    pub beam_width: Option<usize>,
    pub radius: f32,
    pub inner_radius: Option<f32>,
    pub initial_search_slack: f32,
    pub range_search_slack: f32,
}

#[derive(Debug, Error)]
pub enum RangeSearchParamsError {
    #[error("beam width cannot be zero")]
    BeamWidthZero,
    #[error("l_value cannot be zero")]
    LZero,
    #[error("initial_search_slack must be between 0 and 1.0")]
    StartingListSlackValueError,
    #[error("range_search_slack must be greater than or equal to 1.0")]
    RangeSearchSlackValueError,
    #[error("inner_radius must be less than or equal to radius")]
    InnerRadiusValueError,
}

impl From<RangeSearchParamsError> for ANNError {
    fn from(err: RangeSearchParamsError) -> Self {
        Self::new(ANNErrorKind::IndexError, err)
    }
}

impl RangeSearchParams {
    pub fn new(
        max_returned: Option<usize>,
        starting_l_value: usize,
        beam_width: Option<usize>,
        radius: f32,
        inner_radius: Option<f32>,
        initial_search_slack: f32,
        range_search_slack: f32,
    ) -> Result<Self, RangeSearchParamsError> {
        // note that radius is allowed to be negative due to inner product metrics
        if let Some(beam_width) = beam_width {
            ensure_positive(beam_width, RangeSearchParamsError::BeamWidthZero)?;
        }
        ensure_positive(starting_l_value, RangeSearchParamsError::LZero)?;
        if !(0.0..=1.0).contains(&initial_search_slack) {
            return Err(RangeSearchParamsError::StartingListSlackValueError);
        }
        if range_search_slack < 1.0 {
            return Err(RangeSearchParamsError::RangeSearchSlackValueError);
        }
        if let Some(inner_radius) = inner_radius
            && inner_radius > radius
        {
            return Err(RangeSearchParamsError::InnerRadiusValueError);
        }

        Ok(Self {
            max_returned,
            starting_l_value,
            beam_width,
            radius,
            inner_radius,
            initial_search_slack,
            range_search_slack,
        })
    }

    pub fn new_default(
        starting_l_value: usize,
        radius: f32,
    ) -> Result<Self, RangeSearchParamsError> {
        RangeSearchParams::new(None, starting_l_value, None, radius, None, 1.0, 1.0)
    }

    pub fn l_value(&self) -> usize {
        self.starting_l_value
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
    pub diverse_results_k: usize,
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
        attribute_provider: std::sync::Arc<P>,
    ) -> Self {
        Self {
            diverse_attribute_id,
            diverse_results_k,
            attribute_provider,
        }
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

    #[test]
    fn test_range_search_params_error_cases() {
        {
            // test starting list slack factor error
            let res = RangeSearchParams::new(
                None, // max returned
                10,   // starting l value
                None, // beam width
                1.0,  // radius
                None, // inner radius
                1.1,  // initial search slack
                1.0,  // range search slack
            );
            assert!(res.is_err());
            assert_eq!(
                res.unwrap_err().to_string(),
                "initial_search_slack must be between 0 and 1.0"
            );
        }
        {
            // test range search slack factor error
            let res = RangeSearchParams::new(
                None, // max returned
                10,   // starting l value
                None, // beam width
                1.0,  // radius
                None, // inner radius
                1.0,  // initial search slack
                0.9,  // range search slack
            );
            assert!(res.is_err());
            assert_eq!(
                res.unwrap_err().to_string(),
                "range_search_slack must be greater than or equal to 1.0"
            );
        }
        {
            // test inner radius error
            let res = RangeSearchParams::new(
                None,      // max returned
                10,        // starting l value
                None,      // beam width
                1.0,       // radius
                Some(2.0), // inner radius
                1.0,       // initial search slack
                1.0,       // range search slack
            );
            assert!(res.is_err());
            assert_eq!(
                res.unwrap_err().to_string(),
                "inner_radius must be less than or equal to radius"
            );
        }
    }

    #[test]
    fn test_range_search_params_impl() {
        let res = RangeSearchParams::new(
            None, // max returned
            10,   // starting l value
            None, // beam width
            1.0,  // radius
            None, // inner radius
            1.0,  // initial search slack
            1.0,  // range search slack
        )
        .unwrap();

        assert_eq!(res.l_value(), 10);
    }
}
