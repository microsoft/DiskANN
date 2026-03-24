/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Multi-attribute-diversity search post-processing.

use std::future::Future;

use diskann::{
    ANNError,
    graph::{SearchOutputBuffer, glue},
    neighbor::Neighbor,
    provider::BuildQueryComputer,
};

use super::postprocess::{AsDeletionCheck, DeletionCheck};

#[derive(Debug)]
pub enum MultiAttributeDiversityError {
    InvalidTopK { top_k: usize },
    InvalidEta { eta: f64 },
    InvalidPower { power: f64 },
}

impl std::fmt::Display for MultiAttributeDiversityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTopK { top_k } => write!(f, "top_k must be > 0, got {top_k}"),
            Self::InvalidEta { eta } => write!(f, "eta must be >= 0.0, got {eta}"),
            Self::InvalidPower { power } => write!(f, "power must be > 0.0, got {power}"),
        }
    }
}

impl std::error::Error for MultiAttributeDiversityError {}

#[derive(Debug, Clone, Copy)]
pub struct MultiAttributeDiversitySearchParams {
    pub top_k: usize,
    pub multi_attribute_diversity_eta: f64,
    pub multi_attribute_diversity_power: f64,
}

impl MultiAttributeDiversitySearchParams {
    pub fn new(
        top_k: usize,
        multi_attribute_diversity_eta: f64,
        multi_attribute_diversity_power: f64,
    ) -> Result<Self, MultiAttributeDiversityError> {
        if top_k == 0 {
            return Err(MultiAttributeDiversityError::InvalidTopK { top_k });
        }

        if multi_attribute_diversity_eta < 0.0 || !multi_attribute_diversity_eta.is_finite() {
            return Err(MultiAttributeDiversityError::InvalidEta {
                eta: multi_attribute_diversity_eta,
            });
        }

        if multi_attribute_diversity_power <= 0.0 || !multi_attribute_diversity_power.is_finite() {
            return Err(MultiAttributeDiversityError::InvalidPower {
                power: multi_attribute_diversity_power,
            });
        }

        Ok(Self {
            top_k,
            multi_attribute_diversity_eta,
            multi_attribute_diversity_power,
        })
    }
}

impl<A, T> glue::SearchPostProcess<A, [T]> for MultiAttributeDiversitySearchParams
where
    A: BuildQueryComputer<[T], Id = u32> + AsDeletionCheck,
{
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        _query: &[T],
        _computer: &<A as BuildQueryComputer<[T]>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<u32> + Send + ?Sized,
    {
        let checker = accessor.as_deletion_check();
        let mut values: Vec<_> = candidates
            .filter_map(|candidate| {
                if checker.deletion_check(candidate.id) {
                    None
                } else {
                    Some((candidate.id, candidate.distance))
                }
            })
            .collect();

        values.sort_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        values.truncate(self.top_k);

        std::future::ready(Ok(output.extend(values)))
    }
}
