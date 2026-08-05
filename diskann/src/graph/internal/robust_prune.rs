/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Allocation-free RobustPrune selection over prepared candidates.
//!
//! Callers own source-distance sorting, candidate availability, workspace
//! allocation, ID translation, and saturation. This module owns only the
//! resumable alpha-round occlusion state machine shared by Vamana and PiPNN.

use thiserror::Error;

use crate::graph::config::PruneKind;

/// One available candidate prepared by a caller.
#[derive(Debug)]
pub(in crate::graph) struct Candidate<I, V> {
    id: I,
    source_distance: f32,
    value: V,
}

impl<I, V> Candidate<I, V> {
    pub(in crate::graph) fn new(id: I, source_distance: f32, value: V) -> Self {
        Self {
            id,
            source_distance,
            value,
        }
    }

    pub(in crate::graph) fn id(&self) -> &I {
        &self.id
    }
}

/// Per-candidate state reused across alpha rounds.
///
/// `states[i]` initially describes `candidates[i]`. Once a candidate is
/// selected, the prefix `states[..selected]` also stores selected candidate
/// positions in `candidate`. `last_checked` indexes that selected prefix.
#[derive(Debug, Clone, Copy, Default)]
pub(in crate::graph) struct State {
    occlude_factor: f32,
    last_checked: u16,
    candidate: u16,
}

impl State {
    pub(in crate::graph) fn selected_position(&self) -> usize {
        self.candidate as usize
    }
}

/// Structural or caller-distance failure from [`robust_prune`].
#[derive(Debug, Error)]
pub(in crate::graph) enum RobustPruneError<E = std::convert::Infallible> {
    #[error("robust prune supports at most {max} candidates, got {actual}")]
    TooManyCandidates { actual: usize, max: usize },
    #[error(
        "robust prune needs one state per candidate, got {states} states for {candidates} candidates"
    )]
    StateCountMismatch { states: usize, candidates: usize },
    #[error("distance computation failed: {0}")]
    Distance(E),
}

/// Validate the candidate-position representation before preparation/filtering.
pub(in crate::graph) fn validate_candidate_count<E>(
    candidate_count: usize,
) -> Result<(), RobustPruneError<E>> {
    if candidate_count > u16::MAX as usize {
        return Err(RobustPruneError::TooManyCandidates {
            actual: candidate_count,
            max: u16::MAX as usize,
        });
    }
    Ok(())
}

/// Select candidate positions with Vamana's resumable RobustPrune state machine.
///
/// `candidates` must already be sorted by source distance and contain only
/// available, non-excluded values. `states` must have exactly one element per
/// candidate. The function resets that slice, performs no allocation, and
/// returns the selected count. Selected positions are available through
/// `states[..selected]` and [`State::selected_position`].
pub(in crate::graph) fn robust_prune<I, V, E, D>(
    candidates: &[Candidate<I, V>],
    states: &mut [State],
    degree: usize,
    alpha: f32,
    prune_kind: PruneKind,
    mut distance: D,
) -> Result<usize, RobustPruneError<E>>
where
    D: FnMut(&V, &V) -> Result<f32, E>,
{
    validate_candidate_count(candidates.len())?;
    if states.len() != candidates.len() {
        return Err(RobustPruneError::StateCountMismatch {
            states: states.len(),
            candidates: candidates.len(),
        });
    }

    states.fill(State::default());
    if candidates.is_empty() {
        return Ok(0);
    }

    let mut current_alpha = 1.0f32;
    let increment_factor = alpha.min(1.2);
    let mut selected = 0;

    while selected < degree {
        for (index, candidate) in candidates.iter().enumerate() {
            if selected >= degree {
                break;
            }

            let State {
                mut occlude_factor,
                mut last_checked,
                ..
            } = states[index];
            if occlude_factor > current_alpha {
                continue;
            }

            while last_checked as usize != selected {
                let selected_position = states[last_checked as usize].selected_position();
                last_checked += 1;

                if selected_position >= index {
                    states[index].last_checked = last_checked;
                    continue;
                }

                let pair_distance =
                    distance(&candidate.value, &candidates[selected_position].value)
                        .map_err(RobustPruneError::Distance)?;
                occlude_factor = prune_kind.update_occlude_factor(
                    candidate.source_distance,
                    pair_distance,
                    occlude_factor,
                    current_alpha,
                );
                if occlude_factor > current_alpha {
                    break;
                }
            }

            let state = &mut states[index];
            state.last_checked = last_checked;
            if occlude_factor > current_alpha {
                state.occlude_factor = occlude_factor;
                continue;
            }

            state.occlude_factor = f32::MAX;
            states[selected].candidate = index as u16;
            selected += 1;
        }

        if current_alpha == alpha {
            break;
        }
        current_alpha = (current_alpha * increment_factor).min(alpha);
    }

    Ok(selected)
}

#[cfg(test)]
mod tests;
