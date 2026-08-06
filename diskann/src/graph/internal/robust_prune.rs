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

            // TODO: Track selection separately before defining non-finite-alpha
            // behavior. Once `current_alpha >= f32::MAX`, this sentinel can make
            // an already-selected candidate eligible again. This extraction keeps
            // the existing Vamana behavior unchanged.
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
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct DistanceFailure;

    fn candidate(id: u32, source_distance: f32) -> Candidate<u32, u32> {
        Candidate::new(id, source_distance, id)
    }

    fn selected_ids(
        candidates: &[Candidate<u32, u32>],
        states: &[State],
        selected: usize,
    ) -> Vec<u32> {
        states[..selected]
            .iter()
            .map(|state| *candidates[state.selected_position()].id())
            .collect()
    }

    #[test]
    fn propagates_distance_failure() {
        let candidates = [candidate(1, 1.0), candidate(2, 2.0)];
        let mut states = [State::default(); 2];

        let error = robust_prune(
            &candidates,
            &mut states,
            2,
            1.2,
            PruneKind::TriangleInequality,
            |_, _| Err(DistanceFailure),
        )
        .unwrap_err();

        assert!(matches!(error, RobustPruneError::Distance(DistanceFailure)));
    }

    #[test]
    fn accepts_u16_max_candidates_and_rejects_one_more() {
        let mut candidates = (0..u16::MAX as u32)
            .map(|id| candidate(id, id as f32))
            .collect::<Vec<_>>();
        let mut states = vec![State::default(); candidates.len()];
        assert_eq!(
            robust_prune(
                &candidates,
                &mut states,
                0,
                1.2,
                PruneKind::TriangleInequality,
                |_, _| Ok::<_, std::convert::Infallible>(0.0),
            )
            .unwrap(),
            0
        );

        candidates.push(candidate(u16::MAX as u32, u16::MAX as f32));
        states.push(State::default());
        assert!(matches!(
            robust_prune(
                &candidates,
                &mut states,
                1,
                1.2,
                PruneKind::TriangleInequality,
                |_, _| Ok::<_, std::convert::Infallible>(0.0),
            ),
            Err(RobustPruneError::TooManyCandidates { actual, max })
                if actual == u16::MAX as usize + 1 && max == u16::MAX as usize
        ));
    }

    #[test]
    fn rejects_state_count_mismatch() {
        let candidates = [candidate(1, 1.0), candidate(2, 2.0)];
        let mut states = [State::default(); 1];

        assert!(matches!(
            robust_prune(
                &candidates,
                &mut states,
                1,
                1.2,
                PruneKind::TriangleInequality,
                |_, _| Ok::<_, std::convert::Infallible>(0.0),
            ),
            Err(RobustPruneError::StateCountMismatch {
                states: 1,
                candidates: 2
            })
        ));
    }

    #[test]
    fn returns_selected_positions_in_candidate_order() {
        let candidates = [candidate(10, 1.0), candidate(20, 2.0), candidate(30, 3.0)];
        let mut states = [State::default(); 3];

        let selected = robust_prune(
            &candidates,
            &mut states,
            3,
            1.0,
            PruneKind::TriangleInequality,
            |_, _| Ok::<_, std::convert::Infallible>(0.0),
        )
        .unwrap();

        assert_eq!(selected_ids(&candidates, &states, selected), [10]);
    }

    #[test]
    fn revisits_occluded_candidates_at_larger_alpha() {
        let candidates = [candidate(10, 1.0), candidate(20, 1.1)];
        let mut states = [State::default(); 2];

        let selected = robust_prune(
            &candidates,
            &mut states,
            2,
            1.2,
            PruneKind::TriangleInequality,
            |_, _| Ok::<_, std::convert::Infallible>(1.0),
        )
        .unwrap();

        assert_eq!(selected_ids(&candidates, &states, selected), [10, 20]);
    }

    #[test]
    fn empty_candidates_select_nothing() {
        let mut states = [];
        assert_eq!(
            robust_prune(
                &[] as &[Candidate<u32, u32>],
                &mut states,
                usize::MAX,
                1.2,
                PruneKind::TriangleInequality,
                |_, _| Ok::<_, std::convert::Infallible>(0.0),
            )
            .unwrap(),
            0
        );
    }
}
