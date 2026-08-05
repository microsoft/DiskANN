/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;

#[derive(Debug, PartialEq)]
struct DistanceFailure;

fn candidate(id: u32, source_distance: f32) -> Candidate<u32, u32> {
    Candidate::new(id, source_distance, id)
}

fn selected_ids(candidates: &[Candidate<u32, u32>], states: &[State], selected: usize) -> Vec<u32> {
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
        Policy::new(2, 1.2, PruneKind::TriangleInequality),
        |_, _| Err(DistanceFailure),
    )
    .unwrap_err();

    assert!(matches!(error, RobustPruneError::Distance(DistanceFailure)));
}

#[test]
fn rejects_invalid_alpha() {
    for alpha in [f32::NAN, f32::INFINITY, 0.999] {
        let mut states = [];
        let error = robust_prune(
            &[] as &[Candidate<u32, u32>],
            &mut states,
            Policy::new(1, alpha, PruneKind::TriangleInequality),
            |_, _| Ok::<_, std::convert::Infallible>(0.0),
        )
        .unwrap_err();
        assert!(
            matches!(error, RobustPruneError::InvalidAlpha(value) if value.to_bits() == alpha.to_bits())
        );
    }

    let mut states = [];
    assert_eq!(
        robust_prune(
            &[] as &[Candidate<u32, u32>],
            &mut states,
            Policy::new(1, 1.0, PruneKind::TriangleInequality),
            |_, _| Ok::<_, std::convert::Infallible>(0.0),
        )
        .unwrap(),
        0
    );
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
            Policy::new(0, 1.2, PruneKind::TriangleInequality),
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
            Policy::new(1, 1.2, PruneKind::TriangleInequality),
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
            Policy::new(1, 1.2, PruneKind::TriangleInequality),
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
        Policy::new(3, 1.0, PruneKind::TriangleInequality),
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
        Policy::new(2, 1.2, PruneKind::TriangleInequality),
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
            Policy::new(usize::MAX, 1.2, PruneKind::TriangleInequality),
            |_, _| Ok::<_, std::convert::Infallible>(0.0),
        )
        .unwrap(),
        0
    );
}
