/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;
use crate::{
    error::{RankedError, ToRanked, TransientError},
    neighbor::Neighbor,
};

#[derive(Debug, PartialEq)]
struct DistanceFailure;

fn run<I, V, E>(
    scratch: &mut Scratch<I>,
    policy: Policy,
    lookup: impl FnMut(I) -> Option<V>,
    distance: impl FnMut(&V, &V) -> Result<f32, E>,
) -> Result<(), RobustPruneError<E>>
where
    I: VectorId,
{
    let max_candidates = scratch.candidates_mut().len();
    let mut context = scratch.as_context(max_candidates);
    robust_prune(
        &mut context,
        policy,
        &mut Vec::new(),
        lookup,
        distance,
        |_| false,
    )
}

#[test]
fn propagates_distance_failure() {
    let mut scratch = Scratch::new();
    scratch
        .candidates_mut()
        .extend([Neighbor::new(1_u32, 1.0), Neighbor::new(2_u32, 2.0)]);

    let error = run(
        &mut scratch,
        Policy::new(2, 1.2, PruneKind::TriangleInequality, false),
        Some,
        |_, _| Err(DistanceFailure),
    )
    .unwrap_err();

    assert!(matches!(error, RobustPruneError::Distance(DistanceFailure)));
}

#[test]
fn rejects_invalid_alpha() {
    for alpha in [f32::NAN, f32::INFINITY, 0.999] {
        let mut scratch = Scratch::<u32>::new();
        let error = run(
            &mut scratch,
            Policy::new(1, alpha, PruneKind::TriangleInequality, false),
            Some,
            |_, _| Ok::<_, std::convert::Infallible>(0.0),
        )
        .unwrap_err();
        assert!(
            matches!(error, RobustPruneError::InvalidAlpha(value) if value.to_bits() == alpha.to_bits())
        );
    }

    let mut scratch = Scratch::<u32>::new();
    run(
        &mut scratch,
        Policy::new(1, 1.0, PruneKind::TriangleInequality, false),
        Some,
        |_, _| Ok::<_, std::convert::Infallible>(0.0),
    )
    .unwrap();
}

#[test]
fn rejects_candidate_count_above_u16() {
    let mut scratch = Scratch::new();
    scratch
        .candidates_mut()
        .extend((0..u16::MAX as u32).map(|id| Neighbor::new(id, id as f32)));
    run(
        &mut scratch,
        Policy::new(0, 1.2, PruneKind::TriangleInequality, false),
        Some,
        |_, _| Ok::<_, std::convert::Infallible>(0.0),
    )
    .unwrap();

    scratch
        .candidates_mut()
        .push(Neighbor::new(u16::MAX as u32, u16::MAX as f32));
    let error = run(
        &mut scratch,
        Policy::new(1, 1.2, PruneKind::TriangleInequality, false),
        Some,
        |_, _| Ok::<_, std::convert::Infallible>(0.0),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        RobustPruneError::TooManyCandidates {
            actual,
            max
        } if actual == u16::MAX as usize + 1 && max == u16::MAX as usize
    ));
}

#[test]
fn excludes_candidates_before_lookup() {
    let mut scratch = Scratch::default();
    scratch
        .candidates_mut()
        .extend([Neighbor::new(1_u32, 1.0), Neighbor::new(2_u32, 2.0)]);
    let mut context = scratch.as_context(2);
    robust_prune(
        &mut context,
        Policy::new(1, 1.2, PruneKind::TriangleInequality, false),
        &mut Vec::new(),
        Some,
        |left, right| Ok::<_, std::convert::Infallible>(left.abs_diff(*right) as f32),
        |id| id == 1,
    )
    .unwrap();

    assert_eq!(&**scratch.neighbors(), &[2]);
}

#[test]
fn preserves_transient_and_fatal_error_rank() {
    let transient = FailedVectorRetrieval(7_u32);
    let escalated = transient.escalate("test escalation");
    assert!(escalated.to_string().contains("test escalation"));

    assert!(matches!(
        ListError::failed_retrieval(8_u32).to_ranked(),
        RankedError::Transient(FailedVectorRetrieval(8))
    ));
    assert!(matches!(
        <ListError<u32> as ToRanked>::from_transient(FailedVectorRetrieval(9)),
        ListError::FailedVectorRetrieval(FailedVectorRetrieval(9))
    ));

    let fatal = ANNError::new(
        ANNErrorKind::IndexError,
        std::io::Error::other("fatal prune test error"),
    );
    assert!(matches!(ListError::<u32>::from(fatal), ListError::Other(_)));
    let fatal = ANNError::new(
        ANNErrorKind::IndexError,
        std::io::Error::other("fatal prune test error"),
    );
    assert!(matches!(
        <ListError<u32> as ToRanked>::from_error(fatal).to_ranked(),
        RankedError::Error(_)
    ));
}

#[test]
fn saturation_fills_from_original_pool_order_after_occlusion() {
    let candidates = [
        Neighbor::new(1_u32, 1.0),
        Neighbor::new(2_u32, 2.0),
        Neighbor::new(3_u32, 3.0),
    ];

    let mut unsaturated = Scratch::new();
    unsaturated.candidates_mut().extend(candidates);
    run(
        &mut unsaturated,
        Policy::new(3, 1.0, PruneKind::TriangleInequality, false),
        Some,
        |_, _| Ok::<_, std::convert::Infallible>(0.0),
    )
    .unwrap();
    // A zero selected-to-candidate distance occludes every candidate after
    // the first one, so this verifies the prune result before saturation.
    assert_eq!(&**unsaturated.neighbors(), &[1]);

    let mut saturated = Scratch::new();
    saturated.candidates_mut().extend(candidates);
    run(
        &mut saturated,
        Policy::new(3, 1.0, PruneKind::TriangleInequality, true),
        Some,
        |_, _| Ok::<_, std::convert::Infallible>(0.0),
    )
    .unwrap();
    // Saturation walks the original source-distance ordering. AdjacencyList
    // suppresses the already-selected first ID rather than duplicating it.
    assert_eq!(&**saturated.neighbors(), &[1, 2, 3]);
}

#[test]
fn empty_pool_clears_previous_output() {
    let mut scratch = Scratch::new();
    scratch.candidates_mut().push(Neighbor::new(1_u32, 1.0));
    run(
        &mut scratch,
        Policy::new(1, 1.2, PruneKind::TriangleInequality, false),
        Some,
        |_, _| Ok::<_, std::convert::Infallible>(0.0),
    )
    .unwrap();
    assert_eq!(&**scratch.neighbors(), &[1]);

    scratch.candidates_mut().clear();
    run(
        &mut scratch,
        Policy::new(usize::MAX, 1.2, PruneKind::TriangleInequality, false),
        Some,
        |_, _| Ok::<_, std::convert::Infallible>(0.0),
    )
    .unwrap();

    assert!(scratch.neighbors().is_empty());
}
