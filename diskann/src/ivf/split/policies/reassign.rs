/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Score only each source's linked candidates, then apply changes in source order.

use std::fmt::Debug;

use diskann_quantization::{Parallelism, algorithms::kmeans::gemm};
use diskann_utils::{
    Reborrow,
    views::rowmajor::{self, Matrix},
};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

use super::{Buckets, SplitError, TryMap};
use crate::{
    ANNError, ANNResult,
    ivf::{
        region::{AsMatRef, Region},
        split::{Links, Reassign},
    },
};

// Avoid matrix preparation for narrow candidate sets. The crossover with the
// 128-row scorer depends on dimension and point count.
const MIN_GEMM_CANDIDATES: usize = 12;

struct Job<'a> {
    points: rowmajor::Ref<'a, f32>,
    candidates: Vec<(usize, &'a [f32])>,
    norms: Vec<f64>,
}

impl Job<'_> {
    fn assign(&self) -> ANNResult<Vec<usize>> {
        if self.candidates.len() < MIN_GEMM_CANDIDATES || self.points.nrows() == 0 {
            return Ok(self
                .points
                .rows()
                .map(|point| nearest_rows(point, self.candidates.iter().map(|(_, center)| *center)))
                .collect());
        }
        let centers = rowmajor::Owned::try_from_data(
            self.candidates
                .iter()
                .flat_map(|(_, row)| row.iter().copied())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            self.candidates.len(),
            self.points.ncols(),
        )?;
        let mut labels = vec![0; self.points.nrows()];
        gemm::assign(self.points, centers.as_view(), &self.norms, &mut labels)
            .map_err(ANNError::new)?;
        Ok(labels.into_iter().map(|n| n as usize).collect())
    }
}

/// Reassigns points among linked lists using direct L2 or tiled GEMM.
pub(crate) struct Linked;

impl<I, L, R> Reassign<I, L, R> for Linked
where
    I: Copy + Send,
    L: Copy + Debug + Send + Sync + 'static,
    R: Region<I, L>,
    for<'a> R::VectorsRef<'a>: AsMatRef<'a, f32>,
{
    type Error = ANNError;

    fn reassign(
        &mut self,
        region: &mut R,
        links: &Links<L>,
        parallelism: Parallelism,
    ) -> ANNResult<()> {
        let moves = {
            let lists = links
                .lists()
                .iter()
                .map(|&id| region.get_list(id).ok_or(SplitError::UnableToRetrieve(id)))
                .collect::<Result<Vec<_>, _>>()?;
            let vectors = links
                .lists()
                .iter()
                .map(|&id| {
                    region
                        .get_vectors(id)
                        .ok_or(SplitError::UnableToRetrieve(id))
                })
                .collect::<Result<Vec<_>, _>>()?;
            // Centroids are immutable within this stage. Prepare a norm only when a
            // wide source needs it, and reuse it across sources by the filled-list index.
            let mut center_norms = vec![None; lists.len()];
            let jobs: Vec<_> = lists
                .iter()
                .zip(&vectors)
                .enumerate()
                .map(|(source, (list, vectors))| {
                    // Linked centroids win ties before the source itself.
                    let candidates: Vec<_> = links
                        .links(source)
                        .iter()
                        .map(|&neighbor| (neighbor, lists[neighbor].centroid))
                        .chain(std::iter::once((source, list.centroid)))
                        .collect();
                    let points = vectors.reborrow().as_view();
                    let norms = if candidates.len() >= MIN_GEMM_CANDIDATES && points.nrows() > 0 {
                        candidates
                            .iter()
                            .map(|&(id, centroid)| {
                                *center_norms[id].get_or_insert_with(|| gemm::center_norm(centroid))
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    Job {
                        points,
                        candidates,
                        norms,
                    }
                })
                .collect();
            let labels = parallelism.try_map(jobs.len(), |i| jobs[i].assign())?;

            lists
                .iter()
                .zip(&jobs)
                .zip(labels)
                .enumerate()
                .map(|(source, ((list, job), labels))| {
                    list.members
                        .iter()
                        .copied()
                        .zip(labels)
                        .filter_map(|(id, slot)| {
                            let destination = job.candidates[slot].0;
                            (destination != source).then_some((id, destination))
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };

        let mut moved = Buckets::new(links.len());

        for (&list, moves) in links.lists().iter().zip(moves) {
            moved.extend(moves.into_iter());
            region.remove_members(list, moved.seal())?;
        }

        let incoming = moved.group();

        for (i, &list) in links.lists().iter().enumerate() {
            region.append_members(list, incoming.get(i))?;
        }

        Ok(())
    }
}

#[inline(always)]
fn nearest_rows<'a>(point: &[f32], centers: impl Iterator<Item = &'a [f32]>) -> usize {
    let mut best = 0;
    let mut minimum = f32::INFINITY;
    for (position, center) in centers.enumerate() {
        let distance: f32 = SquaredL2::evaluate(point, center);
        if position == 0 || distance < minimum {
            minimum = distance;
            best = position;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_scorers_preserve_allowed_keys_and_candidate_order() {
        let points =
            rowmajor::Owned::try_from_data(vec![0.0, 4.0, 8.0, 6.0].into_boxed_slice(), 4, 1)
                .unwrap();
        // A hypothetical global centroid at zero must not be offered to these points.
        for count in [11, 12] {
            let storage: Vec<_> = std::iter::once(vec![4.0])
                .chain(std::iter::once(vec![8.0]))
                .chain((2..count).map(|j| vec![100.0 + j as f32]))
                .collect();
            for (first, second, expected) in [(0, 1, [91, 91, 13, 91]), (1, 0, [91, 91, 13, 13])] {
                let key = |i| if i == 0 { 91 } else { 13 };
                let candidates: Vec<_> = [
                    (key(first), storage[first].as_slice()),
                    (key(second), storage[second].as_slice()),
                ]
                .into_iter()
                .chain((2..count).map(|j| (100 + j, storage[j].as_slice())))
                .collect();
                let norms = candidates
                    .iter()
                    .map(|(_, row)| gemm::center_norm(row))
                    .collect();
                let job = Job {
                    points: points.as_view(),
                    candidates,
                    norms,
                };
                let actual: Vec<_> = job
                    .assign()
                    .unwrap()
                    .iter()
                    .map(|&slot| job.candidates[slot].0)
                    .collect();
                assert_eq!(actual, expected);
            }
        }
    }
}
