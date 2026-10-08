/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! In-memory regions and strategies for centroid splitting and point assignment.
//!
//! Source membership stays fixed while splitting changes the candidate centroids.
//! Strategies and providers are trusted to preserve row alignment and valid destinations.

use std::ops::Range;

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use rand::rngs::StdRng;

use crate::{
    ANNResult,
    ivf::online::{LloydScratch, assign_nearest, fit_two_means, index_error},
    utils::VectorId,
};

/// Original membership, independent of the region's eventual destinations.
pub(super) struct SourceList<L> {
    pub(super) id: L,
    pub(super) rows: Range<usize>,
    /// Committed members precede staged points within `rows`.
    pub(super) member_count: usize,
}

pub(super) struct Region<I, L> {
    pub(super) lists: Vec<SourceList<L>>,
    pub(super) point_ids: Vec<I>,
    pub(super) points: rowmajor::Owned<f32>,
    /// `None` denotes a proposed centroid whose list ID has not been staged.
    pub(super) centroid_ids: Vec<Option<L>>,
    pub(super) centroids: rowmajor::Owned<f32>,
}

impl<I, L> Region<I, L> {
    fn list_points(&self, index: usize) -> ANNResult<rowmajor::Ref<'_, f32>> {
        self.points
            .subview(self.lists[index].rows.clone())
            .ok_or_else(|| index_error("source rows lie outside the region's point matrix"))
    }

    pub(super) fn member_ids(&self, index: usize) -> &[I] {
        let list = &self.lists[index];
        &self.point_ids[list.rows.start..list.rows.start + list.member_count]
    }

    pub(super) fn staged_ids(&self, index: usize) -> &[I] {
        let list = &self.lists[index];
        &self.point_ids[list.rows.start + list.member_count..list.rows.end]
    }
}

/// Replace parent centroids without changing source point storage or row order.
pub(super) trait Split<I, L>: Send + Sync {
    fn split(&self, region: &mut Region<I, L>, parents: &[L], rng: &mut StdRng) -> ANNResult<()>;
}

/// Return one valid candidate-centroid row per point, in the region's row order.
pub(super) trait Assign<I, L>: Send + Sync {
    fn assign(&self, region: &Region<I, L>) -> ANNResult<Box<[usize]>>;
}

pub(super) struct TwoMeansSplit {
    pub(super) iterations: usize,
}

impl<I, L: VectorId> Split<I, L> for TwoMeansSplit {
    fn split(&self, region: &mut Region<I, L>, parents: &[L], rng: &mut StdRng) -> ANNResult<()> {
        let is_parent = |id: &Option<L>| id.is_some_and(|id| parents.contains(&id));
        let survivors = region
            .centroid_ids
            .iter()
            .filter(|id| !is_parent(id))
            .count();
        let count = 2 * parents.len() + survivors;
        let dim = region.points.ncols();
        let mut centroids = rowmajor::Owned::try_from_element(count, dim, 0.0)?;
        let mut centroid_ids = Vec::with_capacity(count);
        let mut scratch = LloydScratch::default();

        for (index, parent) in parents.iter().enumerate() {
            let source = region
                .lists
                .iter()
                .position(|list| list.id == *parent)
                .ok_or_else(|| {
                    index_error(format!("split parent {parent} is not in the region"))
                })?;
            let start = 2 * index * dim;
            let out = rowmajor::Mut::try_from_data(
                &mut centroids.as_mut_slice()[start..start + 2 * dim],
                2,
                dim,
            )?;
            fit_two_means(
                region.list_points(source)?,
                out,
                self.iterations,
                rng,
                &mut scratch,
            )?;
            centroid_ids.extend([None, None]);
        }
        for (id, centroid) in region.centroid_ids.iter().zip(region.centroids.rows()) {
            if !is_parent(id) {
                centroids
                    .row_mut(centroid_ids.len())
                    .copy_from_slice(centroid);
                centroid_ids.push(*id);
            }
        }
        region.centroids = centroids;
        region.centroid_ids = centroid_ids;
        Ok(())
    }
}

pub(super) struct NearestCentroid;

impl<I, L> Assign<I, L> for NearestCentroid {
    fn assign(&self, region: &Region<I, L>) -> ANNResult<Box<[usize]>> {
        assign_nearest(region.points.as_view(), region.centroids.as_view())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        sync::{Arc, Mutex},
    };

    use hashbrown::HashMap;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::{
        ANNError,
        ivf::{
            Config, IVFIndex,
            traits::{
                Centroids, InsertAccessor, MaintenanceStrategy, Provider, Reader, SelectedList,
                Stage,
            },
            index::{ListInsertion, StagedPoint},
            update::{CentroidDelta, Delta, Deltas},
        },
        provider::DefaultContext,
    };

    struct TestProvider;

    impl Provider for TestProvider {
        type Context = DefaultContext;
        type InternalId = u32;
        type ExternalId = u32;
        type ListId = u32;
        type Error = ANNError;

        fn to_internal_id(&self, _context: &DefaultContext, id: &u32) -> ANNResult<u32> {
            Ok(*id)
        }

        fn to_external_id(&self, _context: &DefaultContext, id: u32) -> ANNResult<u32> {
            Ok(id)
        }
    }

    struct TestList {
        members: Box<[u32]>,
        vectors: rowmajor::Owned<f32>,
        centroid: Box<[f32]>,
    }

    struct TestAccessor {
        lists: HashMap<u32, TestList>,
        reads: Arc<Mutex<Vec<u32>>>,
        stage_calls: Cell<u32>,
        staged: Vec<(u32, Box<[f32]>)>,
        applied: Arc<Mutex<Vec<Deltas<u32, u32>>>>,
        fail_read: Option<u32>,
        fail_stage: Option<u32>,
        missing_centroid: Option<u32>,
    }

    fn matrix(values: &[f32]) -> rowmajor::Owned<f32> {
        rowmajor::Owned::try_from_data(values.into(), values.len(), 1).unwrap()
    }

    fn inputs<'a>(
        vectors: &'a rowmajor::Owned<f32>,
        lists: &[u32],
    ) -> Vec<ListInsertion<'a, u32, u32>> {
        lists
            .iter()
            .map(|&list| {
                let points: &[(u32, usize)] = match list {
                    10 => &[(503, 3), (500, 0)],
                    20 => &[(501, 1)],
                    30 => &[(502, 2)],
                    _ => unreachable!(),
                };
                ListInsertion {
                    list,
                    staged: points
                        .iter()
                        .map(|&(id, row)| StagedPoint {
                            id,
                            vector: vectors.row(row),
                        })
                        .collect(),
                }
            })
            .collect()
    }

    impl TestAccessor {
        fn new() -> Self {
            Self {
                lists: [
                    (
                        10,
                        TestList {
                            members: Box::new([100, 101]),
                            vectors: matrix(&[0.0, 2.0]),
                            centroid: Box::new([1.0]),
                        },
                    ),
                    (
                        20,
                        TestList {
                            members: Box::new([200, 201]),
                            vectors: matrix(&[10.0, 12.0]),
                            centroid: Box::new([11.0]),
                        },
                    ),
                    (
                        30,
                        TestList {
                            members: Box::new([300]),
                            vectors: matrix(&[100.0]),
                            centroid: Box::new([100.0]),
                        },
                    ),
                ]
                .into_iter()
                .collect(),
                reads: Arc::default(),
                stage_calls: Cell::new(0),
                staged: Vec::new(),
                applied: Arc::default(),
                fail_read: None,
                fail_stage: None,
                missing_centroid: None,
            }
        }
    }

    struct TestCentroids<'a> {
        lists: &'a HashMap<u32, TestList>,
        missing: Option<u32>,
    }

    impl Centroids for TestCentroids<'_> {
        type Id = u32;
        type Error = ANNError;

        fn len(&self) -> usize {
            self.lists.len()
        }

        fn dim(&self) -> usize {
            self.lists.values().next().unwrap().centroid.len()
        }

        fn centroid(&self, id: u32) -> Option<impl std::ops::Deref<Target = [f32]>> {
            self.lists
                .get(&id)
                .filter(|_| self.missing != Some(id))
                .map(|list| list.centroid.as_ref())
        }

        fn select(&self, query: &[f32], nprobe: usize) -> ANNResult<Vec<SelectedList<u32>>> {
            let mut selected: Vec<_> = self
                .lists
                .iter()
                .map(|(&id, list)| SelectedList {
                    id,
                    distance: query
                        .iter()
                        .zip(list.centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum(),
                })
                .collect();
            selected.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
            selected.truncate(nprobe);
            Ok(selected)
        }
    }

    struct TestReader<'a> {
        lists: &'a HashMap<u32, TestList>,
        reads: &'a Mutex<Vec<u32>>,
        fail: Option<u32>,
    }

    impl Reader for TestReader<'_> {
        type Error = ANNError;
        type Id = u32;

        fn dim(&self) -> usize {
            self.lists.values().next().unwrap().centroid.len()
        }

        async fn read_into(&self, id: u32, mut out: rowmajor::Mut<'_, f32>) -> ANNResult<()> {
            self.reads.lock().unwrap().push(id);
            if self.fail == Some(id) {
                return Err(index_error("fixture read failed"));
            }
            out.as_mut_slice()
                .copy_from_slice(self.lists[&id].vectors.as_slice());
            Ok(())
        }
    }

    impl Stage<TestProvider, f32> for TestAccessor {
        async fn stage_point(&mut self, id: &u32, element: f32, out: &mut [f32]) -> ANNResult<u32> {
            out[0] = element;
            Ok(*id + 10_000)
        }

        async fn stage_centroid(&mut self, centroid: &[f32]) -> ANNResult<u32> {
            let call = self.stage_calls.get();
            self.stage_calls.set(call + 1);
            if self.fail_stage == Some(call) {
                return Err(index_error("fixture staging failed"));
            }
            let id = 1000 + call;
            self.staged.push((id, centroid.into()));
            Ok(id)
        }
    }

    impl InsertAccessor<TestProvider, f32> for TestAccessor {
        type Centroids<'a> = TestCentroids<'a>;
        type Reader<'a> = TestReader<'a>;
        type Error = ANNError;

        fn reader(&self) -> TestReader<'_> {
            TestReader {
                lists: &self.lists,
                reads: &self.reads,
                fail: self.fail_read,
            }
        }

        fn centroids(&self) -> TestCentroids<'_> {
            TestCentroids {
                lists: &self.lists,
                missing: self.missing_centroid,
            }
        }

        fn dim(&self) -> usize {
            self.lists.values().next().unwrap().centroid.len()
        }

        fn list_size(&self, list: u32) -> ANNResult<usize> {
            Ok(self.get_members(list)?.len())
        }

        fn get_members(&self, list: u32) -> ANNResult<&[u32]> {
            self.lists
                .get(&list)
                .map(|list| list.members.as_ref())
                .ok_or_else(|| index_error(format!("fixture list {list} is missing")))
        }

        async fn update(self, update: Deltas<u32, u32>) -> ANNResult<()> {
            self.applied.lock().unwrap().push(update);
            Ok(())
        }
    }

    struct TestStrategy(Mutex<Option<TestAccessor>>);

    impl<'a> MaintenanceStrategy<'a, TestProvider, f32> for TestStrategy {
        type MaintenanceAccessor = TestAccessor;
        type Error = ANNError;

        fn maintenance_accessor(
            &'a self,
            _provider: &'a mut TestProvider,
            _context: &'a DefaultContext,
        ) -> ANNResult<TestAccessor> {
            self.0
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| index_error("fixture accessor was consumed"))
        }
    }

    fn require_send<T: Send>(value: T) -> T {
        value
    }

    async fn split_update(
        accessor: &mut TestAccessor,
        inputs: &[ListInsertion<'_, u32, u32>],
    ) -> ANNResult<Vec<Delta<u32, u32>>> {
        let mut region =
            IVFIndex::<TestProvider>::gather_region::<_, f32>(accessor, inputs).await?;
        let parents: Vec<_> = inputs.iter().map(|input| input.list).collect();
        TwoMeansSplit { iterations: 10 }.split(
            &mut region,
            &parents,
            &mut StdRng::seed_from_u64(7),
        )?;
        let assignments = NearestCentroid.assign(&region)?;
        IVFIndex::<TestProvider>::build_deltas::<_, f32>(accessor, &region, &assignments).await
    }

    fn appends(update: &Deltas<u32, u32>) -> HashMap<u32, Vec<u32>> {
        let mut appends = HashMap::new();
        for delta in update.deltas() {
            if let Delta::PointAppends { to, ids } = delta {
                assert!(appends.insert(*to, ids.to_vec()).is_none());
            }
        }
        appends
    }

    fn retired(update: &Deltas<u32, u32>, parent: u32) -> Vec<(u32, u32)> {
        update
            .deltas()
            .iter()
            .find_map(|delta| match delta {
                Delta::CentroidDelta {
                    id,
                    delta: CentroidDelta::Retire { moves },
                } if *id == parent => {
                    Some(moves.iter().map(|point| (point.id(), point.to())).collect())
                }
                _ => None,
            })
            .unwrap()
    }

    #[tokio::test]
    async fn gather_preserves_list_and_staged_order_in_one_buffer() {
        let mut accessor = TestAccessor::new();
        let vectors = matrix(&[1.0, 11.0, 100.0, 3.0]);
        let inputs = inputs(&vectors, &[10, 20, 30]);
        assert_eq!(inputs[0].staged[0].vector.as_ptr(), vectors.row(3).as_ptr());
        assert_eq!(inputs[0].staged[1].vector.as_ptr(), vectors.row(0).as_ptr());
        let region = require_send(IVFIndex::<TestProvider>::gather_region::<_, f32>(
            &mut accessor,
            &inputs,
        ))
        .await
        .unwrap();

        assert_eq!(
            region.point_ids,
            [100, 101, 503, 500, 200, 201, 501, 300, 502]
        );
        assert_eq!(
            region.points.as_slice(),
            &[0.0, 2.0, 3.0, 1.0, 10.0, 12.0, 11.0, 100.0, 100.0]
        );
        assert_eq!(region.centroid_ids, [Some(10), Some(20), Some(30)]);
        assert_eq!(region.centroids.as_slice(), &[1.0, 11.0, 100.0]);
        for (index, (rows, members, staged)) in [
            (0..4, &[100, 101][..], &[503, 500][..]),
            (4..7, &[200, 201][..], &[501][..]),
            (7..9, &[300][..], &[502][..]),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(region.lists[index].rows, rows);
            assert_eq!(region.member_ids(index), members);
            assert_eq!(region.staged_ids(index), staged);
            assert_eq!(
                region.list_points(index).unwrap().as_ptr(),
                region.points.row(rows.start).as_ptr()
            );
        }
        assert_eq!(*accessor.reads.lock().unwrap(), [10, 20, 30]);
    }

    #[tokio::test]
    async fn gather_handles_multiple_dimensions_and_an_empty_member_prefix() {
        let mut accessor = TestAccessor::new();
        accessor.lists = [(
            10,
            TestList {
                members: Box::new([]),
                vectors: rowmajor::Owned::from_element(0, 2, 0.0),
                centroid: Box::new([0.0, 0.0]),
            },
        )]
        .into_iter()
        .collect();
        let staged = rowmajor::Owned::try_from_data(Box::new([1.0, 2.0, 3.0, 4.0]), 2, 2).unwrap();
        let region = IVFIndex::<TestProvider>::gather_region::<_, f32>(
            &mut accessor,
            &[ListInsertion {
                list: 10,
                staged: vec![
                    StagedPoint {
                        id: 501,
                        vector: staged.row(1),
                    },
                    StagedPoint {
                        id: 500,
                        vector: staged.row(0),
                    },
                ],
            }],
        )
        .await
        .unwrap();
        assert_eq!(region.points.as_slice(), &[3.0, 4.0, 1.0, 2.0]);
        assert_eq!(region.point_ids, [501, 500]);
        assert!(region.member_ids(0).is_empty());
        assert_eq!(*accessor.reads.lock().unwrap(), [10]);
    }

    #[tokio::test]
    async fn gather_fills_disjoint_multidimensional_subviews_in_list_order() {
        let mut accessor = TestAccessor::new();
        accessor.lists = [
            (
                10,
                TestList {
                    members: Box::new([100, 101]),
                    vectors: rowmajor::Owned::try_from_data(Box::new([1.0, 2.0, 3.0, 4.0]), 2, 2)
                        .unwrap(),
                    centroid: Box::new([2.0, 3.0]),
                },
            ),
            (
                20,
                TestList {
                    members: Box::new([200]),
                    vectors: rowmajor::Owned::try_from_data(Box::new([5.0, 6.0]), 1, 2).unwrap(),
                    centroid: Box::new([5.0, 6.0]),
                },
            ),
        ]
        .into_iter()
        .collect();
        let staged = rowmajor::Owned::try_from_data(Box::new([7.0, 8.0, 9.0, 10.0]), 2, 2).unwrap();
        let region = require_send(IVFIndex::<TestProvider>::gather_region::<_, f32>(
            &mut accessor,
            &[
                ListInsertion {
                    list: 10,
                    staged: vec![
                        StagedPoint {
                            id: 501,
                            vector: staged.row(1),
                        },
                        StagedPoint {
                            id: 500,
                            vector: staged.row(0),
                        },
                    ],
                },
                ListInsertion {
                    list: 20,
                    staged: Vec::new(),
                },
            ],
        ))
        .await
        .unwrap();

        assert_eq!(region.lists[0].rows, 0..4);
        assert_eq!(region.lists[1].rows, 4..5);
        assert_eq!(region.point_ids, [100, 101, 501, 500, 200]);
        assert_eq!(
            region.points.as_slice(),
            &[1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 7.0, 8.0, 5.0, 6.0],
        );
        assert_eq!(region.centroids.as_slice(), &[2.0, 3.0, 5.0, 6.0]);
        assert!(region.staged_ids(1).is_empty());
        assert_eq!(*accessor.reads.lock().unwrap(), [10, 20]);
    }

    #[tokio::test]
    async fn split_can_retire_an_empty_list_and_assign_only_staged_points() {
        let mut accessor = TestAccessor::new();
        accessor.lists.get_mut(&10).unwrap().members = Box::new([]);
        accessor.lists.get_mut(&10).unwrap().vectors = matrix(&[]);
        let vectors = matrix(&[0.0, 10.0]);
        let update = Deltas::new(
            split_update(
                &mut accessor,
                &[ListInsertion {
                    list: 10,
                    staged: vec![
                        StagedPoint {
                            id: 501,
                            vector: vectors.row(1),
                        },
                        StagedPoint {
                            id: 500,
                            vector: vectors.row(0),
                        },
                    ],
                }],
            )
            .await
            .unwrap(),
        );
        assert!(retired(&update, 10).is_empty());
        assert_eq!(accessor.staged.len(), 2);
        let appended = appends(&update);
        assert_eq!(appended.len(), 2);
        for (to, ids) in appended {
            assert!(to >= 1000);
            assert_eq!(ids.len(), 1);
            assert!([500, 501].contains(&ids[0]));
        }
    }

    #[tokio::test]
    async fn split_replaces_parents_preserves_points_and_numerical_behavior() {
        let mut accessor = TestAccessor::new();
        let vectors = matrix(&[1.0, 11.0, 100.0, 3.0]);
        let mut region = IVFIndex::<TestProvider>::gather_region::<_, f32>(
            &mut accessor,
            &inputs(&vectors, &[10, 20, 30]),
        )
        .await
        .unwrap();
        let points_ptr = region.points.as_ptr();
        let ids_ptr = region.point_ids.as_ptr();
        let point_values = region.points.as_slice().to_vec();
        let mut rng = StdRng::seed_from_u64(7);
        let mut expected_rng = rng.clone();
        let mut expected = rowmajor::Owned::from_element(4, 1, 0.0);
        for (out, source) in expected.as_mut_slice().chunks_exact_mut(2).zip([1, 0]) {
            fit_two_means(
                region.list_points(source).unwrap(),
                rowmajor::Mut::try_from_data(out, 2, 1).unwrap(),
                10,
                &mut expected_rng,
                &mut LloydScratch::default(),
            )
            .unwrap();
        }

        TwoMeansSplit { iterations: 10 }
            .split(&mut region, &[20, 10], &mut rng)
            .unwrap();
        assert_eq!(region.centroid_ids, [None, None, None, None, Some(30)]);
        assert_eq!(&region.centroids.as_slice()[..4], expected.as_slice());
        assert_eq!(region.centroids.row(4), &[100.0]);
        assert_eq!(region.points.as_ptr(), points_ptr);
        assert_eq!(region.point_ids.as_ptr(), ids_ptr);
        assert_eq!(region.points.as_slice(), point_values);
        assert_eq!(rng.random::<u64>(), expected_rng.random::<u64>());
        assert_eq!(accessor.stage_calls.get(), 0);
    }

    #[tokio::test]
    async fn regional_assignment_can_choose_another_parents_children() {
        let mut accessor = TestAccessor::new();
        accessor.lists.get_mut(&10).unwrap().members = Box::new([100, 101, 102]);
        accessor.lists.get_mut(&10).unwrap().vectors = matrix(&[0.0, 10.0, 100.0]);
        accessor.lists.get_mut(&20).unwrap().vectors = matrix(&[1.0, 2.0]);
        let mut region = IVFIndex::<TestProvider>::gather_region::<_, f32>(
            &mut accessor,
            &[
                ListInsertion {
                    list: 10,
                    staged: Vec::new(),
                },
                ListInsertion {
                    list: 20,
                    staged: Vec::new(),
                },
            ],
        )
        .await
        .unwrap();
        TwoMeansSplit { iterations: 10 }
            .split(&mut region, &[10, 20], &mut StdRng::seed_from_u64(7))
            .unwrap();
        let assigned = NearestCentroid.assign(&region).unwrap();
        assert_eq!(region.centroid_ids, [None, None, None, None]);
        assert!(
            assigned[0] >= 2,
            "A's point at zero should prefer B's child at one"
        );
        assert_eq!(region.centroids.row(assigned[0]), &[1.0]);
        let update = Deltas::new(
            IVFIndex::<TestProvider>::build_deltas::<_, f32>(&mut accessor, &region, &assigned)
                .await
                .unwrap(),
        );
        assert_eq!(retired(&update, 10)[0], (100, 1000 + assigned[0] as u32));
        for parent in [10, 20] {
            assert!(retired(&update, parent).iter().all(|(_, to)| *to >= 1000));
        }
    }

    #[tokio::test]
    async fn regional_deltas_move_parent_members_and_append_only_regional_staged_points() {
        let mut accessor = TestAccessor::new();
        let vectors = matrix(&[1.0, 11.0, 100.0, 3.0]);
        let update = Deltas::new(
            require_send(split_update(&mut accessor, &inputs(&vectors, &[10, 20])))
                .await
                .unwrap(),
        );
        assert_eq!(accessor.staged.len(), 4);
        assert_eq!(*accessor.reads.lock().unwrap(), [10, 20]);
        for (parent, ids) in [(10, [100, 101]), (20, [200, 201])] {
            let moves = retired(&update, parent);
            assert_eq!(moves.iter().map(|(id, _)| *id).collect::<Vec<_>>(), ids);
            assert!(moves.iter().all(|(_, to)| *to >= 1000));
        }
        let appended = appends(&update);
        assert!(!appended.contains_key(&30));
        let mut ids: Vec<_> = appended.values().flatten().copied().collect();
        ids.sort_unstable();
        assert_eq!(ids, [500, 501, 503]);
        let destination = appended.values().find(|ids| ids.contains(&503)).unwrap();
        assert_eq!(destination, &[503]);
    }

    struct ThreeChildren;

    impl Split<u32, u32> for ThreeChildren {
        fn split(
            &self,
            region: &mut Region<u32, u32>,
            _parents: &[u32],
            _rng: &mut StdRng,
        ) -> ANNResult<()> {
            region.centroids = matrix(&[0.0, 2.0, 100.0, 100.0]);
            region.centroid_ids = vec![None, None, None, Some(30)];
            Ok(())
        }
    }

    #[tokio::test]
    async fn custom_split_supports_three_children_and_neighbor_reassignment() {
        let mut accessor = TestAccessor::new();
        let vectors = matrix(&[1.0, 11.0, 100.0, 3.0]);
        let mut region = IVFIndex::<TestProvider>::gather_region::<_, f32>(
            &mut accessor,
            &inputs(&vectors, &[10, 30]),
        )
        .await
        .unwrap();
        ThreeChildren
            .split(&mut region, &[10], &mut StdRng::seed_from_u64(7))
            .unwrap();
        let assigned = NearestCentroid.assign(&region).unwrap();
        let update = Deltas::new(
            IVFIndex::<TestProvider>::build_deltas::<_, f32>(&mut accessor, &region, &assigned)
                .await
                .unwrap(),
        );
        assert_eq!(accessor.staged.len(), 3);
        let neighbor_moves: Vec<_> = update
            .deltas()
            .iter()
            .filter_map(|delta| match delta {
                Delta::PointMoves { from: 30, moves } => Some(moves.as_ref()),
                _ => None,
            })
            .flatten()
            .map(|point| (point.id(), point.to()))
            .collect();
        assert_eq!(neighbor_moves, [(300, 1002)]);
        assert_eq!(appends(&update)[&1002], [502]);
        assert!(
            update
                .deltas()
                .iter()
                .all(|delta| !matches!(delta, Delta::CentroidDelta { id: 30, .. }))
        );
    }

    #[tokio::test]
    async fn surviving_neighbor_members_and_staged_points_can_stay_put() {
        let mut accessor = TestAccessor::new();
        let vectors = matrix(&[1.0, 11.0, 100.0, 3.0]);
        let mut region = IVFIndex::<TestProvider>::gather_region::<_, f32>(
            &mut accessor,
            &inputs(&vectors, &[10, 30]),
        )
        .await
        .unwrap();
        TwoMeansSplit { iterations: 10 }
            .split(&mut region, &[10], &mut StdRng::seed_from_u64(7))
            .unwrap();
        let assigned = NearestCentroid.assign(&region).unwrap();
        let update = Deltas::new(
            IVFIndex::<TestProvider>::build_deltas::<_, f32>(&mut accessor, &region, &assigned)
                .await
                .unwrap(),
        );
        assert_eq!(accessor.staged.len(), 2);
        assert!(
            update
                .deltas()
                .iter()
                .all(|delta| !matches!(delta, Delta::PointMoves { .. }))
        );
        assert_eq!(appends(&update)[&30], [502]);
    }

    #[tokio::test]
    async fn custom_assignment_uses_the_same_flat_row_contract() {
        struct LastCentroid;
        impl Assign<u32, u32> for LastCentroid {
            fn assign(&self, region: &Region<u32, u32>) -> ANNResult<Box<[usize]>> {
                Ok(vec![region.centroids.nrows() - 1; region.points.nrows()].into_boxed_slice())
            }
        }
        let mut accessor = TestAccessor::new();
        let vectors = matrix(&[1.0, 11.0, 100.0, 3.0]);
        let mut region = IVFIndex::<TestProvider>::gather_region::<_, f32>(
            &mut accessor,
            &inputs(&vectors, &[10, 30]),
        )
        .await
        .unwrap();
        TwoMeansSplit { iterations: 10 }
            .split(&mut region, &[10], &mut StdRng::seed_from_u64(7))
            .unwrap();
        let update = Deltas::new(
            IVFIndex::<TestProvider>::build_deltas::<_, f32>(
                &mut accessor,
                &region,
                &LastCentroid.assign(&region).unwrap(),
            )
            .await
            .unwrap(),
        );
        assert_eq!(retired(&update, 10), [(100, 30), (101, 30)]);
        assert_eq!(appends(&update)[&30], [503, 500, 502]);
    }

    #[tokio::test]
    async fn empty_region_inputs_skip_reads_and_centroid_staging() {
        let mut accessor = TestAccessor::new();
        let region = require_send(IVFIndex::<TestProvider>::gather_region::<_, f32>(
            &mut accessor,
            &[],
        ))
        .await
        .unwrap();
        assert!(region.lists.is_empty());
        assert!(region.point_ids.is_empty());
        assert_eq!(region.points.nrows(), 0);
        assert_eq!(region.centroids.nrows(), 0);
        assert!(accessor.reads.lock().unwrap().is_empty());
        assert_eq!(accessor.stage_calls.get(), 0);
    }

    #[tokio::test]
    async fn gathering_errors_propagate_before_centroid_staging() {
        let vectors = matrix(&[1.0, 11.0, 100.0, 3.0]);
        for failure in ["members", "reader", "centroid"] {
            let mut accessor = TestAccessor::new();
            match failure {
                "members" => {
                    accessor.lists.remove(&20);
                }
                "reader" => {
                    accessor.fail_read = Some(20);
                }
                "centroid" => {
                    accessor.missing_centroid = Some(20);
                }
                _ => unreachable!(),
            }
            let err = split_update(&mut accessor, &inputs(&vectors, &[10, 20]))
                .await
                .unwrap_err();
            let expected = match failure {
                "members" => "fixture list 20 is missing",
                "reader" => "fixture read failed",
                "centroid" => "centroid 20 is unavailable",
                _ => unreachable!(),
            };
            assert!(err.to_string().contains(expected), "{err}");
            assert_eq!(accessor.stage_calls.get(), 0);
        }
    }

    #[tokio::test]
    async fn numerical_and_staging_errors_propagate() {
        let mut accessor = TestAccessor::new();
        accessor.lists.get_mut(&10).unwrap().members = Box::new([100]);
        accessor.lists.get_mut(&10).unwrap().vectors = matrix(&[0.0]);
        let err = split_update(
            &mut accessor,
            &[ListInsertion {
                list: 10,
                staged: Vec::new(),
            }],
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("at least two points"), "{err}");
        assert_eq!(accessor.stage_calls.get(), 0);

        let mut region = IVFIndex::<TestProvider>::gather_region::<_, f32>(
            &mut accessor,
            &[ListInsertion {
                list: 10,
                staged: Vec::new(),
            }],
        )
        .await
        .unwrap();
        region.centroids = matrix(&[]);
        region.centroid_ids.clear();
        let err = NearestCentroid.assign(&region).unwrap_err();
        assert!(err.to_string().contains("at least one centroid"), "{err}");

        accessor.fail_stage = Some(1);
        let vectors = matrix(&[1.0, 11.0, 100.0, 3.0]);
        let err = split_update(&mut accessor, &inputs(&vectors, &[10, 20]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("fixture staging failed"), "{err}");
        assert_eq!(accessor.staged.len(), 1);
    }

    #[tokio::test]
    async fn insert_batch_applies_splits_and_unsplit_appends_together() {
        for threshold in [2, 3, 100] {
            let accessor = TestAccessor::new();
            let applied = Arc::clone(&accessor.applied);
            let reads = Arc::clone(&accessor.reads);
            let strategy = TestStrategy(Mutex::new(Some(accessor)));
            let mut index = IVFIndex::new(
                TestProvider,
                Config {
                    split_threshold: threshold,
                    reassign_neighbors: 1,
                    two_means_iterations: 10,
                    seed: 7,
                },
            )
            .unwrap();
            require_send(index.insert_batch(
                &strategy,
                &DefaultContext,
                &[(500, 1.0), (501, 11.0), (502, 100.0), (503, 3.0)],
            ))
            .await
            .unwrap();
            let updates = applied.lock().unwrap();
            assert_eq!(updates.len(), 1);
            let appended = appends(&updates[0]);
            let mut ids: Vec<_> = appended.values().flatten().copied().collect();
            ids.sort_unstable();
            assert_eq!(ids, [10500, 10501, 10502, 10503]);
            assert_eq!(appended[&30], [10502]);
            if threshold < 100 {
                assert_eq!(retired(&updates[0], 10).len(), 2);
                let mut read_ids = reads.lock().unwrap().clone();
                read_ids.sort_unstable();
                if threshold == 2 {
                    assert_eq!(retired(&updates[0], 20).len(), 2);
                    assert_eq!(read_ids, [10, 20]);
                } else {
                    assert_eq!(appended[&20], [10501]);
                    assert_eq!(read_ids, [10]);
                }
            } else {
                assert_eq!(appended[&10], [10500, 10503]);
                assert_eq!(appended[&20], [10501]);
                assert!(reads.lock().unwrap().is_empty());
                assert_eq!(updates[0].deltas().len(), 3);
            }
        }
    }

    #[tokio::test]
    async fn regional_failure_does_not_commit_already_planned_ordinary_appends() {
        let mut accessor = TestAccessor::new();
        accessor.fail_read = Some(10);
        let applied = Arc::clone(&accessor.applied);
        let strategy = TestStrategy(Mutex::new(Some(accessor)));
        let mut index = IVFIndex::new(
            TestProvider,
            Config {
                split_threshold: 3,
                reassign_neighbors: 1,
                two_means_iterations: 10,
                seed: 7,
            },
        )
        .unwrap();
        let err = require_send(index.insert_batch(
            &strategy,
            &DefaultContext,
            &[(500, 1.0), (501, 11.0), (502, 100.0), (503, 3.0)],
        ))
        .await
        .unwrap_err();
        assert!(err.to_string().contains("fixture read failed"), "{err}");
        assert!(applied.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn empty_insert_does_not_construct_an_accessor_or_apply_an_update() {
        let strategy = TestStrategy(Mutex::new(None));
        let mut index = IVFIndex::new(
            TestProvider,
            Config {
                split_threshold: 2,
                reassign_neighbors: 1,
                two_means_iterations: 10,
                seed: 7,
            },
        )
        .unwrap();
        let result = require_send(index.insert_batch(&strategy, &DefaultContext, &[]))
            .await
            .unwrap();
        assert_eq!(result.inserted, 0);
    }
}
