/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Benchmark-local storage for exercising the real IVF insertion algorithm.
//! Build routing is exhaustive; a centroid graph is built once after ingestion.

use std::{collections::BTreeMap, sync::Arc};

use diskann::{
    ivf::{
        pending::{Membership, PendingLists},
        traits::{
            BuildAccessor, BuildStrategy, CoarseAccessor, InsertStrategy, ListAccessor, Provider,
            SelectedList,
        },
        workingset::{Batch, ListView, PointRef, WorkingSet},
    },
    provider::DefaultContext,
    ANNError, ANNResult,
};
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::{distance::SquaredL2, PureDistanceFunction};

pub(super) struct StoredList {
    pub centroid: Vec<f32>,
    pub members: Vec<u32>,
}

pub(super) struct Store {
    dim: usize,
    batches: Vec<Arc<Batch<u32>>>,
    point_batches: Vec<usize>,
    pub lists: BTreeMap<u32, StoredList>,
    next_list: u32,
}

impl Store {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            batches: Vec::new(),
            point_batches: Vec::new(),
            lists: BTreeMap::new(),
            next_list: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.point_batches.len()
    }

    pub fn vector(&self, id: u32) -> ANNResult<&[f32]> {
        self.point_batches
            .get(id as usize)
            .and_then(|&batch| self.batches[batch].get(id))
            .ok_or_else(|| ANNError::message(format!("missing point {id}")))
    }

    /// Run outside all timers: dropped or duplicated points invalidate the measurement.
    pub fn validate_membership(&self, expected: usize) -> anyhow::Result<()> {
        anyhow::ensure!(self.len() == expected, "incomplete point publication");
        let mut seen = vec![false; expected];
        for list in self.lists.values() {
            anyhow::ensure!(
                list.centroid.len() == self.dim && list.centroid.iter().all(|x| x.is_finite()),
                "invalid centroid"
            );
            for &id in &list.members {
                let slot = seen
                    .get_mut(id as usize)
                    .ok_or_else(|| anyhow::anyhow!("invalid member ID {id}"))?;
                anyhow::ensure!(!*slot, "duplicate member ID {id}");
                *slot = true;
            }
        }
        anyhow::ensure!(
            seen.iter().all(|seen| *seen),
            "some points have no list membership"
        );
        Ok(())
    }
}

impl Provider for Store {
    type Context = DefaultContext;
    type InternalId = u32;
    type ExternalId = u32;
    type ListId = u32;
    type Error = ANNError;

    fn to_internal_id(&self, _: &DefaultContext, id: &u32) -> ANNResult<u32> {
        self.vector(*id).map(|_| *id)
    }

    fn to_external_id(&self, context: &DefaultContext, id: u32) -> ANNResult<u32> {
        self.to_internal_id(context, &id)
    }
}

pub(super) struct Build<'p> {
    store: &'p mut Store,
    pending: PendingLists<u32, u32>,
    created: BTreeMap<u32, Vec<f32>>,
    batch: Option<Arc<Batch<u32>>>,
    next_list: u32,
}

impl Build<'_> {
    fn stored_members(&self, list: u32) -> ANNResult<&[u32]> {
        if let Some(stored) = self.store.lists.get(&list) {
            Ok(&stored.members)
        } else if self.created.contains_key(&list) {
            Ok(&[])
        } else {
            Err(ANNError::message(format!("unknown list {list}")))
        }
    }
}

pub(super) struct Coarse<'a> {
    stored: &'a BTreeMap<u32, StoredList>,
    created: &'a BTreeMap<u32, Vec<f32>>,
    pending: &'a PendingLists<u32, u32>,
}

impl CoarseAccessor for Coarse<'_> {
    type ListId = u32;

    fn len(&self) -> usize {
        self.stored
            .keys()
            .chain(self.created.keys())
            .filter(|&&id| !self.pending.is_retired(id))
            .count()
    }

    async fn select(&mut self, query: &[f32], nprobe: usize) -> ANNResult<Vec<SelectedList<u32>>> {
        let mut selected: Vec<_> = self
            .stored
            .iter()
            .map(|(&id, list)| (id, list.centroid.as_slice()))
            .chain(
                self.created
                    .iter()
                    .map(|(&id, centroid)| (id, centroid.as_slice())),
            )
            .filter(|(id, _)| !self.pending.is_retired(*id))
            .map(|(id, centroid)| SelectedList {
                id,
                distance: SquaredL2::evaluate(query, centroid),
            })
            .collect();
        let compare = |a: &SelectedList<u32>, b: &SelectedList<u32>| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        };
        if nprobe < selected.len() {
            selected.select_nth_unstable_by(nprobe, compare);
            selected.truncate(nprobe);
        }
        selected.sort_unstable_by(compare);
        Ok(selected)
    }
}

pub(super) struct Lists<'a, 'p>(&'a mut Build<'p>);

impl ListAccessor for Lists<'_, '_> {
    type InternalId = u32;
    type ListId = u32;

    async fn list_size(&mut self, list: u32) -> ANNResult<usize> {
        self.0
            .pending
            .list_size(list, self.0.stored_members(list)?.len())
    }

    async fn append_members(&mut self, list: u32, members: Vec<u32>) -> ANNResult<()> {
        self.0.stored_members(list)?;
        self.0.pending.append(list, members)
    }

    async fn set_members(&mut self, list: u32, members: Vec<u32>) -> ANNResult<()> {
        self.0.stored_members(list)?;
        self.0.pending.set(list, members)
    }

    async fn create_list(&mut self, centroid: &[f32], members: Vec<u32>) -> ANNResult<u32> {
        if centroid.len() != self.0.store.dim {
            return Err(ANNError::message("wrong centroid dimension"));
        }
        let id = self.0.next_list;
        self.0.next_list = id
            .checked_add(1)
            .ok_or_else(|| ANNError::message("list ID overflow"))?;
        self.0.created.insert(id, centroid.to_vec());
        self.0.pending.set(id, members)?;
        Ok(id)
    }

    async fn retire_list(&mut self, list: u32) -> ANNResult<()> {
        self.0.stored_members(list)?;
        self.0.pending.retire(list)
    }
}

impl<'p> BuildAccessor for Build<'p> {
    type InternalId = u32;
    type ListId = u32;
    type Coarse<'a>
        = Coarse<'a>
    where
        Self: 'a;
    type Lists<'a>
        = Lists<'a, 'p>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.store.dim
    }

    fn coarse(&mut self) -> Self::Coarse<'_> {
        Coarse {
            stored: &self.store.lists,
            created: &self.created,
            pending: &self.pending,
        }
    }

    fn lists(&mut self) -> Self::Lists<'_> {
        Lists(self)
    }

    async fn fill(&mut self, lists: &[u32]) -> ANNResult<WorkingSet<'_, u32, u32>> {
        let mut view = Vec::with_capacity(lists.len());
        for &list in lists {
            let mut points = Vec::with_capacity(
                self.pending
                    .list_size(list, self.stored_members(list)?.len())?,
            );
            for id in self.pending.members(list, self.stored_members(list)?)? {
                let vector = match self.batch.as_ref().and_then(|batch| batch.get(id)) {
                    Some(vector) => vector,
                    None => self.store.vector(id)?,
                };
                points.push(PointRef { id, vector });
            }
            view.push(ListView { id: list, points });
        }
        Ok(view)
    }

    async fn finish(self) -> ANNResult<()> {
        if let Some(batch) = self.batch {
            self.store.point_batches.extend(std::iter::repeat_n(
                self.store.batches.len(),
                batch.vectors().nrows(),
            ));
            self.store.batches.push(batch);
        }
        for (id, centroid) in self.created {
            self.store.lists.insert(
                id,
                StoredList {
                    centroid,
                    members: Vec::new(),
                },
            );
        }
        for (id, change) in self.pending.into_changes() {
            match change {
                Membership::Append(members) => self
                    .store
                    .lists
                    .get_mut(&id)
                    .unwrap()
                    .members
                    .extend(members),
                Membership::Replace(members) => {
                    self.store.lists.get_mut(&id).unwrap().members = members
                }
                Membership::Retire => {
                    self.store.lists.remove(&id).unwrap();
                }
            }
        }
        self.store.next_list = self.next_list;
        Ok(())
    }
}

pub(super) struct Strategy;

impl<'a> BuildStrategy<'a, Store> for Strategy {
    type Accessor = Build<'a>;
    type Error = ANNError;

    fn build_accessor(
        &'a self,
        store: &'a mut Store,
        _: &'a DefaultContext,
    ) -> ANNResult<Build<'a>> {
        let next_list = store.next_list;
        Ok(Build {
            store,
            pending: PendingLists::default(),
            created: BTreeMap::new(),
            batch: None,
            next_list,
        })
    }
}

impl<'a, 'v> InsertStrategy<'a, Store, &'v [f32]> for Strategy {
    async fn prepare(
        &self,
        accessor: &mut Build<'a>,
        points: &[(u32, &'v [f32])],
    ) -> ANNResult<Arc<Batch<u32>>> {
        let mut vectors = rowmajor::Owned::try_from_element(points.len(), accessor.dim(), 0.0)?;
        let mut ids = Vec::with_capacity(points.len());
        for (offset, ((id, vector), out)) in points.iter().zip(vectors.rows_mut()).enumerate() {
            if *id as usize != accessor.store.len() + offset || vector.len() != out.len() {
                return Err(ANNError::message(
                    "benchmark input needs contiguous IDs and matching dimensions",
                ));
            }
            out.copy_from_slice(vector);
            ids.push(*id);
        }
        let batch = Arc::new(Batch::new(ids, vectors)?);
        accessor.batch = Some(Arc::clone(&batch));
        Ok(batch)
    }
}
