/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;

#[derive(Debug, Clone)]
pub(super) struct StoredList {
    pub(super) centroid: Vec<f32>,
    pub(super) members: Vec<u32>,
}

#[derive(Debug, Default)]
pub(super) struct Store {
    pub(super) dim: usize,
    pub(super) points: BTreeMap<u32, Vec<f32>>,
    pub(super) lists: BTreeMap<u32, StoredList>,
    pub(super) next_list: u32,
    pub(super) finishes: usize,
    pub(super) fills: usize,
    pub(super) filled_lists: Vec<u32>,
    pub(super) fail_route: bool,
    pub(super) fail_nearest: bool,
    pub(super) fail_fill: bool,
    pub(super) fail_create: bool,
    pub(super) fail_set_members: bool,
    pub(super) fail_retire: bool,
    pub(super) fail_finish: bool,
}

impl Store {
    pub(super) fn to_internal_id(&self, _: &DefaultContext, id: &u32) -> ANNResult<u32> {
        id.checked_add(10_000)
            .filter(|id| self.points.contains_key(id))
            .ok_or_else(|| ANNError::message("point is not visible"))
    }

    pub(super) fn to_external_id(&self, _: &DefaultContext, id: u32) -> ANNResult<u32> {
        if !self.points.contains_key(&id) {
            return Err(ANNError::message("point is not visible"));
        }
        id.checked_sub(10_000)
            .ok_or_else(|| ANNError::message("invalid internal point ID"))
    }
}

pub(super) struct Build<'p, T> {
    pub(super) store: MutexGuard<'p, Store>,
    pub(super) pending: PendingLists<u32, u32>,
    pub(super) created: Vec<StoredList>,
    pub(super) staged: BTreeMap<u32, Vec<T>>,
    pub(super) sources: BTreeMap<u32, Source>,
    pub(super) fills: Cell<usize>,
    pub(super) filled_lists: Vec<u32>,
}

pub(super) struct Coarse<'a> {
    pub(super) stored: &'a BTreeMap<u32, StoredList>,
    pub(super) fail_route: bool,
    pub(super) fail_nearest: bool,
}

fn squared_distance<T: Copy + Into<f32>>(query: &[T], vector: &[f32]) -> f32 {
    query
        .iter()
        .zip(vector)
        .map(|(&value, &coordinate)| {
            let delta = value.into() - coordinate;
            delta * delta
        })
        .sum()
}

impl Coarse<'_> {
    fn select<T, P, F>(&self, query: &[T], nprobe: NonZeroUsize, pred: P, mut emit: F) -> usize
    where
        T: Copy + Into<f32>,
        P: Fn(&u32) -> bool,
        F: FnMut(&u32, f32),
    {
        let mut selected: Vec<_> = self
            .stored
            .iter()
            .filter(|(id, _)| pred(id))
            .map(|(&id, list)| (id, squared_distance(query, &list.centroid)))
            .collect();
        selected.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        selected.truncate(nprobe.get());
        for (id, distance) in &selected {
            emit(id, *distance);
        }
        selected.len()
    }
    pub(super) fn route<T, F>(
        &self,
        query: &[T],
        nprobe: NonZeroUsize,
        emit: F,
    ) -> impl SendFuture<ANNResult<()>>
    where
        T: Copy + Into<f32> + Sync,
        F: FnMut(&u32, f32) + Send,
    {
        async move {
            if self.fail_route {
                return Err(ANNError::message("injected routing failure"));
            }
            self.select(query, nprobe, |_| true, emit);
            Ok(())
        }
    }
}

impl Nearest for Coarse<'_> {
    type Id = u32;

    fn find<P, F>(
        &self,
        id: Self::Id,
        nprobe: NonZeroUsize,
        pred: P,
        emit: F,
    ) -> impl SendFuture<ANNResult<usize>>
    where
        P: Fn(&Self::Id) -> bool + Send,
        F: FnMut(&Self::Id, f32) + Send,
    {
        async move {
            if self.fail_nearest {
                return Err(ANNError::message("injected nearest failure"));
            }
            let query = &self
                .stored
                .get(&id)
                .ok_or_else(|| ANNError::message(format!("unknown list {id}")))?
                .centroid;
            Ok(self.select(query, nprobe, pred, emit))
        }
    }
}

pub(super) struct Source {
    pub(super) centroid: Vec<f32>,
    pub(super) members: Vec<u32>,
    pub(super) vectors: rowmajor::Owned<f32>,
}

pub(super) struct Update<'a> {
    pub(super) dim: usize,
    pub(super) sources: &'a BTreeMap<u32, Source>,
    pub(super) memberships: &'a mut PendingLists<u32, u32>,
    pub(super) created: &'a mut Vec<StoredList>,
    pub(super) fail_create: bool,
    pub(super) fail_set_members: bool,
    pub(super) fail_retire: bool,
}

impl<'a> Update<'a> {
    pub(super) fn new(
        sources: &'a BTreeMap<u32, Source>,
        dim: usize,
        memberships: &'a mut PendingLists<u32, u32>,
        created: &'a mut Vec<StoredList>,
    ) -> Self {
        Self {
            dim,
            sources,
            memberships,
            created,
            fail_create: false,
            fail_set_members: false,
            fail_retire: false,
        }
    }

    fn check_list(&self, list: u32) -> ANNResult<()> {
        if self.sources.contains_key(&list) {
            Ok(())
        } else {
            Err(ANNError::message(format!("unknown list {list}")))
        }
    }
}

pub(super) struct Region<'a> {
    pub(super) sources: &'a BTreeMap<u32, Source>,
}

impl<'a> Region<'a> {
    pub(super) fn new(sources: &'a BTreeMap<u32, Source>) -> Self {
        Self { sources }
    }
}

impl View<u32, u32> for Region<'_> {
    type VectorsRef<'a> = rowmajor::Ref<'a, f32>;
    type Vectors<'a>
        = rowmajor::Ref<'a, f32>
    where
        Self: 'a;

    fn get_list(&self, list: u32) -> Option<List<'_, u32, u32>> {
        self.sources.get(&list).map(|source| List {
            id: list,
            centroid: &source.centroid,
            members: &source.members,
        })
    }

    fn get_vectors(&self, list: u32) -> Option<rowmajor::Ref<'_, f32>> {
        self.sources
            .get(&list)
            .map(|source| source.vectors.as_view())
    }
}

impl crate::ivf::region::Update<u32, u32> for Update<'_> {
    fn set_members(&mut self, list: u32, members: &[u32]) -> ANNResult<()> {
        if self.fail_set_members {
            return Err(ANNError::message("injected set_members failure"));
        }
        self.check_list(list)?;
        self.memberships.set(list, members.to_vec())
    }

    fn retire(&mut self, list: u32) -> ANNResult<()> {
        if self.fail_retire {
            return Err(ANNError::message("injected retire failure"));
        }
        self.check_list(list)?;
        self.memberships.retire(list)
    }

    fn create(&mut self, centroid: &[f32], members: &[u32]) -> ANNResult<()> {
        if self.fail_create {
            return Err(ANNError::message("injected create failure"));
        }
        if centroid.len() != self.dim || !centroid.iter().all(|value| value.is_finite()) {
            return Err(ANNError::message("invalid centroid"));
        }
        self.created.push(StoredList {
            centroid: centroid.to_vec(),
            members: members.to_vec(),
        });
        Ok(())
    }
}

impl<T> HasIds for Build<'_, T> {
    type Id = u32;
    type ListId = u32;
}

impl<'p, 'b, 'v, T: Copy + Into<f32> + Send + Sync> BuildAccessor<&'b [(u32, &'v [T])]>
    for Build<'p, T>
{
    type Update<'a>
        = Update<'a>
    where
        Self: 'a;
    type Nearest<'a>
        = Coarse<'a>
    where
        Self: 'a;
    type VectorsRef<'a> = rowmajor::Ref<'a, f32>;
    type Region<'a>
        = Region<'a>
    where
        Self: 'a;

    fn dim(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.store.dim).unwrap()
    }

    fn nearest(&mut self) -> Self::Nearest<'_> {
        Coarse {
            stored: &self.store.lists,
            fail_route: self.store.fail_route,
            fail_nearest: self.store.fail_nearest,
        }
    }

    fn route<F>(
        &mut self,
        points: &'b [(u32, &'v [T])],
        mut emit: F,
    ) -> impl SendFuture<ANNResult<NonZeroUsize>>
    where
        F: FnMut(&Self::ListId, NonZeroUsize) + Send,
    {
        async move {
            let inserted = NonZeroUsize::new(points.len())
                .ok_or_else(|| ANNError::message("cannot insert an empty batch"))?;
            let mut staged = BTreeMap::new();
            let mut routed = BTreeMap::<u32, Vec<u32>>::new();
            for &(external, vector) in points {
                if vector.len() != self.store.dim
                    || !vector.iter().all(|&value| value.into().is_finite())
                {
                    return Err(ANNError::message("invalid input vector"));
                }
                let id = external
                    .checked_add(10_000)
                    .ok_or_else(|| ANNError::message("internal point ID overflow"))?;
                if self.store.points.contains_key(&id)
                    || self.staged.contains_key(&id)
                    || staged.contains_key(&id)
                {
                    return Err(ANNError::message("point already exists"));
                }
                let mut count = 0usize;
                {
                    let router = self.nearest();
                    router
                        .route(vector, ONE, |&list, _| {
                            count += 1;
                            routed.entry(list).or_default().push(id);
                        })
                        .await?;
                }
                if count != 1 {
                    return Err(ANNError::message("IVF routing returned no live centroid"));
                }
                staged.insert(id, vector.to_vec());
            }
            self.staged.extend(staged);
            for (list, members) in routed {
                let size = self
                    .store
                    .lists
                    .get(&list)
                    .ok_or_else(|| ANNError::message(format!("unknown list {list}")))?
                    .members
                    .len();
                self.pending.append(list, members)?;
                let size = NonZeroUsize::new(self.pending.list_size(list, size)?)
                    .ok_or_else(|| ANNError::message("appending must leave a nonempty list"))?;
                emit(&list, size);
            }
            Ok(inserted)
        }
    }

    fn fill<Itr>(
        &mut self,
        lists: Itr,
    ) -> impl SendFuture<ANNResult<(Self::Region<'_>, Self::Update<'_>)>>
    where
        Itr: Iterator<Item = Self::ListId> + Clone + Send + Sync,
    {
        async move {
            self.fills.set(self.fills.get() + 1);
            if self.store.fail_fill {
                return Err(ANNError::message("injected fill failure"));
            }
            self.filled_lists.extend(lists.clone());
            self.sources.clear();
            for list in lists {
                let stored = self
                    .store
                    .lists
                    .get(&list)
                    .ok_or_else(|| ANNError::message(format!("unknown list {list}")))?;
                let members: Vec<_> = self.pending.members(list, &stored.members)?.collect();
                let mut vectors =
                    rowmajor::Owned::from_element(members.len(), self.dim().get(), 0.0);
                for (&id, row) in members.iter().zip(vectors.rows_mut()) {
                    if let Some(vector) = self.staged.get(&id) {
                        for (dst, &src) in row.iter_mut().zip(vector) {
                            *dst = src.into();
                        }
                    } else {
                        let vector = self
                            .store
                            .points
                            .get(&id)
                            .ok_or_else(|| ANNError::message(format!("missing point {id}")))?;
                        row.copy_from_slice(vector);
                    }
                }
                self.sources.insert(
                    list,
                    Source {
                        centroid: stored.centroid.clone(),
                        members,
                        vectors,
                    },
                );
            }
            let mut update = Update::new(
                &self.sources,
                self.store.dim,
                &mut self.pending,
                &mut self.created,
            );
            update.fail_create = self.store.fail_create;
            update.fail_set_members = self.store.fail_set_members;
            update.fail_retire = self.store.fail_retire;
            Ok((Region::new(&self.sources), update))
        }
    }

    fn finish(mut self) -> impl SendFuture<ANNResult<()>> {
        async move {
            if self.store.fail_finish {
                return Err(ANNError::message("injected finish failure"));
            }
            let created = u32::try_from(self.created.len())
                .map_err(|_| ANNError::message("list ID overflow"))?;
            let next_list = self
                .store
                .next_list
                .checked_add(created)
                .ok_or_else(|| ANNError::message("list ID overflow"))?;
            for (id, vector) in self.staged {
                self.store
                    .points
                    .insert(id, vector.into_iter().map(Into::into).collect());
            }
            for (id, list) in (self.store.next_list..next_list).zip(self.created) {
                self.store.lists.insert(id, list);
            }
            for (id, change) in self.pending.into_changes() {
                match change {
                    Membership::Append(members) => {
                        self.store
                            .lists
                            .get_mut(&id)
                            .unwrap()
                            .members
                            .extend(members);
                    }
                    Membership::Replace(members) => {
                        self.store.lists.get_mut(&id).unwrap().members = members;
                    }
                    Membership::Retire => {
                        self.store.lists.remove(&id).unwrap();
                    }
                }
            }
            self.store.next_list = next_list;
            self.store.finishes += 1;
            self.store.fills += self.fills.get();
            self.store.filled_lists = self.filled_lists;
            Ok(())
        }
    }
}

pub(super) struct Strategy {
    pub(super) store: Mutex<Store>,
}

impl Strategy {
    pub(super) fn new(store: Store) -> Self {
        Self {
            store: Mutex::new(store),
        }
    }

    pub(super) fn store(&self) -> MutexGuard<'_, Store> {
        self.store.try_lock().unwrap()
    }
}

impl HasIds for Strategy {
    type Id = u32;
    type ListId = u32;
}

impl<'a, T: Copy + Into<f32> + Send + Sync> BuildStrategy<'a, &[(u32, &[T])], DefaultContext>
    for Strategy
{
    type Accessor = Build<'a, T>;
    type Error = ANNError;

    fn build_accessor(&'a self, _: &'a DefaultContext) -> ANNResult<Self::Accessor> {
        Ok(Build {
            store: self.store.try_lock().map_err(ANNError::new)?,
            pending: PendingLists::default(),
            created: Vec::new(),
            staged: BTreeMap::new(),
            sources: BTreeMap::new(),
            fills: Cell::new(0),
            filled_lists: Vec::new(),
        })
    }
}

pub(super) struct Search<'p, 'q, T> {
    pub(super) store: MutexGuard<'p, Store>,
    pub(super) query: &'q [T],
}

impl<T> HasIds for Search<'_, '_, T> {
    type Id = u32;
    type ListId = u32;
}

impl<'q, T: Copy + Into<f32> + Send + Sync> SearchAccessor<&'q [T]> for Search<'_, 'q, T> {
    fn scan<A>(&self, nprobe: NonZeroUsize, results: &mut A) -> impl SendFuture<ANNResult<()>>
    where
        A: Accumulator<Self::Id>,
    {
        async move {
            if self.store.fail_route {
                return Err(ANNError::message("injected routing failure"));
            }
            let coarse = Coarse {
                stored: &self.store.lists,
                fail_route: false,
                fail_nearest: false,
            };
            let mut selected = Vec::new();
            coarse.select(self.query, nprobe, |_| true, |id, _| selected.push(*id));

            for list in selected {
                let source = self
                    .store
                    .lists
                    .get(&list)
                    .ok_or_else(|| ANNError::message(format!("unknown list {list}")))?;
                let mut partial = results.empty_like();
                for id in &source.members {
                    let vector = self
                        .store
                        .points
                        .get(id)
                        .ok_or_else(|| ANNError::message(format!("missing point {id}")))?;
                    partial.push(*id, squared_distance(self.query, vector));
                }
                results.merge(partial);
            }
            Ok(())
        }
    }
}

impl<'a, 'q, T: Copy + Into<f32> + Send + Sync> SearchStrategy<'a, &'q [T], DefaultContext>
    for Strategy
{
    type Accessor = Search<'a, 'q, T>;
    type Error = ANNError;

    fn search_accessor(
        &'a self,
        _: &'a DefaultContext,
        query: &'q [T],
    ) -> ANNResult<Self::Accessor> {
        let store = self.store.try_lock().map_err(ANNError::new)?;
        if query.len() != store.dim || !query.iter().all(|&value| value.into().is_finite()) {
            return Err(ANNError::message("invalid query vector"));
        }
        Ok(Search { store, query })
    }
}

pub(super) fn matrix<const D: usize>(rows: &[[f32; D]]) -> rowmajor::Owned<f32> {
    let data: Box<[f32]> = rows.iter().flatten().copied().collect();
    rowmajor::Owned::try_from_data(data, rows.len(), D).unwrap()
}

pub(super) fn index(store: Store, threshold: usize) -> (IVFIndex<DefaultContext>, Strategy) {
    index_with_neighbors(store, threshold, 0)
}

pub(super) fn index_with_neighbors(
    store: Store,
    threshold: usize,
    neighbors: usize,
) -> (IVFIndex<DefaultContext>, Strategy) {
    let index = IVFIndex::new(Config {
        split_threshold: NonZeroUsize::new(threshold).unwrap(),
        reassign_count: NonZeroUsize::new(neighbors)
            .map_or(ReassignCount::None, ReassignCount::Count),
        split_iterations: NonZeroUsize::new(5).unwrap(),
        seed: 7,
    })
    .unwrap();
    (index, Strategy::new(store))
}

pub(super) fn seeded() -> Store {
    Store {
        dim: 2,
        points: [
            (10_000, vec![0.0, 0.0]),
            (10_001, vec![2.0, 0.0]),
            (10_002, vec![10.0, 0.0]),
            (10_003, vec![12.0, 0.0]),
            (10_004, vec![100.0, 0.0]),
        ]
        .into(),
        lists: [
            (
                1,
                StoredList {
                    centroid: vec![1.0, 0.0],
                    members: vec![10_000, 10_001],
                },
            ),
            (
                2,
                StoredList {
                    centroid: vec![11.0, 0.0],
                    members: vec![10_002, 10_003],
                },
            ),
            (
                3,
                StoredList {
                    centroid: vec![100.0, 0.0],
                    members: vec![10_004],
                },
            ),
        ]
        .into(),
        next_list: 4,
        ..Store::default()
    }
}

pub(super) fn require_send<F: Future + Send>(future: F) -> F {
    future
}
