/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A pedantic in-memory provider for testing the IVF index, written as a reference for
//! providers.

use std::{
    cmp,
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    ops::Range,
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use diskann_utils::{
    future::SendFuture,
    views::rowmajor::{self, Matrix},
};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use thiserror::Error;

use crate::{
    ANNResult, convert_error,
    error::message,
    internal::counter::Counter,
    ivf::{
        region::{self, List},
        traits::{self, Accumulator},
    },
};

//////////////
// Provider //
//////////////

/// An in-memory IVF provider.
///
///
/// [`Metrics`] counts the calls served, and [`Provider::with_fault`] makes one fail.
#[derive(Debug)]
pub struct Provider {
    /// The lists that searches read and that the next build starts from.
    published: Mutex<Arc<Snapshot>>,
    /// Whether a build is live.
    building: AtomicBool,
    counters: Counters,
    fault: Option<Fault>,
}

impl Provider {
    /// Construct a provider that publishes `snapshot`.
    ///
    /// All counters will be initialized to zero.
    pub fn new(snapshot: Snapshot) -> Self {
        Self {
            published: Mutex::new(Arc::new(snapshot)),
            building: AtomicBool::new(false),
            counters: Counters::new(),
            fault: None,
        }
    }

    /// Make the `nth` call of kind `call` fail, counting from zero.
    pub fn with_fault(self, call: Call, nth: usize) -> Self {
        Self {
            fault: Some(Fault::new(call, nth)),
            ..self
        }
    }

    /// Return the published snapshot.
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.published().clone()
    }

    /// Return the calls served so far, including failed ones and those of abandoned builds.
    ///
    /// Each count is read separately, so calls made meanwhile may show in some counts and not
    /// in others.
    pub fn metrics(&self) -> Metrics {
        self.counters.metrics()
    }

    /// Count a call of kind `call`, and fail it if it is the faulty one.
    fn enter(&self, call: Call) -> ANNResult<()> {
        self.counters.get(call).increment();
        match &self.fault {
            Some(fault) => fault.check(call),
            None => Ok(()),
        }
    }

    /// Apply `changes` to the published snapshot: in place, unless a search still holds it.
    ///
    /// The lease kept other builds from publishing, so the changes were made over the
    /// published snapshot.
    fn publish(&self, changes: Changes) {
        Arc::make_mut(&mut self.published()).apply(changes);
    }

    /// Lock the published snapshot. Nothing panics while holding the lock, so poisoning is
    /// ignored.
    fn published(&self) -> MutexGuard<'_, Arc<Snapshot>> {
        self.published
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl traits::HasIds for Provider {
    type Id = u32;
    type ListId = u32;
}

//---------//
// Metrics //
//---------//

/// A kind of call that a [`Provider`] serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Call {
    Route,
    Find,
    Fill,
    AppendMembers,
    RemoveMembers,
    Retire,
    Create,
    /// The finish of a region.
    RegionFinish,
    /// The finish of a build.
    Finish,
    /// The scan of one list by a search.
    Scan,
}

/// The calls a [`Provider`] has served.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct Metrics {
    pub route: usize,
    pub find: usize,
    pub fill: usize,
    pub append_members: usize,
    pub remove_members: usize,
    pub retire: usize,
    pub create: usize,
    pub region_finish: usize,
    pub finish: usize,
    /// Lists scanned by searches.
    pub scan: usize,
    /// Distances computed by scans.
    pub distances: usize,
}

#[cfg(test)]
crate::test::cmp::verbose_eq!(Metrics {
    route,
    find,
    fill,
    append_members,
    remove_members,
    retire,
    create,
    region_finish,
    finish,
    scan,
    distances,
});

/// The calls a provider serves, counted atomically.
#[derive(Debug)]
struct Counters {
    route: Counter,
    find: Counter,
    fill: Counter,
    append_members: Counter,
    remove_members: Counter,
    retire: Counter,
    create: Counter,
    region_finish: Counter,
    finish: Counter,
    scan: Counter,
    distances: Counter,
}

impl Counters {
    /// Construct counters that are all zero.
    fn new() -> Self {
        Self {
            route: Counter::new(),
            find: Counter::new(),
            fill: Counter::new(),
            append_members: Counter::new(),
            remove_members: Counter::new(),
            retire: Counter::new(),
            create: Counter::new(),
            region_finish: Counter::new(),
            finish: Counter::new(),
            scan: Counter::new(),
            distances: Counter::new(),
        }
    }

    /// Return the counter of calls of kind `call`.
    fn get(&self, call: Call) -> &Counter {
        match call {
            Call::Route => &self.route,
            Call::Find => &self.find,
            Call::Fill => &self.fill,
            Call::AppendMembers => &self.append_members,
            Call::RemoveMembers => &self.remove_members,
            Call::Retire => &self.retire,
            Call::Create => &self.create,
            Call::RegionFinish => &self.region_finish,
            Call::Finish => &self.finish,
            Call::Scan => &self.scan,
        }
    }

    /// Return the counts so far.
    fn metrics(&self) -> Metrics {
        Metrics {
            route: self.route.value(),
            find: self.find.value(),
            fill: self.fill.value(),
            append_members: self.append_members.value(),
            remove_members: self.remove_members.value(),
            retire: self.retire.value(),
            create: self.create.value(),
            region_finish: self.region_finish.value(),
            finish: self.finish.value(),
            scan: self.scan.value(),
            distances: self.distances.value(),
        }
    }
}

//-------//
// Fault //
//-------//

/// Fails the `nth` call of one kind, counting from zero.
#[derive(Debug)]
struct Fault {
    call: Call,
    nth: usize,
    /// The calls of kind `call` checked so far.
    seen: AtomicUsize,
}

impl Fault {
    /// Construct a fault that fails the `nth` call of kind `call`.
    fn new(call: Call, nth: usize) -> Self {
        Self {
            call,
            nth,
            seen: AtomicUsize::new(0),
        }
    }

    /// Fail if `call` is the faulty call.
    fn check(&self, call: Call) -> ANNResult<()> {
        if call == self.call && self.seen.fetch_add(1, Ordering::Relaxed) == self.nth {
            Err(message!("injected {call:?} failure"))
        } else {
            Ok(())
        }
    }
}

//////////////
// Snapshot //
//////////////

/// The lists and points that a provider publishes.
///
/// Vectors are stored by point ID, so changes can name points by ID alone.
#[derive(Debug, Clone)]
pub struct Snapshot {
    dim: usize,
    lists: BTreeMap<u32, StoredList>,
    /// The vector of every point.
    vectors: BTreeMap<u32, Vec<f32>>,
    /// The ID of the next list created. IDs are never reused.
    next_id: u32,
}

impl Snapshot {
    /// Construct a snapshot of `dim`-dimensional vectors with no lists.
    pub fn new(dim: NonZeroUsize) -> Self {
        Self {
            dim: dim.get(),
            lists: BTreeMap::new(),
            vectors: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// Return the dimension of the vectors and centroids.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the live lists, by ID.
    pub fn lists(&self) -> &BTreeMap<u32, StoredList> {
        &self.lists
    }

    /// Return the vector of point `id`, if it exists.
    pub fn vector(&self, id: u32) -> Option<&[f32]> {
        self.vectors.get(&id).map(Vec::as_slice)
    }

    /// Return the ID of the next list created.
    pub fn next_list(&self) -> u32 {
        self.next_id
    }

    /// Apply the changes of a build.
    fn apply(&mut self, changes: Changes) {
        changes.lists.apply(&mut self.lists);
        self.vectors.extend(changes.vectors);
        self.next_id = changes.next_id;
    }
}

/// A list: its centroid and its members.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredList {
    centroid: Vec<f32>,
    members: Vec<u32>,
}

impl StoredList {
    /// Return the centroid.
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    /// Return the members.
    pub fn members(&self) -> &[u32] {
        &self.members
    }
}

/// The accessor of one build.
///
/// It records its changes in an overlay over the snapshot published when it began, copying a
/// list when it first changes it, and applies them when it finishes: in place, unless a search
/// still holds the snapshot, which is then copied.
#[derive(Debug)]
pub struct Build<'p> {
    provider: &'p Provider,
    overlay: Overlay,
    _lease: Lease<'p>,
}

impl<'p> Build<'p> {
    /// Begin a build over the published snapshot, failing while another build is live.
    fn new(provider: &'p Provider) -> Result<Self, BuildIsLive> {
        // Take the lease first, so that no other build publishes over the snapshot read.
        let lease = Lease::take(&provider.building)?;
        Ok(Self {
            provider,
            overlay: Overlay::new(provider.snapshot()),
            _lease: lease,
        })
    }
}

impl traits::HasIds for Build<'_> {
    type Id = u32;
    type ListId = u32;
}

impl<'p, 'b, 'v> traits::BuildAccessor<&'b [(u32, &'v [f32])]> for Build<'p> {
    type VectorsRef<'a> = rowmajor::Ref<'a, f32>;

    type Region<'a>
        = Region<'a>
    where
        Self: 'a;

    fn find<P, F>(
        &self,
        id: u32,
        nprobe: NonZeroUsize,
        pred: P,
        mut emit: F,
    ) -> impl SendFuture<ANNResult<usize>>
    where
        P: Fn(u32) -> bool + Send,
        F: FnMut(u32, f32) + Send,
    {
        async move {
            self.provider.enter(Call::Find)?;
            let list = self
                .overlay
                .list(id)
                .ok_or_else(|| message!("list {id} is not live"))?;

            let found = nearest(self.overlay.lists(), &list.centroid, nprobe.get(), pred);
            for &(id, distance) in &found {
                emit(id, distance);
            }
            Ok(found.len())
        }
    }

    fn route<F>(
        &mut self,
        points: &'b [(u32, &'v [f32])],
        emit: F,
    ) -> impl SendFuture<ANNResult<NonZeroUsize>>
    where
        F: FnMut(u32, NonZeroUsize) + Send,
    {
        async move {
            self.provider.enter(Call::Route)?;
            let count = NonZeroUsize::new(points.len())
                .ok_or_else(|| message!("cannot insert an empty batch"))?;

            self.overlay.route(points, emit)?;
            Ok(count)
        }
    }

    fn fill<Itr>(&mut self, lists: Itr) -> impl SendFuture<ANNResult<Region<'_>>>
    where
        Itr: Iterator<Item = u32> + Clone + Send + Sync,
    {
        async move {
            self.provider.enter(Call::Fill)?;
            Region::new(self.provider, &mut self.overlay, lists)
        }
    }

    fn finish(self) -> impl SendFuture<ANNResult<()>> {
        async move {
            self.provider.enter(Call::Finish)?;
            self.provider.publish(self.overlay.into_changes());
            Ok(())
        }
    }
}

//-------//
// Lease //
//-------//

/// Marks a build as live until it is dropped.
#[derive(Debug)]
struct Lease<'p>(&'p AtomicBool);

impl<'p> Lease<'p> {
    /// Take the lease of `building`, failing while another build holds it.
    fn take(building: &'p AtomicBool) -> Result<Self, BuildIsLive> {
        if building.swap(true, Ordering::Acquire) {
            Err(BuildIsLive)
        } else {
            Ok(Self(building))
        }
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// The error of [`build_accessor`](traits::BuildStrategy::build_accessor) while another
/// build is live.
#[derive(Debug, Clone, Copy, Error)]
#[error("another build is live")]
pub struct BuildIsLive;

convert_error!(BuildIsLive);

//---------//
// Overlay //
//---------//

/// The changes of a build over the snapshot it began from.
#[derive(Debug)]
struct Changes {
    /// The lists routed to, written or created, or `None` for those retired.
    lists: Edits,
    /// The vectors of the points routed.
    vectors: BTreeMap<u32, Vec<f32>>,
    /// The ID of the next list created.
    next_id: u32,
}

/// A snapshot as a build has changed it.
#[derive(Debug)]
struct Overlay {
    /// The snapshot published when the build began.
    base: Arc<Snapshot>,
    changes: Changes,
}

impl Overlay {
    /// Begin changing `base`.
    fn new(base: Arc<Snapshot>) -> Self {
        let changes = Changes {
            lists: Edits::default(),
            vectors: BTreeMap::new(),
            next_id: base.next_id,
        };
        Self { base, changes }
    }

    /// Return the dimension of the vectors and centroids.
    fn dim(&self) -> usize {
        self.base.dim
    }

    /// Return live list `id`, as the build has changed it.
    fn list(&self, id: u32) -> Option<&StoredList> {
        match self.changes.lists.get(id) {
            Some(edited) => edited,
            None => self.base.lists.get(&id),
        }
    }

    /// Return every live list, as the build has changed it.
    fn lists(&self) -> impl Iterator<Item = (&u32, &StoredList)> {
        let edits = &self.changes.lists;
        let unedited = self
            .base
            .lists
            .iter()
            .filter(|&(&id, _)| !edits.contains(id));
        unedited.chain(edits.live())
    }

    /// Return the vector of point `id`, if it exists.
    fn vector(&self, id: u32) -> Option<&[f32]> {
        let vector = self.changes.vectors.get(&id);
        vector
            .or_else(|| self.base.vectors.get(&id))
            .map(Vec::as_slice)
    }

    /// Return live list `id` to change, copying it into the changes on the first change.
    fn list_mut(&mut self, id: u32) -> Option<&mut StoredList> {
        let base = &self.base.lists;
        self.changes.lists.write(id, || base.get(&id).cloned())
    }

    /// Add each of `points` to the list with the closest centroid, and call `emit` with every
    /// list that receives points and its new size.
    fn route(
        &mut self,
        points: &[(u32, &[f32])],
        mut emit: impl FnMut(u32, NonZeroUsize),
    ) -> ANNResult<()> {
        let mut routed = BTreeMap::<u32, Vec<u32>>::new();
        let mut batch = BTreeSet::new();
        for &(id, vector) in points {
            if self.vector(id).is_some() || !batch.insert(id) {
                return Err(message!("point {id} already exists"));
            }
            let list = closest(self.lists(), vector)
                .ok_or_else(|| message!("there is no list to route point {id} to"))?;
            routed.entry(list).or_default().push(id);
        }

        let vectors = points.iter().map(|&(id, vector)| (id, vector.to_vec()));
        self.changes.vectors.extend(vectors);

        for (id, ids) in routed {
            let list = self
                .list_mut(id)
                .ok_or_else(|| message!("list {id} is not live"))?;
            list.members.extend(ids);
            if let Some(size) = NonZeroUsize::new(list.members.len()) {
                emit(id, size);
            }
        }
        Ok(())
    }

    /// Allocate the ID of a new list.
    fn next_list_id(&mut self) -> ANNResult<u32> {
        let id = self.changes.next_id;
        self.changes.next_id = id
            .checked_add(1)
            .ok_or_else(|| message!("list IDs are exhausted"))?;
        Ok(id)
    }

    /// Add the edits of a finished region.
    fn merge(&mut self, edits: Edits) {
        self.changes.lists.extend(edits);
    }

    /// Return the changes, dropping the reference to the base so that publishing can change
    /// it in place.
    fn into_changes(self) -> Changes {
        self.changes
    }
}

//-------//
// Edits //
//-------//

/// Lists written, created or retired over lists kept elsewhere. An edited list maps to its
/// new contents, or to `None` once retired.
#[derive(Debug, Default)]
struct Edits(BTreeMap<u32, Option<StoredList>>);

impl Edits {
    /// Return the edit of list `id`, or `None` if it was not edited.
    fn get(&self, id: u32) -> Option<Option<&StoredList>> {
        self.0.get(&id).map(Option::as_ref)
    }

    /// Return whether list `id` was edited.
    fn contains(&self, id: u32) -> bool {
        self.0.contains_key(&id)
    }

    /// Return the edited lists that are live.
    fn live(&self) -> impl Iterator<Item = (&u32, &StoredList)> {
        self.0
            .iter()
            .filter_map(|(id, list)| Some((id, list.as_ref()?)))
    }

    /// Return list `id` to write, starting from `original` on the first write, or `None` if
    /// the list is retired.
    fn write(
        &mut self,
        id: u32,
        original: impl FnOnce() -> Option<StoredList>,
    ) -> Option<&mut StoredList> {
        self.0.entry(id).or_insert_with(original).as_mut()
    }

    /// Record list `id`, created as `list`.
    fn create(&mut self, id: u32, list: StoredList) {
        self.0.insert(id, Some(list));
    }

    /// Retire list `id`, and return whether it was live.
    fn retire(&mut self, id: u32) -> bool {
        !matches!(self.0.insert(id, None), Some(None))
    }

    /// Add the edits of `later`, which supersede these.
    fn extend(&mut self, later: Self) {
        self.0.extend(later.0);
    }

    /// Apply the edits to `lists`.
    fn apply(self, lists: &mut BTreeMap<u32, StoredList>) {
        for (id, list) in self.0 {
            if let Some(list) = list {
                lists.insert(id, list);
            } else {
                lists.remove(&id);
            }
        }
    }
}

////////////
// Region //
////////////

/// The lists of one fill.
///
/// The fill gathers the vectors of the lists into one matrix, which the region owns and lends
/// out by list. Reads see the lists as filled, and writes are recorded as drafts, which
/// [`finish`](region::Region::finish) adds to the build's changes; dropping the region
/// discards them.
#[derive(Debug)]
pub struct Region<'a> {
    provider: &'a Provider,
    overlay: &'a mut Overlay,
    /// The filled lists, which reads and writes may target, and their vectors.
    filled: Filled,
    /// The lists this region wrote to or created, or `None` for those it retired.
    drafts: Edits,
}

impl<'a> Region<'a> {
    /// Fill `lists` from `overlay`, failing if a list is not live or a member has no vector.
    fn new(
        provider: &'a Provider,
        overlay: &'a mut Overlay,
        lists: impl Iterator<Item = u32>,
    ) -> ANNResult<Self> {
        let filled = Filled::new(overlay, lists)?;
        Ok(Self {
            provider,
            overlay,
            filled,
            drafts: Edits::default(),
        })
    }

    /// Return list `id` as filled, or `None` if it was not filled.
    fn get(&self, id: u32) -> Option<&StoredList> {
        if self.filled.contains(id) {
            self.overlay.list(id)
        } else {
            None
        }
    }

    /// Fail unless list `id` was filled.
    fn check_filled(&self, id: u32) -> ANNResult<()> {
        if self.filled.contains(id) {
            Ok(())
        } else {
            Err(message!("list {id} was not filled"))
        }
    }

    /// Return the draft of list `id`, copying the list as filled on the first write.
    fn draft(&mut self, id: u32) -> ANNResult<&mut StoredList> {
        self.check_filled(id)?;
        let overlay = &*self.overlay;
        self.drafts
            .write(id, || overlay.list(id).cloned())
            .ok_or_else(|| message!("list {id} is retired"))
    }
}

impl region::Region<u32, u32> for Region<'_> {
    type VectorsRef<'a> = rowmajor::Ref<'a, f32>;

    type Vectors<'a>
        = rowmajor::Ref<'a, f32>
    where
        Self: 'a;

    fn get_list(&self, id: u32) -> Option<List<'_, u32>> {
        self.get(id).map(|list| List {
            centroid: &list.centroid,
            members: &list.members,
        })
    }

    fn get_vectors(&self, id: u32) -> Option<rowmajor::Ref<'_, f32>> {
        self.filled.vectors(id)
    }

    fn create(&mut self, centroid: &[f32], members: &[u32]) -> ANNResult<u32> {
        self.provider.enter(Call::Create)?;
        let id = self.overlay.next_list_id()?;
        let list = StoredList {
            centroid: centroid.to_vec(),
            members: members.to_vec(),
        };
        self.drafts.create(id, list);
        Ok(id)
    }

    fn retire(&mut self, list: u32) -> ANNResult<()> {
        self.provider.enter(Call::Retire)?;
        self.check_filled(list)?;
        if self.drafts.retire(list) {
            Ok(())
        } else {
            Err(message!("list {list} is retired"))
        }
    }

    fn append_members(&mut self, list: u32, ids: &[u32]) -> ANNResult<()> {
        self.provider.enter(Call::AppendMembers)?;
        let draft = self.draft(list)?;
        if let Some(id) = ids.iter().find(|id| draft.members.contains(id)) {
            return Err(message!("point {id} is already in list {list}"));
        }

        draft.members.extend_from_slice(ids);
        Ok(())
    }

    fn remove_members(&mut self, list: u32, ids: &[u32]) -> ANNResult<()> {
        self.provider.enter(Call::RemoveMembers)?;
        let ids = distinct(ids)?;
        let draft = self.draft(list)?;
        if let Some(id) = ids.iter().find(|id| !draft.members.contains(id)) {
            return Err(message!("point {id} is not in list {list}"));
        }

        draft.members.retain(|id| !ids.contains(id));
        Ok(())
    }

    fn finish(self) -> impl SendFuture<ANNResult<()>> {
        async move {
            self.provider.enter(Call::RegionFinish)?;
            self.overlay.merge(self.drafts);
            Ok(())
        }
    }
}

//--------//
// Filled //
//--------//

/// The lists filled into a region, with their vectors gathered into one matrix.
#[derive(Debug)]
struct Filled {
    /// The rows of each filled list in `vectors`.
    rows: BTreeMap<u32, Range<usize>>,
    vectors: rowmajor::Owned<f32>,
}

impl Filled {
    /// Gather the vectors of `lists` from `overlay`, failing if a list is not live or a
    /// member has no vector.
    fn new(overlay: &Overlay, lists: impl Iterator<Item = u32>) -> ANNResult<Self> {
        let mut rows = BTreeMap::new();
        let mut data = Vec::new();
        let mut end = 0;
        for id in lists {
            let list = overlay
                .list(id)
                .ok_or_else(|| message!("list {id} is not live"))?;
            for &point in &list.members {
                let vector = overlay
                    .vector(point)
                    .ok_or_else(|| message!("point {point} has no vector"))?;
                data.extend_from_slice(vector);
            }

            let start = end;
            end += list.members.len();
            rows.insert(id, start..end);
        }

        let vectors = rowmajor::Owned::try_from_data(data.into(), end, overlay.dim())
            .map_err(|err| message!("cannot gather the vectors of the filled lists: {err}"))?;
        Ok(Self { rows, vectors })
    }

    /// Return whether list `id` was filled.
    fn contains(&self, id: u32) -> bool {
        self.rows.contains_key(&id)
    }

    /// Return the vectors of filled list `id`.
    fn vectors(&self, id: u32) -> Option<rowmajor::Ref<'_, f32>> {
        let rows = self.rows.get(&id)?;
        self.vectors.subview(rows.clone())
    }
}

////////////
// Search //
////////////

/// The accessor of one search, over the snapshot published when it began.
#[derive(Debug)]
pub struct Search<'p> {
    provider: &'p Provider,
    snapshot: Arc<Snapshot>,
    query: Vec<f32>,
}

impl<'p> Search<'p> {
    /// Begin a search for `query` over the published snapshot.
    fn new(provider: &'p Provider, query: &[f32]) -> Self {
        Self {
            provider,
            snapshot: provider.snapshot(),
            query: query.to_vec(),
        }
    }
}

impl traits::HasIds for Search<'_> {
    type Id = u32;
    type ListId = u32;
}

impl traits::SearchAccessor for Search<'_> {
    fn scan<A>(&self, nprobe: NonZeroUsize, results: &mut A) -> impl SendFuture<ANNResult<()>>
    where
        A: Accumulator<Self::Id>,
    {
        async move {
            let Snapshot { lists, vectors, .. } = &*self.snapshot;
            let mut distances = self.provider.counters.distances.local();
            for (id, _) in nearest(lists, &self.query, nprobe.get(), |_| true) {
                self.provider.enter(Call::Scan)?;
                let members = &lists[&id].members;
                for &member in members {
                    results.push(member, distance(&self.query, &vectors[&member]));
                }
                distances.increment_by(members.len());
            }
            Ok(())
        }
    }
}

/////////////
// Helpers //
/////////////

/// Return up to `n` of `lists` that `pred` accepts with their distances, nearest `query`
/// first and then by ID.
fn nearest<'a>(
    lists: impl IntoIterator<Item = (&'a u32, &'a StoredList)>,
    query: &[f32],
    n: usize,
    pred: impl Fn(u32) -> bool,
) -> Vec<(u32, f32)> {
    let mut found: Vec<_> = lists
        .into_iter()
        .filter(|&(&id, _)| pred(id))
        .map(|(&id, list)| (id, distance(query, &list.centroid)))
        .collect();

    found.sort_by(nearer);
    found.truncate(n);
    found
}

/// Return the list with the centroid nearest `query`, breaking ties by the lowest ID.
fn closest<'a>(
    lists: impl IntoIterator<Item = (&'a u32, &'a StoredList)>,
    query: &[f32],
) -> Option<u32> {
    let found = lists
        .into_iter()
        .map(|(&id, list)| (id, distance(query, &list.centroid)));
    found.min_by(nearer).map(|(id, _)| id)
}

/// Order `(id, distance)` pairs by distance, then by ID.
fn nearer(a: &(u32, f32), b: &(u32, f32)) -> cmp::Ordering {
    a.1.total_cmp(&b.1).then(a.0.cmp(&b.0))
}

/// Return `ids` as a set, failing if one repeats.
fn distinct(ids: &[u32]) -> ANNResult<BTreeSet<u32>> {
    let mut set = BTreeSet::new();
    for &id in ids {
        if !set.insert(id) {
            return Err(message!("point {id} repeats"));
        }
    }
    Ok(set)
}

/// Return the squared L2 distance between `x` and `y`.
fn distance(x: &[f32], y: &[f32]) -> f32 {
    SquaredL2::evaluate(x, y)
}
