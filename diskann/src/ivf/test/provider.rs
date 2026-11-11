/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A reference in-memory provider for testing IVF algorithm contracts.

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

/// A reference provider for IVF algorithm tests.
///
/// * Only one build can be live; successful finish publishes its changes atomically.
/// * Searches keep the snapshot captured when their accessor was created.
/// * Centroids are ranked by squared L2 distance, then list ID.
///
/// Regions enforce filled-list access; see [`Region`] for read/write visibility.
/// Use [`Snapshot::is_consistent`] to check the final state.
///
/// Counters and one-shot faults support protocol and recovery assertions.
#[derive(Debug)]
pub struct Provider {
    published: Mutex<Arc<Snapshot>>,
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

    /// Fail the zero-based `nth` subsequent call of kind `call`, once.
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

    /// Publish over the leased base, copying only if another snapshot owner remains.
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

/// A provider call kind that can be counted or fault-injected.
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

/// Provider calls and point distances accumulated across operations.
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
    /// List scans attempted, including injected failures.
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

/// A one-shot fault counted independently for one call kind.
#[derive(Debug)]
struct Fault {
    call: Call,
    nth: usize,
    /// The calls of kind `call` checked so far.
    seen: AtomicUsize,
}

impl Fault {
    fn new(call: Call, nth: usize) -> Self {
        Self {
            call,
            nth,
            seen: AtomicUsize::new(0),
        }
    }

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
    vectors: BTreeMap<u32, Vec<f32>>,
    /// Next list ID; IDs allocated by a published build are not reused.
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

    /// Check dimensions and list-ID bounds, and require every stored point to appear in
    /// exactly one live list. Every member must have a vector.
    ///
    /// # Errors
    ///
    /// Returns the first violation. Not run automatically when a build publishes.
    pub fn is_consistent(&self) -> ANNResult<()> {
        let dim = self.dim;

        // The lists holding each member, once per membership.
        let mut lists_of = BTreeMap::<u32, Vec<u32>>::new();
        for (&id, list) in &self.lists {
            if id >= self.next_id {
                return Err(message!(
                    "list {id} is not below the next list ID {}",
                    self.next_id
                ));
            }
            if list.centroid.len() != dim {
                return Err(message!(
                    "list {id} has a centroid of {} entries, expected {dim}",
                    list.centroid.len()
                ));
            }
            for &member in &list.members {
                if !self.vectors.contains_key(&member) {
                    return Err(message!("member {member} of list {id} has no vector"));
                }
                lists_of.entry(member).or_default().push(id);
            }
        }

        for (&point, vector) in &self.vectors {
            if vector.len() != dim {
                return Err(message!(
                    "point {point} has a vector of {} entries, expected {dim}",
                    vector.len()
                ));
            }
            match lists_of.get(&point).map(Vec::as_slice) {
                Some([_]) => {}
                None => return Err(message!("point {point} is in no list")),
                Some(lists) => return Err(message!("point {point} is in lists {lists:?}")),
            }
        }

        Ok(())
    }

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

/// An exclusive build with unpublished copy-on-write edits.
///
/// [`traits::BuildAccessor::finish`] publishes all edits; dropping the build abandons them
/// and releases the lease. Publication reuses the snapshot unless another reader holds it.
#[derive(Debug)]
pub struct Build<'p> {
    provider: &'p Provider,
    overlay: Overlay,
    _lease: Lease<'p>,
}

impl<'p> Build<'p> {
    /// Begin a build over the published snapshot, failing while another build is live.
    pub fn new(provider: &'p Provider) -> Result<Self, BuildIsLive> {
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

/// The error of [`Build::new`] while another build is live.
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
    lists: Edits,
    vectors: BTreeMap<u32, Vec<f32>>,
    next_id: u32,
}

/// A read-through view where build edits take precedence over the base snapshot.
#[derive(Debug)]
struct Overlay {
    base: Arc<Snapshot>,
    changes: Changes,
}

impl Overlay {
    fn new(base: Arc<Snapshot>) -> Self {
        let changes = Changes {
            lists: Edits::default(),
            vectors: BTreeMap::new(),
            next_id: base.next_id,
        };
        Self { base, changes }
    }

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

    /// Return each live list once, preferring edits over the base snapshot.
    fn lists(&self) -> impl Iterator<Item = (&u32, &StoredList)> {
        let edits = &self.changes.lists;
        let unedited = self
            .base
            .lists
            .iter()
            .filter(|&(&id, _)| !edits.contains(id));
        unedited.chain(edits.live())
    }

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

    fn next_list_id(&mut self) -> ANNResult<u32> {
        let id = self.changes.next_id;
        self.changes.next_id = id
            .checked_add(1)
            .ok_or_else(|| message!("list IDs are exhausted"))?;
        Ok(id)
    }

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
    /// Distinguish unedited lists (`None`) from retired ones (`Some(None)`).
    fn get(&self, id: u32) -> Option<Option<&StoredList>> {
        self.0.get(&id).map(Option::as_ref)
    }

    fn contains(&self, id: u32) -> bool {
        self.0.contains_key(&id)
    }

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

/// A filled view with buffered list edits.
///
/// Only filled lists can be read, edited, or retired; newly created lists are outside
/// this set. Reads keep the fill-time state, even after writes or retirement.
/// [`region::Region::finish`] merges drafts into the build. Dropping the region discards
/// drafts but does not roll back list IDs allocated within that build.
///
/// Vector reads borrow from one matrix gathered at fill time.
#[derive(Debug)]
pub struct Region<'a> {
    provider: &'a Provider,
    overlay: &'a mut Overlay,
    filled: Filled,
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

    fn contains(&self, id: u32) -> bool {
        self.rows.contains_key(&id)
    }

    fn vectors(&self, id: u32) -> Option<rowmajor::Ref<'_, f32>> {
        let rows = self.rows.get(&id)?;
        self.vectors.subview(rows.clone())
    }
}

////////////
// Search //
////////////

/// A search pinned to the snapshot captured at construction.
#[derive(Debug)]
pub struct Search<'p> {
    provider: &'p Provider,
    snapshot: Arc<Snapshot>,
    query: Vec<f32>,
}

impl<'p> Search<'p> {
    /// Begin a search for `query` over the published snapshot.
    pub fn new(provider: &'p Provider, query: &[f32]) -> Self {
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

/// Filter before truncating to `n`; order accepted centroids by squared L2, then list ID.
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

fn distance(x: &[f32], y: &[f32]) -> f32 {
    SquaredL2::evaluate(x, y)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        ANNError,
        ivf::{
            region::Region as _,
            test::harness::{assert_abandoned, assert_fails, assert_scans, batch, layout, line},
            traits::{BuildAccessor as _, SearchAccessor as _},
        },
        test::cmp::assert_eq_verbose,
        utils::ONE,
    };
    use diskann_utils::assert_contains;

    async fn provider() -> Provider {
        line(&[(0.0, &[0.0, 1.0]), (10.0, &[10.0]), (20.0, &[])]).await
    }

    /// Collect routed list sizes, checking point counts and that errors emit nothing.
    async fn route(build: &mut Build<'_>, points: &[(u32, f32)]) -> ANNResult<Vec<(u32, usize)>> {
        let mut reported = Vec::new();
        let routed = build
            .route(&batch(points)[..], |list, size| {
                reported.push((list, size.get()))
            })
            .await;

        match routed {
            Ok(count) => {
                assert_eq!(count.get(), points.len(), "route should count every point");
                Ok(reported)
            }
            Err(err) => {
                assert!(reported.is_empty(), "a failed route should report no list");
                Err(err)
            }
        }
    }

    /// Collect ranked lists and check that `find`'s count matches its emissions.
    async fn find(
        build: &Build<'_>,
        id: u32,
        nprobe: usize,
        pred: impl Fn(u32) -> bool + Send,
    ) -> ANNResult<Vec<(u32, f32)>> {
        let nprobe = NonZeroUsize::new(nprobe).unwrap();
        let mut found = Vec::new();
        let count = build
            .find(id, nprobe, pred, |list, distance| {
                found.push((list, distance))
            })
            .await?;

        assert_eq!(count, found.len(), "find should return how many it emitted");
        Ok(found)
    }

    /// An accumulator that records every push, in order.
    #[derive(Debug, Default)]
    struct Pushes(Vec<(u32, f32)>);

    impl Accumulator<u32> for Pushes {
        fn empty_like(&self) -> Self {
            Self::default()
        }

        fn push(&mut self, id: u32, distance: f32) {
            self.0.push((id, distance));
        }

        fn merge(&mut self, other: Self) {
            self.0.extend(other.0);
        }
    }

    /// Record raw scan emissions, without top-k selection or reordering.
    async fn scan(search: &Search<'_>, nprobe: usize) -> ANNResult<Vec<(u32, f32)>> {
        let nprobe = NonZeroUsize::new(nprobe).unwrap();
        let mut pushes = Pushes::default();
        search.scan(nprobe, &mut pushes).await?;
        Ok(pushes.0)
    }

    //----------//
    // Snapshot //
    //----------//

    /// A change that breaks a snapshot.
    type Corrupt = fn(&mut Snapshot);

    #[tokio::test]
    async fn is_consistent_reports_each_violation() {
        let valid = line(&[(0.0, &[0.0, 1.0]), (5.0, &[5.0])]).await.snapshot();
        valid.is_consistent().unwrap();

        let cases: [(Corrupt, &str); 7] = [
            (|s| s.next_id = 1, "list 1 is not below the next list ID 1"),
            (
                |s| s.lists.get_mut(&1).unwrap().centroid.push(0.0),
                "list 1 has a centroid of 2 entries, expected 1",
            ),
            (
                |s| s.lists.get_mut(&1).unwrap().members.push(9),
                "member 9 of list 1 has no vector",
            ),
            (
                |s| s.vectors.get_mut(&2).unwrap().push(0.0),
                "point 2 has a vector of 2 entries, expected 1",
            ),
            (
                |s| {
                    s.vectors.insert(3, vec![3.0]);
                },
                "point 3 is in no list",
            ),
            (
                |s| s.lists.get_mut(&1).unwrap().members.push(0),
                "point 0 is in lists [0, 1]",
            ),
            (
                |s| s.lists.get_mut(&0).unwrap().members.push(0),
                "point 0 is in lists [0, 0]",
            ),
        ];
        for (corrupt, expected) in cases {
            let mut snapshot = Snapshot::clone(&valid);
            corrupt(&mut snapshot);
            assert_fails(snapshot.is_consistent(), expected);
        }
    }

    //-------//
    // Route //
    //-------//

    #[tokio::test]
    async fn route_appends_each_point_to_its_nearest_list() {
        let provider = provider().await;
        let mut build = Build::new(&provider).unwrap();

        // Point 4 at 5 is as near list 0 as list 1, and goes to the lower ID.
        let reported = route(&mut build, &[(3, 9.0), (4, 5.0), (5, -1.0)]).await;
        assert_eq!(
            reported.unwrap(),
            [(0, 4), (1, 2)],
            "each list that receives points is reported once, with its new size"
        );

        build.finish().await.unwrap();
        assert_eq!(
            layout(&provider.snapshot()),
            [
                (0, 0.0, vec![0.0, 1.0, 5.0, -1.0]),
                (1, 10.0, vec![10.0, 9.0]),
                (2, 20.0, vec![]),
            ]
        );
    }

    #[tokio::test]
    async fn route_rejects_bad_batches_without_changes() {
        let provider = provider().await;
        let before = layout(&provider.snapshot());

        let cases: [(&[(u32, f32)], &str); 3] = [
            (&[], "cannot insert an empty batch"),
            (&[(3, 1.0), (3, 2.0)], "point 3 already exists"),
            (&[(3, 1.0), (2, 1.0)], "point 2 already exists"),
        ];
        for (points, expected) in cases {
            let mut build = Build::new(&provider).unwrap();
            assert_fails(route(&mut build, points).await, expected);

            build.finish().await.unwrap();
            assert_eq!(layout(&provider.snapshot()), before);
        }

        let provider = line(&[]).await;
        let mut build = Build::new(&provider).unwrap();
        assert_fails(
            route(&mut build, &[(0, 1.0)]).await,
            "there is no list to route point 0 to",
        );
    }

    //------//
    // Fill //
    //------//

    #[tokio::test]
    async fn fill_lends_lists_and_one_matrix_of_their_vectors() {
        let provider = provider().await;
        let mut build = Build::new(&provider).unwrap();
        route(&mut build, &[(3, 2.0)]).await.unwrap();

        let region = build.fill([1, 0].into_iter()).await.unwrap();

        // Filled lists include the routed points.
        assert_filled_list(&region, 0, 0.0, &[0, 1, 3], &[0.0, 1.0, 2.0]);
        assert_filled_list(&region, 1, 10.0, &[2], &[10.0]);

        // The vectors are rows of one matrix, list by list in fill order, lent without copies.
        let first = region.get_vectors(1).unwrap();
        let second = region.get_vectors(0).unwrap();
        assert_eq!(
            first.as_slice().as_ptr_range().end,
            second.as_ptr(),
            "list 0's rows should follow list 1's"
        );
        assert_eq!(
            region.get_vectors(0).unwrap().as_ptr(),
            second.as_ptr(),
            "lending should not copy"
        );

        // A list that was not filled cannot be read.
        assert_unfilled(&region, 2);
    }

    #[tokio::test]
    async fn fill_rejects_lists_it_cannot_gather() {
        let provider = provider().await;
        let mut build = Build::new(&provider).unwrap();

        assert_fails(build.fill([0, 3].into_iter()).await, "list 3 is not live");

        // Error: a member without a vector, which regions take as given.
        let mut region = build.fill(std::iter::empty()).await.unwrap();
        assert_eq!(region.create(&[30.0], &[9]).unwrap(), 3);
        region.finish().await.unwrap();
        assert_fails(build.fill([3].into_iter()).await, "point 9 has no vector");
    }

    //--------//
    // Region //
    //--------//

    /// A write through a region.
    type Write = fn(&mut Region<'_>) -> ANNResult<()>;

    /// Check the centroid, member IDs and vector rows of a filled one-dimensional list.
    #[track_caller]
    fn assert_filled_list(
        region: &Region<'_>,
        id: u32,
        centroid: f32,
        members: &[u32],
        vectors: &[f32],
    ) {
        let list = region.get_list(id).unwrap();
        assert_eq!(list.centroid, [centroid], "centroid of list {id}");
        assert_eq!(list.members, members, "members of list {id}");
        assert_eq!(
            region.get_vectors(id).unwrap().as_slice(),
            vectors,
            "vectors of list {id}"
        );
    }

    /// Check that neither read interface exposes a list outside the filled region.
    #[track_caller]
    fn assert_unfilled(region: &Region<'_>, id: u32) {
        assert!(
            region.get_list(id).is_none(),
            "list {id} should not be filled"
        );
        assert!(
            region.get_vectors(id).is_none(),
            "vectors of list {id} should not be filled"
        );
    }

    #[tokio::test]
    async fn region_reads_see_the_lists_as_filled() {
        let provider = provider().await;
        let mut build = Build::new(&provider).unwrap();
        let mut region = build.fill([0, 1].into_iter()).await.unwrap();

        region.remove_members(0, &[1]).unwrap();
        region.retire(1).unwrap();
        let created = region.create(&[30.0], &[1, 2]).unwrap();

        // Writes are drafts until the region finishes; created lists are not filled.
        assert_filled_list(&region, 0, 0.0, &[0, 1], &[0.0, 1.0]);
        assert_filled_list(&region, 1, 10.0, &[2], &[10.0]);
        assert_unfilled(&region, created);
    }

    #[tokio::test]
    async fn region_writes_edit_members_and_lists() {
        let provider = line(&[
            (0.0, &[0.0, 1.0, 2.0, 3.0]),
            (10.0, &[10.0]),
            (20.0, &[20.0]),
        ])
        .await;
        let mut build = Build::new(&provider).unwrap();
        let mut region = build.fill([0, 1, 2].into_iter()).await.unwrap();

        // Removing keeps the other members in order; appending adds in the order given.
        region.remove_members(0, &[2, 0]).unwrap();
        region.append_members(1, &[2, 0]).unwrap();

        // Created lists take the next IDs.
        region.retire(2).unwrap();
        assert_eq!(region.create(&[30.0], &[5]).unwrap(), 3);
        assert_eq!(region.create(&[40.0], &[]).unwrap(), 4);

        region.finish().await.unwrap();
        build.finish().await.unwrap();

        let snapshot = provider.snapshot();
        assert_eq!(
            layout(&snapshot),
            [
                (0, 0.0, vec![1.0, 3.0]),
                (1, 10.0, vec![10.0, 2.0, 0.0]),
                (3, 30.0, vec![20.0]),
                (4, 40.0, vec![]),
            ]
        );
        assert_eq!(snapshot.next_list(), 5);
    }

    #[tokio::test]
    async fn region_writes_reject_contract_violations() {
        let cases: [(Write, &str); 10] = [
            (|r| r.append_members(2, &[0]), "list 2 was not filled"),
            (|r| r.remove_members(2, &[]), "list 2 was not filled"),
            (|r| r.retire(2), "list 2 was not filled"),
            (
                |r| r.append_members(0, &[2, 1]),
                "point 1 is already in list 0",
            ),
            (|r| r.remove_members(0, &[2]), "point 2 is not in list 0"),
            (|r| r.remove_members(0, &[1, 1]), "point 1 repeats"),
            (
                |r| {
                    r.retire(0)?;
                    r.append_members(0, &[2])
                },
                "list 0 is retired",
            ),
            (
                |r| {
                    r.retire(0)?;
                    r.remove_members(0, &[])
                },
                "list 0 is retired",
            ),
            (
                |r| {
                    r.retire(0)?;
                    r.retire(0)
                },
                "list 0 is retired",
            ),
            (
                |r| {
                    let created = r.create(&[30.0], &[])?;
                    r.append_members(created, &[2])
                },
                "list 3 was not filled",
            ),
        ];

        let provider = provider().await;
        for (write, expected) in cases {
            let mut build = Build::new(&provider).unwrap();
            let mut region = build.fill([0, 1].into_iter()).await.unwrap();
            assert_fails(write(&mut region), expected);
        }
    }

    #[tokio::test]
    async fn region_create_fails_once_list_ids_are_exhausted() {
        let mut snapshot = Snapshot::new(ONE);
        snapshot.next_id = u32::MAX - 1;
        let provider = Provider::new(snapshot);
        let mut build = Build::new(&provider).unwrap();
        let mut region = build.fill(std::iter::empty()).await.unwrap();

        assert_eq!(region.create(&[0.0], &[]).unwrap(), u32::MAX - 1);
        assert_fails(region.create(&[0.0], &[]), "list IDs are exhausted");
    }

    #[tokio::test]
    async fn region_finish_shows_its_changes_to_the_rest_of_the_build() {
        let provider = provider().await;
        let mut build = Build::new(&provider).unwrap();
        let mut region = build.fill([0].into_iter()).await.unwrap();
        region.retire(0).unwrap();
        assert_eq!(region.create(&[25.0], &[0, 1]).unwrap(), 3);
        region.finish().await.unwrap();

        // Later fills see list 3 in place of list 0...
        assert_fails(build.fill([0].into_iter()).await, "list 0 is not live");
        {
            let region = build.fill([3].into_iter()).await.unwrap();
            assert_filled_list(&region, 3, 25.0, &[0, 1], &[0.0, 1.0]);
        }

        // ... and so does `find`.
        assert_eq!(
            find(&build, 2, 4, |_| true).await.unwrap(),
            [(2, 0.0), (3, 25.0), (1, 100.0)]
        );

        // Finishing the build publishes the changes.
        build.finish().await.unwrap();
        assert_eq!(
            layout(&provider.snapshot()),
            [
                (1, 10.0, vec![10.0]),
                (2, 20.0, vec![]),
                (3, 25.0, vec![0.0, 1.0])
            ]
        );
    }

    #[tokio::test]
    async fn region_dropped_without_finish_discards_its_changes() {
        let provider = provider().await;
        let before = layout(&provider.snapshot());
        let mut build = Build::new(&provider).unwrap();
        let mut region = build.fill([0, 1].into_iter()).await.unwrap();
        region.remove_members(0, &[1]).unwrap();
        region.append_members(1, &[1]).unwrap();
        assert_eq!(region.create(&[30.0], &[]).unwrap(), 3);
        drop(region);

        {
            let region = build.fill([0, 1].into_iter()).await.unwrap();
            assert_filled_list(&region, 0, 0.0, &[0, 1], &[0.0, 1.0]);
            assert_filled_list(&region, 1, 10.0, &[2], &[10.0]);
        }
        assert_fails(build.fill([3].into_iter()).await, "list 3 is not live");

        build.finish().await.unwrap();
        let snapshot = provider.snapshot();
        assert_eq!(layout(&snapshot), before);
        assert_eq!(
            snapshot.next_list(),
            4,
            "IDs are never reused, even those of discarded lists"
        );
    }

    //------//
    // Find //
    //------//

    #[tokio::test]
    async fn find_emits_the_nearest_live_lists_that_pred_accepts() {
        let provider = line(&[(0.0, &[]), (10.0, &[]), (20.0, &[]), (30.0, &[])]).await;
        let build = Build::new(&provider).unwrap();

        // Nearest first and then by ID, the list itself included.
        let all = [(1, 0.0), (0, 100.0), (2, 100.0), (3, 400.0)];
        for (nprobe, expected) in [(4, &all[..]), (2, &all[..2]), (5, &all[..])] {
            assert_eq!(
                find(&build, 1, nprobe, |_| true).await.unwrap(),
                expected,
                "nprobe {nprobe}"
            );
        }

        // Only lists that `pred` accepts.
        assert_eq!(find(&build, 1, 2, |id| id > 1).await.unwrap(), all[2..]);

        assert_fails(find(&build, 4, 1, |_| true).await, "list 4 is not live");
    }

    //-------//
    // Build //
    //-------//

    #[tokio::test]
    async fn build_new_fails_while_another_build_is_live() {
        let provider = provider().await.with_fault(Call::Finish, 0);

        let build = Build::new(&provider).unwrap();
        assert_fails(Build::new(&provider), "another build is live");
        assert_contains!(
            ANNError::from(BuildIsLive).to_string(),
            "another build is live"
        );

        // Finishing a build ends it, even if finishing fails...
        assert_fails(build.finish().await, "injected Finish failure");

        // ... and so does dropping it.
        drop(Build::new(&provider).unwrap());
        assert!(Build::new(&provider).is_ok());
    }

    #[tokio::test]
    async fn build_dropped_without_finish_publishes_nothing() {
        let provider = provider().await;
        let before = provider.snapshot();

        let mut build = Build::new(&provider).unwrap();
        route(&mut build, &[(3, 4.0)]).await.unwrap();
        let mut region = build.fill([2].into_iter()).await.unwrap();
        region.retire(2).unwrap();
        region.finish().await.unwrap();
        drop(build);

        assert_abandoned(&provider, &before);
    }

    #[tokio::test]
    async fn build_finish_publishes_in_place_unless_the_snapshot_is_held() {
        let provider = provider().await;
        let mut expected = layout(&provider.snapshot());

        // Point 3 at 4 joins list 0, in place.
        let address = Arc::as_ptr(&provider.snapshot());
        let mut build = Build::new(&provider).unwrap();
        route(&mut build, &[(3, 4.0)]).await.unwrap();
        build.finish().await.unwrap();
        expected[0].2.push(4.0);
        assert_eq!(
            Arc::as_ptr(&provider.snapshot()),
            address,
            "publishing should not copy"
        );

        // Point 4 at 14 joins list 1, in a copy: the held snapshot, and a search begun
        // before, keep the lists they had.
        let held = provider.snapshot();
        let search = Search::new(&provider, &[14.0]);
        let mut build = Build::new(&provider).unwrap();
        route(&mut build, &[(4, 14.0)]).await.unwrap();
        build.finish().await.unwrap();

        assert!(!Arc::ptr_eq(&provider.snapshot(), &held));
        assert_eq!(layout(&held), expected);
        assert_eq!(scan(&search, 1).await.unwrap(), [(2, 16.0)]);

        expected[1].2.push(14.0);
        assert_eq!(layout(&provider.snapshot()), expected);
        let search = Search::new(&provider, &[14.0]);
        assert_eq!(scan(&search, 1).await.unwrap(), [(2, 16.0), (4, 0.0)]);
    }

    //--------//
    // Search //
    //--------//

    #[tokio::test]
    async fn scan_pushes_every_member_of_the_nearest_lists() {
        // Query 5 is as near list 0 as list 1, so list 0 comes first; list 2 is empty.
        let all = [(0, 25.0), (1, 16.0), (2, 25.0)];
        for (nprobe, expected) in [(1, &all[..2]), (2, &all[..]), (4, &all[..])] {
            let provider = provider().await;
            let search = Search::new(&provider, &[5.0]);
            assert_eq!(scan(&search, nprobe).await.unwrap(), expected);

            assert_scans(provider.metrics(), nprobe.min(3), expected.len());
        }

        let provider = line(&[]).await;
        let search = Search::new(&provider, &[5.0]);
        assert!(scan(&search, 1).await.unwrap().is_empty());
    }

    //--------------------//
    // Metrics and faults //
    //--------------------//

    #[tokio::test]
    async fn metrics_count_every_call_served() {
        let provider = provider().await;

        // The calls of an abandoned build count, failed ones included.
        let mut build = Build::new(&provider).unwrap();
        route(&mut build, &[(3, 4.0)]).await.unwrap();
        find(&build, 0, 1, |_| true).await.unwrap();
        assert_fails(build.fill([9].into_iter()).await, "list 9 is not live");

        let mut region = build.fill([0, 1, 2].into_iter()).await.unwrap();
        region.remove_members(0, &[3]).unwrap();
        region.append_members(2, &[3]).unwrap();
        region.retire(1).unwrap();
        region.create(&[30.0], &[2]).unwrap();
        region.finish().await.unwrap();
        drop(build);

        Build::new(&provider).unwrap().finish().await.unwrap();
        let search = Search::new(&provider, &[4.0]);
        scan(&search, 2).await.unwrap();

        let expected = Metrics {
            route: 1,
            find: 1,
            fill: 2,
            append_members: 1,
            remove_members: 1,
            retire: 1,
            create: 1,
            region_finish: 1,
            finish: 1,
            scan: 2,
            distances: 3,
        };
        assert_eq_verbose!(provider.metrics(), expected);
    }

    #[tokio::test]
    async fn with_fault_fails_only_the_nth_call_of_its_kind() {
        // Fill 1 fails, and fills 0 and 2 succeed.
        {
            let provider = provider().await.with_fault(Call::Fill, 1);
            let mut build = Build::new(&provider).unwrap();
            build.fill([0].into_iter()).await.unwrap();
            assert_fails(build.fill([0].into_iter()).await, "injected Fill failure");
            build.fill([0].into_iter()).await.unwrap();
            assert_eq!(provider.metrics().fill, 3);
        }

        // A scan stopped by a fault counts the distances of the list it scanned.
        {
            let provider = provider().await.with_fault(Call::Scan, 1);
            let search = Search::new(&provider, &[4.0]);
            assert_fails(scan(&search, 2).await, "injected Scan failure");

            assert_scans(provider.metrics(), 2, 2);
        }
    }
}
