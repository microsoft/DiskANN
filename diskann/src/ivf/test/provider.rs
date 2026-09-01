/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Self-contained, **production-shaped** example backend for the IVF module.
//!
//! This file is intended to double as a worked example of how to implement the three IVF
//! accessor traits ([`ListAccessor`], [`SearchAccessor`], [`InsertAccessor`]) and a
//! [`DataProvider`] for them, with concurrency in mind.
//!
//! # Roles
//!
//! A real deployment may split storage across several collaborators (see
//! `rfcs/00000-ivf-index.md`). For a compact, testable example we fold all of them into one
//! [`Provider`] that plays three roles at once:
//!
//! * **DataProvider** -- owns the external<->internal id mapping and allocates internal ids.
//! * **Coarse quantizer** -- owns the immutable centroids (`list_id -> centroid`).
//! * **Inverted-list store** -- owns the per-list membership. This is a list-owned
//!   **IVF-Flat** layout: each list stores its members' full vectors *contiguously* in a
//!   single flat buffer, in the same order as the list's ids (see [`ListStore`]). The fine
//!   scan therefore streams a list's vectors sequentially out of one cache-friendly block
//!   rather than chasing per-id pointers into a global table.
//!
//! Because the vectors live with their list, there is no global `id -> vector` table: a
//! vector is materialized only when [`InsertAccessor::append`] commits it to a list. The
//! provider's [`provider::SetElement::set_element`] therefore only *allocates* an internal
//! id; the append step does the contiguous store.
//!
//! # Concurrency model
//!
//! The centroids are immutable and shared via [`Arc`], so the fine-scan accessor can hand
//! list scans to independent tasks (see [`ScanAccessor`]'s concurrent path). The mutable
//! state -- each list's contiguous store -- is guarded by a per-list [`RwLock`] so inserts
//! can append to one list while searches read others. This mirrors how a production backend
//! would keep the read (search) path lock-light while still supporting concurrent append.

use std::{
    cmp::Ordering,
    fmt::{self, Debug},
    future::Future,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU32, Ordering as AtomicOrdering},
    },
};

use diskann_utils::{future::SendFuture, views::Matrix};
use diskann_vector::{PreprocessedDistanceFunction, distance::Metric};
use thiserror::Error;

use crate::graph::SearchOutputBuffer;
use crate::{
    ANNError, always_escalate,
    internal::counter::{Counter, LocalCounter},
    ivf::{InsertAccessor, InsertStrategy, ListAccessor, SearchAccessor, SearchStrategy},
    provider::{self, ExecutionContext, HasId, NoopGuard},
    utils::VectorRepr,
};

/// The id type used for inverted lists throughout this example.
pub type ListId = u32;

/// The internal/external id type used by this provider.
pub type Id = u32;

//////////////////////////
// Shared selection core //
//////////////////////////

/// Return the (up to) `nprobe` lists closest to `query` as `(list_id, distance)` pairs,
/// ordered by `(distance asc, list_id asc)`.
///
/// This is the single source of truth for coarse list selection. Both [`CentroidAccessor`]
/// (the production path, via this function) and the test harness's oracle (via
/// [`nearest_lists`]) build on it, which guarantees they agree on exactly which lists a
/// query probes -- essential for an *exact* approximate-search oracle.
fn scored_lists(
    centroids: &Matrix<f32>,
    query: &[f32],
    metric: Metric,
    nprobe: usize,
) -> Vec<(ListId, f32)> {
    let computer = f32::query_distance(query, metric);
    let mut scored: Vec<(ListId, f32)> = centroids
        .row_iter()
        .enumerate()
        .map(|(i, c)| (i as ListId, computer.evaluate_similarity(c)))
        .collect();
    scored.sort_by(|a, b| cmp_dist_id(*a, *b));
    scored.truncate(nprobe);
    scored
}

/// Return the (up to) `nprobe` list ids closest to `query`, ordered by
/// `(distance asc, list_id asc)`. Used by the test harness oracle.
pub fn nearest_lists(
    centroids: &Matrix<f32>,
    query: &[f32],
    metric: Metric,
    nprobe: usize,
) -> Vec<ListId> {
    scored_lists(centroids, query, metric, nprobe)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

/// Order by `(distance asc, id asc)` with NaN treated as equal.
fn cmp_dist_id(a: (ListId, f32), b: (ListId, f32)) -> Ordering {
    a.1.partial_cmp(&b.1)
        .unwrap_or(Ordering::Equal)
        .then(a.0.cmp(&b.0))
}

/// Index of the single nearest list for `vector`.
fn assign_nearest(centroids: &Matrix<f32>, vector: &[f32], metric: Metric) -> ListId {
    let computer = f32::query_distance(vector, metric);
    centroids
        .row_iter()
        .enumerate()
        .map(|(i, c)| (i as ListId, computer.evaluate_similarity(c)))
        .min_by(|a, b| cmp_dist_id(*a, *b))
        .map(|(id, _)| id)
        .expect("provider always has at least one centroid")
}

/// Public re-export of [`assign_nearest`] for tests that want to predict an insert's target
/// list independently of the index.
pub fn assign_nearest_for_test(centroids: &Matrix<f32>, vector: &[f32], metric: Metric) -> ListId {
    assign_nearest(centroids, vector, metric)
}

//////////////
// Provider //
//////////////

/// Error conditions for [`Provider::build`].
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("ivf::test::Provider needs at least one vector")]
    EmptyVectors,
    #[error("ivf::test::Provider needs at least one centroid")]
    EmptyCentroids,
    #[error("ivf::test::Provider vectors must have non-zero dimension")]
    ZeroDimension,
    #[error("ivf::test::Provider dimension mismatch: vectors={vectors}, centroids={centroids}")]
    DimMismatch { vectors: usize, centroids: usize },
}

impl From<ProviderError> for ANNError {
    #[track_caller]
    fn from(err: ProviderError) -> ANNError {
        ANNError::opaque(err)
    }
}

/// One inverted list's contiguous storage.
///
/// `ids[j]` is the internal id of the `j`-th member, and its full vector occupies
/// `data[j * dim .. (j + 1) * dim]`. The two vectors grow together in [`Self::push`], so a
/// member's id and its vector always share the same index -- the property the fine scan
/// relies on to pair a streamed vector back with its id.
#[derive(Debug, Default)]
struct ListStore {
    ids: Vec<Id>,
    data: Vec<f32>,
}

impl ListStore {
    /// Append `(id, vector)`, keeping ids and the contiguous `data` buffer in lockstep.
    fn push(&mut self, id: Id, vector: &[f32]) {
        self.ids.push(id);
        self.data.extend_from_slice(vector);
    }

    /// Iterate `(id, vector)` pairs in storage order, slicing the flat buffer by `dim`.
    fn iter(&self, dim: usize) -> impl Iterator<Item = (Id, &[f32])> {
        self.ids.iter().copied().zip(self.data.chunks_exact(dim))
    }
}

/// In-memory IVF backend.
#[derive(Debug)]
pub struct Provider {
    /// Immutable centroids, one row per inverted list.
    centroids: Arc<Matrix<f32>>,
    /// Per-list contiguous storage (ids + full vectors). Fixed list count; each list grows
    /// on insert.
    lists: Arc<Vec<RwLock<ListStore>>>,
    /// Allocates the next internal id. Identity external<->internal ids means this is just a
    /// monotonically increasing slot counter.
    next_id: Arc<AtomicU32>,
    /// Vector dimension.
    dim: usize,
    /// Distance metric used for both coarse selection and fine scoring (RFC D5).
    metric: Metric,
    /// Number of query->centroid distance computations performed.
    centroid_cmps: Counter,
    /// Number of query->member distance computations performed.
    member_cmps: Counter,
}

impl Provider {
    /// Build a provider from `vectors` and `centroids`, assigning each vector to its
    /// nearest centroid's list (the "bring-your-own-centroids" build of RFC F1/D6).
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] if either matrix is empty, the dimension is zero, or the
    /// vector and centroid dimensions disagree.
    pub fn build(
        vectors: Matrix<f32>,
        centroids: Matrix<f32>,
        metric: Metric,
    ) -> Result<Self, ProviderError> {
        if vectors.nrows() == 0 {
            return Err(ProviderError::EmptyVectors);
        }
        if centroids.nrows() == 0 {
            return Err(ProviderError::EmptyCentroids);
        }
        let dim = vectors.ncols();
        if dim == 0 {
            return Err(ProviderError::ZeroDimension);
        }
        if centroids.ncols() != dim {
            return Err(ProviderError::DimMismatch {
                vectors: dim,
                centroids: centroids.ncols(),
            });
        }

        let n_lists = centroids.nrows();
        let lists: Vec<RwLock<ListStore>> = (0..n_lists)
            .map(|_| RwLock::new(ListStore::default()))
            .collect();
        for (i, vector) in vectors.row_iter().enumerate() {
            let list = assign_nearest(&centroids, vector, metric);
            lists[list as usize]
                .write()
                .expect("list lock poisoned during build")
                .push(i as Id, vector);
        }

        let next_id = vectors.nrows() as u32;

        Ok(Self {
            centroids: Arc::new(centroids),
            lists: Arc::new(lists),
            next_id: Arc::new(AtomicU32::new(next_id)),
            dim,
            metric,
            centroid_cmps: Counter::new(),
            member_cmps: Counter::new(),
        })
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Distance metric.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Number of inverted lists (== number of centroids).
    pub fn n_lists(&self) -> usize {
        self.centroids.nrows()
    }

    /// Number of vectors currently stored (== number of allocated internal ids).
    pub fn len(&self) -> usize {
        self.next_id.load(AtomicOrdering::Relaxed) as usize
    }

    /// Borrow the centroids (used by the harness oracle).
    pub fn centroids(&self) -> &Matrix<f32> {
        &self.centroids
    }

    /// Snapshot the ids in `list`, in storage order (used by the harness oracle and tests).
    pub fn members(&self, list: ListId) -> Vec<Id> {
        self.lists[list as usize]
            .read()
            .expect("list lock poisoned")
            .ids
            .clone()
    }

    /// Snapshot the `(id, vector)` entries in `list`, in storage order (used by the harness
    /// oracle). The vectors are copied out of the list's contiguous buffer.
    pub fn list_entries(&self, list: ListId) -> Vec<(Id, Vec<f32>)> {
        let store = self.lists[list as usize]
            .read()
            .expect("list lock poisoned");
        store
            .iter(self.dim)
            .map(|(id, v)| (id, v.to_vec()))
            .collect()
    }

    /// Snapshot of the per-provider distance-computation counters.
    pub fn metrics(&self) -> ProviderMetrics {
        ProviderMetrics {
            centroid_cmps: self.centroid_cmps.value(),
            member_cmps: self.member_cmps.value(),
        }
    }
}

/// Distance-computation counters tracked by [`Provider`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderMetrics {
    /// Total query->centroid distance computations (coarse step).
    pub centroid_cmps: usize,
    /// Total query->member distance computations (fine step).
    pub member_cmps: usize,
}

/////////////
// Context //
/////////////

/// Per-operation execution context that records how many tasks were spawned through it,
/// so concurrency tests can assert the fine-scan path actually fanned out.
#[derive(Debug, Clone)]
pub struct Context(Arc<Counter>);

impl Context {
    pub fn new() -> Self {
        Self(Arc::new(Counter::new()))
    }

    /// Number of spawns routed through [`ExecutionContext::wrap_spawn`].
    pub fn spawns(&self) -> usize {
        self.0.value()
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext for Context {
    fn wrap_spawn<F, T>(&self, f: F) -> impl Future<Output = T> + Send + 'static
    where
        F: Future<Output = T> + Send + 'static,
    {
        self.0.increment();
        f
    }
}

////////////
// Errors //
////////////

/// Critical id-validation error: a requested id is out of range.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
#[error("ivf::test::Provider has no id {0}")]
pub struct InvalidId(pub u32);

always_escalate!(InvalidId);

impl From<InvalidId> for ANNError {
    #[track_caller]
    fn from(err: InvalidId) -> ANNError {
        ANNError::opaque(err)
    }
}

/// Critical error injected by [`Strategy::failing_on_list`]: scanning a specific list fails.
///
/// Used to verify that a fine-scan failure escalates out of
/// [`crate::ivf::IvfIndex::knn_search`].
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
#[error("ivf::test::Provider injected scan failure on list {0}")]
pub struct ScanFailure(pub ListId);

always_escalate!(ScanFailure);

impl From<ScanFailure> for ANNError {
    #[track_caller]
    fn from(err: ScanFailure) -> ANNError {
        ANNError::opaque(err)
    }
}

/// Dimension-mismatch error from strategy construction.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("dimension mismatch: strategy expects {expected}, got {actual}")]
pub struct StrategyError {
    pub expected: usize,
    pub actual: usize,
}

impl From<StrategyError> for ANNError {
    #[track_caller]
    fn from(err: StrategyError) -> ANNError {
        ANNError::opaque(err)
    }
}

//////////////////
// DataProvider //
//////////////////

impl provider::DataProvider for Provider {
    type Context = Context;
    type InternalId = Id;
    type ExternalId = Id;
    type Error = InvalidId;
    type Guard = NoopGuard<Id>;

    fn to_internal_id(&self, _ctx: &Context, gid: &Id) -> Result<Id, InvalidId> {
        if (*gid as usize) < self.len() {
            Ok(*gid)
        } else {
            Err(InvalidId(*gid))
        }
    }

    fn to_external_id(&self, _ctx: &Context, id: Id) -> Result<Id, InvalidId> {
        if (id as usize) < self.len() {
            Ok(id)
        } else {
            Err(InvalidId(id))
        }
    }
}

impl provider::SetElement<&[f32]> for Provider {
    type SetError = InvalidId;

    async fn set_element(
        &self,
        _context: &Context,
        _id: &Id,
        _element: &[f32],
    ) -> Result<Self::Guard, Self::SetError> {
        // List-owned storage: the vector is *not* stored here -- it is committed to its
        // list's contiguous buffer by `InsertAccessor::append`, once the target list is
        // known. `set_element` therefore only allocates the next internal id. A production
        // provider would honor the supplied external id and return a guard that rolls back
        // the allocation on drop-without-complete.
        let new_id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        Ok(NoopGuard::new(new_id))
    }
}

//////////////////////////////
// ListAccessor (coarse step) //
//////////////////////////////

/// Coarse accessor: selects the `nprobe` lists nearest to a bound query.
///
/// Holds an [`Arc`] clone of the immutable centroids and a preprocessed query computer. The
/// `LocalCounter` flushes the coarse comparison count back to the provider on drop.
pub struct CentroidAccessor<'a> {
    centroids: Arc<Matrix<f32>>,
    metric: Metric,
    nprobe_query: Vec<f32>,
    cmps: LocalCounter<'a>,
}

impl Debug for CentroidAccessor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CentroidAccessor")
            .field("n_lists", &self.centroids.nrows())
            .finish_non_exhaustive()
    }
}

impl ListAccessor for CentroidAccessor<'_> {
    type Id = ListId;
    type Error = InvalidId;

    fn select_lists<B>(
        &mut self,
        nprobe: usize,
        output: &mut B,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        B: SearchOutputBuffer<Self::Id> + Send + ?Sized,
    {
        async move {
            self.cmps.increment_by(self.centroids.nrows());
            for (id, dist) in scored_lists(&self.centroids, &self.nprobe_query, self.metric, nprobe)
            {
                if output.push(id, dist).is_full() {
                    break;
                }
            }
            Ok(())
        }
    }
}

/////////////////////////////////
// SearchAccessor (fine step) //
/////////////////////////////////

/// Whether the fine scan runs sequentially or fans lists out across spawned tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanMode {
    /// Scan lists one at a time on the current task.
    Sequential,
    /// Spawn one task per list via [`ExecutionContext::wrap_spawn`] and merge results.
    Concurrent,
}

/// Fine accessor: scans the members of a set of lists, scoring each against a bound query.
///
/// All data needed for a scan is held behind [`Arc`] so the concurrent path can move clones
/// into `'static` spawned tasks. Each spawned task produces its own `(id, distance)` vector;
/// the driver merges those into the single `f` callback, preserving the trait's
/// single-threaded-callback contract while doing the distance work concurrently.
pub struct ScanAccessor<'a> {
    lists: Arc<Vec<RwLock<ListStore>>>,
    computer: Arc<<f32 as VectorRepr>::QueryDistance>,
    dim: usize,
    context: Context,
    mode: ScanMode,
    fail_on_list: Option<ListId>,
    cmps: LocalCounter<'a>,
}

impl Debug for ScanAccessor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanAccessor")
            .field("mode", &self.mode)
            .field("fail_on_list", &self.fail_on_list)
            .finish_non_exhaustive()
    }
}

impl HasId for ScanAccessor<'_> {
    type Id = Id;
}

impl ScanAccessor<'_> {
    /// Score every member of `list`, returning `(id, distance)` pairs. Shared by both the
    /// sequential driver and the spawned-task driver.
    ///
    /// Streams the list's contiguous vector buffer in storage order, pairing each vector
    /// with the id at the matching index.
    fn score_list(
        lists: &Arc<Vec<RwLock<ListStore>>>,
        computer: &<f32 as VectorRepr>::QueryDistance,
        dim: usize,
        list: ListId,
    ) -> Vec<(Id, f32)> {
        let store = lists[list as usize].read().expect("list lock poisoned");
        store
            .iter(dim)
            .map(|(id, v)| (id, computer.evaluate_similarity(v)))
            .collect()
    }
}

impl SearchAccessor for ScanAccessor<'_> {
    type ListId = ListId;
    type Error = ScanFailure;

    fn scan_lists<Itr, F>(
        &mut self,
        lists: Itr,
        mut f: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        Itr: Iterator<Item = Self::ListId> + Send,
        F: Send + FnMut(Self::Id, f32),
    {
        async move {
            match self.mode {
                ScanMode::Sequential => {
                    for list in lists {
                        if self.fail_on_list == Some(list) {
                            return Err(ScanFailure(list));
                        }
                        for (id, dist) in
                            Self::score_list(&self.lists, &self.computer, self.dim, list)
                        {
                            self.cmps.increment();
                            f(id, dist);
                        }
                    }
                }
                ScanMode::Concurrent => {
                    let mut set = tokio::task::JoinSet::new();
                    for list in lists {
                        if self.fail_on_list == Some(list) {
                            set.shutdown().await;
                            return Err(ScanFailure(list));
                        }
                        let lists = Arc::clone(&self.lists);
                        let computer = Arc::clone(&self.computer);
                        let dim = self.dim;
                        let task = async move { Self::score_list(&lists, &computer, dim, list) };
                        set.spawn(self.context.wrap_spawn(task));
                    }
                    while let Some(joined) = set.join_next().await {
                        let scored = joined.expect("scan task panicked");
                        for (id, dist) in scored {
                            self.cmps.increment();
                            f(id, dist);
                        }
                    }
                }
            }
            Ok(())
        }
    }
}

/////////////////////////////////
// InsertAccessor (append step) //
/////////////////////////////////

/// Append accessor: commits `(id, vector)` to a chosen list's contiguous store.
///
/// This example is **IVF-Flat** with *list-owned* storage: the full vector is appended to
/// the target list's contiguous buffer here (in lockstep with its id), and read back from
/// that same buffer during the fine scan. There is no global `id -> vector` table.
///
/// An **IVF-PQ** backend would slot in here unchanged in shape: quantize `vector` and push
/// the code instead of the raw floats, e.g. `store.push_code(id, codebook.encode(vector))`.
#[derive(Debug)]
pub struct AppendAccessor {
    lists: Arc<Vec<RwLock<ListStore>>>,
    dim: usize,
}

impl HasId for AppendAccessor {
    type Id = Id;
}

impl<'v> InsertAccessor<&'v [f32]> for AppendAccessor {
    type ListId = ListId;
    type Error = InvalidId;

    fn append(
        &mut self,
        list: Self::ListId,
        id: Self::Id,
        vector: &'v [f32],
    ) -> impl SendFuture<Result<(), Self::Error>> {
        async move {
            debug_assert_eq!(
                vector.len(),
                self.dim,
                "append received a vector of the wrong dimension",
            );
            self.lists[list as usize]
                .write()
                .expect("list lock poisoned")
                .push(id, vector);
            Ok(())
        }
    }
}

//////////////
// Strategy //
//////////////

/// Per-call factory wiring centroids (coarse) to scanning/appending (fine) for `&[f32]`
/// queries and vectors. Validates query dimension and can inject a scan failure.
#[derive(Debug, Clone)]
pub struct Strategy {
    dim: usize,
    metric: Metric,
    mode: ScanMode,
    fail_on_list: Option<ListId>,
}

impl Strategy {
    /// Construct a strategy expecting vectors of dimension `dim`, scanning sequentially.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            mode: ScanMode::Sequential,
            fail_on_list: None,
        }
    }

    /// Select the fine-scan mode (sequential vs. concurrent).
    pub fn with_mode(mut self, mode: ScanMode) -> Self {
        self.mode = mode;
        self
    }

    /// Make the fine scan fail when it reaches `list`.
    pub fn failing_on_list(mut self, list: ListId) -> Self {
        self.fail_on_list = Some(list);
        self
    }

    fn check_dim(&self, actual: usize) -> Result<(), StrategyError> {
        if actual != self.dim {
            Err(StrategyError {
                expected: self.dim,
                actual,
            })
        } else {
            Ok(())
        }
    }
}

impl<'a> SearchStrategy<'a, Provider, &'a [f32]> for Strategy {
    type ListId = ListId;
    type SearchAccessor = ScanAccessor<'a>;
    type ListAccessor = CentroidAccessor<'a>;
    type Error = StrategyError;

    fn search_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Context,
        query: &'a [f32],
    ) -> Result<Self::SearchAccessor, Self::Error> {
        self.check_dim(query.len())?;
        Ok(ScanAccessor {
            lists: Arc::clone(&provider.lists),
            computer: Arc::new(f32::query_distance(query, self.metric)),
            dim: provider.dim,
            context: context.clone(),
            mode: self.mode,
            fail_on_list: self.fail_on_list,
            cmps: provider.member_cmps.local(),
        })
    }

    fn list_accessor(
        &'a self,
        provider: &'a Provider,
        _context: &'a Context,
        query: &'a [f32],
    ) -> Result<Self::ListAccessor, Self::Error> {
        self.check_dim(query.len())?;
        Ok(CentroidAccessor {
            centroids: Arc::clone(&provider.centroids),
            metric: self.metric,
            nprobe_query: query.to_vec(),
            cmps: provider.centroid_cmps.local(),
        })
    }
}

impl<'a> InsertStrategy<'a, Provider, &'a [f32]> for Strategy {
    type ListId = ListId;
    type InsertAccessor = AppendAccessor;
    type ListAccessor = CentroidAccessor<'a>;
    type Error = StrategyError;

    fn insert_accessor(
        &'a self,
        provider: &'a Provider,
        _context: &'a Context,
    ) -> Result<Self::InsertAccessor, Self::Error> {
        // The vector is validated by `list_accessor` (the coarse step runs first), and is
        // stored contiguously by `append`, so the append accessor only needs the list
        // storage handle and the dimension (for its lockstep buffer layout).
        Ok(AppendAccessor {
            lists: Arc::clone(&provider.lists),
            dim: provider.dim,
        })
    }

    fn list_accessor(
        &'a self,
        provider: &'a Provider,
        _context: &'a Context,
        vector: &'a [f32],
    ) -> Result<Self::ListAccessor, Self::Error> {
        self.check_dim(vector.len())?;
        Ok(CentroidAccessor {
            centroids: Arc::clone(&provider.centroids),
            metric: self.metric,
            nprobe_query: vector.to_vec(),
            cmps: provider.centroid_cmps.local(),
        })
    }
}
