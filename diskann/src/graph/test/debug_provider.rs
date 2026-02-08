/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::{HashMap, hash_map},
    sync::{
        RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::{
    ANNError, ANNErrorKind, ANNResult,
    graph::{
        AdjacencyList,
        glue::{
            AsElement, ExpandBeam, FillSet, FilterStartPoints, InplaceDeleteStrategy,
            InsertStrategy, Pipeline, PruneStrategy, SearchExt, SearchStrategy,
        },
    },
    provider::{
        self, Accessor, BuildDistanceComputer, BuildQueryComputer, DataProvider, DefaultAccessor,
        DefaultContext, DelegateNeighbor, Delete, ElementStatus, HasId, NeighborAccessor,
        NeighborAccessorMut,
    },
    tracked_warn,
    utils::VectorRepr,
};
use diskann_utils::Reborrow;
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};
use thiserror::Error;

pub trait DebugQuantizer: Send + Sync + 'static {
    type DistanceComputer: for<'a> DistanceFunction<&'a [f32], &'a [u8], f32>
        + for<'a> DistanceFunction<&'a [u8], &'a [u8], f32>
        + Send
        + Sync
        + 'static;
    type QueryComputer: for<'a> PreprocessedDistanceFunction<&'a [u8], f32>
        + Send
        + Sync
        + 'static;

    fn num_chunks(&self) -> usize;
    fn compress_into(&self, input: &[f32], output: &mut [u8]) -> ANNResult<()>;
    fn build_distance_computer(&self, metric: Metric) -> ANNResult<Self::DistanceComputer>;
    fn build_query_computer(&self, metric: Metric, query: &[f32]) -> ANNResult<Self::QueryComputer>;
}

/// Operates entirely in full precision.
///
/// All indexing and search operations use the uncompressed full-precision vectors.
#[derive(Debug, Clone, Copy)]
pub struct FullPrecision;

/// Operates entirely in the quantized space.
///
/// All indexing and search operations use quantized vectors.
/// If full-precision vectors are available, they are only used for the final reranking step.
#[derive(Debug, Clone, Copy)]
pub struct Quantized;

/// Internal variant of above strategy types to avoid start point filtering.
#[derive(Debug)]
pub struct Internal<T>(pub T);

/// A tag type indicating that a method fails via panic instead of returning an error.
///
/// This is an enum with no alternatives, so is impossible to construct. Therefore, we know
/// that there can never be an actual value with this type.
#[derive(Debug)]
pub enum Panics {}

impl std::fmt::Display for Panics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "panics")
    }
}

impl std::error::Error for Panics {}

impl From<Panics> for ANNError {
    #[cold]
    fn from(_: Panics) -> ANNError {
        ANNError::log_async_error("unreachable")
    }
}

crate::always_escalate!(Panics);

#[derive(Debug, Clone)]
pub struct DebugConfig {
    pub start_id: u32,
    pub start_point: Vec<f32>,
    pub max_degree: usize,
    pub metric: Metric,
}

/// A version of `DebugConfig` that has the compressed representation of `start_point` in
/// addition to the full-precision representation.
#[derive(Debug, Clone)]
struct InternalConfig {
    start_id: u32,
    start_point: Datum,
    max_degree: usize,
    metric: Metric,
}

/// A combined full-precision and PQ quantized vector.
#[derive(Debug, Default, Clone)]
pub struct Datum {
    full: Vec<f32>,
    quant: Vec<u8>,
}

impl Datum {
    /// Create a new `Datum`.
    fn new(full: Vec<f32>, quant: Vec<u8>) -> Self {
        Self { full, quant }
    }

    /// Return a reference to the full-precision vector.
    fn full(&self) -> &[f32] {
        &self.full
    }

    /// Return a reference to the quantized vector.
    fn quant(&self) -> &[u8] {
        &self.quant
    }
}

/// A container for `Datum`s within the `Debug` provider.
///
/// This tracks whether items are valid or have been marked as deleted inline.
#[derive(Debug, Clone)]
pub enum Vector {
    Valid(Datum),
    Deleted(Datum),
}

impl Vector {
    /// Change `self` to be `Self::Deleted`, leaving the internal `Datum` unchanged.
    fn mark_deleted(&mut self) {
        *self = match self.take() {
            Self::Valid(v) => Self::Deleted(v),
            Self::Deleted(v) => Self::Deleted(v),
        }
    }

    /// Take the internal `Datum` and construct a new instance of `Self`.
    ///
    /// Leave the caller with an empty `Datum`.
    fn take(&mut self) -> Self {
        match self {
            Self::Valid(v) => Self::Valid(std::mem::take(v)),
            Self::Deleted(v) => Self::Deleted(std::mem::take(v)),
        }
    }

    /// Return `true` if `self` has been marked as deleted. Otherwise, return `false`.
    fn is_deleted(&self) -> bool {
        matches!(self, Self::Deleted(_))
    }
}

impl std::ops::Deref for Vector {
    type Target = Datum;
    fn deref(&self) -> &Datum {
        match self {
            Self::Valid(v) => v,
            Self::Deleted(v) => v,
        }
    }
}

/// An element for two-level datasets that is either full-precision or quantized.
/// This allows a pruning strategy that combines full-precision and quantized distances.
#[derive(Debug, Clone)]
pub enum Hybrid<F, Q> {
    Full(F),
    Quant(Q),
}

impl<F, Q> Hybrid<F, Q> {
    pub fn is_full(&self) -> bool {
        matches!(self, Self::Full(_))
    }
}

impl<'short, F, Q> Reborrow<'short> for Hybrid<F, Q>
where
    F: Reborrow<'short>,
    Q: Reborrow<'short>,
{
    type Target = Hybrid<F::Target, Q::Target>;

    fn reborrow(&'short self) -> Self::Target {
        match self {
            Self::Full(v) => Hybrid::Full(v.reborrow()),
            Self::Quant(v) => Hybrid::Quant(v.reborrow()),
        }
    }
}

/// A distance computer that operates on `Hybrid`.
pub struct HybridComputer<Q, T>
where
    T: VectorRepr,
{
    quant: Q,
    full: T::Distance,
}

impl<Q, T> HybridComputer<Q, T>
where
    T: VectorRepr,
{
    pub fn new(quant: Q, full: T::Distance) -> Self {
        Self { quant, full }
    }
}

/// The implementation of `DistanceFunction` for the hybrid computer.
impl<Q, T> DistanceFunction<Hybrid<&[T], &[u8]>, Hybrid<&[T], &[u8]>, f32>
    for HybridComputer<Q, T>
where
    Q: for<'a> DistanceFunction<&'a [f32], &'a [u8], f32>
        + for<'a> DistanceFunction<&'a [u8], &'a [u8], f32>,
    T: VectorRepr,
{
    #[inline(always)]
    fn evaluate_similarity(&self, x: Hybrid<&[T], &[u8]>, y: Hybrid<&[T], &[u8]>) -> f32 {
        match x {
            Hybrid::Full(x) => match y {
                Hybrid::Full(y) => self.full.evaluate_similarity(x, y),
                Hybrid::Quant(y) => {
                    // SAFETY: This can only panic when T = `MinMaxElement` and the underlying slice is ill-defined.
                    // we are ok with panicking in distance functions for now.
                    #[allow(clippy::unwrap_used)]
                    self.quant
                        .evaluate_similarity(&*T::as_f32(x).unwrap(), y)
                }
            },
            Hybrid::Quant(x) => match y {
                Hybrid::Full(y) => {
                    // SAFETY: This can only panic when T = `MinMaxElement` and the underlying slice is ill-defined.
                    // we are ok with panicking in distance functions for now.
                    #[allow(clippy::unwrap_used)]
                    self.quant
                        .evaluate_similarity(&*T::as_f32(y).unwrap(), x)
                }
                Hybrid::Quant(y) => self.quant.evaluate_similarity(x, y),
            },
        }
    }
}

/// A simple increment-only thread-safe counter.
#[derive(Debug)]
pub struct Counter(AtomicUsize);

impl Counter {
    /// Construct a new counter with a count of 0.
    fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    /// Increment the counter by 1.
    pub(crate) fn increment(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    /// Return the current value of the counter.
    pub fn get(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
}

pub struct DebugProvider<Q> {
    config: InternalConfig,

    pub pq_table: Q,
    pub data: RwLock<HashMap<u32, Vector>>,
    pub neighbors: RwLock<HashMap<u32, Vec<u32>>>,

    // Counters
    pub full_reads: Counter,
    pub quant_reads: Counter,
    pub neighbor_reads: Counter,
    pub data_writes: Counter,
    pub neighbor_writes: Counter,

    // Track whether the `insert_search_accessor` is invoked.
    pub insert_search_accessor_calls: Counter,
}

impl<Q> DebugProvider<Q>
where
    Q: DebugQuantizer,
{
    pub fn new(config: DebugConfig, pq_table: Q) -> ANNResult<Self> {
        // Compress the start point.
        let mut pq = vec![0u8; pq_table.num_chunks()];
        pq_table.compress_into(config.start_point.as_slice(), pq.as_mut_slice())?;

        let config = InternalConfig {
            start_id: config.start_id,
            start_point: Datum::new(config.start_point, pq),
            max_degree: config.max_degree,
            metric: config.metric,
        };

        let mut data = HashMap::new();
        data.insert(config.start_id, Vector::Valid(config.start_point.clone()));

        let mut neighbors = HashMap::new();
        neighbors.insert(config.start_id, Vec::new());

        Ok(Self {
            config,
            pq_table,
            data: RwLock::new(data),
            neighbors: RwLock::new(neighbors),
            full_reads: Counter::new(),
            quant_reads: Counter::new(),
            neighbor_reads: Counter::new(),
            data_writes: Counter::new(),
            neighbor_writes: Counter::new(),
            insert_search_accessor_calls: Counter::new(),
        })
    }

    /// Return the dimension of the full-precision data.
    pub fn dim(&self) -> usize {
        self.config.start_point.full().len()
    }

    /// Return the maximum degree that can be held by this graph.
    pub fn max_degree(&self) -> usize {
        self.config.max_degree
    }

    /// Return the start point stored in this provider.
    pub fn start_point(&self) -> &Datum {
        &self.config.start_point
    }

    /// Return the full-precision start point stored in this provider.
    pub fn start_point_full(&self) -> &[f32] {
        self.config.start_point.full()
    }

    #[expect(
        clippy::expect_used,
        reason = "DebugProvider is not a production data structure"
    )]
    fn data(&self) -> RwLockReadGuard<'_, HashMap<u32, Vector>> {
        self.data.read().expect("cannot recover from lock poison")
    }

    #[expect(
        clippy::expect_used,
        reason = "DebugProvider is not a production data structure"
    )]
    fn data_mut(&self) -> RwLockWriteGuard<'_, HashMap<u32, Vector>> {
        self.data.write().expect("cannot recover from lock poison")
    }

    #[expect(
        clippy::expect_used,
        reason = "DebugProvider is not a production data structure"
    )]
    fn neighbors(&self) -> RwLockReadGuard<'_, HashMap<u32, Vec<u32>>> {
        self.neighbors
            .read()
            .expect("cannot recover from lock poison")
    }

    #[expect(
        clippy::expect_used,
        reason = "DebugProvider is not a production data structure"
    )]
    fn neighbors_mut(&self) -> RwLockWriteGuard<'_, HashMap<u32, Vec<u32>>> {
        self.neighbors
            .write()
            .expect("cannot recover from lock poison")
    }

    fn is_deleted(&self, id: u32) -> Result<bool, InvalidId> {
        match self.data().get(&id) {
            Some(element) => Ok(element.is_deleted()),
            None => Err(InvalidId::Internal(id)),
        }
    }
}

/// Light-weight error type for reporting access to an invalid ID.
#[derive(Debug, Clone, Copy, Error)]
pub enum InvalidId {
    #[error("internal id {0} not initialized")]
    Internal(u32),
    #[error("external id {0} not initialized")]
    External(u32),
    #[error("is start point {0}")]
    IsStartPoint(u32),
}

crate::always_escalate!(InvalidId);

impl From<InvalidId> for ANNError {
    #[track_caller]
    fn from(err: InvalidId) -> ANNError {
        ANNError::opaque(err)
    }
}

//////////////////
// DataProvider //
//////////////////

impl<Q> DataProvider for DebugProvider<Q>
where
    Q: DebugQuantizer,
{
    type Context = DefaultContext;
    type InternalId = u32;
    type ExternalId = u32;
    type Error = InvalidId;

    fn to_internal_id(
        &self,
        _context: &DefaultContext,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error> {
        // Check that the ID actually exists
        let valid = self.data().contains_key(gid);
        if valid {
            Ok(*gid)
        } else {
            Err(InvalidId::External(*gid))
        }
    }

    fn to_external_id(
        &self,
        _context: &DefaultContext,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error> {
        // Check that the ID actually exists
        let valid = self.data().contains_key(&id);
        if valid {
            Ok(id)
        } else {
            Err(InvalidId::External(id))
        }
    }
}

impl<Q> Delete for DebugProvider<Q>
where
    Q: DebugQuantizer,
{
    async fn delete(
        &self,
        _context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<(), Self::Error> {
        if *gid == self.config.start_id {
            return Err(InvalidId::IsStartPoint(*gid));
        }

        let mut guard = self.data_mut();
        match guard.entry(*gid) {
            hash_map::Entry::Occupied(mut occupied) => {
                occupied.get_mut().mark_deleted();
                Ok(())
            }
            hash_map::Entry::Vacant(_) => Err(InvalidId::External(*gid)),
        }
    }

    async fn release(
        &self,
        _context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<(), Self::Error> {
        if id == self.config.start_id {
            return Err(InvalidId::IsStartPoint(id));
        }

        // NOTE: acquire the locks in the order `data` then `neighbors`.
        let mut data = self.data_mut();
        let mut neighbors = self.neighbors_mut();

        let v = data.remove(&id);
        let u = neighbors.remove(&id);

        if v.is_none() || u.is_none() {
            Err(InvalidId::Internal(id))
        } else {
            Ok(())
        }
    }

    async fn status_by_internal_id(
        &self,
        _context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<ElementStatus, Self::Error> {
        if self.is_deleted(id)? {
            Ok(provider::ElementStatus::Deleted)
        } else {
            Ok(provider::ElementStatus::Valid)
        }
    }

    fn status_by_external_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> impl Future<Output = Result<ElementStatus, Self::Error>> + Send {
        self.status_by_internal_id(context, *gid)
    }
}

impl<Q> provider::SetElement<[f32]> for DebugProvider<Q>
where
    Q: DebugQuantizer,
{
    type SetError = ANNError;

    type Guard = provider::NoopGuard<u32>;

    fn set_element(
        &self,
        _context: &Self::Context,
        id: &Self::ExternalId,
        element: &[f32],
    ) -> impl Future<Output = Result<Self::Guard, Self::SetError>> + Send {
        #[derive(Debug, Clone, Copy, Error)]
        #[error("vector id {0} is already assigned")]
        pub struct AlreadyAssigned(u32);

        crate::always_escalate!(AlreadyAssigned);

        impl From<AlreadyAssigned> for ANNError {
            #[track_caller]
            fn from(err: AlreadyAssigned) -> Self {
                Self::new(ANNErrorKind::IndexError, err)
            }
        }

        // NOTE: acquire the locks in the order `vectors` then `neighbors`.
        let result = match self.data_mut().entry(*id) {
            hash_map::Entry::Occupied(_) => Err(AlreadyAssigned(*id).into()),
            hash_map::Entry::Vacant(data) => match self.neighbors_mut().entry(*id) {
                hash_map::Entry::Occupied(_) => Err(AlreadyAssigned(*id).into()),
                hash_map::Entry::Vacant(neighbors) => {
                    self.data_writes.increment();

                    let mut pq = vec![0u8; self.pq_table.num_chunks()];
                    match self.pq_table.compress_into(element, pq.as_mut_slice()) {
                        Ok(()) => {
                            data.insert(Vector::Valid(Datum::new(element.into(), pq)));
                            neighbors.insert(Vec::new());
                            Ok(provider::NoopGuard::new(*id))
                        }
                        Err(err) => Err(err),
                    }
                }
            },
        };

        std::future::ready(result)
    }
}

impl<Q> postprocess::DeletionCheck for DebugProvider<Q>
where
    Q: DebugQuantizer,
{
    fn deletion_check(&self, id: u32) -> bool {
        match self.is_deleted(id) {
            Ok(is_deleted) => is_deleted,
            Err(err) => {
                tracked_warn!("Deletion post-process failed with error {err} - continuing");
                true
            }
        }
    }
}

///////////////
// Accessors //
///////////////

#[derive(Debug, Clone, Copy, Error)]
#[error("Attempt to access an invalid id: {0}")]
pub struct AccessedInvalidId(u32);

crate::always_escalate!(AccessedInvalidId);

impl From<AccessedInvalidId> for ANNError {
    #[track_caller]
    fn from(err: AccessedInvalidId) -> Self {
        Self::opaque(err)
    }
}

impl<Q> DefaultAccessor for DebugProvider<Q>
where
    Q: DebugQuantizer,
{
    type Accessor<'a> = DebugNeighborAccessor<'a, Q>;

    fn default_accessor(&self) -> Self::Accessor<'_> {
        DebugNeighborAccessor::new(self)
    }
}

pub struct DebugNeighborAccessor<'a, Q> {
    provider: &'a DebugProvider<Q>,
}

impl<'a, Q> Copy for DebugNeighborAccessor<'a, Q> {}

impl<'a, Q> Clone for DebugNeighborAccessor<'a, Q> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, Q> DebugNeighborAccessor<'a, Q>
where
    Q: DebugQuantizer,
{
    pub fn new(provider: &'a DebugProvider<Q>) -> Self {
        Self { provider }
    }
}

impl<Q> HasId for DebugNeighborAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Id = u32;
}

impl<Q> NeighborAccessor for DebugNeighborAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        let result = match self.provider.neighbors().get(&id) {
            Some(v) => {
                self.provider.neighbor_reads.increment();
                neighbors.overwrite_trusted(v);
                Ok(self)
            }
            None => Err(ANNError::opaque(AccessedInvalidId(id))),
        };

        std::future::ready(result)
    }
}

impl<Q> NeighborAccessorMut for DebugNeighborAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    fn set_neighbors(
        self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        assert!(neighbors.len() <= self.provider.config.max_degree);
        let result = match self.provider.neighbors_mut().get_mut(&id) {
            Some(v) => {
                self.provider.neighbor_writes.increment();
                v.clear();
                v.extend_from_slice(neighbors);
                Ok(self)
            }
            None => Err(ANNError::opaque(AccessedInvalidId(id))),
        };

        std::future::ready(result)
    }

    fn append_vector(
        self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        let result = match self.provider.neighbors_mut().get_mut(&id) {
            Some(v) => {
                assert!(
                    v.len().checked_add(neighbors.len()).unwrap()
                        <= self.provider.config.max_degree,
                    "current = {:?}, new = {:?}, id = {}",
                    v,
                    neighbors,
                    id
                );

                let check = neighbors.iter().try_for_each(|n| {
                    if v.contains(n) {
                        Err(ANNError::message(
                            ANNErrorKind::Opaque,
                            format!("id {} is duplicated", n),
                        ))
                    } else {
                        Ok(())
                    }
                });

                match check {
                    Ok(()) => {
                        self.provider.neighbor_writes.increment();
                        v.extend_from_slice(neighbors);
                        Ok(self)
                    }
                    Err(err) => Err(err),
                }
            }
            None => Err(ANNError::opaque(AccessedInvalidId(id))),
        };

        std::future::ready(result)
    }
}

//---------------//
// Full Accessor //
//---------------//

pub struct FullAccessor<'a, Q> {
    provider: &'a DebugProvider<Q>,
    buffer: Box<[f32]>,
}

impl<'a, Q> FullAccessor<'a, Q>
where
    Q: DebugQuantizer,
{
    pub fn new(provider: &'a DebugProvider<Q>) -> Self {
        let buffer = (0..provider.dim()).map(|_| 0.0).collect();
        Self { provider, buffer }
    }

    pub fn provider(&self) -> &DebugProvider<Q> {
        self.provider
    }
}

impl<Q> HasId for FullAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Id = u32;
}

impl<Q> Accessor for FullAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Extended = Box<[f32]>;
    type Element<'a>
        = &'a [f32]
    where
        Self: 'a;
    type ElementRef<'a> = &'a [f32];

    type GetError = AccessedInvalidId;

    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        let result = match self.provider.data().get(&id) {
            Some(v) => {
                self.provider.full_reads.increment();
                self.buffer.copy_from_slice(v.full());
                Ok(&*self.buffer)
            }
            None => Err(AccessedInvalidId(id)),
        };

        std::future::ready(result)
    }
}

impl<Q> provider::CacheableAccessor for FullAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Map = diskann_utils::lifetime::Slice<f32>;

    fn as_cached<'a, 'b>(element: &'a &'b [f32]) -> &'a &'b [f32]
    where
        Self: 'a + 'b,
    {
        element
    }

    fn from_cached<'a>(element: &'a [f32]) -> &'a [f32]
    where
        Self: 'a,
    {
        element
    }
}

impl<Q> SearchExt for FullAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        futures_util::future::ok(vec![self.provider.config.start_id])
    }
}

impl<'a, Q> DelegateNeighbor<'a> for FullAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Delegate = DebugNeighborAccessor<'a, Q>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        DebugNeighborAccessor::new(self.provider)
    }
}

impl<Q> BuildDistanceComputer for FullAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type DistanceComputerError = Panics;
    type DistanceComputer = <f32 as VectorRepr>::Distance;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(f32::distance(
            self.provider.config.metric,
            Some(self.provider.dim()),
        ))
    }
}

impl<Q> BuildQueryComputer<[f32]> for FullAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type QueryComputerError = Panics;
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;

    fn build_query_computer(
        &self,
        from: &[f32],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(f32::query_distance(from, self.provider.config.metric))
    }
}

impl<Q> ExpandBeam<[f32]> for FullAccessor<'_, Q> where Q: DebugQuantizer {}
impl<Q> FillSet for FullAccessor<'_, Q> where Q: DebugQuantizer {}

impl<Q> postprocess::AsDeletionCheck for FullAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Checker = DebugProvider<Q>;
    fn as_deletion_check(&self) -> &Self::Checker {
        self.provider
    }
}

//----------------//
// Quant Accessor //
//----------------//

pub struct QuantAccessor<'a, Q> {
    provider: &'a DebugProvider<Q>,
}

impl<'a, Q> QuantAccessor<'a, Q>
where
    Q: DebugQuantizer,
{
    pub fn new(provider: &'a DebugProvider<Q>) -> Self {
        Self { provider }
    }
}

impl<Q> HasId for QuantAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Id = u32;
}

impl<Q> Accessor for QuantAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Extended = Vec<u8>;
    type Element<'a>
        = Vec<u8>
    where
        Self: 'a;
    type ElementRef<'a> = &'a [u8];

    type GetError = AccessedInvalidId;

    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        let result = match self.provider.data().get(&id) {
            Some(v) => {
                self.provider.quant_reads.increment();
                Ok(v.quant().to_owned())
            }
            None => Err(AccessedInvalidId(id)),
        };

        std::future::ready(result)
    }
}

impl<Q> SearchExt for QuantAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        futures_util::future::ok(vec![self.provider.config.start_id])
    }
}

impl<'a, Q> DelegateNeighbor<'a> for QuantAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Delegate = DebugNeighborAccessor<'a, Q>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        DebugNeighborAccessor::new(self.provider)
    }
}

impl<Q> BuildQueryComputer<[f32]> for QuantAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type QueryComputerError = Panics;
    type QueryComputer = Q::QueryComputer;

    fn build_query_computer(
        &self,
        from: &[f32],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        #[allow(clippy::unwrap_used)]
        Ok(self
            .provider
            .pq_table
            .build_query_computer(self.provider.config.metric, from)
            .unwrap())
    }
}

impl<Q> ExpandBeam<[f32]> for QuantAccessor<'_, Q> where Q: DebugQuantizer {}

impl<Q> postprocess::AsDeletionCheck for QuantAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Checker = DebugProvider<Q>;
    fn as_deletion_check(&self) -> &Self::Checker {
        self.provider
    }
}

//-----------------//
// Hybrid Accessor //
//-----------------//

pub struct HybridAccessor<'a, Q> {
    provider: &'a DebugProvider<Q>,
}

impl<'a, Q> HybridAccessor<'a, Q>
where
    Q: DebugQuantizer,
{
    pub fn new(provider: &'a DebugProvider<Q>) -> Self {
        Self { provider }
    }
}

impl<Q> HasId for HybridAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Id = u32;
}

impl<Q> Accessor for HybridAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Extended = Hybrid<Vec<f32>, Vec<u8>>;
    type Element<'a>
        = Hybrid<Vec<f32>, Vec<u8>>
    where
        Self: 'a;
    type ElementRef<'a> = Hybrid<&'a [f32], &'a [u8]>;

    type GetError = AccessedInvalidId;

    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        let result = match self.provider.data().get(&id) {
            Some(v) => {
                self.provider.full_reads.increment();
                Ok(Hybrid::Full(v.full().to_owned()))
            }
            None => Err(AccessedInvalidId(id)),
        };

        std::future::ready(result)
    }
}

impl<Q> SearchExt for HybridAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        futures_util::future::ok(vec![self.provider.config.start_id])
    }
}

impl<'a, Q> DelegateNeighbor<'a> for HybridAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type Delegate = DebugNeighborAccessor<'a, Q>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        DebugNeighborAccessor::new(self.provider)
    }
}

impl<Q> BuildDistanceComputer for HybridAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    type DistanceComputerError = Panics;
    type DistanceComputer = HybridComputer<Q::DistanceComputer, f32>;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        #[allow(clippy::unwrap_used)]
        Ok(HybridComputer::new(
            self.provider
                .pq_table
                .build_distance_computer(self.provider.config.metric)
                .unwrap(),
            f32::distance(self.provider.config.metric, Some(self.provider.dim())),
        ))
    }
}

impl<Q> FillSet for HybridAccessor<'_, Q>
where
    Q: DebugQuantizer,
{
    async fn fill_set<Itr>(
        &mut self,
        set: &mut HashMap<Self::Id, Self::Extended>,
        itr: Itr,
    ) -> Result<(), Self::GetError>
    where
        Itr: Iterator<Item = Self::Id> + Send + Sync,
    {
        let threshold = 1; // one full vec per fill
        let data = self.provider.data();
        itr.enumerate().for_each(|(i, id)| {
            let e = set.entry(id);
            if i < threshold {
                e.and_modify(|v| {
                    if !v.is_full() {
                        let element = data.get(&id).unwrap();
                        *v = Hybrid::Full(element.full().to_owned());
                    }
                })
                .or_insert_with(|| {
                    let element = data.get(&id).unwrap();
                    Hybrid::Full(element.full().to_owned())
                });
            } else {
                e.or_insert_with(|| {
                    let element = data.get(&id).unwrap();
                    Hybrid::Quant(element.quant().to_owned())
                });
            }
        });
        Ok(())
    }
}

mod postprocess {
    use crate::{
        graph::{SearchOutputBuffer, glue},
        neighbor::Neighbor,
        provider::BuildQueryComputer,
    };

    /// A bridge allowing `Accessors` to opt-in to [`RemoveDeletedIdsAndCopy`] by delegating to
    /// an implementation of the [`DeletionCheck`] trait.
    ///
    /// # Note
    ///
    /// This **must not** be used as a general replacement for [`crate::provider::Delete`].
    /// This must only be used as a performance improvement for [`RemoveDeletedIdsAndCopy`].
    pub(crate) trait AsDeletionCheck {
        type Checker: DeletionCheck;
        fn as_deletion_check(&self) -> &Self::Checker;
    }

    /// A light-weight, synchronous alternative to [`crate::provider::Delete`], targeted at
    /// quickly filtering out deleted IDs during search post-processing.
    ///
    /// For the [`crate::graph::test::debug_provider::FullPrecision`] case, we rely on
    /// constant-propagation and dead code elimination to optimize away filters.
    pub(crate) trait DeletionCheck {
        fn deletion_check(&self, id: u32) -> bool;
    }

    /// A [`glue::SearchPostProcess`] routine that fuses the removal of deleted elements with the
    /// copying of IDs into an output buffer.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct RemoveDeletedIdsAndCopy;

    impl<A, T> glue::SearchPostProcess<A, T> for RemoveDeletedIdsAndCopy
    where
        A: BuildQueryComputer<T, Id = u32> + AsDeletionCheck,
        T: ?Sized,
    {
        type Error = std::convert::Infallible;

        fn post_process<I, B>(
            &self,
            accessor: &mut A,
            _query: &T,
            _computer: &<A as BuildQueryComputer<T>>::QueryComputer,
            candidates: I,
            output: &mut B,
        ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
        where
            I: Iterator<Item = Neighbor<u32>> + Send,
            B: SearchOutputBuffer<u32> + Send + ?Sized,
        {
            let checker = accessor.as_deletion_check();
            let count = output.extend(candidates.filter_map(|n| {
                if checker.deletion_check(n.id) {
                    None
                } else {
                    Some((n.id, n.distance))
                }
            }));
            std::future::ready(Ok(count))
        }
    }
}

////////////////
// Strategies //
////////////////

impl<Q> SearchStrategy<DebugProvider<Q>, [f32]> for Internal<FullPrecision>
where
    Q: DebugQuantizer,
{
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;
    type PostProcessor = postprocess::RemoveDeletedIdsAndCopy;
    type SearchAccessorError = Panics;
    type SearchAccessor<'a> = FullAccessor<'a, Q>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider<Q>,
        _context: &'a <DebugProvider<Q> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl<Q> SearchStrategy<DebugProvider<Q>, [f32]> for FullPrecision
where
    Q: DebugQuantizer,
{
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;
    type PostProcessor = Pipeline<FilterStartPoints, postprocess::RemoveDeletedIdsAndCopy>;
    type SearchAccessorError = Panics;
    type SearchAccessor<'a> = FullAccessor<'a, Q>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider<Q>,
        _context: &'a <DebugProvider<Q> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl<Q> SearchStrategy<DebugProvider<Q>, [f32]> for Internal<Quantized>
where
    Q: DebugQuantizer,
{
    type QueryComputer = Q::QueryComputer;
    type PostProcessor = postprocess::RemoveDeletedIdsAndCopy;
    type SearchAccessorError = Panics;
    type SearchAccessor<'a> = QuantAccessor<'a, Q>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider<Q>,
        _context: &'a <DebugProvider<Q> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl<Q> SearchStrategy<DebugProvider<Q>, [f32]> for Quantized
where
    Q: DebugQuantizer,
{
    type QueryComputer = Q::QueryComputer;
    type PostProcessor = Pipeline<FilterStartPoints, postprocess::RemoveDeletedIdsAndCopy>;
    type SearchAccessorError = Panics;
    type SearchAccessor<'a> = QuantAccessor<'a, Q>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider<Q>,
        _context: &'a <DebugProvider<Q> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl<Q> PruneStrategy<DebugProvider<Q>> for FullPrecision
where
    Q: DebugQuantizer,
{
    type DistanceComputer = <f32 as VectorRepr>::Distance;
    type PruneAccessor<'a> = FullAccessor<'a, Q>;
    type PruneAccessorError = crate::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider<Q>,
        _context: &'a <DebugProvider<Q> as DataProvider>::Context,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(FullAccessor::new(provider))
    }
}

impl<'a, Q> AsElement<&'a [f32]> for FullAccessor<'a, Q>
where
    Q: DebugQuantizer,
{
    type Error = Panics;
    fn as_element(
        &mut self,
        vector: &'a [f32],
        _id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::Error>> + Send {
        std::future::ready(Ok(vector))
    }
}

impl<Q> PruneStrategy<DebugProvider<Q>> for Quantized
where
    Q: DebugQuantizer,
{
    type DistanceComputer = HybridComputer<Q::DistanceComputer, f32>;
    type PruneAccessor<'a> = HybridAccessor<'a, Q>;
    type PruneAccessorError = crate::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider<Q>,
        _context: &'a <DebugProvider<Q> as DataProvider>::Context,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(HybridAccessor::new(provider))
    }
}

impl<'a, Q> AsElement<&'a [f32]> for HybridAccessor<'a, Q>
where
    Q: DebugQuantizer,
{
    type Error = Panics;
    fn as_element(
        &mut self,
        vector: &'a [f32],
        _id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'a>, Self::Error>> + Send {
        std::future::ready(Ok(Hybrid::Full(vector.to_vec())))
    }
}

impl<Q> InsertStrategy<DebugProvider<Q>, [f32]> for FullPrecision
where
    Q: DebugQuantizer,
{
    type PruneStrategy = Self;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    fn insert_search_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider<Q>,
        context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        provider.insert_search_accessor_calls.increment();
        self.search_accessor(provider, context)
    }
}

impl<Q> InsertStrategy<DebugProvider<Q>, [f32]> for Quantized
where
    Q: DebugQuantizer,
{
    type PruneStrategy = Self;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    fn insert_search_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider<Q>,
        context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        provider.insert_search_accessor_calls.increment();
        self.search_accessor(provider, context)
    }
}

impl<Q> InplaceDeleteStrategy<DebugProvider<Q>> for FullPrecision
where
    Q: DebugQuantizer,
{
    type DeleteElement<'a> = [f32];
    type DeleteElementGuard = Vec<f32>;
    type DeleteElementError = Panics;
    type PruneStrategy = Self;
    type SearchStrategy = Internal<Self>;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    fn search_strategy(&self) -> Self::SearchStrategy {
        Internal(*self)
    }

    fn get_delete_element<'a>(
        &'a self,
        provider: &'a DebugProvider<Q>,
        _context: &'a <DebugProvider<Q> as DataProvider>::Context,
        id: <DebugProvider<Q> as DataProvider>::InternalId,
    ) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send
    {
        futures_util::future::ok(provider.data().get(&id).unwrap().full().to_vec())
    }
}

impl<Q> InplaceDeleteStrategy<DebugProvider<Q>> for Quantized
where
    Q: DebugQuantizer,
{
    type DeleteElement<'a> = [f32];
    type DeleteElementGuard = Vec<f32>;
    type DeleteElementError = Panics;
    type PruneStrategy = Self;
    type SearchStrategy = Internal<Self>;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    fn search_strategy(&self) -> Self::SearchStrategy {
        Internal(*self)
    }

    fn get_delete_element<'a>(
        &'a self,
        provider: &'a DebugProvider<Q>,
        _context: &'a <DebugProvider<Q> as DataProvider>::Context,
        id: <DebugProvider<Q> as DataProvider>::InternalId,
    ) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send
    {
        futures_util::future::ok(provider.data().get(&id).unwrap().full().to_vec())
    }
}
