/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::{HashMap, hash_map},
    sync::{
        Arc, RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicUsize, Ordering},
    },
};

use diskann::{
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
use diskann_quantization::CompressInto;
use diskann_vector::distance::Metric;
use thiserror::Error;

use crate::{
    model::{
        FixedChunkPQTable,
        distance::{DistanceComputer, QueryComputer},
        graph::provider::async_::{
            common::{FullPrecision, Internal, Panics, Quantized},
            distances::{self, pq::Hybrid},
            postprocess,
        },
        pq,
    },
    utils::BridgeErr,
};

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
    pub(crate) fn get(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
}

pub struct DebugProvider {
    config: InternalConfig,

    pub pq_table: Arc<FixedChunkPQTable>,
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

impl DebugProvider {
    pub fn new(config: DebugConfig, pq_table: Arc<FixedChunkPQTable>) -> ANNResult<Self> {
        // Compress the start point.
        let mut pq = vec![0u8; pq_table.get_num_chunks()];
        pq_table
            .compress_into(config.start_point.as_slice(), pq.as_mut_slice())
            .bridge_err()?;

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
            pq_table: pq_table.clone(),
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

diskann::always_escalate!(InvalidId);

impl From<InvalidId> for ANNError {
    #[track_caller]
    fn from(err: InvalidId) -> ANNError {
        ANNError::opaque(err)
    }
}

//////////////////
// DataProvider //
//////////////////

impl DataProvider for DebugProvider {
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

impl Delete for DebugProvider {
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

impl provider::SetElement<[f32]> for DebugProvider {
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

        diskann::always_escalate!(AlreadyAssigned);

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

                    let mut pq = vec![0u8; self.pq_table.get_num_chunks()];
                    match self
                        .pq_table
                        .compress_into(element, pq.as_mut_slice())
                        .bridge_err()
                    {
                        Ok(()) => {
                            data.insert(Vector::Valid(Datum::new(element.into(), pq)));
                            neighbors.insert(Vec::new());
                            Ok(provider::NoopGuard::new(*id))
                        }
                        Err(err) => Err(ANNError::from(err)),
                    }
                }
            },
        };

        std::future::ready(result)
    }
}

impl postprocess::DeletionCheck for DebugProvider {
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

diskann::always_escalate!(AccessedInvalidId);

impl From<AccessedInvalidId> for ANNError {
    #[track_caller]
    fn from(err: AccessedInvalidId) -> Self {
        Self::opaque(err)
    }
}

impl DefaultAccessor for DebugProvider {
    type Accessor<'a> = DebugNeighborAccessor<'a>;

    fn default_accessor(&self) -> Self::Accessor<'_> {
        DebugNeighborAccessor::new(self)
    }
}

#[derive(Clone, Copy)]
pub struct DebugNeighborAccessor<'a> {
    provider: &'a DebugProvider,
}

impl<'a> DebugNeighborAccessor<'a> {
    pub fn new(provider: &'a DebugProvider) -> Self {
        Self { provider }
    }
}

impl HasId for DebugNeighborAccessor<'_> {
    type Id = u32;
}

impl NeighborAccessor for DebugNeighborAccessor<'_> {
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

impl NeighborAccessorMut for DebugNeighborAccessor<'_> {
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

pub struct FullAccessor<'a> {
    provider: &'a DebugProvider,
    buffer: Box<[f32]>,
}

impl<'a> FullAccessor<'a> {
    pub fn new(provider: &'a DebugProvider) -> Self {
        let buffer = (0..provider.dim()).map(|_| 0.0).collect();
        Self { provider, buffer }
    }

    pub fn provider(&self) -> &DebugProvider {
        self.provider
    }
}

impl HasId for FullAccessor<'_> {
    type Id = u32;
}

impl Accessor for FullAccessor<'_> {
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

impl diskann::provider::CacheableAccessor for FullAccessor<'_> {
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

impl SearchExt for FullAccessor<'_> {
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        futures_util::future::ok(vec![self.provider.config.start_id])
    }
}

impl<'a> DelegateNeighbor<'a> for FullAccessor<'_> {
    type Delegate = DebugNeighborAccessor<'a>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        DebugNeighborAccessor::new(self.provider)
    }
}

impl BuildDistanceComputer for FullAccessor<'_> {
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

impl BuildQueryComputer<[f32]> for FullAccessor<'_> {
    type QueryComputerError = Panics;
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;

    fn build_query_computer(
        &self,
        from: &[f32],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(f32::query_distance(from, self.provider.config.metric))
    }
}

impl ExpandBeam<[f32]> for FullAccessor<'_> {}
impl FillSet for FullAccessor<'_> {}

impl postprocess::AsDeletionCheck for FullAccessor<'_> {
    type Checker = DebugProvider;
    fn as_deletion_check(&self) -> &Self::Checker {
        self.provider
    }
}

//----------------//
// Quant Accessor //
//----------------//

pub struct QuantAccessor<'a> {
    provider: &'a DebugProvider,
}

impl<'a> QuantAccessor<'a> {
    pub fn new(provider: &'a DebugProvider) -> Self {
        Self { provider }
    }
}

impl HasId for QuantAccessor<'_> {
    type Id = u32;
}

impl Accessor for QuantAccessor<'_> {
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

impl SearchExt for QuantAccessor<'_> {
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        futures_util::future::ok(vec![self.provider.config.start_id])
    }
}

impl<'a> DelegateNeighbor<'a> for QuantAccessor<'_> {
    type Delegate = DebugNeighborAccessor<'a>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        DebugNeighborAccessor::new(self.provider)
    }
}

impl BuildQueryComputer<[f32]> for QuantAccessor<'_> {
    type QueryComputerError = Panics;
    type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;

    fn build_query_computer(
        &self,
        from: &[f32],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(QueryComputer::new(
            self.provider.pq_table.clone(),
            self.provider.config.metric,
            from,
            None,
        )
        .unwrap())
    }
}

impl ExpandBeam<[f32]> for QuantAccessor<'_> {}

impl postprocess::AsDeletionCheck for QuantAccessor<'_> {
    type Checker = DebugProvider;
    fn as_deletion_check(&self) -> &Self::Checker {
        self.provider
    }
}

//-----------------//
// Hybrid Accessor //
//-----------------//

pub struct HybridAccessor<'a> {
    provider: &'a DebugProvider,
}

impl<'a> HybridAccessor<'a> {
    pub fn new(provider: &'a DebugProvider) -> Self {
        Self { provider }
    }
}

impl HasId for HybridAccessor<'_> {
    type Id = u32;
}

impl Accessor for HybridAccessor<'_> {
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

impl SearchExt for HybridAccessor<'_> {
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        futures_util::future::ok(vec![self.provider.config.start_id])
    }
}

impl<'a> DelegateNeighbor<'a> for HybridAccessor<'_> {
    type Delegate = DebugNeighborAccessor<'a>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        DebugNeighborAccessor::new(self.provider)
    }
}

impl BuildDistanceComputer for HybridAccessor<'_> {
    type DistanceComputerError = Panics;
    type DistanceComputer = distances::pq::HybridComputer<f32>;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(distances::pq::HybridComputer::new(
            DistanceComputer::new(self.provider.pq_table.clone(), self.provider.config.metric)
                .unwrap(),
            f32::distance(self.provider.config.metric, Some(self.provider.dim())),
        ))
    }
}

impl FillSet for HybridAccessor<'_> {
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

////////////////
// Strategies //
////////////////

impl SearchStrategy<DebugProvider, [f32]> for Internal<FullPrecision> {
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;
    type PostProcessor = postprocess::RemoveDeletedIdsAndCopy;
    type SearchAccessorError = Panics;
    type SearchAccessor<'a> = FullAccessor<'a>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider,
        _context: &'a <DebugProvider as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl SearchStrategy<DebugProvider, [f32]> for FullPrecision {
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;
    type PostProcessor = Pipeline<FilterStartPoints, postprocess::RemoveDeletedIdsAndCopy>;
    type SearchAccessorError = Panics;
    type SearchAccessor<'a> = FullAccessor<'a>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider,
        _context: &'a <DebugProvider as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl SearchStrategy<DebugProvider, [f32]> for Internal<Quantized> {
    type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;
    type PostProcessor = postprocess::RemoveDeletedIdsAndCopy;
    type SearchAccessorError = Panics;
    type SearchAccessor<'a> = QuantAccessor<'a>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider,
        _context: &'a <DebugProvider as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl SearchStrategy<DebugProvider, [f32]> for Quantized {
    type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;
    type PostProcessor = Pipeline<FilterStartPoints, postprocess::RemoveDeletedIdsAndCopy>;
    type SearchAccessorError = Panics;
    type SearchAccessor<'a> = QuantAccessor<'a>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider,
        _context: &'a <DebugProvider as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl PruneStrategy<DebugProvider> for FullPrecision {
    type DistanceComputer = <f32 as VectorRepr>::Distance;
    type PruneAccessor<'a> = FullAccessor<'a>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider,
        _context: &'a <DebugProvider as DataProvider>::Context,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(FullAccessor::new(provider))
    }
}

impl<'a> AsElement<&'a [f32]> for FullAccessor<'a> {
    type Error = Panics;
    fn as_element(
        &mut self,
        vector: &'a [f32],
        _id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::Error>> + Send {
        std::future::ready(Ok(vector))
    }
}

impl PruneStrategy<DebugProvider> for Quantized {
    type DistanceComputer = distances::pq::HybridComputer<f32>;
    type PruneAccessor<'a> = HybridAccessor<'a>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider,
        _context: &'a <DebugProvider as DataProvider>::Context,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(HybridAccessor::new(provider))
    }
}

impl<'a> AsElement<&'a [f32]> for HybridAccessor<'a> {
    type Error = Panics;
    fn as_element(
        &mut self,
        vector: &'a [f32],
        _id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'a>, Self::Error>> + Send {
        std::future::ready(Ok(Hybrid::Full(vector.to_vec())))
    }
}

impl InsertStrategy<DebugProvider, [f32]> for FullPrecision {
    type PruneStrategy = Self;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    fn insert_search_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider,
        context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        provider.insert_search_accessor_calls.increment();
        self.search_accessor(provider, context)
    }
}

impl InsertStrategy<DebugProvider, [f32]> for Quantized {
    type PruneStrategy = Self;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    fn insert_search_accessor<'a>(
        &'a self,
        provider: &'a DebugProvider,
        context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        provider.insert_search_accessor_calls.increment();
        self.search_accessor(provider, context)
    }
}

impl InplaceDeleteStrategy<DebugProvider> for FullPrecision {
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
        provider: &'a DebugProvider,
        _context: &'a <DebugProvider as DataProvider>::Context,
        id: <DebugProvider as DataProvider>::InternalId,
    ) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send
    {
        futures_util::future::ok(provider.data().get(&id).unwrap().full().to_vec())
    }
}

impl InplaceDeleteStrategy<DebugProvider> for Quantized {
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
        provider: &'a DebugProvider,
        _context: &'a <DebugProvider as DataProvider>::Context,
        id: <DebugProvider as DataProvider>::InternalId,
    ) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send
    {
        futures_util::future::ok(provider.data().get(&id).unwrap().full().to_vec())
    }
}


// TODO: These tests need to be migrated to use diskann-inmem test helpers.
// The test module has been temporarily commented out because it depends on
// `crate::index::diskann_async::tests` which was moved to diskann-inmem.
/*
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use diskann::{
        graph::{self, DiskANNIndex},
        provider::{Guard, SetElement},
        utils::async_tools::VectorIdBoxSlice,
    };
    use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
    use rstest::rstest;

    use super::*;
    use crate::{
        index::diskann_async::{
            tests::{
                GenerateGrid, PagedSearch, check_grid_search, populate_data, populate_graph, squish,
            },
            train_pq,
        },
        test_utils::groundtruth,
        utils,
    };

    #[tokio::test]
    async fn basic_operations() {
        let dim = 2;
        let ctx = &DefaultContext;

        let debug_config = DebugConfig {
            start_id: u32::MAX,
            start_point: vec![0.0; dim],
            max_degree: 10,
            metric: Metric::L2,
        };

        let vectors = [vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let pq_table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim), // Number of PQ chunks is bounded by the dimension.
            &mut crate::utils::create_rnd_from_seed_in_tests(0x04a8832604476965),
            1usize,
        )
        .unwrap();
        let provider = DebugProvider::new(debug_config, Arc::new(pq_table)).unwrap();

        provider
            .set_element(ctx, &0, &[1.0, 1.0])
            .await
            .unwrap()
            .complete()
            .await;

        // internal id = external id
        assert_eq!(provider.to_internal_id(ctx, &0).unwrap(), 0);
        assert_eq!(provider.to_external_id(ctx, 0).unwrap(), 0);

        let mut accessor = FullAccessor::new(&provider);

        let res = accessor.get_element(0).await;
        assert!(res.is_ok());
        assert_eq!(provider.full_reads.get(), 1);

        let mut neighbors = AdjacencyList::new();

        let accessor = provider.default_accessor();
        let res = accessor.get_neighbors(0, &mut neighbors).await;
        assert!(res.is_ok());
        assert_eq!(provider.neighbor_reads.get(), 1);

        let accessor = provider.default_accessor();
        let res = accessor.set_neighbors(0, &[1, 2, 3]).await;
        assert!(res.is_ok());
        assert_eq!(provider.neighbor_writes.get(), 1);

        // delete and release vector 0
        let res = provider.delete(&DefaultContext, &0).await;
        assert!(res.is_ok());
        assert_eq!(
            ElementStatus::Deleted,
            provider
                .status_by_external_id(&DefaultContext, &0)
                .await
                .unwrap()
        );

        let mut accessor = FullAccessor::new(&provider);
        let res = accessor.get_element(0).await;
        assert!(res.is_ok());
        assert_eq!(provider.full_reads.get(), 2);

        let mut accessor = HybridAccessor::new(&provider);
        let res = accessor.get_element(0).await;
        assert!(res.is_ok());
        assert_eq!(provider.full_reads.get(), 3);

        // Releasing should make the element unreachable.
        let res = provider.release(&DefaultContext, 0).await;
        assert!(res.is_ok());
        assert!(
            provider
                .status_by_external_id(&DefaultContext, &0)
                .await
                .is_err()
        );
    }

    pub fn new_quant_index(
        index_config: graph::Config,
        debug_config: DebugConfig,
        pq_table: FixedChunkPQTable,
    ) -> Arc<DiskANNIndex<DebugProvider>> {
        let data = DebugProvider::new(debug_config, Arc::new(pq_table)).unwrap();
        Arc::new(DiskANNIndex::new(index_config, data, None))
    }

    #[rstest]
    #[case(1, 100)]
    #[case(3, 7)]
    #[case(4, 5)]
    #[tokio::test]
    async fn grid_search(#[case] dim: usize, #[case] grid_size: usize) {
        let l = 10;
        let max_degree = 2 * dim;
        let num_points = (grid_size).pow(dim as u32);
        let start_id = u32::MAX;

        let index_config = graph::config::Builder::new(
            max_degree,
            graph::config::MaxDegree::default_slack(),
            l,
            (Metric::L2).into(),
        )
        .build()
        .unwrap();

        let debug_config = DebugConfig {
            start_id,
            start_point: vec![grid_size as f32; dim],
            max_degree,
            metric: Metric::L2,
        };

        let adjacency_lists = match dim {
            1 => utils::generate_1d_grid_adj_list(grid_size as u32),
            3 => utils::genererate_3d_grid_adj_list(grid_size as u32),
            4 => utils::generate_4d_grid_adj_list(grid_size as u32),
            _ => panic!("Unsupported number of dimensions"),
        };
        let mut vectors = f32::generate_grid(dim, grid_size);

        assert_eq!(adjacency_lists.len(), num_points);
        assert_eq!(vectors.len(), num_points);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim), // Number of PQ chunks is bounded by the dimension.
            &mut crate::utils::create_rnd_from_seed_in_tests(0x04a8832604476965),
            1usize,
        )
        .unwrap();

        let index = new_quant_index(index_config, debug_config, table);
        {
            let mut neighbor_accessor = index.provider().default_accessor();
            populate_data(index.provider(), &DefaultContext, &vectors).await;
            populate_graph(&mut neighbor_accessor, &adjacency_lists).await;

            // Set the adjacency list for the start point.
            neighbor_accessor
                .set_neighbors(start_id, &[num_points as u32 - 1])
                .await
                .unwrap();
        }

        // The corpus of actual vectors consists of all but the last point, which we use
        // as the start point.
        //
        // So, when we compute the corpus used during groundtruth generation, we take all
        // but this last point.
        let corpus: diskann_utils::views::Matrix<f32> =
            squish(vectors.iter().take(num_points), dim);

        let mut paged_tests = Vec::new();

        // Test with the zero query.
        let query = vec![0.0; dim];
        let gt = groundtruth(corpus.as_view(), &query, |a, b| SquaredL2::evaluate(a, b));
        paged_tests.push(PagedSearch::new(query, gt));

        // Test with the start point to ensure it is filtered out.
        let query = vectors.last().unwrap();
        let gt = groundtruth(corpus.as_view(), query, |a, b| SquaredL2::evaluate(a, b));
        paged_tests.push(PagedSearch::new(query.clone(), gt));

        // Unfortunately - this is needed for the `check_grid_search` test.
        vectors.push(index.provider().config.start_point.full().to_owned());
        check_grid_search(&index, &vectors, &paged_tests, FullPrecision, Quantized).await;
    }

    #[rstest]
    #[tokio::test]
    async fn grid_search_with_build(
        #[values((1, 100), (3, 7), (4, 5))] dim_and_size: (usize, usize),
    ) {
        let dim = dim_and_size.0;
        let grid_size = dim_and_size.1;
        let start_id = u32::MAX;

        let l = 10;

        // NOTE: Be careful changing `max_degree`. It needs to be high enough that the
        // graph is navigable, but low enough that the batch parallel handling inside
        // `multi_insert` is needed for the multi-insert graph to be navigable.
        //
        // With the current configured values, removing the other elements in the batch
        // from the visited set during `multi_insert` results in a graph failure.
        let max_degree = 2 * dim;

        let num_points = (grid_size).pow(dim as u32);

        let index_config = graph::config::Builder::new_with(
            max_degree,
            graph::config::MaxDegree::default_slack(),
            l,
            (Metric::L2).into(),
            |b| {
                b.max_minibatch_par(10);
            },
        )
        .build()
        .unwrap();

        let debug_config = DebugConfig {
            start_id,
            start_point: vec![grid_size as f32; dim],
            max_degree: index_config.max_degree().into(),
            metric: Metric::L2,
        };

        let mut vectors = f32::generate_grid(dim, grid_size);
        assert_eq!(vectors.len(), num_points);

        // This is a little subtle, but we need `vectors` to contain the start point as
        // its last element, but we **don't** want to include it in the index build.
        //
        // This basically means that we need to be careful with index initialization.
        vectors.push(vec![grid_size as f32; dim]);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim), // Number of PQ chunks is bounded by the dimension.
            &mut crate::utils::create_rnd_from_seed_in_tests(0x04a8832604476965),
            1usize,
        )
        .unwrap();

        // Initialize an index for a new round of building.
        let init_index =
            || new_quant_index(index_config.clone(), debug_config.clone(), table.clone());

        // Build with full-precision single insert
        {
            let index = init_index();
            let ctx = DefaultContext;
            for (i, v) in vectors.iter().take(num_points).enumerate() {
                index
                    .insert(FullPrecision, &ctx, &(i as u32), v.as_slice())
                    .await
                    .unwrap();
            }

            // Ensure the `insert_search_accessor` API is invoked.
            assert_eq!(
                index.provider().insert_search_accessor_calls.get(),
                num_points,
                "insert should invoke `insert_search_accessor`",
            );

            check_grid_search(&index, &vectors, &[], FullPrecision, Quantized).await;
        }

        // Build with quantized single insert
        {
            let index = init_index();
            let ctx = DefaultContext;
            for (i, v) in vectors.iter().take(num_points).enumerate() {
                index
                    .insert(Quantized, &ctx, &(i as u32), v.as_slice())
                    .await
                    .unwrap();
            }

            // Ensure the `insert_search_accessor` API is invoked.
            assert_eq!(
                index.provider().insert_search_accessor_calls.get(),
                num_points,
                "insert should invoke `insert_search_accessor`",
            );

            check_grid_search(&index, &vectors, &[], FullPrecision, Quantized).await;
        }

        // Build with full-precision multi-insert
        {
            let index = init_index();
            let ctx = DefaultContext;
            let batch: Box<[_]> = vectors
                .iter()
                .take(num_points)
                .enumerate()
                .map(|(id, v)| VectorIdBoxSlice::new(id as u32, v.as_slice().into()))
                .collect();

            index
                .multi_insert(FullPrecision, &ctx, batch)
                .await
                .unwrap();

            // Ensure the `insert_search_accessor` API is invoked.
            assert_eq!(
                index.provider().insert_search_accessor_calls.get(),
                num_points,
                "multi-insert should invoke `insert_search_accessor`",
            );

            check_grid_search(&index, &vectors, &[], FullPrecision, Quantized).await;
        }

        // Build with quantized multi-insert
        {
            let index = init_index();
            let ctx = DefaultContext;
            let batch: Box<[_]> = vectors
                .iter()
                .take(num_points)
                .enumerate()
                .map(|(id, v)| VectorIdBoxSlice::new(id as u32, v.as_slice().into()))
                .collect();

            index.multi_insert(Quantized, &ctx, batch).await.unwrap();

            // Ensure the `insert_search_accessor` API is invoked.
            assert_eq!(
                index.provider().insert_search_accessor_calls.get(),
                num_points,
                "multi-insert should invoke `insert_search_accessor`",
            );

            check_grid_search(&index, &vectors, &[], FullPrecision, Quantized).await;
        }
    }
}
*/
