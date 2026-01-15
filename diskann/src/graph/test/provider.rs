/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A pedantic provider implementation used for testing alorithmic logic.

use std::{
    collections::{HashMap, HashSet, hash_map},
    num::NonZeroUsize,
    sync::Arc,
};

use diskann_vector::distance::Metric;
use thiserror::Error;

use crate::{
    ANNError, ANNResult,
    error::{Infallible, message},
    graph::{AdjacencyList, glue},
    internal::{
        buckets::Buckets,
        counter::{Counter, LocalCounter},
    },
    provider,
    utils::{IntoUsize, VectorRepr},
};

/// A starting point for graph search algorithms.
///
/// # Examples
///
/// ```rust
/// use diskann::graph::test::provider::StartPoint;
///
/// // Create a starting point with ID 1 and a 3-dimensional vector
/// let start_point = StartPoint::new(1, vec![0.5, 1.2, -0.8]);
///
/// assert_eq!(start_point.id(), 1);
/// assert_eq!(start_point.vector(), &[0.5, 1.2, -0.8]);
/// ```
#[derive(Debug)]
pub struct StartPoint {
    id: u32,
    vector: Vec<f32>,
}

impl StartPoint {
    /// Construct a new start point with the given ID and vector.
    pub fn new(id: u32, vector: Vec<f32>) -> Self {
        Self { id, vector }
    }

    //// Return the ID of the start point.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Return the vector of the start point.
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }
}

impl IntoIterator for StartPoint {
    type Item = Self;
    type IntoIter = std::iter::Once<Self>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self)
    }
}

/// Configuration for the test provider.
///
/// # Examples
///
/// ```rust
/// use diskann::graph::test::provider::{Config, StartPoint};
/// use diskann_vector::distance::Metric;
///
/// let start_points = vec![
///     StartPoint::new(0, vec![1.0, 2.0, 3.0]),
///     StartPoint::new(1, vec![4.0, 5.0, 6.0]),
/// ];
///
/// let config = Config::new(Metric::L2, 10, start_points).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Config {
    start_points: HashMap<u32, Vec<f32>>,
    max_degree: NonZeroUsize,
    dim: NonZeroUsize,
    metric: Metric,
}

impl Config {
    /// Create a new [`Config`] instance.
    ///
    /// # Errors
    ///
    /// Returns an error in the following cases:
    ///
    /// * `max_degree` is zero.
    /// * No start points are provided.
    /// * The dimensions of the provided start points do not match or are zero.
    /// * The provided start points are not unique.
    pub fn new<I>(metric: Metric, max_degree: usize, start_points: I) -> Result<Self, ConfigError>
    where
        I: IntoIterator<Item = StartPoint>,
    {
        let max_degree = match NonZeroUsize::new(max_degree) {
            Some(max_degree) => max_degree,
            None => return Err(ConfigError::MaxDegreeCannotBeZero),
        };

        let mut dim: Option<NonZeroUsize> = None;
        let mut count = 0;
        let start_points = start_points
            .into_iter()
            .map(|point| {
                match dim {
                    None => {
                        dim = NonZeroUsize::new(point.vector.len());
                    }
                    Some(dim) => {
                        if dim.get() != point.vector.len() {
                            return Err(ConfigError::MismatchedDims);
                        }
                    }
                }
                count += 1;
                Ok((point.id, point.vector))
            })
            .collect::<Result<HashMap<u32, Vec<f32>>, ConfigError>>()?;

        if start_points.is_empty() {
            return Err(ConfigError::NeedStartPoint);
        }

        if start_points.len() != count {
            return Err(ConfigError::StartPointsNotUnique);
        }

        let dim = match dim {
            None => return Err(ConfigError::DimCannotBeZero),
            Some(dim) => dim,
        };

        Ok(Self {
            start_points,
            max_degree,
            dim,
            metric,
        })
    }
}

/// Error conditions for [`Config::new`].
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("at least one start point must be specified")]
    NeedStartPoint,
    #[error("start points must be unique")]
    StartPointsNotUnique,
    #[error("not all start points have the same dimension")]
    MismatchedDims,
    #[error("start point dimension must be non-zero")]
    DimCannotBeZero,
    #[error("max degree must be non-zero")]
    MaxDegreeCannotBeZero,
}

// The number of buckets to use for the light-weight concurrent hash map.
const BUCKETS: usize = 64;

/// A test data provider for validating DiskANN API guarantees.
///
/// The following is a list of properties this provider seeks to maintain:
///
/// * Start points provided at construction time are immutable and cannot be deleted or modified.
/// * All calls to [`provider::NeighborAccessorMut::set_neighbors`] do not contain duplicates.
/// * All calls to [`provider::NeighborAccessorMut::append_vector`] do not contain duplicates
///   and are disjoint with the current adjacency list.
/// * Vectors can be marked as deleted, but their data remains accessible.
/// * Vectors that are deleted but not [`provider::Delete::release`]d cannot be overwritten.
/// * Attempting to retrieve and ID that is not present is an error.
/// * All attempts to mutate the graph via [`provider::NeighborAccessorMut`] must be preceeded
///   by [`provider::SetElement`].
///
/// This provider allows for some amount of concurrent access, but is not optimized for performance.
#[derive(Debug)]
pub struct Provider {
    terms: Buckets<HashMap<u32, Term>, BUCKETS>,
    config: Config,

    // Counters
    pub(crate) get_vector: Counter,
    pub(crate) set_vector: Counter,
    pub(crate) get_neighbors: Counter,
    pub(crate) set_neighbors: Counter,
    pub(crate) append_neighbors: Counter,
}

impl Provider {
    /// Create a new [`Provider`].
    ///
    /// All counters will be initialized to zero.
    pub fn new(config: Config) -> Self {
        let mut this = Self {
            terms: Buckets::new(),
            config,
            get_vector: Counter::new(),
            set_vector: Counter::new(),
            get_neighbors: Counter::new(),
            set_neighbors: Counter::new(),
            append_neighbors: Counter::new(),
        };

        for (id, value) in this.config.start_points.iter() {
            this.terms.get_mut(id.into_usize()).insert(
                *id,
                Term {
                    data: Vector::Valid(value.clone()),
                    neighbors: AdjacencyList::new(),
                },
            );
        }

        this
    }

    /// Create a new [`Provider`] from the given configuration, start points, and points.
    ///
    /// This method is used to pre-initialize a provider to assist with search-only tests
    /// and performs the following checks:
    ///
    /// * All IDs yielded by the `start_points` iterator must indeed be start points in `config`.
    /// * The IDs in the `points` iterator must not overlap with the start points.
    /// * All data vectors in the `points` iterator must be equal to the dimension in `config`.
    /// * All adjacency lists must be within the maximum degree specified in `config`.
    ///
    /// After initialization, all adjacency lists are verified to ensure that they only
    /// contain either start point IDs or IDs yielded by the `points` iterator.
    pub fn new_from<I, T>(config: Config, start_points: I, points: T) -> ANNResult<Self>
    where
        I: IntoIterator<Item = (u32, AdjacencyList<u32>)>,
        T: IntoIterator<Item = (u32, Vec<f32>, AdjacencyList<u32>)>,
    {
        let mut this = Self::new(config);
        let max_degree = this.config.max_degree.get();

        // Add the start points.
        for (id, neighbors) in start_points {
            if neighbors.len() > max_degree {
                return Err(message!(
                    "start point {} has neighbors with length {} when max degree is {}",
                    id,
                    neighbors.len(),
                    max_degree
                ));
            }

            if let Some(term) = this.terms.get_mut(id.into_usize()).get_mut(&id) {
                term.neighbors = neighbors;
            } else {
                return Err(message!("id {} is not a valid start point", id));
            }
        }

        // Add the remaining points.
        for (id, data, neighbors) in points {
            if this.is_start_point(id) {
                return Err(message!(
                    "cannot assign start point {} through a regular point",
                    id
                ));
            }

            if neighbors.len() > max_degree {
                return Err(message!(
                    "point {} has neighbors with length {} when max degree is {}",
                    id,
                    neighbors.len(),
                    max_degree
                ));
            }

            if data.len() != this.dim() {
                return Err(message!(
                    "data for id {} has length {} but the provider is expecting dim {}",
                    id,
                    data.len(),
                    this.dim(),
                ));
            }

            let term = Term {
                data: Vector::Valid(data),
                neighbors,
            };

            this.terms.get_mut(id.into_usize()).insert(id, term);
        }

        // Now that we have inserted all the points - ensure our graph is consistent.
        this.is_consistent()?;
        Ok(this)
    }

    /// Return the dimensionality of data contained by this provider.
    pub fn dim(&self) -> usize {
        self.config.dim.get()
    }

    /// Return the largest degree this provider is capable of holding.
    pub fn max_degree(&self) -> usize {
        self.config.max_degree.get()
    }

    /// Return `true` is `id` is a start point. Otherwise, return `false`.
    fn is_start_point(&self, id: u32) -> bool {
        self.config.start_points.contains_key(&id)
    }

    /// Return an approximation of the collection of internal IDs currently in the index.
    ///
    /// This is approximate as it is not atomic. Thus, it is possible for other threads to
    /// update the collection of internal IDs while this operation is executing.
    pub fn all_internal_ids(&self) -> HashSet<u32> {
        let mut all = HashSet::<u32>::new();
        for i in 0..BUCKETS {
            let bucket = self.terms.blocking_read(i);
            all.extend(bucket.keys());
        }
        all
    }

    /// Check whether all adjacency lists in `self` point to valid IDs.
    pub fn is_consistent(&self) -> ANNResult<()> {
        let all = self.all_internal_ids();
        for i in 0..BUCKETS {
            let bucket = self.terms.blocking_read(i);
            for (id, term) in bucket.iter() {
                for neighbor in term.neighbors.iter() {
                    if !all.contains(neighbor) {
                        return Err(message!(
                            "term with id {} has neighbors {:?} \
                             but neighbor {} is not in the provider",
                            id,
                            term.neighbors,
                            neighbor,
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Return `true` if `id` is present in the provider but marked as deleted.
    ///
    /// If `id` is present but not marked deleted, returns `false`.
    ///
    /// An error is returned if `id` is not present in the provider.
    async fn is_deleted(&self, id: u32) -> Result<bool, InvalidId> {
        if let Some(term) = self.terms.read(id.into_usize()).await.get(&id) {
            Ok(term.is_deleted())
        } else {
            Err(InvalidId::Internal(id))
        }
    }

    /// Return the metrics recorded in the provider.
    pub fn metrics(&self) -> Metrics {
        Metrics {
            get_vector: self.get_vector.value(),
            set_vector: self.set_vector.value(),
            get_neighbors: self.get_neighbors.value(),
            set_neighbors: self.set_neighbors.value(),
            append_neighbors: self.append_neighbors.value(),
        }
    }
}

/// Provider level metrics.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct Metrics {
    pub get_vector: usize,
    pub set_vector: usize,
    pub get_neighbors: usize,
    pub set_neighbors: usize,
    pub append_neighbors: usize,
}

#[cfg(test)]
crate::test::cmp::verbose_eq!(Metrics {
    get_vector,
    set_vector,
    get_neighbors,
    set_neighbors,
    append_neighbors
});

#[derive(Debug)]
struct Term {
    neighbors: AdjacencyList<u32>,
    data: Vector,
}

impl Term {
    fn mark_deleted(&mut self) {
        self.data.mark_deleted()
    }

    fn is_deleted(&self) -> bool {
        self.data.is_deleted()
    }
}

/// A data vector that records whether or not it has been deleted.
#[derive(Debug)]
enum Vector {
    Valid(Vec<f32>),
    Deleted(Vec<f32>),
}

impl Vector {
    /// Change `self` to be `Self::Deleted`, leaving the internal data unchanged.
    fn mark_deleted(&mut self) {
        *self = match self.take() {
            Self::Valid(v) => Self::Deleted(v),
            Self::Deleted(v) => Self::Deleted(v),
        }
    }

    /// Take the internal data and construct a new instance of `Self`.
    ///
    /// Leave the caller with an empty data.
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
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match self {
            Self::Valid(v) => v,
            Self::Deleted(v) => v,
        }
    }
}

/////////////
// Context //
/////////////

/// The execution context used by the test provider.
///
/// This records the number of task spawns launched by this context.
#[derive(Debug)]
pub struct Context(Arc<ContextInner>);

impl Context {
    /// Create a new context.
    pub fn new() -> Self {
        let inner = ContextInner {
            spawns: Counter::new(),
            clones: Counter::new(),
        };

        Self(Arc::new(inner))
    }

    /// Return the number of spawns made through this context.
    pub fn spawns(&self) -> usize {
        self.0.spawns.value()
    }

    /// Return the number of clones made of the context.
    pub fn clones(&self) -> usize {
        self.0.clones.value()
    }

    /// Aggregate the context level metrics.
    pub fn metrics(&self) -> ContextMetrics {
        ContextMetrics {
            spawns: self.spawns(),
            clones: self.clones(),
        }
    }
}

/// Metrics recorded by [`Context`].
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct ContextMetrics {
    pub spawns: usize,
    pub clones: usize,
}

#[cfg(test)]
crate::test::cmp::verbose_eq!(ContextMetrics { spawns, clones });

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl provider::ExecutionContext for Context {
    fn wrap_spawn<F, T>(&self, f: F) -> impl Future<Output = T> + Send + 'static
    where
        F: Future<Output = T> + Send + 'static,
    {
        self.0.spawns.increment();
        f
    }
}

impl Clone for Context {
    fn clone(&self) -> Self {
        self.0.clones.increment();
        Self(self.0.clone())
    }
}

#[derive(Debug)]
struct ContextInner {
    spawns: Counter,
    clones: Counter,
}

//////////////////
// DataProvider //
//////////////////

/// Light-weight error type for reporting access to an invalid ID.
#[derive(Debug, Clone, Copy, Error)]
pub enum InvalidId {
    #[error("internal id {0} is not initialized")]
    Internal(u32),
    #[error("external id {0} is not initialized")]
    External(u32),
    #[error("cannot delete start point {0}")]
    IsStartPoint(u32),
}

crate::always_escalate!(InvalidId);

impl From<InvalidId> for ANNError {
    #[track_caller]
    fn from(err: InvalidId) -> ANNError {
        ANNError::opaque(err)
    }
}

impl provider::DataProvider for Provider {
    type Context = Context;
    type InternalId = u32;
    type ExternalId = u32;

    type Error = InvalidId;

    fn to_internal_id(&self, _context: &Context, gid: &u32) -> Result<u32, InvalidId> {
        let valid = self.terms.blocking_read(gid.into_usize()).contains_key(gid);
        if valid {
            Ok(*gid)
        } else {
            Err(InvalidId::External(*gid))
        }
    }

    fn to_external_id(&self, _context: &Context, id: u32) -> Result<u32, InvalidId> {
        let valid = self.terms.blocking_read(id.into_usize()).contains_key(&id);
        if valid {
            Ok(id)
        } else {
            Err(InvalidId::Internal(id))
        }
    }
}

impl provider::Delete for Provider {
    async fn delete(
        &self,
        _context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<(), Self::Error> {
        if self.config.start_points.contains_key(gid) {
            return Err(InvalidId::IsStartPoint(*gid));
        }

        let mut guard = self.terms.write(gid.into_usize()).await;
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
        if self.config.start_points.contains_key(&id) {
            return Err(InvalidId::IsStartPoint(id));
        }

        let mut guard = self.terms.write(id.into_usize()).await;
        if guard.remove(&id).is_none() {
            Err(InvalidId::Internal(id))
        } else {
            Ok(())
        }
    }

    async fn status_by_internal_id(
        &self,
        _context: &Context,
        id: u32,
    ) -> Result<provider::ElementStatus, Self::Error> {
        if self.is_deleted(id).await? {
            Ok(provider::ElementStatus::Deleted)
        } else {
            Ok(provider::ElementStatus::Valid)
        }
    }

    fn status_by_external_id(
        &self,
        context: &Context,
        gid: &u32,
    ) -> impl Future<Output = Result<provider::ElementStatus, Self::Error>> + Send {
        self.status_by_internal_id(context, *gid)
    }
}

impl provider::SetElement<[f32]> for Provider {
    type SetError = ANNError;
    type Guard = provider::NoopGuard<u32>;

    async fn set_element(
        &self,
        _context: &Context,
        id: &Self::ExternalId,
        element: &[f32],
    ) -> Result<Self::Guard, Self::SetError> {
        #[derive(Debug, Clone, Copy, Error)]
        enum SetError {
            #[error("vector id {0} is already assigned")]
            AlreadyAssigned(u32),
            #[error("wrong dim - got {0}, expected {1}")]
            WrongDim(usize, usize),
        }

        crate::always_escalate!(SetError);

        impl From<SetError> for ANNError {
            #[track_caller]
            fn from(err: SetError) -> Self {
                Self::new(crate::ANNErrorKind::IndexError, err)
            }
        }

        // Ensure that the assigned vector has the correct length.
        if element.len() != self.dim() {
            return Err(SetError::WrongDim(element.len(), self.dim()).into());
        }

        match self.terms.write(id.into_usize()).await.entry(*id) {
            hash_map::Entry::Occupied(_) => Err(SetError::AlreadyAssigned(*id).into()),
            hash_map::Entry::Vacant(term) => {
                term.insert(Term {
                    neighbors: AdjacencyList::new(),
                    data: Vector::Valid(element.into()),
                });
                self.set_vector.increment();
                Ok(provider::NoopGuard::new(*id))
            }
        }
    }
}

//////////////////////
// NeighborAccessor //
//////////////////////

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

impl provider::DefaultAccessor for Provider {
    type Accessor<'a> = NeighborAccessor<'a>;

    fn default_accessor(&self) -> Self::Accessor<'_> {
        NeighborAccessor::new(self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NeighborAccessor<'a> {
    provider: &'a Provider,
}

impl<'a> NeighborAccessor<'a> {
    pub fn new(provider: &'a Provider) -> Self {
        Self { provider }
    }
}

impl provider::HasId for NeighborAccessor<'_> {
    type Id = u32;
}

impl provider::NeighborAccessor for NeighborAccessor<'_> {
    async fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> ANNResult<Self> {
        match self.provider.terms.read(id.into_usize()).await.get(&id) {
            Some(v) => {
                self.provider.get_neighbors.increment();
                neighbors.overwrite_trusted(&v.neighbors);
                Ok(self)
            }
            None => Err(ANNError::opaque(AccessedInvalidId(id))),
        }
    }
}

impl provider::NeighborAccessorMut for NeighborAccessor<'_> {
    async fn set_neighbors(self, id: Self::Id, neighbors: &[Self::Id]) -> ANNResult<Self> {
        if neighbors.len() > self.provider.max_degree() {
            return Err(message!(
                "trying to assign neighbors with length {} when max degree is {}",
                neighbors.len(),
                self.provider.max_degree()
            ));
        }

        match self
            .provider
            .terms
            .write(id.into_usize())
            .await
            .get_mut(&id)
        {
            Some(term) => {
                term.neighbors.clear();
                term.neighbors.extend_from_slice(neighbors);

                // Even if we return an error due to duplicate neighbors, the assignment
                // still sticks. Since it doesn't exceed the graph degree, it's still a valid
                // assignment.
                self.provider.set_neighbors.increment();

                // Check whether or not the input slice was unique. We can do this with
                // a length check because `extend_from_slice` deduplicates.
                //
                // If `neighbors` did have duplicates, then the final length of `v` will
                // be smaller.
                if term.neighbors.len() != neighbors.len() {
                    Err(message!("duplicate neighbors detected"))
                } else {
                    Ok(self)
                }
            }
            None => Err(ANNError::opaque(AccessedInvalidId(id))),
        }
    }

    async fn append_vector(self, id: Self::Id, neighbors: &[Self::Id]) -> ANNResult<Self> {
        match self
            .provider
            .terms
            .write(id.into_usize())
            .await
            .get_mut(&id)
        {
            Some(term) => {
                // Do not allow `append_vector` to exceed the max degree.
                if let Some(estimate) = term.neighbors.len().checked_add(neighbors.len()) {
                    if estimate > self.provider.max_degree() {
                        return Err(message!(
                            "append neighbors to {} will exceed the max degree",
                            id
                        ));
                    }
                } else {
                    return Err(message!("the number of neighbors is way too high"));
                }

                let added = term.neighbors.extend_from_slice(neighbors);
                self.provider.append_neighbors.increment();
                if added != neighbors.len() {
                    Err(message!("duplicate ids in append-vector"))
                } else {
                    Ok(self)
                }
            }
            None => Err(ANNError::opaque(AccessedInvalidId(id))),
        }
    }
}

//////////////
// Accessor //
//////////////

#[derive(Debug)]
pub struct Accessor<'a> {
    provider: &'a Provider,
    buffer: Box<[f32]>,
    get_vector: LocalCounter<'a>,
}

impl<'a> Accessor<'a> {
    pub fn new(provider: &'a Provider) -> Self {
        let buffer = (0..provider.dim()).map(|_| 0.0).collect();
        Self {
            provider,
            buffer,
            get_vector: provider.get_vector.local(),
        }
    }
}

impl provider::HasId for Accessor<'_> {
    type Id = u32;
}

impl provider::Accessor for Accessor<'_> {
    type Extended = Box<[f32]>;
    type Element<'a>
        = &'a [f32]
    where
        Self: 'a;
    type ElementRef<'a> = &'a [f32];
    type GetError = AccessedInvalidId;

    async fn get_element(&mut self, id: u32) -> Result<&[f32], AccessedInvalidId> {
        match self.provider.terms.read(id.into_usize()).await.get(&id) {
            Some(term) => {
                self.get_vector.increment();
                self.buffer.copy_from_slice(&term.data);
                Ok(&*self.buffer)
            }
            None => Err(AccessedInvalidId(id)),
        }
    }
}

impl<'a> provider::DelegateNeighbor<'a> for Accessor<'_> {
    type Delegate = NeighborAccessor<'a>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        NeighborAccessor::new(self.provider)
    }
}

impl provider::BuildQueryComputer<[f32]> for Accessor<'_> {
    type QueryComputerError = Infallible;
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;

    fn build_query_computer(
        &self,
        from: &[f32],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(f32::query_distance(from, self.provider.config.metric))
    }
}

//------//
// Glue //
//------//

impl glue::SearchExt for Accessor<'_> {
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> + Send {
        futures_util::future::ok(self.provider.config.start_points.keys().copied().collect())
    }
}

impl glue::ExpandBeam<[f32]> for Accessor<'_> {}
impl glue::FillSet for Accessor<'_> {}

#[derive(Debug, Default, Clone, Copy)]
pub struct Strategy {
    _phantom: (),
}

impl Strategy {
    pub fn new() -> Self {
        Self { _phantom: () }
    }
}

impl glue::SearchStrategy<Provider, [f32]> for Strategy {
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;
    type PostProcessor = glue::CopyIds;
    type SearchAccessorError = Infallible;
    type SearchAccessor<'a> = Accessor<'a>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        _context: &'a Context,
    ) -> Result<Accessor<'a>, Infallible> {
        Ok(Accessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test::{assert_message_contains, tokio::current_thread_runtime};

    #[test]
    fn test_start_point() {
        let start_point = StartPoint::new(42, vec![1.0, 2.0, 3.0]);

        assert_eq!(start_point.id(), 42);
        assert_eq!(start_point.vector(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_config_new() {
        let metric = Metric::L2;

        // Happy path - valid configuration
        {
            let start_points = [
                StartPoint::new(0, vec![1.0, 2.0, 3.0]),
                StartPoint::new(1, vec![4.0, 5.0, 6.0]),
            ];
            let config = Config::new(metric, 10, start_points).unwrap();
            assert_eq!(config.max_degree.get(), 10);
            assert_eq!(config.dim.get(), 3);
            assert_eq!(config.metric, metric);
            assert_eq!(config.start_points.len(), 2);
        }

        // Error: max_degree is zero
        {
            let err = Config::new(metric, 0, StartPoint::new(0, vec![1.0, 2.0])).unwrap_err();
            assert!(matches!(err, ConfigError::MaxDegreeCannotBeZero));
        }

        // Error: no start points provided
        {
            let err = Config::new(metric, 10, []).unwrap_err();
            assert!(matches!(err, ConfigError::NeedStartPoint));
        }

        // Error: mismatched dimensions
        {
            let start_points = [
                StartPoint::new(0, vec![1.0, 2.0, 3.0]),
                StartPoint::new(1, vec![4.0, 5.0]), // Different dimension
            ];
            let err = Config::new(metric, 10, start_points).unwrap_err();
            assert!(matches!(err, ConfigError::MismatchedDims));
        }

        // Error: zero dimension (empty vectors)
        {
            let err = Config::new(metric, 10, StartPoint::new(0, vec![])).unwrap_err();
            assert!(matches!(err, ConfigError::DimCannotBeZero));
        }

        // Error: duplicate start point IDs
        {
            let start_points = [
                StartPoint::new(0, vec![1.0, 2.0]),
                StartPoint::new(0, vec![3.0, 4.0]), // Same ID
            ];
            let err = Config::new(metric, 10, start_points).unwrap_err();
            assert!(matches!(err, ConfigError::StartPointsNotUnique));
        }
    }

    #[test]
    fn test_vector() {
        let vector = vec![1.0, 2.0, 3.0];
        let ptr = vector.as_ptr();
        let mut v = Vector::Valid(vector);
        assert!(!v.is_deleted());

        // `v.as_ptr` goes through `Deref` to a slice.
        assert_eq!(v.as_ptr(), ptr);

        v.mark_deleted();
        assert!(v.is_deleted());
        assert_eq!(v.as_ptr(), ptr);

        v.mark_deleted();
        assert!(v.is_deleted());
        assert_eq!(v.as_ptr(), ptr);
    }

    #[test]
    fn test_term() {
        let vector = vec![1.0, 2.0, 3.0];
        let ptr = vector.as_ptr();
        let mut t = Term {
            neighbors: AdjacencyList::new(),
            data: Vector::Valid(vector),
        };

        assert!(!t.is_deleted());
        assert_eq!(t.data.as_ptr(), ptr);

        t.mark_deleted();
        assert!(t.is_deleted());
        assert_eq!(t.data.as_ptr(), ptr);

        t.mark_deleted();
        assert!(t.is_deleted());
        assert_eq!(t.data.as_ptr(), ptr);
    }

    #[test]
    fn test_context() {
        use provider::ExecutionContext;

        let context = Context::default();
        let ContextMetrics { spawns, clones } = context.metrics();
        assert_eq!(spawns, 0);
        assert_eq!(clones, 0);

        // Test clones are recorded
        {
            let c0 = context.clone();
            let _c1 = c0.clone();
        }

        let ContextMetrics { spawns, clones } = context.metrics();
        assert_eq!(spawns, 0);
        assert_eq!(clones, 2);

        // Test spawns are recorded.
        let rt = current_thread_runtime();
        let v = rt.block_on(context.clone().wrap_spawn(async { 2usize }));
        assert_eq!(v, 2);

        let ContextMetrics { spawns, clones } = context.metrics();
        assert_eq!(spawns, 1);
        assert_eq!(clones, 3);
    }

    #[test]
    fn test_provider_new_from() {
        // Happy path - valid provider with start points and additional points
        {
            let config = Config::new(
                Metric::L2,
                3,
                [
                    StartPoint::new(0, vec![1.0, 0.0]),
                    StartPoint::new(1, vec![0.0, 1.0]),
                ],
            )
            .unwrap();

            let start_points = [(0, AdjacencyList::from_iter_untrusted([1, 2]))];
            let points = [(
                2,
                vec![1.0, 1.0],
                AdjacencyList::from_iter_untrusted([0, 1]),
            )];

            let provider = Provider::new_from(config, start_points, points).unwrap();
            assert_eq!(provider.dim(), 2);
            assert_eq!(provider.max_degree(), 3);
        }

        // Happy Path: empty iterators (only start points from config)
        {
            let config = Config::new(Metric::L2, 5, [StartPoint::new(0, vec![1.0])]).unwrap();
            let provider = Provider::new_from(config, [], []).unwrap();
            assert_eq!(provider.dim(), 1);
        }

        // Error: start point neighbors exceed max degree
        {
            let config = Config::new(Metric::L2, 2, [StartPoint::new(0, vec![1.0])]).unwrap();
            // Exceeds max degree of 2
            let start_points = [(0, AdjacencyList::from_iter_untrusted([1, 2, 3]))];
            let err = Provider::new_from(config, start_points, []).unwrap_err();
            assert_message_contains!(err.to_string(), "max degree");
        }

        // Error: invalid start point ID
        {
            let config = Config::new(Metric::L2, 5, [StartPoint::new(0, vec![1.0])]).unwrap();
            let start_points = [(999, AdjacencyList::new())]; // 999 is not a valid start point
            let err = Provider::new_from(config, start_points, []).unwrap_err();
            assert_message_contains!(err.to_string(), "not a valid start point");
        }

        // Error: regular point neighbors exceed max degree
        {
            let config = Config::new(Metric::L2, 2, [StartPoint::new(0, vec![1.0])]).unwrap();

            // Exceeds max degree
            let points = [(1, vec![2.0], AdjacencyList::from_iter_untrusted([0, 2, 3]))];
            let err = Provider::new_from(config, [], points).unwrap_err();
            assert_message_contains!(err.to_string(), "max degree");
        }

        // Error: trying to assign start point through regular points
        {
            let config = Config::new(Metric::L2, 5, [StartPoint::new(0, vec![1.0])]).unwrap();
            let points = [(0, vec![2.0], AdjacencyList::new())]; // 0 is already a start point
            let err = Provider::new_from(config, [], points).unwrap_err();
            assert_message_contains!(err.to_string(), "cannot assign start point");
        }

        // Error: dimension mismatch in regular points
        {
            let config = Config::new(Metric::L2, 5, [StartPoint::new(0, vec![1.0, 2.0])]).unwrap();
            let points = [(1, vec![3.0], AdjacencyList::new())]; // Wrong dimension (1 instead of 2)
            let err = Provider::new_from(config, [], points).unwrap_err();
            assert_message_contains!(err.to_string(), "expecting dim");
        }

        // Error: inconsistent graph (neighbor points to non-existent ID)
        {
            let config = Config::new(Metric::L2, 5, [StartPoint::new(0, vec![1.0])]).unwrap();
            let points = [(
                1,
                vec![2.0],
                AdjacencyList::from_iter_unique(std::iter::once(999)),
            )]; // 999 doesn't exist
            let err = Provider::new_from(config, [], points).unwrap_err();
            assert_message_contains!(err.to_string(), "not in the provider");
        }
    }

    fn create_test_provider() -> Provider {
        // Edge case: complex valid graph
        let config = Config::new(
            Metric::Cosine,
            4,
            [
                StartPoint::new(0, vec![1.0, 0.0]),
                StartPoint::new(1, vec![0.0, 1.0]),
            ],
        )
        .unwrap();
        let start_points = [
            (0, AdjacencyList::from_iter_untrusted([1, 2, 3])),
            (1, AdjacencyList::from_iter_untrusted([0, 3])),
        ];

        let points = [
            (
                2,
                vec![0.5, 0.5],
                AdjacencyList::from_iter_untrusted([0, 3]),
            ),
            (
                3,
                vec![-1.0, 1.0],
                AdjacencyList::from_iter_untrusted([0, 1, 2]),
            ),
        ];
        let provider = Provider::new_from(config, start_points, points).unwrap();
        assert_eq!(provider.dim(), 2);
        assert_eq!(provider.max_degree(), 4);

        provider
    }

    #[test]
    fn id_conversion() {
        use provider::DataProvider;

        let provider = create_test_provider();

        let context = Context::default();
        for i in 0u32..3u32 {
            let internal = provider.to_internal_id(&context, &i).unwrap();
            assert_eq!(internal, i);

            let external = provider.to_external_id(&context, i).unwrap();
            assert_eq!(external, i);
        }

        let err = provider.to_internal_id(&context, &5).unwrap_err();
        let message = err.to_string();
        assert_eq!(
            message, "external id 5 is not initialized",
            "got {}",
            message
        );

        let err = provider.to_external_id(&context, 5).unwrap_err();
        let message = err.to_string();
        assert_eq!(
            message, "internal id 5 is not initialized",
            "got {}",
            message
        );
    }

    #[test]
    fn test_set_element() {
        use provider::{Accessor, Guard, SetElement};

        let provider = create_test_provider();
        let rt = current_thread_runtime();

        let context = Context::new();
        let mut accessor = super::Accessor::new(&provider);
        let id = 5;

        assert!(rt.block_on(accessor.get_element(5)).is_err());

        // Setting with the wrong dimension is an error.
        {
            let v = vec![1.0f32; provider.dim() + 1];
            let err = rt
                .block_on(provider.set_element(&context, &id, &v))
                .unwrap_err();
            let msg = err.to_string();
            assert_message_contains!(msg, "wrong dim");
            assert!(rt.block_on(accessor.get_element(id)).is_err());
        }

        // Setting with the correct dimension is successful.
        {
            let v = vec![1.0f32; provider.dim()];
            let guard = rt
                .block_on(provider.set_element(&context, &id, &v))
                .unwrap();
            rt.block_on(guard.complete());

            let element = rt.block_on(accessor.get_element(id)).unwrap();
            assert_eq!(v, element);
        }

        // Setting again is an error.
        {
            let v = vec![1.0f32; provider.dim()];
            let err = rt
                .block_on(provider.set_element(&context, &id, &v))
                .unwrap_err();
            let msg = err.to_string();
            assert_message_contains!(msg, "vector id 5 is already assigned");
        }
    }

    #[test]
    fn test_neighbor_accessor() {
        use provider::{DefaultAccessor, NeighborAccessor};

        let provider = create_test_provider();
        let accessor = provider.default_accessor();
        let mut v = AdjacencyList::new();

        let rt = current_thread_runtime();

        // The following accesses should all be successful and reflact the initialization
        // performed in `create_test_provider`.
        rt.block_on(accessor.get_neighbors(0, &mut v)).unwrap();
        assert_eq!(&*v, &[1, 2, 3]);

        rt.block_on(accessor.get_neighbors(1, &mut v)).unwrap();
        assert_eq!(&*v, &[0, 3]);

        rt.block_on(accessor.get_neighbors(2, &mut v)).unwrap();
        assert_eq!(&*v, &[0, 3]);

        rt.block_on(accessor.get_neighbors(3, &mut v)).unwrap();
        assert_eq!(&*v, &[0, 1, 2]);

        // Accessing an uninitialized vector is an error.
        let err = rt.block_on(accessor.get_neighbors(4, &mut v)).unwrap_err();
        assert_message_contains!(err.to_string(), "Attempt to access an invalid id");
    }

    #[test]
    fn test_set_neighbors() {
        use provider::{DefaultAccessor, NeighborAccessor, NeighborAccessorMut};

        let provider = create_test_provider();
        let accessor = provider.default_accessor();
        let mut v = AdjacencyList::new();

        let rt = current_thread_runtime();

        // Test that emptying neighbors works.
        rt.block_on(accessor.set_neighbors(2, &[])).unwrap();
        rt.block_on(accessor.get_neighbors(2, &mut v)).unwrap();
        assert!(v.is_empty());
        assert_eq!(provider.set_neighbors.value(), 1);

        // Adding a few neighbors behaves well.
        rt.block_on(accessor.set_neighbors(2, &[1, 3])).unwrap();
        rt.block_on(accessor.get_neighbors(2, &mut v)).unwrap();
        assert_eq!(&*v, &[1, 3]);
        assert_eq!(provider.set_neighbors.value(), 2);

        // Adding too many neighbors is an error.
        {
            assert_eq!(
                provider.max_degree(),
                4,
                "if this changes - update this mini test"
            );
            let err = rt
                .block_on(accessor.set_neighbors(2, &[1, 2, 3, 4, 5]))
                .unwrap_err();
            rt.block_on(accessor.get_neighbors(2, &mut v)).unwrap();
            assert_eq!(&*v, &[1, 3], "original neighbors should be unchanged");

            let msg = err.to_string();
            assert_message_contains!(msg, "trying to assign neighbors with length 5");

            assert_eq!(
                provider.set_neighbors.value(),
                2,
                "number of successful `set_neighbors` should not change"
            );
        }

        // Assigning duplicates is an error.
        {
            let err = rt
                .block_on(accessor.set_neighbors(2, &[1, 2, 3, 2]))
                .unwrap_err();

            rt.block_on(accessor.get_neighbors(2, &mut v)).unwrap();
            assert_eq!(
                &*v,
                &[1, 2, 3],
                "final neighbors should still be deduplicated"
            );
            let msg = err.to_string();
            assert_message_contains!(msg, "duplicate neighbors detected");

            assert_eq!(
                provider.set_neighbors.value(),
                3,
                "number of successful `set_neighbors` should change"
            );
        }

        // Invalid ID is caught
        {
            let err = rt
                .block_on(accessor.set_neighbors(10, &[1, 2]))
                .unwrap_err();

            let msg = err.to_string();
            assert_message_contains!(msg, "access an invalid id");
        }
    }

    #[test]
    fn test_append_vector() {
        use provider::{DefaultAccessor, NeighborAccessor, NeighborAccessorMut};

        let provider = create_test_provider();
        let accessor = provider.default_accessor();
        let mut v = AdjacencyList::new();

        let rt = current_thread_runtime();

        rt.block_on(accessor.get_neighbors(2, &mut v)).unwrap();
        assert_eq!(&*v, &[0, 3]);

        // We can successfully add neighbors.
        rt.block_on(accessor.append_vector(2, &[1])).unwrap();
        rt.block_on(accessor.get_neighbors(2, &mut v)).unwrap();
        assert_eq!(&*v, &[0, 3, 1]);
        assert_eq!(provider.append_neighbors.value(), 1);

        // We can add multiple neighbors.
        {
            rt.block_on(accessor.set_neighbors(2, &[])).unwrap();
            rt.block_on(accessor.append_vector(2, &[1, 3, 4])).unwrap();
            rt.block_on(accessor.get_neighbors(2, &mut v)).unwrap();
            assert_eq!(&*v, &[1, 3, 4]);
            assert_eq!(provider.append_neighbors.value(), 2);
        }

        // Appending duplicates is an error.
        {
            let err = rt.block_on(accessor.append_vector(2, &[1])).unwrap_err();
            rt.block_on(accessor.get_neighbors(2, &mut v)).unwrap();
            assert_eq!(&*v, &[1, 3, 4]);

            let msg = err.to_string();
            assert_message_contains!(msg, "duplicate ids in append-vector");
            assert_eq!(
                provider.append_neighbors.value(),
                3,
                "number of append calls should still increase",
            );
        }

        // Appending fully deduplicates.
        {
            rt.block_on(accessor.set_neighbors(2, &[])).unwrap();
            let err = rt
                .block_on(accessor.append_vector(2, &[1, 1, 1]))
                .unwrap_err();
            rt.block_on(accessor.get_neighbors(2, &mut v)).unwrap();
            assert_eq!(&*v, &[1]);

            let msg = err.to_string();
            assert_message_contains!(msg, "duplicate ids in append-vector");
            assert_eq!(
                provider.append_neighbors.value(),
                4,
                "number of append calls should still increase",
            );
        }

        // Adding too many neighbors is an error.
        {
            let err = rt
                .block_on(accessor.append_vector(2, &[2, 3, 4, 5]))
                .unwrap_err();
            rt.block_on(accessor.get_neighbors(2, &mut v)).unwrap();
            assert_eq!(&*v, &[1]);

            let msg = err.to_string();
            assert_message_contains!(msg, "will exceed the max degree");
            assert_eq!(provider.append_neighbors.value(), 4);
        }

        // Invalid IDs are caught.
        {
            let err = rt
                .block_on(accessor.append_vector(10, &[1, 2]))
                .unwrap_err();

            let msg = err.to_string();
            assert_message_contains!(msg, "access an invalid id");
        }
    }

    #[test]
    fn test_delete() {
        use provider::Delete;

        let provider = create_test_provider();
        let rt = current_thread_runtime();

        let ids = [0, 1, 2, 3];
        let invalid_id = 5;

        // Ensure the test provider is not updated too dramatically.
        {
            let mut check: Vec<_> = provider.all_internal_ids().into_iter().collect();
            check.sort();
            assert_eq!(&*check, &ids);

            assert!(provider.is_start_point(0));
            assert!(provider.is_start_point(1));
            assert!(!provider.is_start_point(2));
            assert!(!provider.is_start_point(3));
        }

        let context = Context::new();
        for i in ids {
            let is_deleted = rt.block_on(provider.is_deleted(i)).unwrap();
            assert!(!is_deleted);

            let status = rt
                .block_on(provider.status_by_internal_id(&context, i))
                .unwrap();
            assert_eq!(status, provider::ElementStatus::Valid);

            let status = rt
                .block_on(provider.status_by_external_id(&context, &i))
                .unwrap();
            assert_eq!(status, provider::ElementStatus::Valid);
        }

        // Accessing an invalid ID in all APIs returns an error.
        {
            let err = rt.block_on(provider.is_deleted(invalid_id)).unwrap_err();
            assert_message_contains!(err.to_string(), "not initialized");

            let err = rt
                .block_on(provider.status_by_internal_id(&context, invalid_id))
                .unwrap_err();
            assert_message_contains!(err.to_string(), "not initialized");

            let err = rt
                .block_on(provider.status_by_external_id(&context, &invalid_id))
                .unwrap_err();
            assert_message_contains!(err.to_string(), "not initialized");
        }

        // Deleting works.
        {
            let id = 3;
            rt.block_on(provider.delete(&context, &id)).unwrap();
            let is_deleted = rt.block_on(provider.is_deleted(id)).unwrap();
            assert!(is_deleted);

            let status = rt
                .block_on(provider.status_by_internal_id(&context, id))
                .unwrap();
            assert_eq!(status, provider::ElementStatus::Deleted);

            let status = rt
                .block_on(provider.status_by_external_id(&context, &id))
                .unwrap();
            assert_eq!(status, provider::ElementStatus::Deleted);
        }

        // Releasing works and completely remove the id.
        {
            let id = 3;
            rt.block_on(provider.release(&context, id)).unwrap();
            let err = rt.block_on(provider.is_deleted(id)).unwrap_err();
            assert_message_contains!(err.to_string(), "not initialized");

            let err = rt
                .block_on(provider.status_by_internal_id(&context, id))
                .unwrap_err();
            assert_message_contains!(err.to_string(), "not initialized");

            let err = rt
                .block_on(provider.status_by_external_id(&context, &id))
                .unwrap_err();
            assert_message_contains!(err.to_string(), "not initialized");
        }
    }

    #[test]
    fn test_start_points_cannot_be_deleted() {
        use provider::Delete;

        let provider = create_test_provider();
        let rt = current_thread_runtime();

        assert!(provider.is_start_point(0));
        assert!(provider.is_start_point(1));

        let context = Context::new();
        let err = rt.block_on(provider.delete(&context, &0)).unwrap_err();
        let msg = err.to_string();
        assert_message_contains!(msg, "cannot delete start point");
        assert!(!rt.block_on(provider.is_deleted(0)).unwrap());

        let err = rt.block_on(provider.release(&context, 0)).unwrap_err();
        let msg = err.to_string();
        assert_message_contains!(msg, "cannot delete start point");
        assert!(!rt.block_on(provider.is_deleted(0)).unwrap());
    }
}
