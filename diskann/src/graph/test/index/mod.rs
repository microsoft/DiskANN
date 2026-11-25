/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A pedantic provider implementation used for testing alorithmic logic.

use std::{
    borrow::BorrowMut,
    collections::{HashMap, HashSet},
    mem::ManuallyDrop,
    num::NonZeroUsize,
    sync::Arc,
};

use dashmap::{DashMap, mapref::entry::Entry};
use diskann_utils::{
    lazy_format,
    views::rowmajor::{self, Matrix},
};
use diskann_vector::{PreprocessedDistanceFunction, distance::Metric};
use thiserror::Error;

use crate::{
    ANNResult, convert_error, default_post_processor,
    error::message,
    graph::{
        self, AdjacencyList, SearchOutputBuffer, adjacencylist::Empty, algorithms, ext::labeled,
        glue, search::PagedSearch, test::synthetic, workingset,
    },
    internal::counter::{Counter, LocalCounter},
    neighbor::Neighbor,
    provider,
    utils::{Bool, VectorRepr},
};

#[cfg(any(test, feature = "testing"))]
use crate::ANNError;

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
    graph: graph::Config,
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
    /// * No start points are provided.
    /// * The dimensions of the provided start points do not match or are zero.
    /// * The provided start points are not unique.
    pub fn new<I>(
        graph: graph::Config,
        metric: Metric,
        start_points: I,
    ) -> Result<Self, ConfigError>
    where
        I: IntoIterator<Item = StartPoint>,
    {
        let max_degree = graph.max_degree();
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
            graph,
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
}

convert_error!(ConfigError);

/// A test data provider for validating DiskANN API guarantees.
///
/// The following is a list of properties this provider seeks to maintain:
///
/// * Start points provided at construction time are immutable and cannot be deleted or modified.
/// * All calls to [`provider::NeighborAccessorMut::set_neighbors`] do not contain duplicates.
/// * All calls to [`provider::NeighborAccessorMut::append_vector`] do not contain duplicates
///   and are disjoint with the current adjacency list.
/// * Vectors can be marked as deleted, removing access to their data but keeping access to
///   their adjacency lists.
/// * Scoped operatios to protect IDs during critical sections.
/// * Attempting to retrieve and ID that is not present is an error.
/// * All attempts to mutate the graph via [`provider::NeighborAccessorMut`] must be preceeded
///   by [`provider::SetElement`].
///
/// This provider allows for some amount of concurrent access, but is not optimized for performance.
#[derive(Debug)]
pub struct Index {
    terms: DashMap<u32, Term>,
    config: Config,

    // Counters
    pub(crate) get_vector: Counter,
    pub(crate) set_vector: Counter,
    pub(crate) get_neighbors: Counter,
    pub(crate) set_neighbors: Counter,
    pub(crate) append_neighbors: Counter,
}

impl Index {
    /// Create a new [`Index`].
    ///
    /// All counters will be initialized to zero.
    pub fn new(config: Config) -> Self {
        let this = Self {
            terms: DashMap::new(),
            config,
            get_vector: Counter::new(),
            set_vector: Counter::new(),
            get_neighbors: Counter::new(),
            set_neighbors: Counter::new(),
            append_neighbors: Counter::new(),
        };

        for (id, value) in this.config.start_points.iter() {
            this.terms.insert(
                *id,
                Term {
                    data: Vector::Valid(value.clone()),
                    neighbors: AdjacencyList::new(),
                },
            );
        }

        this
    }

    /// Create a new [`Index`] from the given configuration, start points, and points.
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
        let this = Self::new(config);
        let max_degree = this.config.graph.max_degree().get();

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

            if let Some(mut term) = this.terms.get_mut(&id) {
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

            this.terms.insert(id, term);
        }

        // Now that we have inserted all the points - ensure our graph is consistent.
        this.is_consistent()?;
        Ok(this)
    }

    /// Return a fully-formed provider with a grid of points.
    ///
    /// The start point will have internal ID `u32::MAX`.
    ///
    /// See the documentation for [`synthetic::Grid`] and [`synthetic::Grid::data`] for
    /// more details.
    pub fn grid(grid: synthetic::Grid, size: usize) -> ANNResult<Self> {
        let max_degree: usize = (grid.dim() * 2).into();
        let start_id = u32::MAX;

        let setup = grid.setup(size, start_id);

        let graph_config = graph::config::Builder::new(
            max_degree,
            graph::config::MaxDegree::same(),
            100,
            (Metric::L2).into(),
        )
        .build()?;

        // Create the provider config with the grid start point.
        let provider_config = Config::new(
            graph_config,
            Metric::L2,
            StartPoint::new(setup.start_id(), setup.start_point()),
        )?;

        // Initialize the provider.
        Self::new_from(provider_config, setup.start_neighbors(), setup.setup())
    }

    /// Return the dimensionality of data contained by this provider.
    pub fn dim(&self) -> usize {
        self.config.dim.get()
    }

    /// Return the largest degree this provider is capable of holding.
    pub fn max_degree(&self) -> usize {
        self.config.graph.max_degree().get()
    }

    /// Return the [`Metric`] used by this provider.
    pub fn distance_metric(&self) -> Metric {
        self.config.metric
    }

    /// Return the current [`DegreeStats`].
    pub fn degree_stats(&self) -> graph::stats::DegreeStats {
        self.terms.iter().map(|ref_multi| ref_multi.value().neighbors.len()).collect()
    }

    /// Return `true` is `id` is a start point. Otherwise, return `false`.
    pub(crate) fn is_start_point(&self, id: u32) -> bool {
        self.config.start_points.contains_key(&id)
    }

    /// Return an approximation of the collection of internal IDs currently in the index.
    ///
    /// This is approximate as it is not atomic. Thus, it is possible for other threads to
    /// update the collection of internal IDs while this operation is executing.
    pub fn all_internal_ids(&self) -> HashSet<u32> {
        self.terms
            .iter()
            .map(|ref_multi| *ref_multi.key())
            .collect()
    }

    /// Check whether all adjacency lists in `self` point to valid IDs.
    pub fn is_consistent(&self) -> ANNResult<()> {
        let all = self.all_internal_ids();
        for ref_multi in self.terms.iter() {
            let id = ref_multi.key();
            let term = ref_multi.value();
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

        Ok(())
    }

    /// Return `true` if `id` is present in the provider but marked as deleted.
    ///
    /// If `id` is present but not marked deleted, returns `false`.
    ///
    /// An error is returned if `id` is not present in the provider.
    fn is_deleted(&self, id: u32) -> Result<bool, InvalidId> {
        if let Some(term) = self.terms.get(&id) {
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

    /// Return a [`provider::NeighborAccessor`] for this provider.
    pub fn neighbors(&self) -> NeighborAccessor<'_> {
        NeighborAccessor::new(self)
    }

    /// Return the adjacency lists in `self` as `source -> { destinations ... }` pairs.
    ///
    /// If `sort == true` then:
    ///
    /// * The returned vector will be ordered with the `source` entry increasing.
    /// * Each `destination` adjacency list will be sorted by increasing ID.
    pub fn dump_neighbors(&self, sort: bool) -> Vec<(u32, AdjacencyList<u32>)> {
        let mut neighbors: Vec<_> = self
            .terms
            .iter()
            .map(|ref_multi| {
                let mut neighbors = ref_multi.value().neighbors.clone();
                if sort {
                    neighbors.sort();
                }
                (*ref_multi.key(), neighbors)
            })
            .collect();

        if sort {
            // Unstable sort is fine: `DashMap` ensures the keys are unique.
            neighbors.sort_unstable_by_key(|(id, _)| *id);
        }

        neighbors
    }

    /// Get the neighbors for `id`.
    ///
    /// Return `true` if the `id` is undeleted - otherwise return false.
    pub(crate) fn get_neighbors(
        &self,
        id: u32,
        neighbors: Empty<'_, u32>,
    ) -> Result<bool, AccessedInvalidId> {
        match self.terms.get(&id) {
            Some(v) => {
                self.get_neighbors.increment();
                neighbors.take().overwrite_trusted(&v.neighbors);
                Ok(!v.data.is_deleted())
            }
            None => Err(AccessedInvalidId(id)),
        }
    }

    /// Return an iterator over all IDs, including soft deleted points and fixed start points.
    pub fn all_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.terms.iter().map(|ref_multi| *ref_multi.key())
    }

    /// Return an iterator over all non-start-point IDs, including soft deleted points.
    pub fn non_start_points_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.terms
            .iter()
            .map(|ref_multi| *ref_multi.key())
            .filter(|id| !self.is_start_point(*id))
    }

    /// Return an iterator over the start point IDs.
    pub fn start_point_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.config.start_points.keys().copied()
    }

    /// Return a search accesor.
    pub fn search_accessor(&self, query: &[f32]) -> Result<Accessor<'_>, DimMismatch> {
        Accessor::new(self, query)
    }

    /// Return a prune accessor.
    fn prune_accessor<'a>(
        &'a self,
        options: &'a Options,
        overlay: Option<&workingset::map::Overlay<u32, workingset::map::Ref<[f32]>>>,
    ) -> PruneAccessor<'a> {
        let cap = if options.working_set_reuse {
            workingset::map::Capacity::Default
        } else {
            workingset::map::Capacity::None
        };

        let builder = workingset::map::Builder::new(cap);
        let builder = if let Some(overlay) = overlay {
            builder.with_overlay(overlay.clone())
        } else {
            builder
        };

        let set = builder.build(self.config.graph.max_occlusion_size().get());
        PruneAccessor::new(self, set)
    }

    /// Run the provided closure on the associated term.
    fn get<F, R>(&self, id: u32, f: F) -> Result<Option<R>, AccessedInvalidId>
    where
        F: FnOnce(&[f32]) -> R,
    {
        match self.terms.get(&id) {
            Some(term) => Ok(term.data.get().map(f)),
            None => Err(AccessedInvalidId(id)),
        }
    }

    fn set(&self, vector: &[f32], id: u32) -> ANNResult<SetGuard<'_>> {
        #[derive(Debug, Clone, Copy, Error)]
        enum SetError {
            #[error("vector id {0} is already assigned")]
            AlreadyAssigned(u32),
            #[error("wrong dim - got {0}, expected {1}")]
            WrongDim(usize, usize),
        }

        convert_error!(SetError);

        // Ensure that the assigned vector has the correct length.
        if vector.len() != self.dim() {
            return Err(SetError::WrongDim(vector.len(), self.dim()).into());
        }

        match self.terms.entry(id) {
            Entry::Occupied(_) => Err(SetError::AlreadyAssigned(id).into()),
            Entry::Vacant(term) => {
                term.insert(Term {
                    neighbors: AdjacencyList::new(),
                    data: Vector::Protected(vector.into()),
                });
                self.set_vector.increment();
                Ok(SetGuard::new(self, id))
            }
        }
    }

    /// Attempt to protect the `id`.
    pub(super) fn try_protect(&self, id: u32) -> Result<Option<Protected<'_>>, InvalidId> {
        if self.is_start_point(id) {
            return Err(InvalidId::IsStartPoint(id));
        }

        match self.terms.entry(id) {
            Entry::Occupied(mut occupied) => {
                if occupied.get_mut().try_protect() {
                    Ok(Some(Protected::new(
                        self,
                        occupied
                            .get_mut()
                            .get()
                            .expect("data is in the protected state"),
                        id,
                    )))
                } else {
                    Ok(None)
                }
            }
            Entry::Vacant(_) => Err(InvalidId::External(id)),
        }
    }

    fn partition_live(&self, ids: &mut AdjacencyList<u32>, deleted: Option<Empty<'_, u32>>) {
        let mut deleted = deleted.map(Empty::take);

        ids.retain(|i| {
            let retain = match self.is_deleted(*i) {
                Ok(true) => false,
                Ok(false) => true,
                Err(_) => false,
            };

            if !retain && let Some(deleted) = deleted.as_mut() {
                deleted.push(*i);
            }

            retain
        });
    }

    //--------//
    // Search //
    //--------//

    pub async fn search<'a, P>(
        &'a self,
        search: P,
        vector: &'a [f32],
        knn: Option<NonZeroUsize>,
    ) -> ANNResult<(Vec<Neighbor<u32>>, P::Output)>
    where
        P: graph::Search,
    {
        let mut accessor = self.search_accessor(vector)?;
        let mut buf = MaybeCapped::new(knn);
        let post_processor = glue::DefaultPostProcessor::default_post_processor(&accessor);
        let output = search
            .search(&mut accessor, post_processor, &mut buf)
            .await?;
        Ok((buf.into_inner(), output))
    }

    pub async fn paged<'a>(
        &'a self,
        vector: &'a [f32],
        l_value: NonZeroUsize,
    ) -> ANNResult<PagedSearch<Accessor<'a>>> {
        let accessor = self.search_accessor(vector)?;
        PagedSearch::new(accessor, l_value).await
    }

    pub async fn filtered<'a, P>(
        &'a self,
        search: P,
        vector: &'a [f32],
        predicate: &'a dyn labeled::QueryLabelProvider<u32>,
        knn: Option<NonZeroUsize>,
    ) -> ANNResult<(Vec<Neighbor<u32>>, P::Output)>
    where
        P: graph::FilteredSearch,
    {
        let mut accessor = labeled::FilteredAccessor::new(self.search_accessor(vector)?, predicate);

        let mut buf = MaybeCapped::new(knn);
        let post_processor = glue::DefaultPostProcessor::default_post_processor(&accessor);
        let output = search
            .filtered_search(&mut accessor, post_processor, &mut buf)
            .await?;
        Ok((buf.into_inner(), output))
    }

    //--------//
    // Insert //
    //--------//

    fn insert_accessor<'a>(
        &'a self,
        vector: &'a [f32],
        options: &'a Options,
        overlay: Option<&workingset::map::Overlay<u32, workingset::map::Ref<[f32]>>>,
    ) -> Result<Insert<'a>, DimMismatch> {
        Ok(Insert {
            search: self.search_accessor(vector)?,
            prune: self.prune_accessor(options, overlay),
        })
    }

    pub async fn insert(&self, vector: &[f32], id: u32, options: &Options) -> ANNResult<()> {
        // Step 1 - try to insert the vector.
        let guard = self.set(vector, id)?;

        // Step 2 - run the insertion algorithm.
        algorithms::insert::insert(
            &mut self.insert_accessor(vector, options, None)?,
            id,
            &self.config.graph,
        )
        .await?;

        guard.finish();
        Ok(())
    }

    pub async fn batch_insert(
        self: &Arc<Self>,
        data: Arc<rowmajor::Owned<f32>>,
        ids: Arc<[u32]>,
        options: &Options,
        handle: &tokio::runtime::Handle,
    ) -> ANNResult<SpawnCount> {
        // Did the provide us with sensible number of values.
        if data.nrows() != ids.len() {
            return Err(ANNError::message(
                "number of data rows does not match the number of ids",
            ));
        }

        // Insert all IDs.
        let mut guards = Vec::new();
        for (vector, id) in std::iter::zip(data.rows(), ids.iter()) {
            guards.push(self.set(vector, *id)?);
        }

        let counter = Counter::new();
        let spawner = CountSpawns::new(&counter, handle);
        let overlay = workingset::map::Overlay::from_batch(data.clone(), ids.iter().cloned());
        algorithms::batch::insert::standard(&spawner, ids, &self.config.graph, || BatchInsert {
            provider: self.clone(),
            overlay: overlay.clone(),
            options: options.clone(),
        })
        .await?;

        // Release our protection on the items we've inserted.
        guards.into_iter().for_each(|guard| guard.finish());
        Ok(SpawnCount(counter.value()))
    }

    pub(super) async fn prune_id(&self, id: u32) -> ANNResult<()> {
        let mut workspace = algorithms::insert::Workspace::new();
        let options = Options::default();
        let mut accessor = self.prune_accessor(&options, None);
        workspace.prune_node(&mut accessor, id, &self.config.graph).await
    }

    //--------//
    // Delete //
    //--------//

    pub async fn delete(
        &self,
        id: u32,
        kind: algorithms::inplace_delete::InplaceDeleteMethod,
        options: &Options,
    ) -> ANNResult<Bool> {
        use algorithms::inplace_delete::{self, InplaceDeleteMethod as IDM};

        // Try to protect the item we are going to delete.
        let Some(guard) = self.try_protect(id)? else {
            return Ok(false.into());
        };

        let mut access = InplaceDelete::new(self.prune_accessor(options, None));
        let mut workspace = inplace_delete::Workspace::new();

        match kind {
            IDM::OneHop(kind) => {
                workspace
                    .one_hop(&kind, &mut access, id, &self.config.graph)
                    .await?;
            }
            IDM::TwoHopAndOneHop(kind) => {
                workspace
                    .two_hop_and_one_hop(&kind, &mut access, id, &self.config.graph)
                    .await?;
            }
            IDM::VisitedAndTopK(kind) => {
                let search = self.search_accessor(guard.data())?;
                let mut accessor = TopKInplaceDelete::new(access, search);
                workspace
                    .visited_and_topk(&kind, &mut accessor, id, &self.config.graph)
                    .await?;
            }
        }

        guard.delete();
        Ok(true.into())
    }

    pub async fn batch_delete(
        self: &Arc<Self>,
        ids: Arc<[u32]>,
        kind: algorithms::inplace_delete::InplaceDeleteMethod,
        options: &Options,
        handle: &tokio::runtime::Handle,
    ) -> ANNResult<SpawnCount> {
        // Attempt to protect all the IDs we are about to delete.
        //
        // This will both ensure that the IDs exist and that they are unique. However, the
        // resulting error message is less than great.
        let guards: Vec<_> = ids
            .iter()
            .map(|&i| -> ANNResult<_> {
                match self.try_protect(i)? {
                    Some(guard) => Ok(guard),
                    None => Err(ANNError::message(lazy_format!(
                        move,
                        "internal ID {i} is already protected"
                    ))),
                }
            })
            .collect::<ANNResult<_>>()?;

        // With all the guards now held, run the batch algorithm.
        let counter = Counter::new();
        let spawner = CountSpawns::new(&counter, handle);
        algorithms::batch::inplace_delete::standard(
            &spawner,
            ids.clone(),
            &self.config.graph,
            || BatchInplaceDelete {
                index: self.clone(),
                kind: kind.clone(),
                options: options.clone(),
            },
        )
        .await?;

        // Mark all the protected data as deleted.
        guards.into_iter().for_each(|g| g.delete());

        Ok(SpawnCount(counter.value()))
    }

    //-------------//
    // Consolidate //
    //-------------//

    /// Run consolidation on the `id`.
    pub async fn consolidate(
        &self,
        id: u32,
        options: &Options,
    ) -> ANNResult<algorithms::consolidate::ConsolidateKind> {
        // FIXME: We do not aquire any kind of protection guard because the consolidate
        // algorithm needs to be tested in the presence of trying to access a deleted ID.

        let mut consolidate = Consolidate {
            parent: self,
            prune: self.prune_accessor(options, None),
        };

        algorithms::consolidate::Workspace::new()
            .run(&mut consolidate, id, &self.config.graph)
            .await
    }
}

//--------//
// Guards //
//--------//

/// A guard used with [`Index::set`] that protects the newly inserted ID for the duration
/// of an insertion. If not explicitly finished, it will mark the newly inserted data as
/// deleted.
#[derive(Debug)]
struct SetGuard<'a> {
    provider: &'a Index,
    id: u32,
}

impl<'a> SetGuard<'a> {
    fn new(provider: &'a Index, id: u32) -> Self {
        Self { provider, id }
    }

    fn id(&self) -> u32 {
        self.id
    }

    fn finish(self) {
        let me = ManuallyDrop::new(self);

        // Intentionally panic for out-of-bounds IDs.
        match me.provider.terms.get_mut(&me.id) {
            Some(mut term) => term.protected_to_valid(),
            None => panic!("id {} should have been properly protected", me.id),
        }
    }
}

impl Drop for SetGuard<'_> {
    fn drop(&mut self) {
        match self.provider.terms.get_mut(&self.id) {
            Some(mut term) => term.protected_to_deleted(),
            None => panic!("id {} should have been properly protected", self.id),
        }
    }
}

#[derive(Debug)]
pub(super) struct Protected<'a> {
    provider: &'a Index,
    data: Vec<f32>,
    id: u32,
}

impl<'a> Protected<'a> {
    fn new(provider: &'a Index, data: &[f32], id: u32) -> Self {
        Self {
            provider,
            data: data.into(),
            id,
        }
    }

    pub(super) fn data(&self) -> &[f32] {
        &self.data
    }

    pub(super) fn delete(self) {
        let me = ManuallyDrop::new(self);

        match me.provider.terms.get_mut(&me.id) {
            Some(mut term) => term.protected_to_deleted(),
            None => panic!("id {} should have been properly protected", me.id),
        }
    }

    pub(super) fn release(self) {
        // Let `Drop` return this back to `Valid`.
    }
}

impl Drop for Protected<'_> {
    fn drop(&mut self) {
        match self.provider.terms.get_mut(&self.id) {
            Some(mut term) => term.protected_to_valid(),
            None => panic!("id {} should have been properly protected", self.id),
        }
    }
}

//---------//
// Spawner //
//---------//

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct SpawnCount(usize);

impl std::ops::AddAssign for SpawnCount {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

#[cfg(test)]
impl crate::test::cmp::VerboseEq for SpawnCount {
    fn verbose_eq(&self, other: &Self) -> ANNResult<()> {
        self.0.verbose_eq(&other.0)
    }
}

#[derive(Debug)]
struct CountSpawns<'a> {
    counter: &'a Counter,
    handle: &'a tokio::runtime::Handle,
}

impl<'a> CountSpawns<'a> {
    fn new(counter: &'a Counter, handle: &'a tokio::runtime::Handle) -> Self {
        Self { counter, handle }
    }
}

impl crate::tokio::Spawn for CountSpawns<'_> {
    fn spawn<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future<Output: Send + 'static> + Send + 'static,
    {
        self.counter.increment();
        self.handle.spawn(future)
    }
}

//--------//
// Insert //
//--------//

#[derive(Debug)]
struct Insert<'a> {
    search: Accessor<'a>,
    prune: PruneAccessor<'a>,
}

provider::has_id!(Insert<'_> = u32);

impl<'a> algorithms::insert::Access for Insert<'a> {
    type SearchAccessor<'b>
        = Accessor<'a>
    where
        Self: 'b;

    type PruneAccessor<'b>
        = PruneAccessor<'a>
    where
        Self: 'b;

    fn search_accessor(&mut self) -> impl BorrowMut<Accessor<'a>> + Send {
        &mut self.search
    }
    fn prune_accessor(&mut self) -> impl BorrowMut<PruneAccessor<'a>> + Send {
        &mut self.prune
    }
}

//-------------//
// BatchInsert //
//-------------//

#[derive(Debug)]
struct BatchInsert {
    provider: Arc<Index>,
    overlay: workingset::map::Overlay<u32, workingset::map::Ref<[f32]>>,
    options: Options,
}

provider::has_id!(BatchInsert = u32);

impl algorithms::batch::insert::Standard for BatchInsert {
    type InsertAccessor<'a> = Insert<'a>;
    type PruneAccessor<'a> = PruneAccessor<'a>;

    fn insert_accessor(
        &mut self,
        id: u32,
    ) -> ANNResult<impl std::borrow::BorrowMut<Self::InsertAccessor<'_>> + Send> {
        let Some(vector) = self.overlay.get(&id) else {
            return Err(ANNError::message(format!(
                "id {id} is not part of the batch"
            )));
        };

        Ok(self
            .provider
            .insert_accessor(vector, &self.options, Some(&self.overlay))?)
    }

    fn prune_accessor(
        &mut self,
    ) -> ANNResult<impl std::borrow::BorrowMut<Self::PruneAccessor<'_>> + Send> {
        Ok(self
            .provider
            .prune_accessor(&self.options, Some(&self.overlay)))
    }
}

//-------------//
// Consolidate //
//-------------//

#[derive(Debug)]
struct Consolidate<'a> {
    parent: &'a Index,
    prune: PruneAccessor<'a>,
}

provider::has_id!(Consolidate<'_> = u32);

impl<'a> algorithms::consolidate::Access for Consolidate<'a> {
    type PruneAccessor<'b>
        = PruneAccessor<'a>
    where
        Self: 'b;

    fn prune_accessor(&mut self) -> impl BorrowMut<Self::PruneAccessor<'_>> + Send {
        &mut self.prune
    }

    async fn neighbors(&mut self, id: u32, neighbors: Empty<'_, u32>) -> ANNResult<Bool> {
        match self.parent.get_neighbors(id, neighbors) {
            Ok(is_valid) => Ok(is_valid.into()),
            Err(AccessedInvalidId(_)) => Ok(false.into()),
        }
    }

    async fn partition_live(
        &mut self,
        ids: &mut AdjacencyList<u32>,
        deleted: Option<Empty<'_, u32>>,
    ) -> ANNResult<()> {
        self.parent.partition_live(ids, deleted);
        Ok(())
    }
}

//----------------//
// Inplace Delete //
//----------------//

#[derive(Debug)]
struct InplaceDelete<'a> {
    prune: PruneAccessor<'a>,
}

impl<'a> InplaceDelete<'a> {
    fn new(prune: PruneAccessor<'a>) -> Self {
        Self { prune }
    }
}

provider::has_id!(InplaceDelete<'_> = u32);

impl<'a> algorithms::inplace_delete::Access for InplaceDelete<'a> {
    type PruneAccessor<'x>
        = PruneAccessor<'a>
    where
        Self: 'x;

    fn prune_accessor(&mut self) -> impl BorrowMut<PruneAccessor<'a>> + Send {
        &mut self.prune
    }

    async fn keep_live(&mut self, ids: &mut AdjacencyList<u32>) -> ANNResult<()> {
        self.prune.provider().partition_live(ids, None);
        Ok(())
    }
}

#[derive(Debug)]
struct TopKInplaceDelete<'a> {
    inplace: InplaceDelete<'a>,
    search: Accessor<'a>,
}

impl<'a> TopKInplaceDelete<'a> {
    fn new(inplace: InplaceDelete<'a>, search: Accessor<'a>) -> Self {
        Self { inplace, search }
    }
}

provider::has_id!(TopKInplaceDelete<'_> = u32);

impl<'a> algorithms::inplace_delete::Access for TopKInplaceDelete<'a> {
    type PruneAccessor<'x>
        = PruneAccessor<'a>
    where
        Self: 'x;

    fn prune_accessor(&mut self) -> impl BorrowMut<PruneAccessor<'a>> + Send {
        self.inplace.prune_accessor()
    }

    async fn keep_live(&mut self, ids: &mut AdjacencyList<u32>) -> ANNResult<()> {
        self.inplace.keep_live(ids).await
    }
}

impl<'a> algorithms::inplace_delete::TopKAccess for TopKInplaceDelete<'a> {
    type SearchAccessor<'x>
        = Accessor<'a>
    where
        Self: 'x;

    fn search_accessor(&mut self) -> impl BorrowMut<Accessor<'a>> + Send {
        &mut self.search
    }
}

//--------------------//
// BatchInplaceDelete //
//--------------------//

#[derive(Debug)]
struct BatchInplaceDelete {
    index: Arc<Index>,
    kind: algorithms::inplace_delete::InplaceDeleteMethod,
    options: Options,
}

provider::has_id!(BatchInplaceDelete = u32);

impl algorithms::batch::inplace_delete::Standard for BatchInplaceDelete {
    type PruneAccessor<'a> = PruneAccessor<'a>;

    async fn discover(
        &mut self,
        id: u32,
        workspace: &mut algorithms::batch::inplace_delete::Workspace<u32>,
        edges: &mut algorithms::batch::inplace_delete::Edges<u32>,
        config: &graph::Config,
    ) -> ANNResult<()> {
        use algorithms::inplace_delete::InplaceDeleteMethod as IDM;

        let mut access = InplaceDelete::new(self.index.prune_accessor(&self.options, None));
        match &self.kind {
            IDM::OneHop(one_hop) => {
                workspace
                    .find_one_hop(edges, &mut access, one_hop, id)
                    .await?;
            }
            IDM::TwoHopAndOneHop(two_hop) => {
                workspace
                    .find_two_hop_and_one_hop(edges, &mut access, two_hop, id)
                    .await?;
            }
            IDM::VisitedAndTopK(topk) => {
                // Attempt to retrieve the data to create a search accessor.
                //
                // Contractually, these ids should be protected so we expect retrieval to
                // succeed.
                let Some(data) = self.index.get(id, |v| -> Box<[_]> { v.into() })? else {
                    return Err(ANNError::message(
                        "expected batch inplace delete IDs to be protected",
                    ));
                };
                let search = self.index.search_accessor(&data)?;
                let mut access = TopKInplaceDelete::new(access, search);
                workspace
                    .find_visited_and_top_k(edges, &mut access, topk, id, config)
                    .await?;
            }
        }
        Ok(())
    }

    fn prune_accessor(&mut self) -> ANNResult<impl BorrowMut<Self::PruneAccessor<'_>> + Send> {
        Ok(self.index.prune_accessor(&self.options, None))
    }
}

#[derive(Debug)]
struct MaybeCapped {
    data: Vec<Neighbor<u32>>,
    max: Option<NonZeroUsize>,
}

impl MaybeCapped {
    fn new(max: Option<NonZeroUsize>) -> Self {
        Self {
            data: Vec::with_capacity(max.unwrap_or(NonZeroUsize::MIN).get()),
            max,
        }
    }

    fn into_inner(self) -> Vec<Neighbor<u32>> {
        self.data
    }
}

impl SearchOutputBuffer<u32> for MaybeCapped {
    fn size_hint(&self) -> Option<usize> {
        match self.max {
            Some(max) => Some(max.get().saturating_sub(self.data.len())),
            None => None,
        }
    }

    fn push(&mut self, neighbor: Neighbor<u32>) -> graph::BufferState {
        match self.size_hint() {
            Some(remaining) => {
                if remaining != 0 {
                    self.data.push(neighbor);
                }
                if remaining > 1 {
                    graph::BufferState::Available
                } else {
                    graph::BufferState::Full
                }
            }
            None => {
                self.data.push(neighbor);
                graph::BufferState::Available
            }
        }
    }

    fn current_len(&self) -> usize {
        self.data.len()
    }

    fn extend<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: IntoIterator<Item = Neighbor<u32>>,
    {
        let start = self.data.len();
        let mut itr = itr.into_iter();
        while let Some(neighbor) = itr.next()
            && self.push(neighbor) == graph::BufferState::Available
        {}

        self.data.len() - start
    }
}

/// Index level metrics.
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
    #[must_use]
    fn try_protect(&mut self) -> bool {
        self.data.try_protect()
    }

    fn protected_to_deleted(&mut self) {
        self.data.protected_to_deleted()
    }

    fn protected_to_valid(&mut self) {
        self.data.protected_to_valid()
    }

    fn get(&self) -> Option<&[f32]> {
        self.data.get()
    }

    fn is_deleted(&self) -> bool {
        self.data.is_deleted()
    }
}

/// A data vector that records whether or not it has been deleted.
#[derive(Debug)]
enum Vector {
    /// Vector data is valid and available for reading and deleting.
    Valid(Vec<f32>),

    /// Vector data is valid for reading, but a task is protecting it against mutation.
    ///
    /// Only the task that set the data to `Protected` is allowed to move this out of the
    /// `Protected` state.
    Protected(Vec<f32>),

    /// Vector data is deleted.
    Deleted,
}

impl Vector {
    /// Move `self` from [`Self::Valid`] to [`Self::Protected`] state. Return `true` only if
    /// the transition was successful.
    #[must_use]
    fn try_protect(&mut self) -> bool {
        let (replacement, success) = match std::mem::replace(self, Self::Deleted) {
            Self::Valid(data) => (Self::Protected(data), true),
            Self::Protected(data) => (Self::Protected(data), false),
            Self::Deleted => (Self::Deleted, false),
        };
        *self = replacement;

        success
    }

    /// Transition from the [`Self::Protected`] state to the [`Self::Deleted`] state.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not protected.
    fn protected_to_deleted(&mut self) {
        if matches!(self, Self::Protected(_)) {
            *self = Self::Deleted;
        } else {
            panic!("invalid state: {:?}", self)
        }
    }

    /// Transition from the [`Self::Protected`] state to the [`Self::Valid`] state.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not protected.
    fn protected_to_valid(&mut self) {
        let (replacement, success) = match std::mem::replace(self, Self::Deleted) {
            Self::Valid(data) => (Self::Valid(data), false),
            Self::Protected(data) => (Self::Valid(data), true),
            Self::Deleted => (Self::Deleted, false),
        };
        *self = replacement;

        if !success {
            panic!("invalid state: {:?}", self)
        }
    }

    /// Return `true` if `self` has been marked as deleted. Otherwise, return `false`.
    fn is_deleted(&self) -> bool {
        matches!(self, Self::Deleted)
    }

    /// Return the contained data if possible.
    fn get(&self) -> Option<&[f32]> {
        match self {
            Self::Valid(data) => Some(&data),
            Self::Protected(data) => Some(&data),
            Self::Deleted => None,
        }
    }
}

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
convert_error!(InvalidId);

//////////////////////
// NeighborAccessor //
//////////////////////

#[derive(Debug, Clone, Copy, Error)]
#[error("Attempt to access an invalid id: {0}")]
pub struct AccessedInvalidId(u32);

crate::always_escalate!(AccessedInvalidId);
convert_error!(AccessedInvalidId);

/// View into the neighbors for [`Index`].
#[derive(Debug, Clone, Copy)]
pub struct NeighborAccessor<'a> {
    provider: &'a Index,
}

impl<'a> NeighborAccessor<'a> {
    pub fn new(provider: &'a Index) -> Self {
        Self { provider }
    }
}

provider::has_id!(NeighborAccessor<'_> = u32);

impl provider::NeighborAccessor for NeighborAccessor<'_> {
    async fn get_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: Empty<'_, Self::Id>,
    ) -> ANNResult<()> {
        self.provider.get_neighbors(id, neighbors)?;
        Ok(())
    }
}

impl provider::NeighborAccessorMut for NeighborAccessor<'_> {
    async fn set_neighbors(&mut self, id: Self::Id, neighbors: &[Self::Id]) -> ANNResult<()> {
        if neighbors.len() > self.provider.max_degree() {
            return Err(message!(
                "trying to assign neighbors with length {} when max degree is {}",
                neighbors.len(),
                self.provider.max_degree()
            ));
        }

        match self.provider.terms.get_mut(&id) {
            Some(mut term) => {
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
                    Ok(())
                }
            }
            None => Err(ANNError::new(AccessedInvalidId(id))),
        }
    }

    async fn append_vector(&mut self, id: Self::Id, neighbors: &[Self::Id]) -> ANNResult<()> {
        match self.provider.terms.get_mut(&id) {
            Some(mut term) => {
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
                    Ok(())
                }
            }
            None => Err(ANNError::new(AccessedInvalidId(id))),
        }
    }
}

///////////////////
// PruneAccessor //
///////////////////

#[derive(Debug)]
pub struct PruneAccessor<'a> {
    provider: &'a Index,
    get_vector: LocalCounter<'a>,
    distance: <f32 as VectorRepr>::Distance,
    set: WorkingSet,
}

type WorkingSet = workingset::Map<u32, Box<[f32]>, workingset::map::Ref<[f32]>>;
type View<'a> = workingset::map::View<'a, u32, Box<[f32]>, workingset::map::Ref<[f32]>>;

impl<'a> PruneAccessor<'a> {
    fn new(provider: &'a Index, set: WorkingSet) -> Self {
        Self {
            provider,
            get_vector: provider.get_vector.local(),
            distance: f32::distance(provider.distance_metric(), Some(provider.dim())),
            set,
        }
    }

    fn provider(&self) -> &Index {
        self.provider
    }
}

provider::has_id!(PruneAccessor<'_> = u32);

impl glue::PruneAccessor for PruneAccessor<'_> {
    type ElementRef<'a> = &'a [f32];

    type View<'a>
        = View<'a>
    where
        Self: 'a;

    type Distance<'a>
        = <f32 as VectorRepr>::Distance
    where
        Self: 'a;

    type Neighbors<'a>
        = NeighborAccessor<'a>
    where
        Self: 'a;

    async fn fill<Itr>(&mut self, itr: Itr) -> ANNResult<(Self::View<'_>, Self::Distance<'_>)>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
    {
        let view = self.set.fill(itr, |i| -> Result<_, AccessedInvalidId> {
            let f = |data: &[f32]| -> Box<[f32]> {
                self.get_vector.increment();
                data.into()
            };

            self.provider.get(i, f)
        })?;

        Ok((view, self.distance))
    }

    fn neighbors(&mut self) -> impl BorrowMut<Self::Neighbors<'_>> + Send {
        NeighborAccessor::new(self.provider)
    }
}

//////////////
// Accessor //
//////////////

#[derive(Debug)]
pub struct Accessor<'a> {
    provider: &'a Index,
    get_vector: LocalCounter<'a>,
    distance: <f32 as VectorRepr>::QueryDistance,
    neighbors: AdjacencyList<u32>,
}

impl<'a> Accessor<'a> {
    /// Return the underlying [`Index`] reference.
    pub fn provider(&self) -> &'a Index {
        self.provider
    }

    /// Return the number of `get_vector` calls made by this accessor (local, not yet
    /// flushed to the provider).
    pub fn get_vector_count(&self) -> usize {
        self.get_vector.value()
    }

    /// Creates an accessor with no flaky behavior (backward-compatible).
    pub fn new(provider: &'a Index, query: &[f32]) -> Result<Self, DimMismatch> {
        if query.len() != provider.dim() {
            Err(DimMismatch {
                got: query.len(),
                expected: provider.dim(),
            })
        } else {
            Ok(Self {
                provider,
                get_vector: provider.get_vector.local(),
                distance: f32::query_distance(query, provider.distance_metric()),
                neighbors: AdjacencyList::new(),
            })
        }
    }

    fn get_distance(&mut self, id: u32) -> Result<Option<f32>, AccessedInvalidId> {
        let f = |data: &[f32]| -> f32 {
            self.get_vector.increment();
            self.distance.evaluate_similarity(data)
        };

        self.provider.get(id, f)
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("query has dim {} but the provider expected {}", self.got, self.expected)]
pub struct DimMismatch {
    got: usize,
    expected: usize,
}

convert_error!(DimMismatch);

provider::has_id!(Accessor<'_> = u32);

impl glue::SearchAccessor for Accessor<'_> {
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> + Send {
        futures_util::future::ok(self.provider.start_point_ids().collect())
    }

    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        for i in self.provider().start_point_ids() {
            match self.get_distance(i)? {
                Some(distance) => f(i, distance),
                None => return Err(ANNError::message("start points must exist")),
            }
        }
        Ok(())
    }

    async fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        mut on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        // Temporarily pilfer `neighbors` so we can call `self.get_distance` while
        // iterating over `neighbors` and not run into a borrowing conflict.
        //
        // We put it back when we're done.
        let mut neighbors = std::mem::take(&mut self.neighbors);
        for id in ids {
            self.provider().get_neighbors(id, neighbors.as_empty())?;
            for &n in neighbors.iter().filter(|i| pred.eval_mut(i)) {
                if let Some(distance) = self.get_distance(n)? {
                    on_neighbors(n, distance)
                }
            }
        }
        self.neighbors = neighbors;
        Ok(())
    }
}

/////////////
// Options //
/////////////

#[derive(Debug, Clone)]
pub struct Options {
    // Set this flag to enable reuse within the [`workingset::Map`]. For multi-threaded
    // baseline tests, this must be set to `false` to obtain repeatable `get_vector` calls.
    working_set_reuse: bool,
}

impl Options {
    pub fn new() -> Self {
        Self {
            working_set_reuse: true,
        }
    }

    pub fn with_options(working_set_reuse: bool) -> Self {
        Self { working_set_reuse }
    }
}

impl Default for Options {
    fn default() -> Self {
        Self::new()
    }
}

impl glue::DefaultPostProcessor<u32> for Accessor<'_> {
    default_post_processor!(glue::Pipeline<glue::FilterStartPoints, glue::CopyIds>);
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test::tokio::current_thread_runtime;
    use diskann_utils::assert_contains;

    #[test]
    fn test_start_point() {
        let start_point = StartPoint::new(42, vec![1.0, 2.0, 3.0]);

        assert_eq!(start_point.id(), 42);
        assert_eq!(start_point.vector(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_config_new() {
        let metric = Metric::L2;
        let graph =
            graph::config::Builder::new(10, graph::config::MaxDegree::same(), 100, metric.into())
                .build()
                .unwrap();

        // Happy path - valid configuration
        {
            let start_points = [
                StartPoint::new(0, vec![1.0, 2.0, 3.0]),
                StartPoint::new(1, vec![4.0, 5.0, 6.0]),
            ];
            let config = Config::new(graph.clone(), metric, start_points).unwrap();
            assert_eq!(config.dim.get(), 3);
            assert_eq!(config.metric, metric);
            assert_eq!(config.start_points.len(), 2);
        }

        // Error: no start points provided
        {
            let err = Config::new(graph.clone(), metric, []).unwrap_err();
            assert!(matches!(err, ConfigError::NeedStartPoint));
        }

        // Error: mismatched dimensions
        {
            let start_points = [
                StartPoint::new(0, vec![1.0, 2.0, 3.0]),
                StartPoint::new(1, vec![4.0, 5.0]), // Different dimension
            ];
            let err = Config::new(graph.clone(), metric, start_points).unwrap_err();
            assert!(matches!(err, ConfigError::MismatchedDims));
        }

        // Error: zero dimension (empty vectors)
        {
            let err = Config::new(graph.clone(), metric, StartPoint::new(0, vec![])).unwrap_err();
            assert!(matches!(err, ConfigError::DimCannotBeZero));
        }

        // Error: duplicate start point IDs
        {
            let start_points = [
                StartPoint::new(0, vec![1.0, 2.0]),
                StartPoint::new(0, vec![3.0, 4.0]), // Same ID
            ];
            let err = Config::new(graph.clone(), metric, start_points).unwrap_err();
            assert!(matches!(err, ConfigError::StartPointsNotUnique));
        }
    }

    // #[test]
    // fn test_vector() {
    //     let vector = vec![1.0, 2.0, 3.0];
    //     let ptr = vector.as_ptr();
    //     let mut v = Vector::Valid(vector);
    //     assert!(!v.is_deleted());

    //     // `v.as_ptr` goes through `Deref` to a slice.
    //     assert_eq!(v.as_ptr(), ptr);

    //     v.mark_deleted();
    //     assert!(v.is_deleted());
    //     assert_eq!(v.as_ptr(), ptr);

    //     v.mark_deleted();
    //     assert!(v.is_deleted());
    //     assert_eq!(v.as_ptr(), ptr);
    // }

    // #[test]
    // fn test_term() {
    //     let vector = vec![1.0, 2.0, 3.0];
    //     let ptr = vector.as_ptr();
    //     let mut t = Term {
    //         neighbors: AdjacencyList::new(),
    //         data: Vector::Valid(vector),
    //     };

    //     assert!(!t.is_deleted());
    //     assert_eq!(t.data.as_ptr(), ptr);

    //     t.mark_deleted();
    //     assert!(t.is_deleted());
    //     assert_eq!(t.data.as_ptr(), ptr);

    //     t.mark_deleted();
    //     assert!(t.is_deleted());
    //     assert_eq!(t.data.as_ptr(), ptr);
    // }

    #[test]
    fn test_provider_new_from() {
        // A dummy graph config.
        let graph = graph::config::Builder::new(
            2,
            graph::config::MaxDegree::same(),
            100,
            (Metric::L2).into(),
        )
        .build()
        .unwrap();

        // Happy path - valid provider with start points and additional points
        {
            let config = Config::new(
                graph.clone(),
                Metric::L2,
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

            let provider = Index::new_from(config, start_points, points).unwrap();
            assert_eq!(provider.dim(), 2);
            assert_eq!(provider.max_degree(), 2);
        }

        // Happy Path: empty iterators (only start points from config)
        {
            let config =
                Config::new(graph.clone(), Metric::L2, [StartPoint::new(0, vec![1.0])]).unwrap();
            let provider = Index::new_from(config, [], []).unwrap();
            assert_eq!(provider.dim(), 1);
        }

        // Error: start point neighbors exceed max degree
        {
            let config =
                Config::new(graph.clone(), Metric::L2, [StartPoint::new(0, vec![1.0])]).unwrap();
            // Exceeds max degree of 5
            let start_points = [(0, AdjacencyList::from_iter_untrusted([1, 2, 3]))];
            let err = Index::new_from(config, start_points, []).unwrap_err();
            assert_contains!(err.to_string(), "max degree");
        }

        // Error: invalid start point ID
        {
            let config =
                Config::new(graph.clone(), Metric::L2, [StartPoint::new(0, vec![1.0])]).unwrap();
            let start_points = [(999, AdjacencyList::new())]; // 999 is not a valid start point
            let err = Index::new_from(config, start_points, []).unwrap_err();
            assert_contains!(err.to_string(), "not a valid start point");
        }

        // Error: regular point neighbors exceed max degree
        {
            let config =
                Config::new(graph.clone(), Metric::L2, [StartPoint::new(0, vec![1.0])]).unwrap();

            // Exceeds max degree
            let points = [(1, vec![2.0], AdjacencyList::from_iter_untrusted([0, 2, 3]))];
            let err = Index::new_from(config, [], points).unwrap_err();
            assert_contains!(err.to_string(), "max degree");
        }

        // Error: trying to assign start point through regular points
        {
            let config =
                Config::new(graph.clone(), Metric::L2, [StartPoint::new(0, vec![1.0])]).unwrap();
            let points = [(0, vec![2.0], AdjacencyList::new())]; // 0 is already a start point
            let err = Index::new_from(config, [], points).unwrap_err();
            assert_contains!(err.to_string(), "cannot assign start point");
        }

        // Error: dimension mismatch in regular points
        {
            let config = Config::new(
                graph.clone(),
                Metric::L2,
                [StartPoint::new(0, vec![1.0, 2.0])],
            )
            .unwrap();
            let points = [(1, vec![3.0], AdjacencyList::new())]; // Wrong dimension (1 instead of 2)
            let err = Index::new_from(config, [], points).unwrap_err();
            assert_contains!(err.to_string(), "expecting dim");
        }

        // Error: inconsistent graph (neighbor points to non-existent ID)
        {
            let config =
                Config::new(graph.clone(), Metric::L2, [StartPoint::new(0, vec![1.0])]).unwrap();
            let points = [(
                1,
                vec![2.0],
                AdjacencyList::from_iter_unique(std::iter::once(999)),
            )]; // 999 doesn't exist
            let err = Index::new_from(config, [], points).unwrap_err();
            assert_contains!(err.to_string(), "not in the provider");
        }
    }

    fn create_test_provider() -> Index {
        let graph_config = graph::config::Builder::new(
            4,
            graph::config::MaxDegree::same(),
            100,
            (Metric::L2).into(),
        )
        .build()
        .unwrap();

        // Edge case: complex valid graph
        let config = Config::new(
            graph_config,
            Metric::L2,
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
        let provider = Index::new_from(config, start_points, points).unwrap();

        assert_eq!(provider.dim(), 2);
        assert_eq!(provider.max_degree(), 4);
        assert_eq!(provider.distance_metric(), Metric::L2);

        provider
    }

    // #[test]
    // fn test_set() {
    //     let provider = create_test_provider();

    //     let query = vec![0.0; provider.dim()];
    //     let mut accessor = super::Accessor::new(&provider, &query).unwrap();
    //     let id = 5;

    //     assert!(accessor.get_distance(5).is_err());

    //     // Setting with the wrong dimension is an error.
    //     {
    //         let v = vec![1.0f32; provider.dim() + 1];
    //         let err = provider.set(&v, id).unwrap_err();
    //         let msg = err.to_string();
    //         assert_contains!(msg, "wrong dim");
    //         assert!(accessor.get_distance(id).is_err());
    //     }

    //     // Setting with the correct dimension is successful.
    //     {
    //         let v = vec![1.0f32; provider.dim()];
    //         provider.set(&v, id).unwrap();

    //         let distance = accessor.get_distance(id).unwrap();
    //         assert_eq!(distance, provider.dim() as f32);
    //     }

    //     // Setting again is an error.
    //     {
    //         let v = vec![1.0f32; provider.dim()];
    //         let err = provider.set(&v, id).unwrap_err();
    //         let msg = err.to_string();
    //         assert_contains!(msg, "vector id 5 is already assigned");
    //     }
    // }

    #[test]
    fn test_neighbor_accessor() {
        use provider::NeighborAccessor;

        let provider = create_test_provider();
        let mut accessor = provider.neighbors();
        let mut v = AdjacencyList::new();

        let rt = current_thread_runtime();

        // The following accesses should all be successful and reflact the initialization
        // performed in `create_test_provider`.
        rt.block_on(accessor.get_neighbors(0, v.as_empty()))
            .unwrap();
        assert_eq!(&*v, &[1, 2, 3]);

        rt.block_on(accessor.get_neighbors(1, v.as_empty()))
            .unwrap();
        assert_eq!(&*v, &[0, 3]);

        rt.block_on(accessor.get_neighbors(2, v.as_empty()))
            .unwrap();
        assert_eq!(&*v, &[0, 3]);

        rt.block_on(accessor.get_neighbors(3, v.as_empty()))
            .unwrap();
        assert_eq!(&*v, &[0, 1, 2]);

        // Accessing an uninitialized vector is an error.
        let err = rt
            .block_on(accessor.get_neighbors(4, v.as_empty()))
            .unwrap_err();
        assert_contains!(err.to_string(), "Attempt to access an invalid id");
    }

    #[test]
    fn test_set_neighbors() {
        use provider::{NeighborAccessor, NeighborAccessorMut};

        let provider = create_test_provider();
        let mut accessor = provider.neighbors();
        let mut v = AdjacencyList::new();

        let rt = current_thread_runtime();

        // Test that emptying neighbors works.
        rt.block_on(accessor.set_neighbors(2, &[])).unwrap();
        rt.block_on(accessor.get_neighbors(2, v.as_empty()))
            .unwrap();
        assert!(v.is_empty());
        assert_eq!(provider.set_neighbors.value(), 1);

        // Adding a few neighbors behaves well.
        rt.block_on(accessor.set_neighbors(2, &[1, 3])).unwrap();
        rt.block_on(accessor.get_neighbors(2, v.as_empty()))
            .unwrap();
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
            rt.block_on(accessor.get_neighbors(2, v.as_empty()))
                .unwrap();
            assert_eq!(&*v, &[1, 3], "original neighbors should be unchanged");

            let msg = err.to_string();
            assert_contains!(msg, "trying to assign neighbors with length 5");

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

            rt.block_on(accessor.get_neighbors(2, v.as_empty()))
                .unwrap();
            assert_eq!(
                &*v,
                &[1, 2, 3],
                "final neighbors should still be deduplicated"
            );
            let msg = err.to_string();
            assert_contains!(msg, "duplicate neighbors detected");

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
            assert_contains!(msg, "access an invalid id");
        }
    }

    #[test]
    fn test_append_vector() {
        use provider::{NeighborAccessor, NeighborAccessorMut};

        let provider = create_test_provider();
        let mut accessor = provider.neighbors();
        let mut v = AdjacencyList::new();

        let rt = current_thread_runtime();

        rt.block_on(accessor.get_neighbors(2, v.as_empty()))
            .unwrap();
        assert_eq!(&*v, &[0, 3]);

        // We can successfully add neighbors.
        rt.block_on(accessor.append_vector(2, &[1])).unwrap();
        rt.block_on(accessor.get_neighbors(2, v.as_empty()))
            .unwrap();
        assert_eq!(&*v, &[0, 3, 1]);
        assert_eq!(provider.append_neighbors.value(), 1);

        // We can add multiple neighbors.
        {
            rt.block_on(accessor.set_neighbors(2, &[])).unwrap();
            rt.block_on(accessor.append_vector(2, &[1, 3, 4])).unwrap();
            rt.block_on(accessor.get_neighbors(2, v.as_empty()))
                .unwrap();
            assert_eq!(&*v, &[1, 3, 4]);
            assert_eq!(provider.append_neighbors.value(), 2);
        }

        // Appending duplicates is an error.
        {
            let err = rt.block_on(accessor.append_vector(2, &[1])).unwrap_err();
            rt.block_on(accessor.get_neighbors(2, v.as_empty()))
                .unwrap();
            assert_eq!(&*v, &[1, 3, 4]);

            let msg = err.to_string();
            assert_contains!(msg, "duplicate ids in append-vector");
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
            rt.block_on(accessor.get_neighbors(2, v.as_empty()))
                .unwrap();
            assert_eq!(&*v, &[1]);

            let msg = err.to_string();
            assert_contains!(msg, "duplicate ids in append-vector");
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
            rt.block_on(accessor.get_neighbors(2, v.as_empty()))
                .unwrap();
            assert_eq!(&*v, &[1]);

            let msg = err.to_string();
            assert_contains!(msg, "will exceed the max degree");
            assert_eq!(provider.append_neighbors.value(), 4);
        }

        // Invalid IDs are caught.
        {
            let err = rt
                .block_on(accessor.append_vector(10, &[1, 2]))
                .unwrap_err();

            let msg = err.to_string();
            assert_contains!(msg, "access an invalid id");
        }
    }

    // #[test]
    // fn test_delete() {
    //     let provider = create_test_provider();

    //     let ids = [0, 1, 2, 3];
    //     let invalid_id = 5;

    //     // Ensure the test provider is not updated too dramatically.
    //     {
    //         let mut check: Vec<_> = provider.all_internal_ids().into_iter().collect();
    //         check.sort();
    //         assert_eq!(&*check, &ids);

    //         assert!(provider.is_start_point(0));
    //         assert!(provider.is_start_point(1));
    //         assert!(!provider.is_start_point(2));
    //         assert!(!provider.is_start_point(3));
    //     }

    //     for i in ids {
    //         let is_deleted = provider.is_deleted(i).unwrap();
    //         assert!(!is_deleted);
    //     }

    //     // Accessing an invalid ID in all APIs returns an error.
    //     {
    //         let err = provider.is_deleted(invalid_id).unwrap_err();
    //         assert_contains!(err.to_string(), "not initialized");
    //     }

    //     // Deleting works.
    //     {
    //         let id = 3;
    //         provider.soft_delete(id).unwrap();
    //         let is_deleted = provider.is_deleted(id).unwrap();
    //         assert!(is_deleted);
    //     }

    //     // Releasing works and completely remove the id.
    //     {
    //         let id = 3;
    //         provider.hard_delete(id).unwrap();
    //         let err = provider.is_deleted(id).unwrap_err();
    //         assert_contains!(err.to_string(), "not initialized");
    //     }
    // }

    // #[test]
    // fn test_start_points_cannot_be_deleted() {
    //     let provider = create_test_provider();

    //     assert!(provider.is_start_point(0));
    //     assert!(provider.is_start_point(1));

    //     let err = provider.soft_delete(0).unwrap_err();
    //     let msg = err.to_string();
    //     assert_contains!(msg, "cannot delete start point");
    //     assert!(!provider.is_deleted(0).unwrap());

    //     let err = provider.hard_delete(0).unwrap_err();
    //     let msg = err.to_string();
    //     assert_contains!(msg, "cannot delete start point");
    //     assert!(!provider.is_deleted(0).unwrap());
    // }
}
