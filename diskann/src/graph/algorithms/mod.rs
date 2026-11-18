/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{borrow::BorrowMut, fmt::Debug, num::NonZeroUsize};

use diskann_utils::{
    Reborrow,
    future::{AssertSend, SendFuture},
};
use diskann_vector::DistanceFunction;
use hashbrown::HashSet;

use super::{
    AdjacencyList, Config,
    glue::{self, PruneAccessor, SearchAccessor},
    internal::prune,
    search::{
        record::{self, SearchRecord},
        scratch::{self, PriorityQueueConfiguration, SearchScratch},
    },
    workingset::View,
};

use crate::{
    ANNResult,
    error::ErrorExt,
    internal,
    neighbor::{Neighbor, NeighborQueue},
    provider::{HasId, NeighborAccessor, NeighborAccessorMut},
    tracked_trace,
    utils::{IntoUsize, VectorId},
};

//-------------//
// Sub-modules //
//-------------//

pub mod inplace_delete;
pub mod multi_insert;

/// Statistics collected during a search operation.
///
/// This struct provides detailed metrics about the search process, including
/// the number of nodes visited, the number of distance computations performed,
/// the number of hops taken during the search, and the total number of results returned.
#[derive(Debug, Clone, Copy)]
pub struct SearchStats {
    /// The total number of distance computations performed during the search.
    pub cmps: u32,

    /// The number of hops (iterations) taken during the search traversal.
    pub hops: u32,

    /// The total number of results returned by the search.
    pub result_count: u32,

    /// Whether the range search continued to the second round
    pub range_search_second_round: bool,
}

// FIXME: Can we do better?
pub(crate) fn search_scratch<I>(config: &Config, l: usize, additional: usize) -> SearchScratch<I>
where
    I: VectorId,
{
    SearchScratch::new(
        PriorityQueueConfiguration::Fixed(l + additional),
        Some(scratch::estimate_node_visited_set_size(
            config.max_degree().get(),
            l,
        )),
    )
}

pub(crate) fn estimate_visited_set_capacity(config: &Config, search_l: Option<usize>) -> usize {
    let effective_l = search_l.unwrap_or(config.l_build().get());
    scratch::estimate_node_visited_set_size(config.pruned_degree().get(), effective_l)
}

//--------//
// Search //
//--------//

/// Internal struct for recording search statistics.
pub(crate) struct InternalSearchStats {
    pub(crate) cmps: u32,
    pub(crate) hops: u32,
    pub(crate) range_search_second_round: bool,
}

impl InternalSearchStats {
    pub(crate) fn finish(self, result_count: u32) -> SearchStats {
        SearchStats {
            cmps: self.cmps,
            hops: self.hops,
            result_count,
            range_search_second_round: self.range_search_second_round,
        }
    }
}

pub(crate) async fn search<A, SR, Q>(
    accessor: &mut A,
    beam_width: NonZeroUsize,
    scratch: &mut SearchScratch<A::Id, Q>,
    search_record: &mut SR,
) -> ANNResult<InternalSearchStats>
where
    A: SearchAccessor,
    SR: SearchRecord<A::Id> + ?Sized,
    Q: NeighborQueue<A::Id>,
{
    // paged search can call search_internal multiple times, we only need to initialize
    // state if not already initialized.
    if scratch.visited.is_empty() {
        accessor
            .start_point_distances(|id, distance| {
                scratch.visited.insert(id);
                scratch.best.insert(Neighbor::new(id, distance));
                scratch.cmps += 1;
            })
            .await?;
    }

    let mut neighbors =
        Vec::with_capacity(accessor.max_degree_hint().map(|v| v.get()).unwrap_or(32));

    while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
        scratch.beam_nodes.clear();

        while scratch.beam_nodes.len() < beam_width.get()
            && let Some(closest_node) = scratch.best.closest_notvisited()
        {
            search_record.record(closest_node, scratch.hops, scratch.cmps);
            scratch.beam_nodes.push(*closest_node.id());
        }

        neighbors.clear();
        accessor
            .expand_beam(
                scratch.beam_nodes.iter().copied(),
                glue::NotInMut::new(&mut scratch.visited),
                |id, distance| neighbors.push(Neighbor::new(id, distance)),
            )
            .await?;

        // The predicate ensures that the contents of `neighbors` are unique.
        //
        // We insert into the priority queue outside of the expansion for
        // code-locality purposes.
        neighbors
            .iter()
            .for_each(|neighbor| scratch.best.insert(*neighbor));

        scratch.cmps += neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;
    }

    Ok(InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: false,
    })
}

//--------//
// Insert //
//--------//

pub trait Insert: HasId + Send {
    type SearchAccessor<'a>: SearchAccessor<Id = Self::Id>
    where
        Self: 'a;

    type PruneAccessor<'a>: PruneAccessor<Id = Self::Id>
    where
        Self: 'a;

    fn search_accessor(&mut self) -> impl std::borrow::BorrowMut<Self::SearchAccessor<'_>> + Send;
    fn prune_accessor(&mut self) -> impl std::borrow::BorrowMut<Self::PruneAccessor<'_>> + Send;
}

/// Insert a vector into the index with the given `id` and `vector`.
pub fn insert<A>(accessor: &mut A, id: A::Id, config: &Config) -> impl SendFuture<ANNResult<()>>
where
    A: Insert,
{
    async move {
        let mut find = FindInsertCandidates::new();
        find.find_insert_candidates(accessor, id, config, std::iter::empty())
            .await?;
        let new_neighbors = std::mem::take(find.candidates());

        tracked_trace!(
            "Inserting out edges for vector_id: {} new_out_neighbors: {:?}",
            id,
            new_neighbors,
        );

        let mut prune = accessor.prune_accessor();

        // insert out edges
        prune
            .borrow_mut()
            .neighbors()
            .set_neighbors(id, &new_neighbors)
            .await?;

        // Number of back edges this insert is allowed to add
        let num_back_edges = config.max_backedges().get();

        // add edges from `new_neighbors` to `internal_id` and prune if necessary
        for source in new_neighbors.iter().take(num_back_edges) {
            add_edge_and_prune(
                prune.borrow_mut(),
                std::slice::from_ref(&id),
                *source,
                find.prune(),
                None,
                config,
            )
            .await?;
        }
        Ok(())
    }
}

//----------------------//
// FindInsertCandidates //
//----------------------//

#[derive(Debug)]
pub(crate) struct FindInsertCandidates<I>
where
    I: VectorId,
{
    search: SearchScratch<I>,
    prune: prune::Scratch<I>,
    additional: Vec<I>,
}

impl<I> FindInsertCandidates<I>
where
    I: VectorId,
{
    pub(crate) fn new() -> Self {
        Self {
            search: SearchScratch::new(PriorityQueueConfiguration::Fixed(0), None),
            prune: prune::Scratch::new(),
            additional: Vec::new(),
        }
    }

    /// Return the candidates discovered during search.
    ///
    /// This is only valid after a successful run of [`Self::find_insert_candidates`].
    pub(crate) fn candidates(&mut self) -> &mut AdjacencyList<I> {
        &mut self.prune.neighbors
    }

    /// Return the prune scratch in an undefined state.
    ///
    /// The returned [`prune::Scratch`] should **only** be used for unrelated pruning operations.
    pub(crate) fn prune(&mut self) -> &mut prune::Scratch<I> {
        &mut self.prune
    }

    /// # FIXME
    ///
    /// * Get rid of the insert retry loop. No-one uses is.
    pub(crate) async fn find_insert_candidates<A, Itr>(
        &mut self,
        accessor: &mut A,
        id: I,
        config: &Config,
        extras: Itr,
    ) -> ANNResult<()>
    where
        A: Insert<Id = I>,
        Itr: IntoIterator<Item = I>,
    {
        // Materialize the extras iterator into `candidates` and then drop into the inner
        // method. This avoids some unnecessary monomorphizations for the main work function.
        self.additional.clear();
        self.additional.extend(extras);
        self.find_insert_candidates_inner(accessor, id, config)
            .await
    }

    /// # Implicit Arguments
    ///
    /// Requires [`Self::additional`] to be pre-populated with additional candidates to mix
    /// in with those discovered during search. This can be empty if no such candidates
    /// are needed.
    async fn find_insert_candidates_inner<A>(
        &mut self,
        accessor: &mut A,
        id: I,
        config: &Config,
    ) -> ANNResult<()>
    where
        A: Insert<Id = I>,
    {
        let insert_retry = config.experimental_insert_retry();
        let num_insert_attempts = insert_retry.map_or(1, |v| v.max_retries().get());

        // This will be over-written in the first iteration of search.
        let mut l_build = 0;

        for attempt in 0..num_insert_attempts {
            // Search Phase
            {
                let mut searcher = accessor.search_accessor();

                // Decide whether or not we've sized our scratch data structures.
                //
                // This requires access to the `SearchAccessor`, so we wait until the first attempt
                // to avoid calling `Insert::search_accessor` unnecessarily.
                if attempt == 0 {
                    let num_start_ids = searcher.borrow_mut().num_starting_points().await?;

                    self.search.clear();
                    self.search.resize(config.l_build().get() + num_start_ids);
                    l_build = self.search.best.search_l();

                    self.prune
                        .pool
                        .reserve(estimate_visited_set_capacity(config, Some(l_build)));
                }

                search(
                    searcher.borrow_mut(),
                    NonZeroUsize::MIN,
                    &mut self.search,
                    &mut record::VisitedRef::new_cleared(&mut self.prune.pool),
                )
                .await?;
            }

            let options = prune::Options {
                force_saturate: insert_retry.is_some_and(|v| v.should_saturate(attempt)),
            };

            // Search candidates are pushed directly into `self.prune.pool`, so `self.prune`
            // can be passed directly.
            //
            // Pruned results are available in `self.prune.neighbors`.
            robust_prune_with(
                accessor.prune_accessor().borrow_mut(),
                id,
                &self.additional,
                &mut self.prune,
                options,
                config,
            )
            .await
            .escalate("self element must be retrievable during insert algorithms")?;

            let should_retry =
                insert_retry.is_some_and(|v| v.should_retry(attempt, self.prune.neighbors.len()));
            if !should_retry {
                break;
            }

            l_build *= 2;
            self.search.resize(l_build);
            self.search.clear();
        }

        Ok(())
    }
}

/// A method to add edges from `source` to `targets`, and prune the adjacency list
/// `source` if its degree exceeds `degree*graph_slack_factor`.
/// Uses `append_neighbors` to append the new edge if a prune is not required.
/// Uses `set_neighbors` to set the new adjacency list of `source` if a prune is required.
/// This method mutates the adjacency list of the `source` node and no other node.
///
/// * `context` - The index context passed through from upstream to providers.
/// * `targets` - A list of indices of points to link to from the `source` point
/// * `source` is the index of the source point
/// * `scratch_option` is an option with mutable reference to a scratch space that can be reused for intermediate computations
/// * `working_set` is an option with cache of vectors that can be used to avoid repeated reads
/// * `to_remove` is an option to remove vectors from the adjacency list before adding the new edges
pub(crate) fn add_edge_and_prune<A>(
    accessor: &mut A,
    targets: &[A::Id],
    source: A::Id,
    scratch: &mut prune::Scratch<A::Id>,
    to_remove: Option<&HashSet<A::Id>>,
    config: &Config,
) -> impl SendFuture<ANNResult<()>>
where
    A: PruneAccessor,
{
    async move {
        let mut adj_list = AdjacencyList::with_capacity(config.max_degree().get());
        accessor
            .neighbors()
            .get_neighbors(source, &mut adj_list)
            .await?;

        let mut did_remove = false;
        if let Some(set) = to_remove {
            adj_list.retain(|i| {
                let contains = set.contains(i);
                did_remove |= contains;
                !contains
            })
        }

        let num_new_edges = adj_list.extend_from_slice(targets);
        if num_new_edges == 0 && !did_remove {
            tracked_trace!(
                "Skipping edge insertion from {} to {:?} all edges already exist",
                source,
                targets,
            );
            return Ok(());
        }

        if adj_list.len() <= config.max_degree().get() {
            // No pruning is needed; we can just append.
            tracked_trace!("Appending back-edge from {} to {:?}.", source, targets,);
            if did_remove {
                accessor
                    .neighbors()
                    .set_neighbors(source, &adj_list)
                    .await?;
            } else if let Some(edges) = adj_list.last(num_new_edges) {
                accessor.neighbors().append_vector(source, edges).await?;
            }
        } else {
            // During back-edge insertion, we are working with a limited number
            // of candidates and trying to keep the best of said candidates.
            //
            // No need to enable saturation.
            let options = prune::Options {
                force_saturate: false,
            };

            robust_prune_list(accessor, source, &adj_list, scratch, options, config)
                .await
                .allow_transient("vector may already be unavailable in hard-delete providers")?;

            tracked_trace!(
                "Setting new AdjList for vector_id {} to {:?}.",
                source,
                scratch.neighbors
            );

            accessor
                .neighbors()
                .set_neighbors(source, &scratch.neighbors)
                .await?;
        }

        Ok(())
    }
}

//-------//
// Prune //
//-------//

/// Run the pruning algorithm on [`prune::Context::pool`] and write the pruned
/// list into [`prune::Context::neighbors`].
///
/// # Errors
///
/// Forwards errors from [`PruneAccessor::fill`].
fn robust_prune<A>(
    accessor: &mut A,
    location: A::Id,
    mut context: prune::Context<'_, A::Id>,
    options: prune::Options,
    config: &Config,
) -> impl SendFuture<ANNResult<()>>
where
    A: PruneAccessor,
{
    async move {
        // Early exit.
        if context.pool.is_empty() {
            context.neighbors.clear();
            return Ok(());
        }

        let (view, computer) = accessor
            .fill(context.pool.iter().map(|n| *n.id()))
            .send()
            .await?;

        occlude_list::<A::View<'_>, A::Id, _, _>(
            &computer,
            &mut context,
            view,
            |id| id == location,
            options,
            config,
        );

        Ok(())
    }
}

/// A specialization of [`Self::robust_prune`] that prunes the IDs in an [`AdjacencyList`].
///
/// The resulting candidates list will be placed into `scratch`'s
/// [`prune::Scratch::neighbors`] field.
///
/// All other fields of `scratch` are clobbered.
///
/// This works by first retrieving `location` and all ids in `list` via
/// [`PruneAccessor::fill`] and then computing distances to populate the candidate pool.
///
/// # Errors
///
/// Forwards errors from [`PruneAccessor::fill`]. If `location` was not made available
/// during [`PruneAccessor::fill`], [`prune::ListError::FailedVectorRetrieval`] is
/// returned to delegate escalation to the caller.
fn robust_prune_list<A>(
    accessor: &mut A,
    location: A::Id,
    list: &AdjacencyList<A::Id>,
    scratch: &mut prune::Scratch<A::Id>,
    options: prune::Options,
    config: &Config,
) -> impl SendFuture<Result<(), prune::ListError<A::Id>>>
where
    A: PruneAccessor,
{
    async move {
        // Early exit.
        if list.is_empty() {
            scratch.neighbors.clear();
            return Ok(());
        }

        // Fetch into the working set, including the source location.
        let (view, computer) = accessor
            .fill(internal::chain(
                std::iter::once(location),
                list.iter().copied(),
            ))
            .send()
            .await?;

        scratch.pool.clear();
        scratch.pool.reserve(list.len());

        // Look up the source vector from the working set and compute distances.
        {
            let vector = match view.get(location) {
                Some(v) => v,
                None => return Err(prune::ListError::failed_retrieval(location)),
            };

            for id in list.iter().filter(|&&i| i != location) {
                if let Some(other) = view.get(*id) {
                    scratch.pool.push(Neighbor::new(
                        *id,
                        computer.evaluate_similarity(vector.reborrow(), other.reborrow()),
                    ));
                }
            }
        }

        // Note: Turbofish needed to help inference.
        occlude_list::<A::View<'_>, A::Id, _, _>(
            &computer,
            &mut scratch.as_context(config.max_occlusion_size().get()),
            view,
            |id| id == location,
            options,
            config,
        );

        Ok(())
    }
}

/// A specialization of [`Self::robust_prune`] that prunes using a search record and
/// optional extra candidates.
///
/// The resulting candidates list will be placed into `scratch`'s
/// [`prune::Scratch::neighbors`] field.
///
/// All other fields of `scratch` are clobbered.
///
/// This works by filling `working_set` with `internal_id`, `extras`, and the visited
/// candidates from `record`, capped at [`Self::max_occlusion_size`]. Items are filled in
/// priority order: `internal_id` first, then `extras`, then `record` candidates.
///
/// If `extras` is non-empty, distances from `internal_id` to each extra are computed and
/// pushed into `record` before pruning. This allows batch neighbors to participate in
/// the candidate pool alongside search results.
///
/// # Errors
///
/// Forwards errors from [`PruneAccessor::fill`]. If `internal_id` cannot be retrieved,
/// [`prune::ListError::FailedVectorRetrieval`] is returned.
///
/// # FIXME
///
/// This method is awkward since it manually uses [`PruneAccessor::fill`] to populate extra
/// distances. This functionality should be rolled into a prune trait to both move it behind
/// the user controlled for efficiency and make the intent clearer and to avoid fetching `id`.
fn robust_prune_with<A>(
    accessor: &mut A,
    id: A::Id,
    extras: &[A::Id],
    scratch: &mut prune::Scratch<A::Id>,
    options: prune::Options,
    config: &Config,
) -> impl SendFuture<Result<(), prune::ListError<A::Id>>>
where
    A: PruneAccessor,
{
    async move {
        if extras.is_empty() {
            robust_prune(
                accessor,
                id,
                scratch.as_context(config.max_occlusion_size().get()),
                options,
                config,
            )
            .await?;

            return Ok(());
        }

        // FIXME: We should really sort `scratch.pool`. This is kept to be behavior
        // preserving with an earlier version of the code during a large refactor.
        let (view, computer) = accessor
            .fill(
                internal::chain(
                    std::iter::once(id),
                    internal::chain(extras.iter().copied(), scratch.pool.iter().map(|n| *n.id())),
                )
                .take(config.max_occlusion_size().get()),
            )
            .await?;

        if extras.len() != 0 {
            let this_vector = view
                .get(id)
                .ok_or_else(|| prune::ListError::failed_retrieval(id))?;

            for &extra in extras.iter() {
                if let Some(element) = view.get(extra) {
                    scratch.pool.push(Neighbor::new(
                        extra,
                        computer.evaluate_similarity(this_vector.reborrow(), element.reborrow()),
                    ))
                }
            }
        }

        occlude_list::<A::View<'_>, A::Id, _, _>(
            &computer,
            &mut scratch.as_context(config.max_occlusion_size().get()),
            view,
            |i| i == id,
            options,
            config,
        );

        Ok(())
    }
}

/// Private implementation of the DiskANN pruning algorithm.
///
/// Run the pruning algorithm using [`prune::Context::pool`] as the list of candidate.
/// The `computer` will be used to perform distance computations, pulling elements
/// from `map`. The closure `exclude` can be used to filter out unwanted ids from
/// the candidates list.
///
/// # Output
///
/// After returning, the pruned neighbors can be found in [`prune::Context::neighbors`]
///
/// ## Clobbers
///
/// Clobbers the following scratch fields:
///
/// * [`prune::Context::states`]
///
/// ## Note
///
/// The API for this function is a little awkward in an effort to minimize
/// allocations. Non-trivial inputs, scratch space, and output all reside within
/// [`prune::Context`].
///
/// After `occlude_list` returns, the pruned neighbors can be found in
/// [`prune::Context::neighbors`].
///
/// # Options
///
/// This algorithm saturates the adjacency list if configured in the global
/// configuration or if forced via [`prune::Options`].
fn occlude_list<M, I, C, F>(
    computer: &C,
    context: &mut prune::Context<'_, I>,
    map: M,
    exclude: F,
    options: prune::Options,
    config: &Config,
) where
    M: View<I>,
    I: VectorId,
    C: for<'a, 'b> DistanceFunction<M::ElementRef<'a>, M::ElementRef<'b>, f32>,
    F: Fn(I) -> bool,
{
    assert!(
        context.pool.len() <= u16::MAX.into_usize(),
        "this has an upper bound set by `diskann::graph::Config` and should not exceed `u16::MAX`"
    );

    let prune::Context {
        pool,
        states,
        neighbors,
    } = context;

    if pool.is_empty() {
        neighbors.clear();
        return;
    }

    let alpha = config.alpha();
    let degree = config.pruned_degree().get();

    states.clear();
    states.resize(pool.len(), prune::State::default());

    // To avoid many hash lookups, we pull out just the candidates we're going to prune
    // into an auxiliary vector which can be accessed linearly.
    //
    // Iteration preserves the pool's nearest-first order. Pruning stores results
    // by cache position and resolves each position to `local_id` afterward.
    let mut storage = Vec::with_capacity(pool.len());
    let cache = pool.map_in(
        &mut storage,
        |id| if exclude(*id) { None } else { map.get(*id) },
    );

    let found = prune::robust_prune(
        cache,
        states,
        degree,
        alpha,
        config.prune_kind(),
        |neighbor: &M::Element<'_>, result: &M::Element<'_>| {
            computer.evaluate_similarity((*neighbor).reborrow(), result.reborrow())
        },
    );

    let mut guard = neighbors.resize(found);
    std::iter::zip(guard.iter_mut(), states.iter()).for_each(|(d, s)| {
        *d = *pool[s.neighbor.into_usize()].id();
    });
    guard.finish(found);

    debug_assert!(neighbors.len() <= degree, "max degree bound violated");

    // Post processing saturation if enabled.
    if options.force_saturate || (config.saturate_after_prune() && alpha > 1.0f32) {
        for neighbor in context.pool.iter() {
            if neighbors.len() >= degree {
                break;
            }

            if !exclude(*neighbor.id()) {
                // `AdjacencyList` filters out duplicates. No need to explicitly
                // check.
                neighbors.push(*neighbor.id());
            }
        }
    }
}
