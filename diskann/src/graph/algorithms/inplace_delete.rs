/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{borrow::BorrowMut, fmt::Debug, num::NonZeroUsize};

use diskann_utils::{Reborrow, future::SendFuture};
use diskann_vector::DistanceFunction;
use hashbrown::{HashMap, HashSet};

use crate::graph::{
    AdjacencyList, Config, algorithms,
    glue::{PruneAccessor, SearchAccessor},
    internal::{BackedgeBuffer, prune},
    workingset::View,
    search::record::NoopSearchRecord,
};

use crate::{
    ANNResult,
    error::IntoANNResult,
    internal,
    neighbor::{self, Neighbor},
    provider::{NeighborAccessor, NeighborAccessorMut, HasId},
    utils::VectorId,
};

pub trait InplaceDelete: HasId + Send {
    type PruneAccessor<'a>: PruneAccessor<Id = Self::Id>
    where
        Self: 'a;

    fn prune_accessor(&mut self) -> impl std::borrow::BorrowMut<Self::PruneAccessor<'_>> + Send;

    /// Keep only the non-deleted entries of `ids` without changing relative ordering.
    ///
    /// It is important that order of the non-deleted entries is preserved. Use
    /// [`AdjacencyList::retain`] for this purpose.
    ///
    /// Out-of-bounds IDs may be considered as deleted.
    fn keep_live(&mut self, ids: &mut AdjacencyList<Self::Id>);
}

pub trait SearchInplaceDelete: InplaceDelete {
    type SearchAccessor<'a>: SearchAccessor<Id = Self::Id>
    where
        Self: 'a;

    fn search_accessor(&mut self) -> impl std::borrow::BorrowMut<Self::SearchAccessor<'_>> + Send;

    //----------//
    // Provided //
    //----------//

    /// Return `true` if [`Self::SearchAccessor`] already removes deleted items from its results.
    ///
    /// When in doubt, do not extend this method. Incorrectly implementing it can severely
    /// degrade the quality of the deletion algorithms.
    fn hint_deleted_already_excluded(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct Scratch<I>
where
    I: VectorId,
{
    /// Output: The delta to apply as a result of running the algorithm.
    edges: HashMap<I, BackedgeBuffer<I>>,

    prune: prune::Scratch<I>,
    pool: Vec<Neighbor<I>>,
    out_neighbors: AdjacencyList<I>,
    in_neighbors: AdjacencyList<I>,
    replace: AdjacencyList<I>,
    // Algorithm specific
    set: HashSet<I>,
}

impl<I> Scratch<I>
where
    I: VectorId,
{
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            prune: prune::Scratch::new(),
            pool: Vec::new(),
            out_neighbors: AdjacencyList::new(),
            in_neighbors: AdjacencyList::new(),
            replace: AdjacencyList::new(),
            set: HashSet::new(),
        }
    }

    pub(crate) fn delta(&self) -> &HashMap<I, BackedgeBuffer<I>> {
        &self.edges
    }

    pub async fn one_hop<A>(
        &mut self,
        one_hop: &OneHop,
        accessor: &mut A,
        id: A::Id,
        config: &Config,
    ) -> ANNResult<()>
    where
        A: InplaceDelete<Id = I>,
    {
        one_hop.discover(
            accessor,
            id,
            config,
            self
        ).await?;

        add_edges_and_drop(
            accessor.prune_accessor().borrow_mut(),
            id,
            config,
            &self.edges,
            &mut self.prune,
        ).await?;

        Ok(())
    }

    pub async fn two_hop<A>(
        &mut self,
        two_hop: &TwoHop,
        accessor: &mut A,
        id: A::Id,
        config: &Config,
    ) -> ANNResult<()>
    where
        A: InplaceDelete<Id = I>,
    {
        two_hop.discover(
            accessor,
            id,
            config,
            self
        ).await?;

        add_edges_and_drop(
            accessor.prune_accessor().borrow_mut(),
            id,
            config,
            &self.edges,
            &mut self.prune,
        ).await?;

        Ok(())
    }

    pub async fn visited_and_topk<A>(
        &mut self,
        visited_and_topk: &VisitedAndTopK,
        accessor: &mut A,
        id: A::Id,
        config: &Config,
    ) -> ANNResult<()>
    where
        A: SearchInplaceDelete<Id = I>,
    {
        visited_and_topk.discover(
            accessor,
            id,
            config,
            self
        ).await?;

        add_edges_and_drop(
            accessor.prune_accessor().borrow_mut(),
            id,
            config,
            &self.edges,
            &mut self.prune,
        ).await?;

        Ok(())
    }
}

//--------//
// OneHop //
//--------//

#[derive(Debug, Clone)]
pub struct OneHop {
    replacements: NonZeroUsize,
}

impl OneHop {
    pub fn new(replacements: NonZeroUsize) -> Self {
        Self { replacements }
    }

    pub(crate) async fn discover<A>(
        &self,
        accessor: &mut A,
        id: A::Id,
        config: &Config,
        scratch: &mut Scratch<A::Id>,
    ) -> ANNResult<()>
    where
        A: InplaceDelete,
    {
        let Self { replacements } = self.clone();

        let max_degree = config.max_degree().get();

        accessor
            .prune_accessor()
            .borrow_mut()
            .neighbors()
            .get_neighbors(id, &mut scratch.out_neighbors)
            .await?;

        accessor.keep_live(&mut scratch.out_neighbors);

        scratch.in_neighbors.clear();
        scratch
            .in_neighbors
            .overwrite_trusted(&scratch.out_neighbors);

        let mut temp = &mut scratch.replace;
        keep_items_referencing(
            &mut scratch.in_neighbors,
            id,
            &mut accessor.prune_accessor().borrow_mut().neighbors(),
            temp,
        )
        .await?;

        // The candidates are just the filtered `out_neighbors`.
        //
        // We can reuse `buf` as the target to save an allocation.
        scratch.replace.overwrite_trusted(&scratch.out_neighbors);

        let mut prune = accessor.prune_accessor();
        FilterCandidates { replacements }
            .filter_candidates(prune.borrow_mut(), scratch)
            .await?;

        Ok(())
    }
}

fn assert_send_one_hop<A>(
    one_hop: OneHop,
    accessor: &mut A,
    id: A::Id,
    config: &Config,
    scratch: &mut Scratch<A::Id>,
) -> impl SendFuture<ANNResult<()>>
where
    A: InplaceDelete,
{
    async move { one_hop.discover(accessor, id, config, scratch).await }
}

//--------//
// TwoHop //
//--------//

#[derive(Debug, Clone)]
pub struct TwoHop {
    replacements: NonZeroUsize,
}

impl TwoHop {
    pub fn new(replacements: NonZeroUsize) -> Self {
        Self { replacements }
    }

    pub(crate) async fn discover<A>(
        &self,
        accessor: &mut A,
        id: A::Id,
        config: &Config,
        scratch: &mut Scratch<A::Id>,
    ) -> ANNResult<()>
    where
        A: InplaceDelete,
    {
        let Self { replacements } = self.clone();
        accessor
            .prune_accessor()
            .borrow_mut()
            .neighbors()
            .get_neighbors(id, &mut scratch.out_neighbors)
            .await?;

        accessor.keep_live(&mut scratch.out_neighbors);

        let set = &mut scratch.set;
        set.clear();

        scratch.in_neighbors.clear();
        let mut temp = &mut scratch.replace;

        let mut prune = accessor.prune_accessor();

        {
            let mut neighbors = prune.borrow_mut().neighbors();
            for neighbor in scratch.out_neighbors.iter() {
                set.insert(*neighbor);
                neighbors.get_neighbors(*neighbor, &mut temp).await?;
                set.extend(temp.iter().copied());

                if temp.contains(id) {
                    // We can use `push_unique` because this is already coming from an
                    // adjacency list.
                    scratch.in_neighbors.push_unique(id);
                }
            }
        }

        scratch
            .replace
            .replace_from_iter_unique(set.iter().copied());

        FilterCandidates { replacements }
            .filter_candidates(prune.borrow_mut(), scratch)
            .await?;

        Ok(())
    }
}

//----------------//
// VisitedAndTopK //
//----------------//

#[derive(Debug, Clone)]
pub struct VisitedAndTopK {
    replacements: NonZeroUsize,
    l_value: NonZeroUsize,
    k_value: NonZeroUsize,
}

impl VisitedAndTopK {
    pub fn new(
        replacements: NonZeroUsize,
        l_value: NonZeroUsize,
        k_value: NonZeroUsize,
    ) -> Self {
        Self { replacements, l_value, k_value }
    }


    pub(crate) async fn discover<A>(
        &self,
        accessor: &mut A,
        id: A::Id,
        config: &Config,
        scratch: &mut Scratch<A::Id>,
    ) -> ANNResult<()>
    where
        A: SearchInplaceDelete,
    {
        let Self { replacements, l_value, k_value } = self;

        // Find the replacement candidates using graph search.
        {
            let mut search_accessor = accessor.search_accessor();
            let num_start_points = search_accessor.borrow_mut().num_starting_points().await?;

            let mut search_scratch = algorithms::search_scratch(
                config,
                l_value.get(),
                num_start_points,
            );

            algorithms::search(
                search_accessor.borrow_mut(),
                NonZeroUsize::MIN,
                &mut search_scratch,
                &mut NoopSearchRecord::new(),
            ).await?;

            scratch.replace.replace_from_iter_trusted(
                search_scratch.best.iter().take(l_value.get()).map(|n| *n.id())
            );
        }


        // Get the current out-neighbors and filter out deleted points from the candidate
        // stream (if needed) as well as the out-neighbors.
        accessor
            .prune_accessor()
            .borrow_mut()
            .neighbors()
            .get_neighbors(id, &mut scratch.out_neighbors)
            .await?;

        if !accessor.hint_deleted_already_excluded() {
            accessor.keep_live(&mut scratch.replace);
        }

        accessor.keep_live(&mut scratch.out_neighbors);

        scratch.in_neighbors.clear();
        scratch.in_neighbors.overwrite_trusted(&scratch.replace);

        let mut prune = accessor.prune_accessor();

        // The in-neighbors are those in the replacement stream that have an edge `id`.
        keep_items_referencing(
            &mut scratch.in_neighbors,
            id,
            &mut prune.borrow_mut().neighbors(),
            &mut AdjacencyList::with_capacity(config.max_degree().get()),
        ).await?;

        scratch.replace.truncate(k_value.get());
        FilterCandidates { replacements: *replacements }
            .filter_candidates(prune.borrow_mut(), scratch)
            .await?;

        Ok(())
    }
}

/////////////
// Helpers //
/////////////

#[derive(Debug, Clone, Copy)]
struct FilterCandidates {
    replacements: NonZeroUsize,
}

impl FilterCandidates {
    async fn filter_candidates<A>(
        &self,
        prune: &mut A,
        scratch: &mut Scratch<A::Id>,
    ) -> ANNResult<()>
    where
        A: PruneAccessor,
    {
        // This is the total union of elements that we consider for neighbors.
        // It's possible there is some repitition, but implementations of `Fill` should
        // already properly handle that.
        let (view, computer) = prune
            .fill(internal::chain(
                internal::chain(
                    scratch.replace.iter().copied(),
                    scratch.in_neighbors.iter().copied(),
                ),
                scratch.out_neighbors.iter().copied(),
            ))
            .await
            .into_ann_result()?;

        self.filter_candidates_inner::<A::View<'_>, A::Id, _>(view, &computer, scratch);

        Ok(())
    }

    fn filter_candidates_inner<M, I, C>(&self, view: M, computer: &C, scratch: &mut Scratch<I>)
    where
        I: VectorId,
        M: View<I>,
        C: for<'a, 'b> DistanceFunction<M::ElementRef<'a>, M::ElementRef<'b>, f32>,
    {
        let Self { replacements } = self;

        // For each candidate `c` with an edge to `p`, compute the closest nodes to `c`
        // from the `best_k_candidates`.
        let edges = &mut scratch.edges;
        let pool = &mut scratch.pool;

        let in_neighbors = &scratch.in_neighbors;
        let out_neighbors = &scratch.out_neighbors;
        let replace = &scratch.replace;

        edges.clear();
        for neighbor in in_neighbors.iter() {
            pool.clear();
            let candidate = match view.get(*neighbor) {
                Some(candidate) => candidate,
                None => continue,
            };

            for other_id in replace.iter() {
                // Don't add a self edge.
                if other_id == neighbor {
                    continue;
                }

                let other = match view.get(*other_id) {
                    Some(other) => other,
                    None => continue,
                };

                pool.push(Neighbor::new(
                    *other_id,
                    computer.evaluate_similarity(candidate.reborrow(), other.reborrow()),
                ));
            }

            pool.sort_unstable_by(neighbor::ord::fast_distance);
            let best = pool.iter().take(replacements.get()).map(|x| *x.id()).collect();
            edges.insert(*neighbor, best);
        }

        for &neighbor in out_neighbors.iter() {
            pool.clear();

            let candidate = match view.get(neighbor) {
                Some(candidate) => candidate,
                None => continue,
            };

            for other_id in replace.iter() {
                // Don't add a self edge.
                if other_id == &neighbor {
                    continue;
                }

                let other = match view.get(*other_id) {
                    Some(other) => other,
                    None => continue,
                };

                pool.push(Neighbor::new(
                    *other_id,
                    computer.evaluate_similarity(candidate.reborrow(), other.reborrow()),
                ));
            }

            pool.sort_unstable_by(neighbor::ord::fast_distance);
            pool.iter().take(replacements.get()).for_each(|n| {
                edges
                    .entry(*n.id())
                    .or_insert_with(BackedgeBuffer::default)
                    .push(neighbor);
            });
        }
    }
}

async fn add_edges_and_drop<A>(
    prune: &mut A,
    id: A::Id,
    config: &Config,
    edges: &HashMap<A::Id, BackedgeBuffer<A::Id>>,
    scratch: &mut prune::Scratch<A::Id>,
) -> ANNResult<()>
where
    A: PruneAccessor,
{
    let delete_set = HashSet::from_iter(std::iter::once(id));
    for (src, dsts) in edges {
        algorithms::add_edge_and_prune(prune, dsts, *src, scratch, Some(&delete_set), config)
            .await?;
    }

    prune.neighbors().clear(id).await?;
    Ok(())
}

/// Keep only the elements in `candidates` whose adjacency lists contain `id`.
async fn keep_items_referencing<A>(
    candidates: &mut AdjacencyList<A::Id>,
    id: A::Id,
    accessor: &mut A,
    buf: &mut AdjacencyList<A::Id>,
) -> ANNResult<()>
where
    A: NeighborAccessor,
{
    let mut i = 0;
    while let Some(candidate) = candidates.get(i).copied() {
        accessor.get_neighbors(candidate, buf).await?;

        if buf.contains(id) {
            // Keep this item, move the index.
            i += 1;
        } else {
            // Remove this item, replacing it with the item at the end which hasn't been
            // checked yet. So no need to increment the index.
            candidates.swap_remove(i);
        }
    }

    Ok(())
}

//-------//
// Batch //
//-------//

#[derive(Debug)]
pub struct Delta<I> {
    delta: HashMap<I, BackedgeBuffer<I>>,
}

impl<I> Delta<I> {
    pub fn new() -> Self {
        Self {
            delta: HashMap::new(),
        }
    }

    fn accumulate(&mut self, other: &HashMap<I, BackedgeBuffer<I>>)
    where
        I: VectorId,
    {
        for (src, dst) in other.iter() {
            self.delta.entry(*src)
                .and_modify(|b| {
                    for d in dst.iter() {
                        b.push(*d);
                    }
                })
                .or_insert_with(|| dst.clone());
        }
    }
}


