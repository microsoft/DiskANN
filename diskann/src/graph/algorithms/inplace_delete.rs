/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    borrow::BorrowMut,
    fmt::Debug,
};

use diskann_utils::{
    Reborrow,
    future::{SendFuture},
};
use diskann_vector::DistanceFunction;
use hashbrown::{HashMap, HashSet};

use crate::graph::{
    AdjacencyList, Config, algorithms,
    glue::{
        PruneAccessor, SearchAccessor,
    },
    internal::{BackedgeBuffer, prune},
    workingset::View,
};

use crate::{
    ANNResult,
    error::{IntoANNResult},
    internal,
    neighbor::{self, Neighbor},
    provider::HasId,
    provider::{
        NeighborAccessor, NeighborAccessorMut,
    },
    utils::{
        VectorId,
    },
};

pub trait InplaceDelete: HasId + Send {
    type PruneAccessor<'a>: PruneAccessor<Id = Self::Id>
    where
        Self: 'a;

    fn prune_accessor(&mut self) -> impl std::borrow::BorrowMut<Self::PruneAccessor<'_>> + Send;

    /// Keep only the non-deleted entries of `ids`.
    ///
    /// Out-of-bounds IDs may be considered as deleted.
    fn keep_live(&mut self, ids: &mut AdjacencyList<Self::Id>);
}

#[derive(Debug)]
pub struct Scratch<I>
where
    I: VectorId,
{
    edges: HashMap<I, BackedgeBuffer<I>>,
    prune: prune::Scratch<I>,
    pool: Vec<Neighbor<I>>,
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
        }
    }
}

//--------//
// OneHop //
//--------//

#[derive(Debug, Clone)]
pub struct OneHop {
    replacements: usize,
}

impl OneHop {
    pub fn new(replacements: usize) -> Self {
        Self { replacements }
    }

    pub async fn one_hop<A>(
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
        let mut out_neighbors = AdjacencyList::with_capacity(max_degree);

        accessor
            .prune_accessor()
            .borrow_mut()
            .neighbors()
            .get_neighbors(id, &mut out_neighbors)
            .await?;

        accessor.keep_live(&mut out_neighbors);

        let mut in_neighbors = out_neighbors.clone();
        let mut buf = AdjacencyList::with_capacity(max_degree);

        keep_items_referencing(
            &mut in_neighbors,
            id,
            &mut accessor.prune_accessor().borrow_mut().neighbors(),
            &mut buf,
        )
        .await?;

        // The candidates are just the filtered `out_neighbors`.
        //
        // We can reuse `buf` as the target to save an allocation.
        buf.overwrite_trusted(&out_neighbors);

        let candidates = Candidates {
            out_neighbors,
            in_neighbors,
            replace: buf,
        };

        let mut prune = accessor.prune_accessor();
        FilterCandidates { replacements }
            .filter_candidates(
                prune.borrow_mut(),
                &candidates,
                scratch,
            )
            .await?;

        add_edges_and_drop(
            prune.borrow_mut(),
            id,
            config,
            &scratch.edges,
            &mut scratch.prune,
        ).await?;

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
    async move { one_hop.one_hop(accessor, id, config, scratch).await }
}

/////////////
// Helpers //
/////////////

#[derive(Debug)]
struct Candidates<I> {
    // The out-neighbors of the node.
    out_neighbors: AdjacencyList<I>,
    // The in-neighbors of the node.
    in_neighbors: AdjacencyList<I>,
    // Replacement candidates for all `in_neighbors`.
    replace: AdjacencyList<I>,
}

#[derive(Debug, Clone, Copy)]
struct FilterCandidates {
    replacements: usize,
}

impl FilterCandidates {
    // FIXME: This has one small async bit and then a large synchronous bit.
    //
    // It would be nice to split these two once prune gets overhauled.
    async fn filter_candidates<A>(
        &self,
        prune: &mut A,
        candidates: &Candidates<A::Id>,
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
                    candidates.replace.iter().copied(),
                    candidates.in_neighbors.iter().copied(),
                ),
                candidates.out_neighbors.iter().copied(),
            ))
            .await
            .into_ann_result()?;

        self.filter_candidates_inner::<A::View<'_>, A::Id, _>(view, &computer, candidates, scratch);


        Ok(())
    }

    fn filter_candidates_inner<M, I, C>(
        &self,
        view: M,
        computer: &C,
        candidates: &Candidates<I>,
        scratch: &mut Scratch<I>,
    ) where
        I: VectorId,
        M: View<I>,
        C: for<'a, 'b> DistanceFunction<M::ElementRef<'a>, M::ElementRef<'b>, f32>,
    {
        let Self { replacements } = self;

        // For each candidate `c` with an edge to `p`, compute the closest nodes to `c`
        // from the `best_k_candidates`.
        let edges = &mut scratch.edges;
        let pool = &mut scratch.pool;

        edges.clear();
        for neighbor in candidates.in_neighbors.iter() {
            pool.clear();
            let candidate = match view.get(*neighbor) {
                Some(candidate) => candidate,
                None => continue,
            };

            for other_id in candidates.replace.iter() {
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
            let best = pool.iter().take(*replacements).map(|x| *x.id()).collect();
            edges.insert(*neighbor, best);
        }

        for &neighbor in candidates.out_neighbors.iter() {
            pool.clear();

            let candidate = match view.get(neighbor) {
                Some(candidate) => candidate,
                None => continue,
            };

            for other_id in candidates.replace.iter() {
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
            pool.iter().take(*replacements).for_each(|n| {
                edges.entry(*n.id()).or_insert_with(BackedgeBuffer::default).push(neighbor);
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
        algorithms::add_edge_and_prune(prune, dsts, *src, scratch, Some(&delete_set), config).await?;
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

// /// This method deletes a vertex without needing to loop over the entire graph to find
// /// its out-neighbors. It is meant to be called in conjunction with
// /// [`Self::drop_deleted_neighbors`] run occasionally as a background process.
// ///
// /// See `https://arxiv.org/abs/2502.13826` for full description and experiments
// pub fn inplace_delete<S>(
//     &self,
//     strategy: S,
//     context: &DP::Context,
//     id: &DP::ExternalId,
//     num_to_replace: usize,
//     inplace_delete_method: InplaceDeleteMethod,
// ) -> impl SendFuture<ANNResult<()>>
// where
//     S: InplaceDeleteStrategy<DP> + Sync + Clone,
//     DP: Delete,
// {
//     async move {
//         let vector_id = self
//             .data_provider
//             .to_internal_id(context, id)
//             .escalate("id translation for `inplace_delete` must succeed")?;
//
//         let edges_to_add = self
//             .inplace_delete_inner(
//                 &strategy,
//                 context,
//                 id,
//                 num_to_replace,
//                 &inplace_delete_method,
//             )
//             .await?;
//
//         let mut delete_set = HashSet::with_capacity(1);
//         delete_set.insert(vector_id);
//         let prune_strategy = strategy.prune_strategy();
//
//         let mut accessor = prune_strategy
//             .prune_accessor(self.provider(), context, self.max_occlusion_size())
//             .into_ann_result()?;
//
//         let mut prune_scratch = prune::Scratch::new();
//
//         for (neighbor, edges) in edges_to_add.iter() {
//             self.add_edge_and_prune(
//                 &mut accessor,
//                 edges,
//                 *neighbor,
//                 &mut prune_scratch,
//                 Some(&delete_set), // delete the edge to the deleted point as part of add_edge_and_prune
//             )
//             .await?
//         }
//
//         self.drop_adj_list(&mut accessor.neighbors(), vector_id)
//             .await?;
//
//         Ok(())
//     }
// }
