// /*
//  * Copyright (c) Microsoft Corporation.
//  * Licensed under the MIT license.
//  */
//
// use std::{
//     fmt::Debug,
//     marker::PhantomData,
//     sync::atomic::{AtomicUsize, Ordering},
// };
//
// use hashbrown::HashMap;
//
// use crate::{
//     ANNResult,
//     error::ErrorExt,
//     graph::{
//         algorithms::inplace_delete,
//         AdjacencyList, Config, algorithms,
//         glue::PruneAccessor,
//         internal::{BackedgeBuffer, prune},
//     },
//     utils::{VectorId, async_tools},
// };
//
// //----------//
// // WorkList //
// //----------//
//
// #[derive(Debug)]
// pub struct BatchInplaceDelete<I>
// where
//     I: VectorId,
// {
//     scratch: inplace_delete::Scratch<I>,
// }
//
// impl<I> BatchInplaceDelete<I>
// where
//     I: VectorId,
// {
//     pub fn new() -> Self {
//         Self {
//             scratch: inplace_delete::Scratch::new(),
//         }
//     }
//
//     pub async fn one_hop<A>(
//         &mut self,
//         one_hop: &inplace_delete::OneHop,
//         delta: &mut Delta<I>,
//         accessor: &mut A,
//         id: I,
//         config: &Config,
//     ) -> ANNResult<()>
//     where
//         A: inplace_delete::InplaceDelete<Id = I>,
//     {
//         one_hop.discover(
//             accessor,
//             id,
//             config,
//             &mut self.scratch
//         ).await?;
//
//         delta.accumulate(self.scratch.delta());
//
//         Ok(())
//     }
//
//     pub async fn two_hop<A>(
//         &mut self,
//         two_hop: &inplace_delete::TwoHop,
//         delta: &mut Delta<I>,
//         accessor: &mut A,
//         id: I,
//         config: &Config,
//     ) -> ANNResult<()>
//     where
//         A: inplace_delete::InplaceDelete<Id = I>,
//     {
//         two_hop.discover(
//             accessor,
//             id,
//             config,
//             &mut self.scratch
//         ).await?;
//
//         delta.accumulate(self.scratch.delta());
//         Ok(())
//     }
//
//     pub async fn visited_and_topk<A>(
//         &mut self,
//         visited_and_topk: &inplace_delete::VisitedAndTopK,
//         delta: &mut Delta<I>,
//         accessor: &mut A,
//         id: I,
//         config: &Config,
//     ) -> ANNResult<()>
//     where
//         A: inplace_delete::SearchInplaceDelete<Id = I>,
//     {
//         visited_and_topk.discover(
//             accessor,
//             id,
//             config,
//             &mut self.scratch
//         ).await?;
//
//         delta.accumulate(self.scratch.delta());
//         Ok(())
//     }
// }
//
//
