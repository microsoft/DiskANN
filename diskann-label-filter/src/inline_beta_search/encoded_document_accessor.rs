/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    borrow::Cow,
    sync::{Arc, RwLock},
};

use diskann::{graph::glue, provider::HasId, ANNError, ANNErrorKind, ANNResult};
use roaring::RoaringTreemap;

use crate::traits::attribute_accessor::AttributeAccessor;
use crate::{
    encoded_attribute_provider::{
        attribute_encoder::AttributeEncoder, encoded_attribute_accessor::EncodedAttributeAccessor,
        encoded_filter_expr::EncodedFilterExpr,
    },
    inline_beta_search::inline_beta_filter::InlineBetaComputer,
    set::roaring_set_provider::RoaringTreemapSetProvider,
    ASTExpr,
};

type AttrAccessor<IA> = EncodedAttributeAccessor<RoaringTreemapSetProvider<<IA as HasId>::Id>>;

pub struct EncodedDocumentAccessor<IA>
where
    IA: HasId,
{
    inner_accessor: IA,
    attribute_accessor: AttrAccessor<IA>,
    computer: InlineBetaComputer,
}

impl<IA> EncodedDocumentAccessor<IA>
where
    IA: HasId,
{
    pub(crate) fn new(
        inner_accessor: IA,
        attribute_accessor: AttrAccessor<IA>,
        attribute_map: Arc<RwLock<AttributeEncoder>>,
        filter_expr: &ASTExpr,
        beta_value: f32,
    ) -> ANNResult<Self> {
        let id_query = EncodedFilterExpr::new(filter_expr, attribute_map.clone())?;
        let computer = InlineBetaComputer::new(beta_value, id_query, attribute_map.clone());
        Ok(Self {
            inner_accessor,
            attribute_accessor,
            computer,
        })
    }

    pub fn inner_accessor(&mut self) -> &mut IA {
        &mut self.inner_accessor
    }

    pub(crate) fn attributes_for<F, R>(&mut self, id: IA::Id, f: F) -> ANNResult<R>
    where
        F: FnOnce(&mut InlineBetaComputer, Cow<'_, RoaringTreemap>) -> R,
    {
        match self
            .attribute_accessor
            .visit_labels_of_point(id, |_, opt_set| match opt_set {
                Some(set) => Ok(f(&mut self.computer, set)),
                None => Err(ANNError::message(
                    ANNErrorKind::IndexError,
                    "No labels were found for vector",
                )),
            }) {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(err)) => Err(err),
            Err(err) => Err(err),
        }
    }
}

impl<IA> HasId for EncodedDocumentAccessor<IA>
where
    IA: HasId,
{
    type Id = <IA as HasId>::Id;
}

impl<IA> glue::SearchAccessor for EncodedDocumentAccessor<IA>
where
    IA: glue::SearchAccessor,
{
    fn starting_points(
        &self,
    ) -> impl std::future::Future<Output = diskann::ANNResult<Vec<Self::Id>>> + Send {
        self.inner_accessor.starting_points()
        // let filter_expr = self.current_filter_expr.clone();
        // let attribute_medoids = self.attribute_medoids.clone();
        // let inner = &self.inner_accessor;
        // async move {
        //     // Extract the expression and compute start points while holding the lock,
        //     // then drop the lock before awaiting
        //     let start_points = {
        //         let expr_guard = filter_expr.read().map_err(|_| {
        //             ANNError::message(
        //                 ANNErrorKind::LockPoisonError,
        //                 "Failed to acquire read lock on filter expression",
        //             )
        //         })?;

        //         if let Some(expr) = expr_guard.as_ref() {
        //             // Compute medoid start points from the stored filter expression
        //             let mut start_points = Vec::new();
        //             Self::collect_start_points_static(expr, &attribute_medoids, &mut start_points);
        //             start_points
        //         } else {
        //             Vec::new()
        //         }
        //         // expr_guard is dropped here
        //     };

        //     let mut inner = inner.starting_points().await?;
        //     inner.extend_from_slice(&start_points);
        //     // Fallback to inner accessor's start points if no expression or no medoids found
        //     Ok(inner)
        // }
    }

    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let mut pairs = Vec::new();
        self.inner_accessor
            .start_point_distances(|id, distance| pairs.push((id, distance)))
            .await?;

        for (id, distance) in pairs {
            let filtered =
                self.attributes_for(id, |computer, set| computer.apply(distance, &set))?;
            f(id, filtered)
        }

        Ok(())
    }

    async fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        mut on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        let mut pairs = Vec::new();
        self.inner_accessor
            .expand_beam(ids, pred, |id, distance| pairs.push((id, distance)))
            .await?;

        for (id, distance) in pairs {
            let filtered =
                self.attributes_for(id, |computer, set| computer.apply(distance, &set))?;
            on_neighbors(id, filtered)
        }

        Ok(())
    }

    fn terminate_early(&mut self) -> bool {
        self.inner_accessor.terminate_early()
    }

    fn is_not_start_point(
        &self,
    ) -> impl std::future::Future<
        Output = diskann::ANNResult<impl Fn(Self::Id) -> bool + Send + Sync + 'static>,
    > + Send {
        self.inner_accessor.is_not_start_point()
    }
}
