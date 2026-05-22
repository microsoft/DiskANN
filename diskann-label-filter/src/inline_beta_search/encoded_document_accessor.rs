/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    borrow::Cow,
    sync::{Arc, RwLock},
};

use diskann::{
    error::ErrorExt,
    graph::glue::{ExpandBeam, SearchExt},
    provider::{Accessor, AsNeighbor, DelegateNeighbor, HasId},
    ANNError, ANNErrorKind, ANNResult,
};
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

pub(crate) struct EncodedDocumentAccessor<IA>
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
        let id_query = EncodedFilterExpr::new(filter_expr, attribute_map)?;
        let computer = InlineBetaComputer::new(beta_value, id_query);
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

impl<IA> Accessor for EncodedDocumentAccessor<IA>
where
    IA: Accessor,
{
    type GetError = ANNError;

    async fn get_distance(&mut self, id: Self::Id) -> Result<f32, Self::GetError> {
        let future = self.inner_accessor.get_distance(id);
        let distance = future.await.escalate("Did not find the vector element")?;
        let filtered = self.attributes_for(id, |computer, set| computer.apply(distance, &set))?;

        Ok(filtered)
    }

    async fn distances_unordered<Itr, F>(
        &mut self,
        itr: Itr,
        mut f: F,
    ) -> Result<(), Self::GetError>
    where
        Self: Sync,
        Itr: Iterator<Item = Self::Id> + Send,
        F: Send + FnMut(f32, Self::Id),
    {
        for i in itr {
            let distance = self
                .inner_accessor
                .get_distance(i)
                .await
                .escalate("Failed to get vector from inner accessor")?;
            let _ = self
                .attribute_accessor
                .visit_labels_of_point(i, |_, opt_set| {
                    let set = match opt_set {
                        Some(set) => set,
                        None => {
                            return Err(ANNError::message(
                                ANNErrorKind::IndexError,
                                format!("No attributes found for point.{}", i),
                            ));
                        }
                    };
                    let distance = self.computer.apply(distance, &set);
                    f(distance, i);
                    Ok(())
                });
        }
        Ok(())
    }
}

impl<IA> SearchExt for EncodedDocumentAccessor<IA>
where
    IA: SearchExt,
{
    fn starting_points(
        &self,
    ) -> impl std::future::Future<Output = diskann::ANNResult<Vec<Self::Id>>> + Send {
        self.inner_accessor.starting_points()
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

impl<IA> ExpandBeam for EncodedDocumentAccessor<IA>
where
    IA: Accessor,
    EncodedDocumentAccessor<IA>: AsNeighbor,
{
}

impl<'a, IA> DelegateNeighbor<'a> for EncodedDocumentAccessor<IA>
where
    IA: DelegateNeighbor<'a>,
{
    type Delegate = IA::Delegate;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.inner_accessor.delegate_neighbor()
    }
}
