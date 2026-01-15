/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{Arc, RwLock};

use diskann::{
    error::{ErrorExt, IntoANNResult},
    graph::glue::{ExpandBeam, SearchExt},
    provider::{Accessor, AsNeighbor, BuildQueryComputer, DelegateNeighbor, HasId},
    ANNError, ANNErrorKind,
};
use diskann_utils::{future::AsyncFriendly, Reborrow};
use roaring::RoaringTreemap;

use crate::traits::attribute_accessor::AttributeAccessor;
use crate::{
    document::EncodedDocument,
    encoded_attribute_provider::{
        attribute_encoder::AttributeEncoder, encoded_attribute_accessor::EncodedAttributeAccessor,
        encoded_filter_expr::EncodedFilterExpr,
    },
    inline_beta_search::inline_beta_filter::InlineBetaComputer,
    query::FilteredQuery,
    set::roaring_set_provider::RoaringTreemapSetProvider,
};

type AttrAccessor<IA> = EncodedAttributeAccessor<RoaringTreemapSetProvider<<IA as HasId>::Id>>;

pub(crate) struct EncodedDocumentAccessor<IA>
where
    IA: HasId,
{
    inner_accessor: IA,
    attribute_accessor: AttrAccessor<IA>,
    attribute_map: Arc<RwLock<AttributeEncoder>>,
    beta_value: f32,
}

impl<IA> EncodedDocumentAccessor<IA>
where
    IA: Accessor,
{
    pub(crate) fn new(
        inner_accessor: IA,
        attribute_accessor: AttrAccessor<IA>,
        attribute_map: Arc<RwLock<AttributeEncoder>>,
        beta_value: f32,
    ) -> Self {
        Self {
            attribute_accessor,
            inner_accessor,
            attribute_map,
            beta_value,
        }
    }

    pub fn inner_accessor(&mut self) -> &mut IA {
        &mut self.inner_accessor
    }
}

impl<IA> HasId for EncodedDocumentAccessor<IA>
where
    IA: HasId,
{
    type Id = <IA as HasId>::Id;
}

/// Say, while implementing the [`Accessor`] trait, we want
/// Extended = Element = T, and ElementRef = &T and T is a
/// struct.
/// Now Element has to implement Into<Extended> as per the
/// requirements of the Accessor trait. But Extended == Element,
/// so we run into the Rust orphan rule where T::Into<T> has
/// already been defined for all T.
/// So, we introduce a new layer of abstraction with the Extended
/// struct. This is the same as Element, with a different type name.
/// Now Element needs to implement Into<Extended> which sidesteps
/// the orphan rule issue.
pub struct Extended<T, U> {
    element: T,
    map: U,
}

impl<'this, T, U> Reborrow<'this> for Extended<T, U>
where
    T: Reborrow<'this>,
{
    type Target = EncodedDocument<T::Target, &'this U>;

    fn reborrow(&'this self) -> Self::Target {
        EncodedDocument::new(self.element.reborrow(), &self.map)
    }
}

impl<T, U, V> From<EncodedDocument<T, V>> for Extended<U, V>
where
    T: Into<U>,
    V: Clone,
{
    fn from(value: EncodedDocument<T, V>) -> Self {
        let (vec, attrs) = value.destructure();
        Self {
            element: vec.into(),
            map: attrs.clone(),
        }
    }
}

impl<IA> Accessor for EncodedDocumentAccessor<IA>
where
    IA: Accessor,
{
    type Extended = Extended<IA::Extended, RoaringTreemap>;
    type Element<'a>
        = EncodedDocument<IA::Element<'a>, RoaringTreemap>
    where
        Self: 'a;
    type ElementRef<'a> = EncodedDocument<IA::ElementRef<'a>, &'a RoaringTreemap>;
    type GetError = ANNError;

    async fn get_element(&mut self, id: Self::Id) -> Result<Self::Element<'_>, Self::GetError> {
        let future = self.inner_accessor.get_element(id);
        let elem = future.await.escalate("Did not find the vector element")?;

        let attrs = self
            .attribute_accessor
            .visit_labels_of_point(id, |_, opt_set| {
                match opt_set {
                    //TODO: Currently, there is no way but to copy. So we copy the set from the Cow into a
                    //hydrated object.
                    //IMP NOTE: Removing the copy will also change the signature of "Element" and may cause other
                    //downstream issues, so should be done with care!
                    Some(set) => Ok(set.into_owned()),
                    None => Err(ANNError::message(
                        ANNErrorKind::IndexError,
                        "No labels were found for vector",
                    )),
                }
            })?;

        Ok(EncodedDocument::new(elem, attrs?))
    }

    async fn on_elements_unordered<Itr, F>(
        &mut self,
        itr: Itr,
        mut f: F,
    ) -> Result<(), Self::GetError>
    where
        Self: Sync,
        Itr: Iterator<Item = Self::Id> + Send,
        F: Send + for<'a> FnMut(Self::ElementRef<'a>, Self::Id),
    {
        for i in itr {
            let vec = self
                .inner_accessor
                .get_element(i)
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
                    let elem = EncodedDocument::new(vec.reborrow(), &*set);
                    f(elem, i);
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

impl<IA, Q> BuildQueryComputer<FilteredQuery<Q>> for EncodedDocumentAccessor<IA>
where
    IA: BuildQueryComputer<Q>,
    Q: AsyncFriendly + Clone,
{
    type QueryComputerError = ANNError;
    type QueryComputer = InlineBetaComputer<IA::QueryComputer>;

    fn build_query_computer(
        &self,
        from: &FilteredQuery<Q>,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        let inner_computer = self
            .inner_accessor
            .build_query_computer(from.query())
            .into_ann_result()?;
        let id_query = EncodedFilterExpr::new(from.filter_expr(), self.attribute_map.clone())?;

        Ok(InlineBetaComputer::new(
            inner_computer,
            self.beta_value,
            id_query,
        ))
    }
}

impl<IA, Q> ExpandBeam<Q> for EncodedDocumentAccessor<IA>
where
    IA: Accessor,
    EncodedDocumentAccessor<IA>: BuildQueryComputer<Q> + AsNeighbor,
    Q: Clone + AsyncFriendly,
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
