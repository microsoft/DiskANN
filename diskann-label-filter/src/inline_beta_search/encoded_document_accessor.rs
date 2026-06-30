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

impl<IA> glue::SearchAccessor for EncodedDocumentAccessor<IA>
where
    IA: glue::SearchAccessor,
{
    fn starting_points(
        &self,
    ) -> impl std::future::Future<Output = diskann::ANNResult<Vec<Self::Id>>> + Send {
        self.inner_accessor.starting_points()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::{Attribute, AttributeValue};
    use crate::parser::ast::CompareOp;
    use crate::set::SetProvider;
    use serde_json::Value;

    /// Minimal inner accessor that only carries an id type.
    struct MockAccessor;

    impl HasId for MockAccessor {
        type Id = u64;
    }

    /// Builds an accessor whose filter is `category == "electronics"`, with
    /// vector `match_id` carrying the matching attribute and vector
    /// `other_id` carrying an unrelated attribute.
    fn build_accessor(
        beta: f32,
        match_id: u64,
        other_id: u64,
    ) -> EncodedDocumentAccessor<MockAccessor> {
        let mut encoder = AttributeEncoder::new();
        let matching = Attribute::from_value(
            "category".to_string(),
            AttributeValue::String("electronics".to_string()),
        );
        let unrelated = Attribute::from_value(
            "category".to_string(),
            AttributeValue::String("furniture".to_string()),
        );
        let match_attr = encoder.insert(&matching);
        let other_attr = encoder.insert(&unrelated);
        let map = Arc::new(RwLock::new(encoder));

        let mut provider: RoaringTreemapSetProvider<u64> = RoaringTreemapSetProvider::new();
        provider.insert(&match_id, &match_attr).unwrap();
        provider.insert(&other_id, &other_attr).unwrap();
        let attribute_accessor = EncodedAttributeAccessor::new(Arc::new(RwLock::new(provider)));

        let ast = ASTExpr::Compare {
            field: "category".to_string(),
            op: CompareOp::Eq(Value::String("electronics".to_string())),
        };

        EncodedDocumentAccessor::new(MockAccessor, attribute_accessor, map, &ast, beta).unwrap()
    }

    #[test]
    fn attributes_for_scales_distance_when_filter_matches() {
        let mut accessor = build_accessor(0.5, 7, 8);
        let scaled = accessor
            .attributes_for(7, |computer, set| computer.apply(10.0, &set))
            .unwrap();
        assert_eq!(scaled, 5.0);
    }

    #[test]
    fn attributes_for_keeps_distance_when_filter_does_not_match() {
        let mut accessor = build_accessor(0.5, 7, 8);
        let unchanged = accessor
            .attributes_for(8, |computer, set| computer.apply(10.0, &set))
            .unwrap();
        assert_eq!(unchanged, 10.0);
    }

    #[test]
    fn attributes_for_errors_when_point_has_no_labels() {
        let mut accessor = build_accessor(0.5, 7, 8);
        let result = accessor.attributes_for(99, |computer, set| computer.apply(10.0, &set));
        assert!(result.is_err());
    }
}
