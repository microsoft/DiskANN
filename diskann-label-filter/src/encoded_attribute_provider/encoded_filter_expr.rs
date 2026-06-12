/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{Arc, RwLock};

use diskann::{utils::VectorId, ANNResult};

use crate::{
    encoded_attribute_provider::{
        ast_id_expr::ASTIdExpr, ast_label_id_mapper::ASTLabelIdMapper,
        attribute_encoder::AttributeEncoder, roaring_attribute_store::RoaringAttributeStore,
    },
    ASTExpr,
};

pub struct EncodedFilterExpr {
    ast_id_expr: ASTIdExpr<u64>,
    attribute_map: Arc<RwLock<AttributeEncoder>>,
}

impl EncodedFilterExpr {
    pub(crate) fn new(
        ast_expr: &ASTExpr,
        attribute_map: Arc<RwLock<AttributeEncoder>>,
    ) -> ANNResult<Self> {
        let mut mapper = ASTLabelIdMapper::new(attribute_map.clone());
        let ast_id_expr = ast_expr.accept(&mut mapper)?;
        Ok(Self {
            ast_id_expr,
            attribute_map,
        })
    }

    pub fn from_attribute_store<IT>(
        ast_expr: &ASTExpr,
        attribute_store: &RoaringAttributeStore<IT>,
    ) -> ANNResult<Self>
    where
        IT: VectorId,
    {
        Self::new(ast_expr, attribute_store.attribute_map())
    }

    pub fn encoded_filter_expr(&self) -> &ASTIdExpr<u64> {
        &self.ast_id_expr
    }

    pub(crate) fn attribute_map(&self) -> &Arc<RwLock<AttributeEncoder>> {
        &self.attribute_map
    }
}
