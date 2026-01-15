/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{Arc, RwLock};

use diskann::ANNResult;

use crate::{
    encoded_attribute_provider::{
        ast_id_expr::ASTIdExpr, ast_label_id_mapper::ASTLabelIdMapper,
        attribute_encoder::AttributeEncoder,
    },
    ASTExpr,
};

pub(crate) struct EncodedFilterExpr {
    ast_id_expr: ASTIdExpr<u64>,
}

impl EncodedFilterExpr {
    pub fn new(
        ast_expr: &ASTExpr,
        attribute_map: Arc<RwLock<AttributeEncoder>>,
    ) -> ANNResult<Self> {
        let mut mapper = ASTLabelIdMapper::new(attribute_map);
        let ast_id_expr = ast_expr.accept(&mut mapper)?;
        Ok(Self { ast_id_expr })
    }

    pub(crate) fn encoded_filter_expr(&self) -> &ASTIdExpr<u64> {
        &self.ast_id_expr
    }
}
