/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{Arc, RwLock};

use crate::{
    encoded_attribute_provider::{
        ast_id_expr::ASTIdExpr, ast_label_id_mapper::ASTLabelIdMapper,
        attribute_encoder::AttributeEncoder,
    },
    ASTExpr,
};

pub(crate) struct EncodedFilterExpr {
    ast_id_expr: Option<ASTIdExpr<u64>>,
}

impl EncodedFilterExpr {
    pub fn new(ast_expr: &ASTExpr, attribute_map: Arc<RwLock<AttributeEncoder>>) -> Self {
        let mut mapper = ASTLabelIdMapper::new(attribute_map);
        match ast_expr.accept(&mut mapper) {
            Ok(ast_id_expr) => Self {
                ast_id_expr: Some(ast_id_expr),
            },
            Err(_e) => Self { ast_id_expr: None },
        }
    }

    pub(crate) fn encoded_filter_expr(&self) -> &Option<ASTIdExpr<u64>> {
        &self.ast_id_expr
    }
}
