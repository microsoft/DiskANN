/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::ASTExpr;

/// Type that can be used to specify a query with a filter expression.
/// The Readme.md file in the label-filter folder describes the format
/// of the query expression.
#[derive(Clone)]
pub struct FilteredQuery<V> {
    query: V,
    filter_expr: ASTExpr,
}

impl<V> FilteredQuery<V> {
    pub fn new(query: V, filter_expr: ASTExpr) -> Self {
        Self { query, filter_expr }
    }

    pub(crate) fn query(&self) -> &V {
        &self.query
    }

    pub(crate) fn filter_expr(&self) -> &ASTExpr {
        &self.filter_expr
    }
}
