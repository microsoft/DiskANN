/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::Reborrow;

use crate::ASTExpr;

/// Type that can be used to specify a query with a filter expression.
/// The Readme.md file in the label-filter folder describes the format
/// of the query expression.
pub struct FilteredQuery<'a, V> {
    query: V,
    filter_expr: &'a ASTExpr,
}

impl<'a, V> FilteredQuery<'a, V> {
    pub fn new(query: V, filter_expr: &'a ASTExpr) -> Self {
        Self { query, filter_expr }
    }

    pub(crate) fn query<'b>(&'b self) -> V::Target
    where
        V: Reborrow<'b>,
    {
        self.query.reborrow()
    }

    pub(crate) fn filter_expr(&self) -> &'a ASTExpr {
        self.filter_expr
    }
}
