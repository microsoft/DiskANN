/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Library for parsing label structures and query filter objects as described in the RFC.

// Re-export the AST types
pub use ast::{QueryExpr, CompareOp};

// Re-export all modules
pub mod ast;
pub mod parser;
pub mod evaluator;
pub mod optimized;
pub mod focused_opt;
pub mod pest_parser;

// Re-export the main functions for ease of use
pub use parser::{parse_query_filter, get_value_by_path};
pub use evaluator::eval_query_expr;