/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// Parser modules
pub mod parser {
    pub mod ast;
    pub mod evaluator;
    pub mod format;
    pub mod query_parser;
}

// Utils
pub mod utils {
    pub mod flatten_utils;
    pub mod jsonl_reader;
}

pub mod inline_beta_search {
    pub mod encoded_document_accessor;
    pub mod inline_beta_filter;
    pub mod predicate_evaluator;
}

// Persisent Index Traits
pub mod traits {
    pub mod attribute_accessor;
    pub mod attribute_store;
    pub mod inverted_index_trait;
    pub mod key_codec;
    pub mod kv_store_traits;
    pub mod posting_list_trait;
    pub mod query_evaluator;
}

//Modules that  handle predicates by mapping them
// to integers.
pub mod encoded_attribute_provider {
    pub(crate) mod ast_id_expr;
    pub(crate) mod ast_label_id_mapper;
    pub(crate) mod attribute_encoder;
    pub mod document_provider;
    pub mod encoded_attribute_accessor;
    pub(crate) mod encoded_filter_expr;
    pub mod roaring_attribute_store;
}

pub mod tests {
    #[cfg(test)]
    pub mod attribute_accessor_test;
    #[cfg(test)]
    pub mod common;
    #[cfg(test)]
    pub mod roaring_attribute_store_test;
}

pub mod attribute;
pub mod document;
pub mod query;
pub mod set;

// Index implementations
pub mod kv_index;

// Storage backends
pub mod stores;

// Re-exports for convenience
pub use parser::ast::{ASTExpr, CompareOp};
pub use parser::evaluator::eval_query_expr;
pub use parser::query_parser::{get_value_by_path, parse_query_filter};
pub use traits::inverted_index_trait::InvertedIndexProvider;
pub use traits::key_codec::DefaultKeyCodec;
pub use traits::kv_store_traits::KvStore;
pub use utils::flatten_utils::Attributes;
pub use utils::jsonl_reader::{
    read_and_parse_queries, read_baselabels, read_ground_truth, read_queries, JsonlReadError,
};
