/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{files::InputFile, CheckDeserialization, Checker};
use serde::{Deserialize, Serialize};

use crate::inputs::{as_input, Example, Input};

//////////////
// Registry //
//////////////

as_input!(MetadataIndexBuild);

pub(super) fn register_inputs(
    registry: &mut diskann_benchmark_runner::registry::Inputs,
) -> anyhow::Result<()> {
    registry.register(Input::<MetadataIndexBuild>::new())?;
    Ok(())
}

///////////////////////////////
// Metadata-only Index Build //
///////////////////////////////

#[derive(Default, Debug, Serialize, Deserialize, Clone, Copy)]
pub(crate) enum InvertedIndexKind {
    #[serde(rename = "bftree")]
    #[default]
    BfTree,
}

impl std::fmt::Display for InvertedIndexKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InvertedIndexKind::BfTree => write!(f, "bftree"),
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct FilterParams {
    pub(crate) query_predicates: InputFile,
    pub(crate) data_labels: InputFile,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MetadataIndexBuild {
    /// Filter parameters describing predicate and label file locations. The
    /// actual label file used to build the inverted index is taken from
    /// `filter_params.data_labels`.
    pub(crate) filter_params: FilterParams,

    /// Which inverted-index implementation to use when building/evaluating
    /// bitmap filters. If omitted in input files, defaults to `fast`.
    #[serde(default)]
    pub(crate) inverted_index_type: InvertedIndexKind,
}

impl MetadataIndexBuild {
    pub(crate) const fn tag() -> &'static str {
        "metadata-index-build"
    }
}

impl CheckDeserialization for MetadataIndexBuild {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // Validate filter parameters (which include the paths to queries and label files)
        self.filter_params
            .data_labels
            .check_deserialization(checker)?;
        self.filter_params
            .query_predicates
            .check_deserialization(checker)?;
        Ok(())
    }
}

impl Example for MetadataIndexBuild {
    fn example() -> Self {
        Self {
            filter_params: FilterParams {
                query_predicates: InputFile::new("path/to/query_predicates"),
                data_labels: InputFile::new("path/to/labels.jsonl"),
            },
            inverted_index_type: InvertedIndexKind::BfTree,
        }
    }
}

macro_rules! write_field {
    ($f:ident, $field:tt, $($expr:tt)*) => {
        writeln!($f, "{:>18}: {}", $field, $($expr)*)
    }
}

impl std::fmt::Display for MetadataIndexBuild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Metadata-only Index Build\n")?;
        writeln!(f, "tag: \"{}\"", Self::tag())?;
        write_field!(
            f,
            "query predicates",
            self.filter_params.query_predicates.display()
        )?;
        write_field!(f, "data labels", self.filter_params.data_labels.display())?;
        write_field!(f, "inverted index", self.inverted_index_type)?;
        Ok(())
    }
}
