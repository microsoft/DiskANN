/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::{checker::Checker, input, registry, Any};

#[derive(Debug)]
pub(crate) struct Jobs {
    /// The benchmark jobs to execute.
    jobs: Vec<Any>,
}

impl Jobs {
    /// Return the jobs associated with this benchmark run.
    pub(crate) fn jobs(&self) -> &[Any] {
        &self.jobs
    }

    /// Consume `self`, returning the contained list of jobs.
    pub(crate) fn into_inner(self) -> Vec<Any> {
        self.jobs
    }

    /// Load `self` from a serialized JSON representation at `path`.
    ///
    /// In addition to deserializing the on-disk representation, the method also runs
    /// the post-load validation of the requested runs, including:
    ///
    /// * Resolution of input files.
    pub(crate) fn load(path: &Path, registry: &registry::Inputs) -> anyhow::Result<Self> {
        Self::parse(&Partial::load(path)?, registry)
    }

    /// Parse `self` from a [`Partial`].
    ///
    /// This method also perform deserialization checks on the parsed inputs.
    pub(crate) fn parse(partial: &Partial, registry: &registry::Inputs) -> anyhow::Result<Self> {
        let mut checker = Checker::new(
            partial
                .search_directories
                .iter()
                .map(PathBuf::from)
                .collect(),
            partial.output_directory.as_ref().map(PathBuf::from),
        );

        let num_jobs = partial.jobs.len();
        let jobs: anyhow::Result<Vec<Any>> = partial
            .jobs
            .iter()
            .enumerate()
            .map(|(i, unprocessed)| {
                let context = || {
                    format!(
                        "while processing input {} of {}",
                        i.wrapping_add(1),
                        num_jobs
                    )
                };

                let input = registry
                    .get(&unprocessed.tag)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Unrecognized input tag: \"{}\"", unprocessed.tag)
                    })
                    .with_context(context)?;

                checker.set_tag(input.tag());
                input
                    .try_deserialize(&unprocessed.content, &mut checker)
                    .with_context(context)
            })
            .collect();

        Ok(Self { jobs: jobs? })
    }

    /// Dump an example of `self` as a JSON string.
    pub(crate) fn example() -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&Partial {
            search_directories: vec!["directory/a".into(), "directory/b".into()],
            output_directory: None,
            jobs: Vec::new(),
        })
    }
}

// The naming of these fields is to maintain backwards compatibility with an earlier version
// of the benchmark binary.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Unprocessed {
    #[serde(rename = "type")]
    pub(crate) tag: String,
    pub(crate) content: serde_json::Value,
}

impl Unprocessed {
    pub(crate) fn new(tag: String, content: serde_json::Value) -> Self {
        Self { tag, content }
    }

    pub(crate) fn format_input(example: input::Registered<'_>) -> anyhow::Result<Self> {
        let tag = example.tag().to_string();
        Ok(Self {
            tag,
            content: example.example()?,
        })
    }
}

/// A partially loaded input file.
///
/// To reach this point, we at least the structure of the input JSON to be correct and
/// parseable. However, we have not yet mapped the raw JSON of any of the registered inputs.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Partial {
    /// Directories to search for input files.
    search_directories: Vec<String>,
    /// Directory to search for/write output files
    output_directory: Option<String>,
    jobs: Vec<Unprocessed>,
}

impl Partial {
    /// Load `self` from a serialized JSON representation at `path` without post-load
    /// validation.
    pub(crate) fn load(path: &Path) -> anyhow::Result<Self> {
        crate::internal::load_from_disk(path)
    }

    pub(crate) fn jobs(&self) -> &[Unprocessed] {
        &self.jobs
    }
}
