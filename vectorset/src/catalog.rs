/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashMap,
    ffi::OsStr,
    path::{Path, PathBuf},
};

use thiserror::Error;

use crate::dataset::{Dataset, DatasetError};

#[derive(Debug, Error)]
pub enum CatalogError {
    #[error("dataset key (stem of {0}) is not valid")]
    BadName(PathBuf),
    #[error("dataset error: {0}")]
    Dataset(#[from] DatasetError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub struct Catalog {
    datasets: HashMap<String, Dataset>,
}

impl Catalog {
    pub fn load_directory<P: AsRef<Path>, S: AsRef<Path>>(
        path: P,
        search_paths: Option<&[S]>,
    ) -> Result<Self, CatalogError> {
        let mut datasets = HashMap::new();

        for spec_path in std::fs::read_dir(path)?
            .filter_map(|de| de.ok())
            .filter_map(|de| {
                if let Ok(ft) = de.file_type()
                    && ft.is_file()
                    && de.path().extension().unwrap_or(OsStr::new("")) == "yaml"
                {
                    Some(de.path())
                } else {
                    None
                }
            })
        {
            let name = spec_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if name.is_empty() {
                return Err(CatalogError::BadName(spec_path));
            }

            let ds = Dataset::from_path(spec_path.clone(), search_paths)?;

            datasets.insert(name.to_string(), ds);
        }

        Ok(Self { datasets })
    }

    pub fn dataset(&self, name: &str) -> Option<&Dataset> {
        self.datasets.get(name)
    }
}
