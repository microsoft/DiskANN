/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashMap,
    fmt,
    fs::File,
    path::{Path, PathBuf},
};

use serde::{
    Deserialize, Deserializer,
    de::{MapAccess, Visitor},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RunbookError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parsing error: {0}")]
    Parsing(#[from] serde_saphyr::Error),
    #[error("runbook missing steps for {0}")]
    NoSteps(String),
}

#[derive(Debug, Deserialize)]
#[serde(transparent)]
pub struct Runbook {
    #[serde(skip)]
    path: PathBuf,
    recipes: HashMap<String, Recipe>,
}

#[derive(Debug, Deserialize)]
pub struct Recipe {
    #[serde(rename = "max_pts")]
    max_points: usize,
    #[serde(flatten)]
    steps: Steps,
}

#[derive(Debug)]
struct Steps(Vec<Operation>);

impl<'de> Deserialize<'de> for Steps {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StepsVisitor;

        impl<'de> Visitor<'de> for StepsVisitor {
            type Value = Steps;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a mapping of step numbers to operations")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Steps, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut steps = Vec::new();
                while let Some((idx, operation)) = map.next_entry::<String, Operation>()? {
                    let idx = idx
                        .parse::<usize>()
                        .map_err(|_| serde::de::Error::custom("operation index not an integer"))?;
                    if idx != steps.len() + 1 {
                        return Err(serde::de::Error::custom(format!(
                            "operation index {idx} out of sequence"
                        )));
                    }
                    match operation {
                        Operation::Insert { start, end } if start >= end => {
                            return Err(serde::de::Error::custom(
                                "start/end ranges invalid".to_string(),
                            ));
                        }
                        Operation::Delete { start, end } if start >= end => {
                            return Err(serde::de::Error::custom(
                                "start/end ranges invalid".to_string(),
                            ));
                        }
                        Operation::Replace {
                            tags_start,
                            tags_end,
                            ids_start,
                            ids_end,
                        } => {
                            if tags_start >= tags_end || ids_start >= ids_end {
                                return Err(serde::de::Error::custom(
                                    "tag or id ranges are invalid".to_string(),
                                ));
                            }
                            if tags_end.wrapping_sub(tags_start) != ids_end.wrapping_sub(ids_start)
                            {
                                return Err(serde::de::Error::custom(format!(
                                    "replace operation {idx} has mismatched tag and id range sizes"
                                )));
                            }
                        }
                        _ => {}
                    }
                    steps.push(operation);
                }
                Ok(Steps(steps))
            }
        }

        deserializer.deserialize_map(StepsVisitor)
    }
}

impl Recipe {
    pub fn steps(&self) -> impl Iterator<Item = &Operation> {
        self.steps.0.iter()
    }

    pub fn max_points(&self) -> usize {
        self.max_points
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "operation", rename_all = "lowercase")]
pub enum Operation {
    Insert {
        start: usize,
        end: usize,
    },
    Delete {
        start: usize,
        end: usize,
    },
    Search,
    Replace {
        tags_start: usize,
        tags_end: usize,
        ids_start: usize,
        ids_end: usize,
    },
}

impl Operation {
    pub fn name(&self) -> &str {
        match self {
            Operation::Insert { .. } => "insert",
            Operation::Delete { .. } => "delete",
            Operation::Search => "search",
            Operation::Replace { .. } => "replace",
        }
    }
}

impl Runbook {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, RunbookError> {
        let f = File::open(&path)?;

        let mut rb: Runbook = serde_saphyr::from_reader(f)?;
        if let Some((name, _)) = rb.recipes.iter().find(|(_, r)| r.steps.0.is_empty()) {
            return Err(RunbookError::NoSteps(name.clone()));
        }
        rb.path = path.as_ref().into();
        Ok(rb)
    }

    pub fn recipes(&self) -> impl Iterator<Item = &String> {
        self.recipes.keys()
    }

    pub fn recipe(&self, key: &str) -> Option<&Recipe> {
        self.recipes.get(key)
    }

    pub fn name(&self) -> Option<&str> {
        self.path.file_name().and_then(|s| s.to_str())
    }
}

#[cfg(test)]
mod test {
    use std::assert_matches;

    use crate::{
        runbook::{Operation, Runbook, RunbookError},
        test_utils::create_test_yaml,
    };

    #[test]
    fn basic() {
        let (_tmpdir, path) = create_test_yaml(
            r#"
            dataset:
              max_pts: 1000
              1: 
                operation: "insert"
                start: 0
                end: 1000
              2:
                operation: "search"
        "#,
        );

        let rb = Runbook::from_path(&path).expect("runbook failed parsing");
        assert_eq!(rb.recipes().count(), 1);
        let r = rb.recipe("dataset").expect("missing recipe");
        assert_eq!(r.max_points, 1000);
        assert_eq!(r.steps.0.len(), 2);
        assert_matches!(
            r.steps.0[0],
            Operation::Insert {
                start: 0,
                end: 1000
            },
        );
        assert_matches!(r.steps.0[1], Operation::Search);
    }

    #[test]
    fn bad_op_index() {
        let (_tmpdir, path) = create_test_yaml(
            r#"
            dataset:
              max_pts: 1000
              a:
                operation: "insert"
                start: 0
                end: 1000
              2:
                operation: "search"
        "#,
        );

        let rb = Runbook::from_path(&path);
        assert_matches!(rb, Err(RunbookError::Parsing(_)));
    }

    #[test]
    fn bad_op_index_seq() {
        let (_tmpdir, path) = create_test_yaml(
            r#"
            dataset:
              max_pts: 1000
              1:
                operation: "insert"
                start: 0
                end: 1000
              5:
                operation: "search"
        "#,
        );

        let rb = Runbook::from_path(&path);
        assert_matches!(rb, Err(RunbookError::Parsing(_)));
    }

    #[test]
    fn missing_steps() {
        let (_tmpdir, path) = create_test_yaml(
            r#"
            dataset:
              max_pts: 1000
        "#,
        );

        let rb = Runbook::from_path(&path);
        assert_matches!(rb, Err(RunbookError::NoSteps(name)) if name == "dataset");
    }
}
