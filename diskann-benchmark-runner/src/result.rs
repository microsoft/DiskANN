/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A utility for providing incremental saving of results.

use std::path::Path;

use serde::{ser::SerializeSeq, Serialize, Serializer};

/// A helper to generate incremental snapshots of data while a benchmark is progressing.
///
/// Benchmark implementations may use this to save results as they become available rather
/// than waiting until the end.
#[derive(Debug, Clone, Copy)]
pub struct Checkpoint<'a> {
    inner: Option<CheckpointInner<'a>>,
}

impl<'a> Checkpoint<'a> {
    /// Create a new check-point that serializes the zip-combination of `input` and `results`
    /// to `path`.
    ///
    /// This is meant to be used in context where we wish to incrementally save new results
    /// along with all the results generated so far. As such, this requires
    /// ```text
    /// inputs.len() <= results.len()
    /// ```
    /// Subsequent calls to `checkpoint` will be assumed to belong to the input at
    /// `results.len() + 1` and will be saved at that position.
    pub(crate) fn new(
        input: &'a [serde_json::Value],
        results: &'a [serde_json::Value],
        path: &'a Path,
    ) -> anyhow::Result<Self> {
        if results.len() > input.len() {
            Err(anyhow::Error::msg(format!(
                "internal error - results len ({}) is greater than input len ({})",
                results.len(),
                input.len(),
            )))
        } else {
            Ok(Self {
                inner: Some(CheckpointInner {
                    input,
                    results,
                    path,
                }),
            })
        }
    }

    /// Create an empty checkpointer that turns calls to `checkpoint` into a no-op.
    pub(crate) fn empty() -> Self {
        Self { inner: None }
    }

    /// Atomically save the zip of the inputs and results to the configured path.
    pub fn save(&self) -> anyhow::Result<()> {
        if let Some(inner) = &self.inner {
            atomic_save(inner.path, &inner)
        } else {
            Ok(())
        }
    }

    /// Treat `partial` as a new partial result for the current contents of the checkpoint.
    ///
    /// All previously generated results will be saved and `partial` will be grouped at
    /// the input at `self.inner.results.len() + 1`.
    ///
    /// This function should only be called if `self` is not full (as in, there is at least
    /// one input that does not have a corresponding result.
    pub fn checkpoint<T: Serialize + ?Sized>(&self, partial: &T) -> anyhow::Result<()> {
        if let Some(inner) = &self.inner {
            if inner.results.len() == inner.input.len() {
                Err(anyhow::Error::msg("internal error - checkpoint is full"))
            } else {
                let appended = Appended {
                    checkpoint: *inner,
                    partial: serde_json::to_value(partial)?,
                };
                atomic_save(inner.path, &appended)
            }
        } else {
            Ok(())
        }
    }
}

/// Atomically save the serializable `object` to a JSON file at `path`.
///
/// This function works by first serializing to `format!("{}.temp", path)` and then using
/// `std::fs::rename`, making the operation safe from interrupts.
///
/// This can fail for a number of reasons:
///
/// 1. `path` is not an valid file path.
/// 2. The temporary file `format!("{}.temp", path)` already exists.
/// 3. Serialization fails.
/// 4. Renaming fails.
pub(crate) fn atomic_save<T>(path: &Path, object: &T) -> anyhow::Result<()>
where
    T: Serialize + ?Sized,
{
    let temp = format!("{}.temp", path.display());
    if Path::new(&temp).exists() {
        return Err(anyhow::Error::msg(format!(
            "Temporary file {} already exists. Aborting!",
            temp
        )));
    }

    let buffer = std::fs::File::create(&temp)?;
    serde_json::to_writer_pretty(buffer, object)?;
    std::fs::rename(&temp, path)?;
    Ok(())
}

////////////////////////////
// Implementation Details //
////////////////////////////

#[derive(Debug, Clone, Copy)]
struct CheckpointInner<'a> {
    input: &'a [serde_json::Value],
    results: &'a [serde_json::Value],
    path: &'a Path,
}

// This applies the "zip" like behavior between pairs of `input` and `results` in
// `CheckpointInner`, so the data structure can act as a vector of pairs rather than as a
// pair of vectors.
#[derive(Debug, Serialize)]
struct SingleResult<'a> {
    input: &'a serde_json::Value,
    results: &'a serde_json::Value,
}
impl Serialize for CheckpointInner<'_> {
    /// Serialize up to `self.results.len()` pairs of inputs and results.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.results.len()))?;
        for (input, results) in std::iter::zip(self.input.iter(), self.results.iter()) {
            seq.serialize_element(&SingleResult { input, results })?;
        }
        seq.end()
    }
}

/// A lazily appended partial data result.
///
/// NOTE: The associated `Checkpoint` must "have room" for an additional value, That is,
/// `checkpoint.results.len() < checkpoint.input.len()`.
struct Appended<'a> {
    checkpoint: CheckpointInner<'a>,
    partial: serde_json::Value,
}

impl Serialize for Appended<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.checkpoint.results.len() + 1))?;
        std::iter::zip(
            self.checkpoint.input.iter(),
            self.checkpoint
                .results
                .iter()
                .chain(std::iter::once(&self.partial)),
        )
        .try_for_each(|(input, results)| seq.serialize_element(&SingleResult { input, results }))?;
        seq.end()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use serde_json::value::Value;

    use super::*;
    use crate::{test::TypeInput, utils::datatype::DataType};

    fn load_from_file<T>(path: &std::path::Path) -> T
    where
        T: for<'a> Deserialize<'a>,
    {
        let file = std::fs::File::open(path).unwrap();
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader).unwrap()
    }

    // Check that each result has the form:
    // ```
    // {
    //     input: <input-object>,
    //     results: <result-object>,
    // }
    // ```
    fn check_results(results: &[Value], inputs: &[TypeInput], expected: &[Value]) {
        assert_eq!(results.len(), inputs.len());
        assert_eq!(results.len(), expected.len());

        for i in 0..results.len() {
            match &results[i] {
                Value::Object(map) => {
                    assert_eq!(
                        map.len(),
                        2,
                        "Each serialized result should only have two top level entries"
                    );
                    let input = TypeInput::deserialize(&map["input"]).unwrap();
                    assert_eq!(input, inputs[i].clone());
                    assert_eq!(map["results"], expected[i]);
                }
                _ => panic!("incorrect formatting for output {}", i),
            }
        }
    }

    #[test]
    fn test_atomic_save() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let message: &str = "hello world";
        let full = path.join("file.json");
        assert!(!full.exists());
        assert!(atomic_save(&full, message).is_ok());
        assert!(full.exists());

        // Deserialize
        let deserialized: String = load_from_file(&full);
        assert_eq!(deserialized, message);

        // Atomic save should fail if the temp file already exists.
        std::fs::File::create(path.join("file.json.temp")).unwrap();

        let err = atomic_save(&full, message).unwrap_err();
        let message = format!("{:?}", err);
        assert!(message.contains("Temporary file"));
        assert!(message.contains("already exists"));
    }

    #[test]
    fn test_empty() {
        let checkpoint = Checkpoint::empty();

        // Make sure we can still call "save" and "checkpoint".
        assert!(checkpoint.save().is_ok());
        assert!(checkpoint.checkpoint("hello world").is_ok());
    }

    #[test]
    fn test_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let savepath = path.join("output.json");

        let inputs = [
            TypeInput::new(DataType::Float32, 1, false),
            TypeInput::new(DataType::Float16, 2, false),
            TypeInput::new(DataType::Float64, 3, false),
        ];

        let serialized: Vec<_> = inputs
            .iter()
            .map(|i| serde_json::to_value(i).unwrap())
            .collect();

        // No saved values.
        {
            let checkpoint = Checkpoint::new(&serialized, &[], &savepath).unwrap();
            assert!(!savepath.exists());
            checkpoint.save().unwrap();
            assert!(savepath.exists());
            let reloaded: Vec<Value> = load_from_file(&savepath);
            assert!(reloaded.is_empty());

            // Append a new result.
            checkpoint.checkpoint("some string").unwrap();
            let reloaded: Vec<Value> = load_from_file(&savepath);
            check_results(
                &reloaded,
                &inputs[0..1],
                &[Value::String("some string".into())],
            );
        }

        // One saved value.
        {
            let values = vec![serde_json::to_value("some result").unwrap()];
            let checkpoint = Checkpoint::new(&serialized, &values, &savepath).unwrap();

            checkpoint.save().unwrap();
            {
                let reloaded: Vec<Value> = load_from_file(&savepath);
                check_results(
                    &reloaded,
                    &inputs[0..1],
                    &[Value::String("some result".into())],
                );
            }

            // Checkpointing will now yield 2 elements.
            checkpoint.checkpoint("another result").unwrap();
            {
                let reloaded: Vec<Value> = load_from_file(&savepath);
                check_results(
                    &reloaded,
                    &inputs[0..2],
                    &[
                        Value::String("some result".into()),
                        Value::String("another result".into()),
                    ],
                );
            }
        }

        // Full checkpoint.
        {
            let values = vec![
                serde_json::to_value("a").unwrap(),
                serde_json::to_value("b").unwrap(),
                serde_json::to_value("c").unwrap(),
            ];
            let checkpoint = Checkpoint::new(&serialized, &values, &savepath).unwrap();
            checkpoint.save().unwrap();
            let reloaded: Vec<Value> = load_from_file(&savepath);

            check_results(
                &reloaded,
                &inputs,
                &[
                    Value::String("a".into()),
                    Value::String("b".into()),
                    Value::String("c".into()),
                ],
            );

            // If we try to checkpoint, we should get an error.
            let err = checkpoint.checkpoint("too full").unwrap_err();
            let message = err.to_string();
            assert!(message.contains("internal error - checkpoint is full"));
        }

        // Malformed Input
        {
            let values = vec![
                serde_json::to_value("a").unwrap(),
                serde_json::to_value("b").unwrap(),
                serde_json::to_value("c").unwrap(),
                serde_json::to_value("d").unwrap(),
            ];
            let err = Checkpoint::new(&serialized, &values, &savepath).unwrap_err();
            let message = err.to_string();
            assert!(message.contains("internal error - results len"));
        }
    }
}
