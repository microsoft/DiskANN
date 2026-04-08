/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use anyhow::Context;

pub(crate) mod regression;

/// Attempt to load and deserialize from a JSON file on disk identified with `path`.
pub(crate) fn load_from_disk<T>(path: &std::path::Path) -> anyhow::Result<T>
where
    T: for<'a> serde::Deserialize<'a>,
{
    let file = std::fs::File::open(path)
        .with_context(|| format!("while trying to open {}", path.display()))?;

    let reader = std::io::BufReader::new(file);
    Ok(serde_json::from_reader(reader)?)
}
