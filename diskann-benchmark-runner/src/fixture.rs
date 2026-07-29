/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Cached CLI-fixture support shared by benchmark applications.
//!
//! A fixture directory contains `stdin.txt`, `stdout.txt`, and optional JSON files.
//! Set `DISKANN_TEST=overwrite` to regenerate expected output.

use std::{
    ffi::OsString,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context};

use crate::{output, ux, App, Registry};

const ENV: &str = "DISKANN_TEST";
const STDIN: &str = "stdin.txt";
const STDOUT: &str = "stdout.txt";
const INPUT_FILE: &str = "input.json";
const OUTPUT_FILE: &str = "output.json";
const TOLERANCES_FILE: &str = "tolerances.json";
const REGRESSION_INPUT_FILE: &str = "regression_input.json";
const CHECK_OUTPUT_FILE: &str = "checks.json";
const FEATURES_FILE: &str = "features.txt";
const GENERATED_OUTPUTS: [&str; 2] = [OUTPUT_FILE, CHECK_OUTPUT_FILE];
const WORKSPACE: &str = "$WORKSPACE";

fn read(path: &Path) -> anyhow::Result<String> {
    Ok(ux::normalize(std::fs::read_to_string(path).with_context(
        || format!("failed to read {}", path.display()),
    )?))
}

fn overwrite() -> anyhow::Result<bool> {
    match std::env::var(ENV) {
        Ok(value) if value == "overwrite" => Ok(true),
        Ok(value) => bail!("unknown {ENV} value {value:?}; expected \"overwrite\""),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(error) => Err(error).context(format!("failed to read {ENV}")),
    }
}

fn resolve(value: &str, input: &Path, dir: &Path, tempdir: &Path) -> PathBuf {
    match value {
        "$INPUT" => input.into(),
        "$OUTPUT" => tempdir.join(OUTPUT_FILE),
        "$SETUP_OUTPUT" => tempdir.join("setup-output.json"),
        "$TOLERANCES" => dir.join(TOLERANCES_FILE),
        "$REGRESSION_INPUT" => dir.join(REGRESSION_INPUT_FILE),
        "$CHECK_OUTPUT" => tempdir.join(CHECK_OUTPUT_FILE),
        _ => value.into(),
    }
}

fn parse_apps(dir: &Path, input: &Path, tempdir: &Path) -> anyhow::Result<Vec<App>> {
    let apps: Vec<_> = read(&dir.join(STDIN))?
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| -> anyhow::Result<App> {
            let args = line
                .split_whitespace()
                .map(|value| OsString::from(resolve(value, input, dir, tempdir)));
            App::try_parse_from(std::iter::once(OsString::from("test-app")).chain(args))
        })
        .collect::<anyhow::Result<_>>()?;
    if apps.is_empty() {
        bail!("{}/stdin.txt has no command", dir.display());
    }
    Ok(apps)
}

fn materialize_input(dir: &Path, tempdir: &Path) -> anyhow::Result<PathBuf> {
    let input = dir.join(INPUT_FILE);
    if !input.is_file() {
        return Ok(input);
    }

    let source = read(&input)?;
    if !source.contains(WORKSPACE) && !source.contains("$TEMPDIR") {
        return Ok(input);
    }

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("benchmark-runner manifest directory has no workspace parent")?
        .display()
        .to_string()
        .replace('\\', "/");
    let tempdir_text = tempdir.display().to_string().replace('\\', "/");
    let rendered = source
        .replace("$TEMPDIR", &tempdir_text)
        .replace(WORKSPACE, &workspace);
    let materialized = tempdir.join(INPUT_FILE);
    std::fs::write(&materialized, rendered)
        .with_context(|| format!("failed to write {}", materialized.display()))?;
    Ok(materialized)
}

/// Read the optional feature list attached to a fixture.
pub fn features(dir: &Path) -> anyhow::Result<Vec<String>> {
    match std::fs::read_to_string(dir.join(FEATURES_FILE)) {
        Ok(contents) => Ok(contents
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(error).context("failed to read fixture features"),
    }
}

/// Run a cached CLI fixture with the supplied production registry.
pub fn run(dir: &Path, registry: &Registry) -> anyhow::Result<()> {
    let tempdir = tempfile::tempdir().context("failed to create fixture directory")?;
    let input = materialize_input(dir, tempdir.path())?;
    let apps = parse_apps(dir, &input, tempdir.path())?;
    let mut buffer = output::Memory::new();

    for (index, app) in apps.iter().enumerate() {
        let last = index + 1 == apps.len();
        let mut sink = output::Sink::new();
        let mut target: &mut dyn crate::Output = if last { &mut buffer } else { &mut sink };
        if let Err(error) = app.run(registry, target) {
            if last {
                write!(target, "{error:?}")?;
            } else {
                return Err(error).context("fixture setup command failed");
            }
        }
    }

    let actual = ux::scrub_path(
        ux::normalize(ux::strip_backtrace(
            String::from_utf8(buffer.into_inner()).context("fixture output is not UTF-8")?,
        )),
        tempdir.path(),
        "$TEMPDIR",
    );
    let expected_path = dir.join(STDOUT);
    let overwrite = overwrite()?;
    if overwrite {
        std::fs::write(&expected_path, &actual)
            .with_context(|| format!("failed to write {}", expected_path.display()))?;
    } else if actual != read(&expected_path)? {
        bail!("fixture output differs from {}", expected_path.display());
    }

    for filename in GENERATED_OUTPUTS {
        let generated = tempdir.path().join(filename);
        let expected = dir.join(filename);
        match (overwrite, generated.is_file(), expected.is_file()) {
            (true, true, _) => {
                std::fs::copy(&generated, &expected).with_context(|| {
                    format!("failed to copy fixture output {}", expected.display())
                })?;
            }
            (true, false, true) => std::fs::remove_file(&expected).with_context(|| {
                format!(
                    "failed to remove stale fixture output {}",
                    expected.display()
                )
            })?,
            (false, true, true) if read(&generated)? != read(&expected)? => {
                bail!("generated {filename} differs from fixture");
            }
            (false, true, false) => bail!("{filename} was generated unexpectedly"),
            (false, false, true) => bail!("{filename} was not generated"),
            _ => {}
        }
    }
    Ok(())
}
