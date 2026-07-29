/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Capture the source revision so benchmark results can be traced back to the code that
//! produced them.
//!
//! This runs at compile time rather than at run time on purpose: a benchmark binary is
//! frequently built once and then run for hours, possibly long after the working tree has
//! moved on. Baking the revision into the binary records the commit that was *actually
//! compiled*, which is the question a result file needs to answer. It also keeps the
//! benchmark itself free of process spawns.
//!
//! Everything here is best-effort. Building outside a git checkout (a vendored crate, a
//! release tarball, a machine without `git` installed) is a supported configuration and
//! simply leaves the revision unrecorded rather than failing the build.

use std::{path::Path, process::Command};

fn main() {
    // Cargo's default is to rerun this script whenever any file in the package changes.
    // Emitting any `rerun-if-changed` replaces that default, so `src` is re-declared
    // below to keep it.
    watch_git_head(Path::new("../.git"));
    println!("cargo:rerun-if-changed=src");

    if let Some(sha) = git(&["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=DISKANN_GIT_SHA={sha}");
    }

    // `--untracked-files=no` deliberately ignores untracked files: a stray scratch file in
    // the tree does not change the code that was compiled, and flagging it would make the
    // dirty bit noisy enough that developers would learn to ignore it.
    if let Some(status) = git(&["status", "--porcelain", "--untracked-files=no"]) {
        println!("cargo:rustc-env=DISKANN_GIT_DIRTY={}", !status.is_empty());
    }
}

/// Run `git` with `args` and return trimmed stdout, or `None` if git is unavailable or the
/// command failed.
fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Rerun this script when the checked-out commit changes.
///
/// `HEAD` only changes when the *branch* changes; committing on the current branch moves
/// the ref that `HEAD` points at, so that file has to be watched as well. A packed ref has
/// no file to watch, and a nonexistent path in `rerun-if-changed` forces a rebuild on every
/// invocation, so both are guarded by an existence check.
fn watch_git_head(git_dir: &Path) {
    let head = git_dir.join("HEAD");
    if !head.is_file() {
        return;
    }
    println!("cargo:rerun-if-changed={}", head.display());

    let Ok(contents) = std::fs::read_to_string(&head) else {
        return;
    };
    if let Some(reference) = contents.trim().strip_prefix("ref: ") {
        let path = git_dir.join(reference);
        if path.is_file() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
