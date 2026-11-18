/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fs::File, io::Write, path::PathBuf};

pub fn create_test_yaml(s: &str) -> (tempfile::TempDir, PathBuf) {
    let tmpdir = tempfile::tempdir().unwrap();
    let path = tmpdir.path().join("test.yaml");

    {
        let mut f = File::create(&path).unwrap();
        f.write_all(s.as_bytes()).unwrap();
    }

    (tmpdir, path)
}
