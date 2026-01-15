/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Compile the flat-buffer protos into Rust code if the `flatbuffers` feature is enabled.
//!
//! Requires that the environment variable `QUANTIZATION_FLATBUFFERS` environment variable
//! is set and points to the `flatc` executable.
//!
//! The `flatc` executable can be downloaded from the official
//! [releases](https://github.com/google/flatbuffers/releases) page.

// Lints: This happens at build time instead of run time.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

fn main() {
    #[cfg(feature = "flatbuffers-build")]
    compile::run();
}

#[cfg(feature = "flatbuffers-build")]
mod compile {
    use std::{env, ffi::OsStr, path::Path, process::Command};

    const FLATC: &str = "FLATC_EXE";
    const SCHEMAS: &str = "schemas";

    pub(super) fn run() {
        let flatc = env::var_os(FLATC).unwrap_or_else(|| {
            panic!(
                "The environmant variable \"{}\" must be set and point to the \"flatc\" executable.",
                FLATC
            )
        });
        let out_dir = env::var_os("OUT_DIR").expect("This should be set by Cargo");
        let input_files = ["transforms_v001.fbs", "spherical_v001.fbs"];

        // Let `cargo` know that we want to rerun under the following conditions:
        //
        // * The `FLATC` environment variable changes.
        // * `build.rs` changes.
        // * Any of the schema files is updated.
        println!("cargo::rerun-if-env-changed={}", FLATC);
        println!("cargo::rerun-if-changed=build.rs");
        for file in input_files {
            println!("cargo::rerun-if-changed={}/{}", SCHEMAS, file);
        }

        // Generate the Rust code for the `.fbs` files.
        let qualified_files = input_files.map(|file| format!("{}/{}", SCHEMAS, file));
        compile(&flatc, &out_dir, &qualified_files);

        // Copy the generated code into the source tree.
        //
        // We first remove the current contents of the generated `flatbuffers` directory
        // to ensure we don't leave cruft behind.
        let output_dirs = ["transforms", "spherical"];

        let generated_folder = "./src/flatbuffers";

        if std::fs::remove_dir_all(generated_folder).is_err() {
            eprintln!("directory did not exist - ignoring");
        }
        std::fs::create_dir(generated_folder).unwrap();

        // This relies on nothing changing too strangely during the copy process.
        //
        // While not ideal, we control the entire build process, so can keep rerunning if
        // needed until the right thing happens.
        //
        // In any case, the lock used by Cargo should protect against most possible
        // interactions.
        for dir in output_dirs {
            // Create the directory in the generated folder.
            std::fs::create_dir(Path::new(generated_folder).join(dir)).unwrap();

            for src in std::fs::read_dir(Path::new(&out_dir).join(dir)).unwrap() {
                let src = src.unwrap();
                println!("processing {:?}", src);
                let file_name = src.file_name();
                let dst = Path::new(generated_folder).join(dir).join(file_name);
                println!("copying to {:?}", dst);
                std::fs::copy(src.path(), dst).unwrap();
            }
        }
    }

    fn compile(flatc: &OsStr, out_dir: &OsStr, files: &[String]) {
        let mut cmd = Command::new(flatc);
        cmd.args(["--rust", "--rust-module-root-file", "-I", SCHEMAS, "-o"])
            .arg(out_dir)
            .args(files);

        eprintln!("compilation command = {:?}", cmd);
        let output = cmd.output().expect("schema compilation failed");

        eprintln!(
            "compilation stdout\n{}",
            std::str::from_utf8(&output.stdout).unwrap()
        );
        eprintln!(
            "compilation stderr\n{}",
            std::str::from_utf8(&output.stderr).unwrap()
        );
    }
}
