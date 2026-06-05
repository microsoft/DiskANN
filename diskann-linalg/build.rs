/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Build script: links OpenBLAS when the `openblas` feature is enabled.
//!
//! OpenBLAS is preferred over Intel MKL because MKL's CPUID-based dispatcher
//! historically selected slower codepaths on non-Intel CPUs (the "anti-AMD"
//! issue documented in many places). OpenBLAS picks the best kernel per
//! CPU at process start without vendor checks.
//!
//! On Ubuntu 24.04 `apt install libopenblas-dev` puts the library in
//! `/usr/lib/x86_64-linux-gnu/openblas-*/`. We let pkg-config find it; if
//! that fails (e.g. building in a stripped container), the bare
//! `-lopenblas` resolves against the loader cache for the standard
//! `/lib/x86_64-linux-gnu/libopenblas.so.0`.

fn main() {
    if std::env::var_os("CARGO_FEATURE_OPENBLAS").is_some() {
        // pkg-config is best-effort. If it can't find openblas (no .pc file),
        // we still emit a bare link directive and let the system linker resolve.
        let pkg_ok = std::process::Command::new("pkg-config")
            .args(["--libs", "openblas"])
            .output()
            .ok()
            .and_then(|o| if o.status.success() { Some(String::from_utf8_lossy(&o.stdout).into_owned()) } else { None });

        if let Some(libs) = pkg_ok {
            // pkg-config returns e.g. "-L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas".
            for token in libs.split_whitespace() {
                if let Some(rest) = token.strip_prefix("-L") {
                    println!("cargo:rustc-link-search=native={}", rest);
                } else if let Some(rest) = token.strip_prefix("-l") {
                    println!("cargo:rustc-link-lib={}", rest);
                }
            }
        } else {
            // Fallback: standard Ubuntu layout.
            println!("cargo:rustc-link-lib=openblas");
        }
        println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
