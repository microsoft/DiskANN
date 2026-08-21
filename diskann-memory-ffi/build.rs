#![allow(clippy::expect_used)]

use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let output = crate_dir.join("include").join("diskann_memory_ffi.h");
    fs::create_dir_all(output.parent().expect("header parent")).expect("create include directory");

    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("load cbindgen configuration");
    let bindings = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("generate C++ bindings");
    bindings.write_to_file(&output);

    let generated = fs::read_to_string(&output).expect("read generated bindings");
    let compatible = generated
        .replacen("uint32_t dist_metric;", "Metric dist_metric;", 1)
        .replace(
            "enum class DiskANNError : int32_t",
            "enum class DiskANNError",
        )
        .replace("enum class Metric : uint32_t", "enum class Metric")
        .replace("enum class TagType : uint32_t", "enum class TagType")
        .replace(
            "enum class DeleteMethod : uint32_t",
            "enum class DeleteMethod",
        );
    fs::write(output, compatible).expect("write compatibility bindings");
}
