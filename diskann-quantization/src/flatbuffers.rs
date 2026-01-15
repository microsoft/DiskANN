/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Import the `flatbuffers` files generated during the `flatbuffers-build` process.
//!
//! These files get copied from a temporary build-time directory like
//! `target/debug/build/quantization-[hash]/out` and copied into the in-source `flatbuffers`
//! directory.
//!
//! That directory gets completely rebuilt each time, hence the need to maintain a separate
//! `flatbuffers.rs` file with a custom import tree.

macro_rules! import_schema {
    ($path:literal) => {
        include!(concat!("flatbuffers/", $path));
    };
    ($module:ident, $path:literal) => {
        // The generated files don't compile cleanly - so we need to suppress some lints.
        #[allow(
            dead_code,
            unused_imports,
            clippy::extra_unused_lifetimes,
            clippy::needless_lifetimes,
            clippy::unwrap_used
        )]
        mod $module {
            import_schema!($path);
        }

        pub(crate) use $module::*;
    };
}

pub(crate) mod transforms {
    // Union
    import_schema!(transform_generated, "transforms/transform_generated.rs");

    // Enum discriminant.
    import_schema!(
        transform_kind_generated,
        "transforms/transform_kind_generated.rs"
    );

    // Individual transxforms.
    import_schema!(
        null_transform_generated,
        "transforms/null_transform_generated.rs"
    );
    import_schema!(
        padding_hadamard_generated,
        "transforms/padding_hadamard_generated.rs"
    );
    import_schema!(
        double_hadamard_generated,
        "transforms/double_hadamard_generated.rs"
    );
    import_schema!(
        random_rotation_generated,
        "transforms/random_rotation_generated.rs"
    );
}

pub(crate) mod spherical {
    use super::*;
    import_schema!(
        supported_metric_generated,
        "spherical/supported_metric_generated.rs"
    );

    import_schema!(
        spherical_quantizer_generated,
        "spherical/spherical_quantizer_generated.rs"
    );

    import_schema!(quantizer_generated, "spherical/quantizer_generated.rs");
}

/// Create a `FlatBufferBuilder` and pass it to the closure.
///
/// After the closure runs, finish the serialization with the returned offset and pass
/// the finished bytes as a vector.
#[cfg(test)]
pub(crate) fn to_flatbuffer<'a, T, F>(packer: F) -> Vec<u8>
where
    F: FnOnce(&mut flatbuffers::FlatBufferBuilder<'a>) -> flatbuffers::WIPOffset<T>,
    T: 'a,
{
    let mut buf = flatbuffers::FlatBufferBuilder::new();
    let offset = packer(&mut buf);
    buf.finish(offset, None);
    buf.finished_data().to_vec()
}
