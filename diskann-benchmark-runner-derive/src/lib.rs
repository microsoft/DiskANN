/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, parse_macro_input, parse_quote};

mod attributes;
mod repr;

fn crate_name() -> syn::Path {
    syn::parse_quote!(::diskann_benchmark_runner::reflect)
}

/// Derive macro for the `Reflect` trait.
///
/// Supports named structs, tuple structs, and enums. Doc comments on the item
/// and its fields/variants are captured as reflection metadata.
///
/// # Example
///
/// ```ignore
/// use diskann_benchmark_runner::Reflect;
///
/// /// A test aggregate.
/// #[derive(Reflect)]
/// struct MyInput {
///     /// The number of threads.
///     threads: usize,
/// }
/// ```
///
/// # Serde Compatibility
///
/// This is meant to be used in conjunction with `serde` for documenting benchmark inputs
/// and as such, it respects several common `serde` attributes:
///
/// * rename_all = "lowercase" | "snake_case" | "kebab-case"
/// * rename = "..."
/// * tag = "..."
/// * tag = "...", content = "..."
#[proc_macro_derive(Reflect, attributes(reflect, serde))]
pub fn derive_reflect(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let input = repr::Input::parse(input)?;
    Ok(input.emit(&crate_name()))
}
