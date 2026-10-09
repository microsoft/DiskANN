/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use proc_macro2::TokenStream;
use syn::DeriveInput;

mod attributes;
mod repr;

pub(crate) fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let input = repr::Input::parse(input)?;
    let crate_name = syn::parse_quote!(::diskann_benchmark_runner::reflect);
    let ts = input.emit(&crate_name);
    Ok(ts)
}
