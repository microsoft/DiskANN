/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, parse_macro_input, spanned::Spanned};

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
/// use diskann_benchmark_runner::reflect::Reflect;
///
/// /// A test aggregate.
/// #[derive(Reflect)]
/// struct MyInput {
///     /// The number of threads.
///     threads: usize,
/// }
/// ```
#[proc_macro_derive(Reflect, attributes(reflect))]
pub fn derive_reflect(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let output = match &input.data {
        Data::Struct(s) => process_struct(&input, s),
        _ => todo!("need to figure this out"),
    };
    output.into()
}

fn process_struct(input: &DeriveInput, s: &syn::DataStruct) -> proc_macro2::TokenStream {
    let type_name = &input.ident;
    let type_name_str = type_name.to_string();
    let path = crate_name();
    let doc = format_docstrings(&input.attrs);

    match &s.fields {
        Fields::Named(named) => {
            let fields = named.named.iter().map(|f| {
                let ty = &f.ty;
                let ident = f.ident.as_ref().unwrap().to_string();
                let doc = format_docstrings(&f.attrs);

                quote_spanned! { ty.span()=> #path::Field::new::<#ty>(#ident, #doc) }
            });

            quote! {
                impl #path::Reflect for #type_name {
                    fn reflect() -> #path::Type {
                        #path::Type::aggregate(
                            #type_name_str,
                            [#(#fields),*],
                            #doc,
                        )
                    }
                }
            }
        }
        _ => todo!("more todos"),
    }
}

fn format_docstrings(attributes: &[syn::Attribute]) -> proc_macro2::TokenStream {
    match extract_docs(attributes) {
        None => quote!{ ::std::option::Option::None },
        Some(docs) => quote! { ::std::option::Option::Some(#docs.into()) },
    }
}

fn extract_docs(attributes: &[syn::Attribute]) -> Option<String> {
    let docstrings = attributes
        .iter()
        .filter_map(|a| {
            if a.path().is_ident("doc")
                && let syn::Meta::NameValue(name) = &a.meta
                && let syn::Expr::Lit(literal) = &name.value
                && let syn::Lit::Str(s) = &literal.lit
            {
                let value = s.value();
                let processed = match value.strip_prefix(" ") {
                    Some(stripped) => stripped.to_owned(),
                    None => value,
                };
                Some(processed)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    if docstrings.is_empty() {
        None
    } else {
        Some(docstrings.join("\n"))
    }
}
