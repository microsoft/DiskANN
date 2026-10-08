/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use proc_macro2::TokenStream;
use quote::{quote};
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
    // Error as early as possible.
    if matches!(input.data, Data::Union(_)) {
        return Err(syn::Error::new_spanned(
            input,
            "Reflect cannot be derived for unions",
        ));
    }

    let doc = format_docstrings(&input.attrs);
    let mut generics = input.generics.clone();
    add_generic_bounds(&mut generics);
    let container = attributes::Container::parse(&input.attrs)?;

    let format_type_name = generate_type_name_body(&input, container.type_name())?;

    let common = DeriveCommon {
        doc,
        generics,
        format_type_name,
        container,
    };

    match &input.data {
        Data::Struct(s) => process_struct(&input, s, common),
        Data::Enum(e) => process_enum(&input, e, common),
        Data::Union(_) => unreachable!("this has already been checked"),
    }
}

/// Common pre-processed items.
struct DeriveCommon {
    /// Documentation for the top-level derive input.
    doc: TokenStream,
    /// The generics with a `where T: Reflect` added for all generic parameters.
    generics: syn::Generics,
    /// The implementation of `format_type_name`.
    format_type_name: TokenStream,
    /// Serde container-level attributes.
    container: attributes::Container,
}

/// Add a bound `T: Reflect` for each type parameter in the generic list.
///
/// This is needed to correctly render type-names.
///
/// For example
/// ```
/// struct Foo<T> {
///     bar: Vec<T>,
/// }
/// ```
/// should have a type name like `Foo<u32>`. We get the value in the brackets from the
/// `Reflect` bound as well, so we need to add the following bounds:
/// ```text
/// impl<T> Reflect for Foo<T>
/// where
///     T: Reflect,
/// {
///     ...
/// }
/// ```
fn add_generic_bounds(generics: &mut syn::Generics) {
    // Get the raw type parameters.
    let type_params: Vec<_> = generics.type_params().map(|p| p.ident.clone()).collect();

    let predicates = &mut generics.make_where_clause().predicates;
    let path = crate_name();

    for p in type_params {
        predicates.push(parse_quote!(#p: #path::Reflect));
    }
}

/// Add a bound `T: Reflect`.
fn add_type_bound(generics: &mut syn::Generics, path: &syn::Path, ty: &syn::Type) {
    generics.make_where_clause()
        .predicates
        .push(parse_quote!(#ty: #path::Reflect))
}

/// Generate the expression for a type name.
///
/// The main complexity comes from dealing with generics.
///
/// When there are generics (say `Foo<T, const N: usize>`), we want the following implement
/// to look something like this when `T == u32` and `N == 10`:
/// ```
/// fn type_name(f: &mut dyn std::fmt::Write) -> std::fmt::Result {
///     f.write_str("Foo");
///     f.write_str("<");
///     // This would come from the `Reflection::type_name` instead.
///     type_name_u32(f);
///     f.write_str(">")
/// }
///
/// fn type_name_u32(f: &mut dyn std::fmt::Write) -> std::fmt::Result {
///     f.write_str("u32")
/// }
/// ```
/// When there are no generics - we can print the type name directly.
///
/// # Compatibility with type-name attributes
///
/// There are two type-name attributes supported:
///
/// * `reflect(prefix = "...")`: Apply the prefix to the final type-name.
/// * `reflect(type_name = "...")`: Use the given type name literal instead.
///
///   To prevent mayhem, `type_name` may only be used on non-genric types.
fn generate_type_name_body(
    input: &DeriveInput,
    type_name: attributes::TypeName,
) -> syn::Result<TokenStream> {
    let path = crate_name();
    let name = input.ident.to_string();
    let arguments: Vec<_> = input
        .generics
        .params
        .iter()
        .filter_map(|p| match p {
            syn::GenericParam::Type(p) => {
                let ident = &p.ident;
                Some(quote! {
                    ::std::write!(
                        f,
                        "{}",
                        #path::Reflection::new::<#ident>().type_name(),
                    )?;
                })
            }
            syn::GenericParam::Const(p) => {
                let ident = &p.ident;
                Some(quote! {
                    ::std::write!(f, "{}", #ident)?;
                })
            }
            syn::GenericParam::Lifetime(_) => None,
        })
        .collect();

    // Check that the type-name attributes are compatible with the struct.
    let prefix: Option<TokenStream> = match type_name {
        attributes::TypeName::Rename(rename) => {
            if arguments.is_empty() {
                let ts = quote! {
                    f.write_str(#rename)
                };
                return Ok(ts);
            } else {
                return Err(syn::Error::new_spanned(
                    rename,
                    "The `type_name` attribute cannot be applied to types with generics",
                ));
            }
        }
        attributes::TypeName::Prefix(prefix) => {
            let ts = quote! {
                f.write_str(#prefix)?;
            };
            Some(ts)
        }
        attributes::TypeName::None => None,
    };

    // If there are no generics, we can dump the typename directly.
    if arguments.is_empty() {
        let ts = quote! {
            #prefix
            f.write_str(#name)
        };
        Ok(ts)
    } else {
        let writes = arguments.iter().enumerate().map(|(index, argument)| {
            if index == 0 {
                quote! {
                    #argument
                }
            } else {
                quote! {
                    f.write_str(", ")?;
                    #argument
                }
            }
        });

        let ts = quote! {
            #prefix
            f.write_str(#name)?;
            f.write_str("<")?;
            #(#writes)*
            f.write_str(">")
        };
        Ok(ts)
    }
}

/// Generate the `Reflect` implementation.
fn process_struct(
    input: &DeriveInput,
    s: &syn::DataStruct,
    common: DeriveCommon,
) -> syn::Result<TokenStream> {
    let type_name = &input.ident;

    let DeriveCommon {
        doc,
        mut generics,
        format_type_name,
        container,
    } = common;

    let s = repr::Struct::parse(s, container.try_as_struct()?, doc)?;

    let path = crate_name();
    s.for_each_type(|ty| add_type_bound(&mut generics, &path, ty));

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let emit = s.emit(&path);

    let ts = quote! {
        impl #impl_generics #path::Reflect for #type_name #ty_generics #where_clause {
            fn ty() -> #path::Type {
                #emit
            }

            fn format_type_name(f: &mut dyn ::std::fmt::Write) -> ::std::fmt::Result {
                #format_type_name
            }
        }
    };

    Ok(ts)
}

//-------//
// Enums //
//-------//

fn process_enum(
    input: &DeriveInput,
    e: &syn::DataEnum,
    common: DeriveCommon,
) -> syn::Result<TokenStream> {
    let type_name = &input.ident;

    let DeriveCommon {
        doc,
        mut generics,
        format_type_name,
        container,
    } = common;

    let e = repr::Enum::parse(e, container.as_enum(), doc)?;

    let path = crate_name();
    e.for_each_type(|ty| add_type_bound(&mut generics, &path, ty));

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let emit = e.emit(&path);
    let ts = quote! {
        impl #impl_generics #path::Reflect for #type_name #ty_generics #where_clause {
            fn ty() -> #path::Type {
                #emit
            }

            fn format_type_name(f: &mut dyn ::std::fmt::Write) -> ::std::fmt::Result {
                #format_type_name
            }
        }
    };

    Ok(ts)
}

//-------------//
// Doc Strings //
//-------------//

fn format_docstrings(attributes: &[syn::Attribute]) -> TokenStream {
    match extract_docs(attributes) {
        None => quote! { ::std::option::Option::None },
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

//-----//
// raw //
//-----//

fn strip_raw_prefix(s: &str) -> &str {
    s.strip_prefix("r#").unwrap_or(s)
}
