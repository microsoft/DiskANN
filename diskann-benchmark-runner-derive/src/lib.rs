/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, parse_macro_input, parse_quote, spanned::Spanned};

mod attributes;

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

/// Add a bound `T: Reflect` for each type in the field.
///
/// For example, if a struct definition looks like this:
/// ```
/// struct Foo {
///     bar: usize,
/// }
/// ```
/// This will add bounds like this
/// ```text
/// impl Reflect for Foo
/// where
///     usize: Reflect
/// {
///    ...
/// }
/// ```
fn add_field_bounds<'a, I>(generics: &mut syn::Generics, fields: I)
where
    I: IntoIterator<Item = &'a syn::Field>,
{
    let path = crate_name();
    for field in fields {
        let ty = &field.ty;
        generics
            .make_where_clause()
            .predicates
            .push(parse_quote!(#ty: #path::Reflect));
    }
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

fn build_fields(
    fields: &syn::Fields,
    generics: &mut syn::Generics,
    rename_all: attributes::RenameAll,
    enum_repr: Option<&attributes::EnumRepr>,
) -> syn::Result<TokenStream> {
    let path = crate_name();

    match fields {
        Fields::Named(fields) => {
            add_field_bounds(generics, &fields.named);
            let list = named_fields(&fields.named, rename_all, enum_repr)?;
            Ok(quote!(#path::tree::Fields::Named(vec![#(#list),*])))
        }
        Fields::Unnamed(fields) => {
            add_field_bounds(generics, &fields.unnamed);
            let list = unnamed_fields(&fields.unnamed, enum_repr)?;

            // Unnamed fields of length 1 become new-types instead.
            let ts = if list.len() == 1 {
                let new_type = &list[0];
                quote!(#path::tree::Fields::NewType(#new_type))
            } else {
                quote!(#path::tree::Fields::Unnamed(vec![#(#list),*]))
            };

            Ok(ts)
        }
        Fields::Unit => Ok(quote!(#path::tree::Fields::Unit)),
    }
}

fn named_fields<'a, I>(
    fields: I,
    rename_all: attributes::RenameAll,
    enum_repr: Option<&attributes::EnumRepr>,
) -> syn::Result<Vec<TokenStream>>
where
    I: IntoIterator<Item = &'a syn::Field>,
{
    use attributes::EnumRepr;

    let path = crate_name();

    // Track the names of fields that have been emitted.
    // If a rename causes a collision, we return an error.
    let mut seen = HashSet::<String>::new();

    fields
        .into_iter()
        .map(move |f| {
            let ty = &f.ty;
            let ident = f
                .ident
                .as_ref()
                .expect("named fields should have identifiers");

            let name = syn::LitStr::new(strip_raw_prefix(&ident.to_string()), ident.span());

            let doc = format_docstrings(&f.attrs);
            let attributes::Field { rename_field } = attributes::Field::parse(&f.attrs)?;
            let name = rename_field.apply_to_field(name, rename_all);

            // If we are dealing with an enum - we need to rule out the situation where:
            //
            // 1. There is an internally tagged enum.
            // 2. The tag field conflicts with the name of the struct.
            if let Some(enum_repr) = enum_repr {
                match enum_repr {
                    EnumRepr::External | EnumRepr::Adjacent { .. } => {}
                    EnumRepr::Internal { tag } => {
                        if name.value() == tag.value() {
                            return Err(syn::Error::new_spanned(
                                name,
                                "field conflicts with internally tagged discriminant",
                            ));
                        }
                    }
                }
            }

            // Ensure uniqueness.
            if seen.insert(name.value()) {
                Ok(quote_spanned! {
                    ty.span()=> #path::tree::NamedField::new::<#ty>(#name, #doc)
                })
            } else {
                Err(syn::Error::new_spanned(
                    &name,
                    format!("Field name \"{}\" found more than once", name.value()),
                ))
            }
        })
        .collect()
}

fn unnamed_fields<P>(
    fields: &syn::punctuated::Punctuated<syn::Field, P>,
    enum_repr: Option<&attributes::EnumRepr>,
) -> syn::Result<Vec<TokenStream>>
where
    P: quote::ToTokens,
{
    use attributes::EnumRepr;

    // Rejects non-newtype tuple fields when "internal" tagging is used, since there is no
    // field to use for the tag.
    //
    // Like serde, we support newtypes since we cannot rule out syntactically whether
    // or not the tag can be embedded in the internal value.
    if let Some(enum_repr) = enum_repr {
        match enum_repr {
            EnumRepr::External | EnumRepr::Adjacent { .. } => {}
            EnumRepr::Internal { .. } => {
                if fields.len() != 1 {
                    return Err(syn::Error::new_spanned(
                        fields,
                        "non-newtype tuple structs are not compatible with internal enum tagging",
                    ));
                }
            }
        }
    }

    let path = crate_name();
    let ts: Vec<_> = fields.iter().map(move |f| {
        let ty = &f.ty;
        let doc = format_docstrings(&f.attrs);
        quote_spanned! { ty.span()=> #path::tree::UnnamedField::new::<#ty>(#doc) }
    })
    .collect();

    Ok(ts)
}

/// Generate the `Reflect` implementation.
fn process_struct(
    input: &DeriveInput,
    s: &syn::DataStruct,
    common: DeriveCommon,
) -> syn::Result<TokenStream> {
    let DeriveCommon {
        doc,
        mut generics,
        format_type_name,
        container,
    } = common;

    // Validate that the attributes we parsed are compatible with a `struct` definition.
    let attributes::Struct { rename_all } = container.try_as_struct()?;

    let type_name = &input.ident;
    let path = crate_name();

    let fields = build_fields(&s.fields, &mut generics, rename_all, None)?;

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let ts = quote! {
        impl #impl_generics #path::Reflect for #type_name #ty_generics #where_clause {
            fn ty() -> #path::Type {
                #path::Type::aggregate(
                    #fields,
                    #doc,
                )
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
    let DeriveCommon {
        doc,
        mut generics,
        format_type_name,
        container,
    } = common;

    // Validate that the attributes we parsed are compatible with an `enum` definition.
    let attributes::Enum {
        rename_all,
        enum_repr,
    } = container.as_enum();

    let type_name = &input.ident;
    let path = crate_name();

    let mut seen = HashSet::<String>::new();
    let variants = e
        .variants
        .iter()
        .map(|v| -> syn::Result<TokenStream> {
            let doc = format_docstrings(&v.attrs);
            let name = syn::LitStr::new(strip_raw_prefix(&v.ident.to_string()), v.ident.span());
            let attributes::Variant {
                rename_variant,
                rename_variant_fields,
            } = attributes::Variant::parse(&v.attrs)?;

            let fields = build_fields(
                &v.fields,
                &mut generics,
                rename_variant_fields,
                Some(&enum_repr),
            )?;

            // Rename the variant as needed.
            let name = rename_variant.apply_to_variant(name, rename_all);
            if seen.insert(name.value()) {
                Ok(quote!(#path::tree::Variant::new(#name, #fields, #doc)))
            } else {
                Err(syn::Error::new_spanned(
                    &name,
                    format!("Variant name \"{}\" found more than once", name.value()),
                ))
            }
        })
        .collect::<syn::Result<Vec<TokenStream>>>()?;

    // Build the enum representation.
    let enum_repr = match enum_repr {
        attributes::EnumRepr::External => quote!(#path::tree::EnumRepr::External),
        attributes::EnumRepr::Internal { tag } => {
            quote!(#path::tree::EnumRepr::Internal { tag: #tag })
        }
        attributes::EnumRepr::Adjacent { tag, content } => {
            quote!(#path::tree::EnumRepr::Adjacent { tag: #tag, content: #content })
        }
    };

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let ts = quote! {
        impl #impl_generics #path::Reflect for #type_name #ty_generics #where_clause {
            fn ty() -> #path::Type {
                #path::Type::enum_(
                    #enum_repr,
                    [#(#variants),*],
                    #doc,
                )
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
