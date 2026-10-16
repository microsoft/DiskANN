/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! This macro is made annoying complicated by the need to be compatible with `serde` and
//! more-or-less consistent with its corner cases. Particularly tricky or non-obvious (to
//! the author at time of writing) aspects of macro writing are recorded here for posterity.
//!
//! ## Generic Bounds
//!
//! If we have a struct like
//! ```
//! struct Container<T>(T);
//!
//! struct Foo<'a, T, U> {
//!     baz: Container<T>,
//!     bar: String,
//!     buz: U,
//! }
//! ```
//! And we want to derive `Reflect`, we do so by recursing through all fields. For fields
//! like `buz`, this will fail unless we add a bound `U: Reflect` to `Foo`'s implementation
//! of `Reflect`. In addition, we don't know that types like `Container<T>` implement
//! `Reflect`. A simple bound like `T: Reflect` doesn't work because `Container`'s
//! implementation may not follow from `T`. So we need to add the following
//! ```text
//! impl<T, U> Reflect for Foo<'_, T, U>
//! where
//!    T: Reflect,
//!    U: Reflect,
//!    Containter<T>: Reflect,
//!    String: Reflect,
//! ```
//! We can do this by walking the generics for `Foo` as well as all field types and adding
//! bounds for all discovered items. Note that this can result in duplicates, but that's
//! fine.
//!
//! The struct [`Generics`] is responsible for this aggregation and for creating the expanded
//! where-clause.
//!
//! ## Renaming
//!
//! The attributes `rename_all` and `rename` are supported. `rename_all` can go on:
//!
//! * Structs: All fields are renamed according to the rule. Tuple-like and unit structs
//!   do not support this attribute and should return an error.
//!
//!   [`Fields::parse`] enforces this.
//!
//! * Enums: All variants are renamed according to the rule.
//!
//! * Variants: Struct-like will have their fields renamed. Tuple-like and unit structs
//!   do not support this attribute and should return an error.
//!
//! Attribute `rename` can go on struct fields, variants, and fields of struct-like variants.
//! When present, it supersedes an enclosing `rename_all`.
//!
//! ## New Types
//!
//! Serde treats tuple-like structs with a single member as special new-types. These are not
//! modeled as single-element containers and instead modeled as transparent types.
//!
//! We don't have to do anything special during parsing to handle this. However, during code
//! emission, [`Fields::Unnamed`] needs to emit a `NewType` when there is just a single field.
//!
//! ## Enum Woes
//!
//! Serde supports four different enum tagging mechanism. We support the following three:
//!
//! ### Externally tagged
//!
//! Externally tagged enums are modeled as
//! ```json
//! "discriminant": {
//!     "key0": 0,
//!     "key1": 1,
//! }
//! ```
//! where "discriminant" is the name for the active enum variant. This is relatively easy
//! to support and doesn't require additional legalization checks.
//!
//! External tagging is the default for enum when neither `serde(tag = "")` nor
//! `serde(content = "") attributes are provided.
//!
//! ### Adjacently Tagged
//!
//! Adjacently tagged enums look like
//! ```json
//! {
//!     "tag": "discriminant",
//!     "content": {
//!         "key0": 0,
//!         "key1": 1,
//!     }
//! }
//! ```
//!
//! This gets selected when both the "tag" and "content" serde attributes are provided. The
//! values for these attributes are used as the actual strings in the "tag" and "content"
//! positions in the example JSON.
//!
//! Like externally tagged enums, there are no additional legalization steps.
//!
//! ### Internally Tagged
//!
//! These are evil. These look like
//! ```text
//! {
//!     "tag": "discriminant", <--- Embedded with the content
//!     "key0": 0,
//!     "key1": 1,
//! }
//! ```
//! This gets selected when just the `serde(tag = "")` attribute is applied to an enum, with
//! the actual "tag" string in the sample replaced with the value of said attribute.
//!
//! There are several additional checks that internally tagged enums need beyond the other two:
//!
//! * Internal Tagging can only be used if *all* variants are either new-types, struct-like,
//!   or units. Tuple-like variants don't have any named fields, so a tag doesn't work.
//!
//!   Serde supports new-types but throws a runtime error if the new-type is not struct-like.
//!   We don't throw this run time error - we just assume it's going to work.
//!
//! * We need to check that the "tag" string does not conflict with any field of the
//!   struct-like variant. This needs to happen *after* all renames have been applied.
//!
//! ### Untagged
//!
//! We don't support untagged enums.
//!
//! ## Name Uniqueness
//!
//! We also need to ensure that after renaming, all final field names are unique and all
//! final variant names are unique.
//!
//! ## Type Names
//!
//! Type names are what we use to provide unique names for reflected types. Two attributes
//! are supported:
//!
//! * `reflect(prefix = "")`: Apply the prefix to the type-name.
//! * `reflect(type_name = "")`: Completely rewrite the type name.
//!
//! Since type registries assume that type names are unique, we don't allow `type_name` to
//! be applied to types with generics. Otherwise, each instantiation of the generic would
//! have the same name, limiting usefulness.
//!

use std::collections::HashSet;

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{parse_quote, spanned::Spanned};

use super::attributes;

pub(crate) struct Input<'a> {
    type_name: &'a syn::Ident,
    generics: Generics<'a>,
    format_type_name: TypeName<'a>,
    container: Container<'a>,
}

impl<'a> Input<'a> {
    pub(crate) fn parse(input: &'a syn::DeriveInput) -> syn::Result<Self> {
        let attrs = attributes::Container::parse(&input.attrs)?;

        let type_name = &input.ident;
        let format_type_name = TypeName::parse(type_name, &input.generics, attrs.type_name())?;

        // Parse the type tree for the container.
        let container = Container::parse(&input.data, attrs, input.span())?;
        let generics = Generics::parse(&input.generics, &container);

        Ok(Self {
            type_name,
            generics,
            format_type_name,
            container,
        })
    }

    pub(crate) fn emit(&self, path: &syn::Path) -> TokenStream {
        let type_name = self.type_name;
        let generics = self.generics.emit(path);
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
        let format_type_name = self.format_type_name.emit(path);
        let container = self.container.emit(path);

        quote! {
            impl #impl_generics #path::Reflect for #type_name #ty_generics #where_clause {
                fn ty() -> #path::Type {
                    #container
                }

                fn format_type_name(f: &mut dyn ::std::fmt::Write) -> ::std::fmt::Result {
                    #format_type_name
                }
            }
        }
    }
}

/// Emit a `where` clause predicate for all generics and field types.
struct Generics<'a> {
    generics: &'a syn::Generics,
    /// The type parameters of the original object.
    params: Vec<&'a syn::Ident>,
    /// All field types.
    types: Vec<&'a syn::Type>,
}

impl<'a> Generics<'a> {
    fn parse(generics: &'a syn::Generics, container: &Container<'a>) -> Self {
        let params = generics.type_params().map(|p| &p.ident).collect();
        let mut types = Vec::new();
        container.for_each_type(|ty| types.push(ty));
        Self {
            generics,
            params,
            types,
        }
    }

    fn emit(&self, path: &syn::Path) -> syn::Generics {
        let mut generics = self.generics.clone();

        let predicates = &mut generics.make_where_clause().predicates;
        for ident in self.params.iter() {
            predicates.push(parse_quote!(#ident: #path::Reflect));
        }

        for ty in self.types.iter() {
            predicates.push(parse_quote!(#ty: #path::Reflect));
        }

        generics
    }
}

enum Container<'a> {
    Struct(Struct<'a>),
    Enum(Enum<'a>),
}

impl<'a> Container<'a> {
    fn parse(data: &'a syn::Data, attrs: attributes::Container, span: Span) -> syn::Result<Self> {
        match data {
            syn::Data::Struct(s) => Ok(Self::Struct(Struct::parse(s, attrs.try_as_struct()?)?)),
            syn::Data::Enum(e) => Ok(Self::Enum(Enum::parse(e, attrs.into_enum())?)),
            syn::Data::Union(_) => Err(syn::Error::new(
                span,
                "Reflect cannot be derived for unions",
            )),
        }
    }

    pub(crate) fn for_each_type<F>(&self, f: F)
    where
        F: FnMut(&'a syn::Type),
    {
        match self {
            Self::Struct(s) => s.for_each_type(f),
            Self::Enum(e) => e.for_each_type(f),
        }
    }

    fn emit(&self, path: &syn::Path) -> TokenStream {
        match self {
            Self::Struct(s) => s.emit(path),
            Self::Enum(e) => e.emit(path),
        }
    }
}

enum TypeName<'a> {
    Override(syn::LitStr),
    Generate {
        prefix: Option<syn::LitStr>,
        raw_name: &'a syn::Ident,
        generics: Vec<Generic<'a>>,
    },
}

impl<'a> TypeName<'a> {
    fn parse(
        raw_name: &'a syn::Ident,
        generics: &'a syn::Generics,
        type_name: attributes::TypeName,
    ) -> syn::Result<Self> {
        let generics: Vec<_> = generics
            .params
            .iter()
            .filter_map(|p| match p {
                syn::GenericParam::Type(p) => Some(Generic::Type(&p.ident)),
                syn::GenericParam::Const(p) => Some(Generic::Const(&p.ident)),
                syn::GenericParam::Lifetime(_) => None,
            })
            .collect();

        let prefix = match type_name {
            attributes::TypeName::Rename(rename) => {
                // Reject `reflect(type_name = "...")` on types with generic parameters.
                if !generics.is_empty() {
                    return Err(syn::Error::new_spanned(
                        rename,
                        "The `type_name` attribute cannot be applied to types with generics",
                    ));
                } else {
                    return Ok(Self::Override(rename));
                };
            }
            attributes::TypeName::Prefix(prefix) => Some(prefix),
            attributes::TypeName::None => None,
        };

        Ok(Self::Generate {
            prefix,
            raw_name,
            generics,
        })
    }

    /// When generics are absent, do the easy thing and apply the prefix with the raw type
    /// name from the [`syn::DeriveInput`].
    ///
    /// When generics are present (for example)
    /// ```
    /// struct Foo<T, U, const N: usize> {
    ///    // ...
    /// }
    /// ```
    /// use the type-names for each generic to get the final type. The generated code roughly
    /// ```text
    /// f.write_str("Foo")?;
    /// f.write_str("<")?;
    /// <T as Reflect>::type_name(f)?; // The type name for the generic
    /// f.write_str(", ")?;
    /// <U as Reflect>::type_name(f)?; // The type name for the generic
    /// f.write_str(", ")?;
    /// f.write!("{}", N)?;
    /// f.write_str(">")?;
    /// ```
    fn emit(&self, path: &syn::Path) -> TokenStream {
        match self {
            Self::Override(type_name) => quote!(f.write_str(#type_name)),
            Self::Generate {
                prefix,
                raw_name,
                generics,
            } => {
                let raw_name = raw_name.to_string();
                let prefix = prefix.as_ref().map(|p| quote!(f.write_str(#p)?;));
                if generics.is_empty() {
                    quote! {
                        #prefix
                        f.write_str(#raw_name)?;
                        Ok(())
                    }
                } else {
                    let generics = generics.iter().enumerate().map(|(i, g)| {
                        let separator = if i == 0 {
                            None
                        } else {
                            Some(quote!(f.write_str(", ")?;))
                        };
                        let ts = g.emit(path);
                        quote! {
                            #separator
                            #ts
                        }
                    });

                    quote! {
                        #prefix
                        f.write_str(#raw_name)?;
                        f.write_str("<")?;
                        #(#generics)*
                        f.write_str(">")?;
                        Ok(())
                    }
                }
            }
        }
    }
}

enum Generic<'a> {
    Type(&'a syn::Ident),
    Const(&'a syn::Ident),
}

impl Generic<'_> {
    fn emit(&self, path: &syn::Path) -> TokenStream {
        match self {
            Self::Type(ident) => quote! {
                ::std::write!(f, "{}", #path::Reflection::new::<#ident>().type_name())?;
            },
            Self::Const(ident) => quote!(::std::write!(f, "{}", #ident)?;),
        }
    }
}

////////////
// Struct //
////////////

pub(crate) struct Struct<'a> {
    doc: attributes::Doc,
    fields: Fields<'a>,
}

impl<'a> Struct<'a> {
    pub(crate) fn parse(s: &'a syn::DataStruct, attrs: attributes::Struct) -> syn::Result<Self> {
        let attributes::Struct { rename_all, doc } = attrs;
        let fields = Fields::parse(&s.fields, rename_all)?;
        Ok(Self { doc, fields })
    }

    pub(crate) fn for_each_type<F>(&self, f: F)
    where
        F: FnMut(&'a syn::Type),
    {
        self.fields.for_each_type(f)
    }

    pub(crate) fn emit(&self, path: &syn::Path) -> TokenStream {
        let doc = self.doc.emit();
        let fields = self.fields.emit(path);
        quote! {
            #path::Type::aggregate(
                #fields,
                #doc
            )
        }
    }
}

//////////
// Enum //
//////////

pub(crate) struct Enum<'a> {
    doc: attributes::Doc,
    enum_repr: attributes::EnumRepr,
    variants: Vec<Variant<'a>>,
}

impl<'a> Enum<'a> {
    pub(crate) fn parse(enum_: &'a syn::DataEnum, attr: attributes::Enum) -> syn::Result<Self> {
        let attributes::Enum {
            rename_all,
            enum_repr,
            doc,
        } = attr;

        // Parse all variants.
        let variants = enum_
            .variants
            .iter()
            .map(|v| Variant::parse(v, rename_all))
            .collect::<syn::Result<Vec<_>>>()?;

        // Ensure all variant names are unique.
        let mut seen_variants = HashSet::new();
        for v in variants.iter() {
            let variant_name = v.name.value();
            if !seen_variants.insert(variant_name.clone()) {
                return Err(syn::Error::new_spanned(
                    &v.name,
                    format!("Variant name \"{}\" seen more than once", variant_name),
                ));
            }
        }

        // Ensure that:
        //
        // 1. No field name conflicts with the tag.
        // 2. Non-newtype tuple variants are rejected.
        if let attributes::EnumRepr::Internal { tag } = &enum_repr {
            let tag = tag.value();
            for v in variants.iter() {
                match &v.fields {
                    Fields::Named(named) => {
                        for field in named.iter() {
                            if tag == field.name.value() {
                                return Err(syn::Error::new_spanned(
                                    &field.name,
                                    format!(
                                        "field \"{}\" conflicts with internal discriminant tag",
                                        tag
                                    ),
                                ));
                            }
                        }
                    }
                    Fields::Unnamed(unnamed) => {
                        if unnamed.len() != 1 {
                            return Err(syn::Error::new_spanned(
                                &v.name,
                                "non-newtype tuple type variants are not allowed with internal tagging",
                            ));
                        }
                    }
                    Fields::Unit => {}
                }
            }
        }

        Ok(Self {
            doc,
            enum_repr,
            variants,
        })
    }

    pub(crate) fn for_each_type<F>(&self, mut f: F)
    where
        F: FnMut(&'a syn::Type),
    {
        self.variants.iter().for_each(|v| v.for_each_type(&mut f))
    }

    pub(crate) fn emit(&self, path: &syn::Path) -> TokenStream {
        let enum_repr = match &self.enum_repr {
            attributes::EnumRepr::External => quote!(#path::tree::EnumRepr::External),
            attributes::EnumRepr::Internal { tag } => {
                quote!(#path::tree::EnumRepr::Internal { tag: #tag })
            }
            attributes::EnumRepr::Adjacent { tag, content } => {
                quote!(#path::tree::EnumRepr::Adjacent { tag: #tag, content: #content })
            }
        };

        let variants = self.variants.iter().map(|v| v.emit(path));
        let doc = self.doc.emit();
        quote! {
            #path::Type::enum_(
                #enum_repr,
                [#(#variants),*],
                #doc
            )
        }
    }
}

////////////////////
// Implementation //
////////////////////

pub(crate) struct UnnamedField<'a> {
    doc: attributes::Doc,
    ty: &'a syn::Type,
}

impl<'a> UnnamedField<'a> {
    fn parse(field: &'a syn::Field) -> syn::Result<Self> {
        assert!(
            field.ident.is_none(),
            "expected `syn` to not attach names to `FieldsNamed` entries"
        );

        let attributes::Field { rename_field, doc } = attributes::Field::parse(&field.attrs)?;

        // Rejact renames on unnamed fields.
        if let Some(span) = rename_field.span() {
            return Err(syn::Error::new(
                span,
                "rename attributes are not supported on unnamed fields",
            ));
        }

        Ok(Self { doc, ty: &field.ty })
    }

    fn emit(&self, path: &syn::Path) -> TokenStream {
        let ty = self.ty;
        let doc = self.doc.emit();
        quote_spanned! { ty.span()=> #path::tree::UnnamedField::new::<#ty>(#doc) }
    }
}

pub(crate) struct NamedField<'a> {
    doc: attributes::Doc,
    name: syn::LitStr,
    ty: &'a syn::Type,
}

impl<'a> NamedField<'a> {
    fn parse(field: &'a syn::Field, rename_all: attributes::RenameAll) -> syn::Result<Self> {
        let ident = field
            .ident
            .as_ref()
            .expect("named fields should have identifiers");

        let name = syn::LitStr::new(strip_raw_prefix(&ident.to_string()), ident.span());
        let attributes::Field { rename_field, doc } = attributes::Field::parse(&field.attrs)?;
        let name = rename_field.apply_to_field(name, rename_all);

        Ok(Self {
            doc,
            name,
            ty: &field.ty,
        })
    }

    fn emit(&self, path: &syn::Path) -> TokenStream {
        let ty = self.ty;
        let doc = self.doc.emit();
        let name = &self.name;
        quote_spanned! {
            ty.span()=> #path::tree::NamedField::new::<#ty>(#name, #doc)
        }
    }
}

pub(crate) enum Fields<'a> {
    Named(Vec<NamedField<'a>>),
    Unnamed(Vec<UnnamedField<'a>>),
    Unit,
}

impl<'a> Fields<'a> {
    fn parse(fields: &'a syn::Fields, rename_all: attributes::RenameAll) -> syn::Result<Self> {
        let me = match fields {
            syn::Fields::Named(fields) => {
                let named = fields
                    .named
                    .iter()
                    .map(|f| NamedField::parse(f, rename_all))
                    .collect::<syn::Result<Vec<_>>>()?;

                let mut seen = HashSet::new();
                for f in named.iter() {
                    let field_name = f.name.value();
                    if !seen.insert(field_name.clone()) {
                        return Err(syn::Error::new_spanned(
                            &f.name,
                            format!("field name \"{}\" seen more than once", field_name),
                        ));
                    }
                }

                Self::Named(named)
            }
            syn::Fields::Unnamed(fields) => {
                if let Some(span) = rename_all.span_if_present() {
                    return Err(syn::Error::new(
                        span,
                        "`rename_all` cannot be applied to tuple structs",
                    ));
                }

                Self::Unnamed(
                    fields
                        .unnamed
                        .iter()
                        .map(UnnamedField::parse)
                        .collect::<syn::Result<_>>()?,
                )
            }
            syn::Fields::Unit => {
                if let Some(span) = rename_all.span_if_present() {
                    return Err(syn::Error::new(
                        span,
                        "`rename_all` cannot be applied to unit structs",
                    ));
                }
                Self::Unit
            }
        };
        Ok(me)
    }

    fn for_each_type<F>(&self, mut f: F)
    where
        F: FnMut(&'a syn::Type),
    {
        match self {
            Self::Named(named) => named.iter().for_each(|field| f(field.ty)),
            Self::Unnamed(unnamed) => unnamed.iter().for_each(|field| f(field.ty)),
            Self::Unit => {}
        }
    }

    fn emit(&self, path: &syn::Path) -> TokenStream {
        match self {
            Self::Named(named) => {
                let ts = named.iter().map(|f| f.emit(path));
                quote!(#path::tree::Fields::Named(vec![#(#ts),*]))
            }
            Self::Unnamed(unnamed) => {
                if unnamed.len() == 1 {
                    let ts = unnamed[0].emit(path);
                    quote!(#path::tree::Fields::NewType(#ts))
                } else {
                    let ts = unnamed.iter().map(|f| f.emit(path));
                    quote!(#path::tree::Fields::Unnamed(vec![#(#ts),*]))
                }
            }
            Self::Unit => quote!(#path::tree::Fields::Unit),
        }
    }
}

struct Variant<'a> {
    doc: attributes::Doc,
    name: syn::LitStr,
    fields: Fields<'a>,
}

impl<'a> Variant<'a> {
    fn parse(variant: &'a syn::Variant, rename_all: attributes::RenameAll) -> syn::Result<Self> {
        let attributes::Variant {
            rename_variant,
            rename_variant_fields,
            doc,
        } = attributes::Variant::parse(&variant.attrs)?;

        let fields = Fields::parse(&variant.fields, rename_variant_fields)?;

        let name = syn::LitStr::new(
            strip_raw_prefix(&variant.ident.to_string()),
            variant.ident.span(),
        );
        let name = rename_variant.apply_to_variant(name, rename_all);

        Ok(Self { doc, name, fields })
    }

    fn for_each_type<F>(&self, f: F)
    where
        F: FnMut(&'a syn::Type),
    {
        self.fields.for_each_type(f)
    }

    fn emit(&self, path: &syn::Path) -> TokenStream {
        let name = &self.name;
        let fields = self.fields.emit(path);
        let doc = self.doc.emit();
        quote!(#path::tree::Variant::new(#name, #fields, #doc))
    }
}

//-----//
// raw //
//-----//

fn strip_raw_prefix(s: &str) -> &str {
    s.strip_prefix("r#").unwrap_or(s)
}
