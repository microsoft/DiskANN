/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;

use crate::{attributes, format_docstrings, strip_raw_prefix};

////////////
// Struct //
////////////

pub(crate) struct Struct<'a> {
    doc: TokenStream,
    fields: Fields<'a>,
}

impl<'a> Struct<'a> {
    pub(crate) fn parse(
        s: &'a syn::DataStruct,
        doc: TokenStream,
        attrs: attributes::Struct,
    ) -> syn::Result<Self> {
        let attributes::Struct { rename_all } = attrs;
        let fields = Fields::parse(&s.fields, rename_all)?;
        Ok(Self {
            doc,
            fields,
        })
    }

    pub(crate) fn for_each_type<F, R>(&self, mut f: F)
    where
        F: FnMut(&syn::Type),
    {
        self.fields.for_each_type(f)
    }

    pub(crate) fn emit(&self, path: &syn::Path) -> TokenStream {
        let doc = &self.doc;
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

struct Enum<'a> {
    doc: TokenStream,
    enum_repr: attributes::EnumRepr,
    variants: Vec<Variant<'a>>,
}

impl<'a> Enum<'a> {
    pub(crate) fn parse(
        enum_: &'a syn::DataEnum,
        attr: attributes::Enum,
        doc: TokenStream,
    ) -> syn::Result<Self> {
        let attributes::Enum {
            rename_all,
            enum_repr,
        } = attr;

        // Parse all variants.
        let mut variants = enum_
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

        let internal_tag = match &enum_repr {
            attributes::EnumRepr::Internal { tag } => Some(tag.value()),
            attributes::EnumRepr::External | attributes::EnumRepr::Adjacent { .. } => None,
        };

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
                            // Check if the field name conflicts with the internal tag.
                            if let Some(tag) = internal_tag.as_ref()
                                && tag == &field.name.value()
                            {
                                return Err(syn::Error::new_spanned(
                                    &field.name,
                                    "field conflicts with internal discriminant tag",
                                ));
                            }
                        }
                    }
                    Fields::Unnamed(unnamed) => if unnamed.len() != 1 {
                            return Err(syn::Error::new_spanned(
                                &v.name,
                                "non-newtype tuple type variants are now allowed with internal tagging",
                            ));
                        }
                    Fields::Unit => {}
                }
            }
        }

        Ok(Self { doc, enum_repr, variants })
    }

    pub(crate) fn for_each_type<F, R>(&self, mut f: F)
    where
        F: FnMut(&syn::Type),
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
        let doc = &self.doc;
        quote! {
            #path::type::enum_(
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
    doc: TokenStream,
    ty: &'a syn::Type,
}

impl<'a> UnnamedField<'a> {
    fn parse(field: &'a syn::Field) -> syn::Result<Self> {
        assert!(
            field.ident.is_none(),
            "expected `syn` to not attach names to `FieldsNamed` entries"
        );

        let attributes::Field { rename_field } = attributes::Field::parse(&field.attrs)?;

        // Rejact renames on unnamed fields.
        if let Some(span) = rename_field.span() {
            return Err(syn::Error::new(
                span,
                "rename attributes are not supported on unnamed fields",
            ));
        }

        Ok(Self {
            doc: format_docstrings(&field.attrs),
            ty: &field.ty,
        })
    }

    fn emit(&self, path: &syn::Path) -> TokenStream {
        let ty = self.ty;
        let doc = &self.doc;
        quote_spanned! { ty.span()=> #path::tree::UnnamedField::new::<#ty>(#doc) }
    }
}

pub(crate) struct NamedField<'a> {
    doc: TokenStream,
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
        let attributes::Field { rename_field } = attributes::Field::parse(&field.attrs)?;
        let name = rename_field.apply_to_field(name, rename_all);

        Ok(Self {
            doc: format_docstrings(&field.attrs),
            name,
            ty: &field.ty,
        })
    }

    fn emit(&self, path: &syn::Path) -> TokenStream {
        let ty = self.ty;
        let doc = &self.doc;
        let name = &self.name;
        quote_spanned! {
            ty.span()=> #path::tree::NamedFields::new::<#ty>(#name, #doc)
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
            },
            syn::Fields::Unnamed(fields) => {
                if rename_all != attributes::RenameAll::None {
                    todo!("propagate the span correctly");
                }

                Self::Unnamed(
                    fields
                        .unnamed
                        .iter()
                        .map(UnnamedField::parse)
                        .collect::<syn::Result<_>>()?
                )
            },
            syn::Fields::Unit => {
                if rename_all != attributes::RenameAll::None {
                    todo!("propagate the span correctly");
                }

                Self::Unit
            }
        };
        Ok(me)
    }

    fn for_each_type<F>(&self, mut f: F)
    where
        F: FnMut(&syn::Type),
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
    doc: TokenStream,
    name: syn::LitStr,
    fields: Fields<'a>,
}

impl<'a> Variant<'a> {
    fn parse(variant: &'a syn::Variant, rename_all: attributes::RenameAll) -> syn::Result<Self> {
        let attributes::Variant {
            rename_variant,
            rename_variant_fields,
        } = attributes::Variant::parse(&variant.attrs)?;

        let fields = Fields::parse(&variant.fields, rename_variant_fields)?;

        let name = syn::LitStr::new(
            strip_raw_prefix(&variant.ident.to_string()),
            variant.ident.span(),
        );
        let name = rename_variant.apply_to_variant(name, rename_all);

        Ok(Self {
            doc: format_docstrings(&variant.attrs),
            name,
            fields,
        })
    }

    fn for_each_type<F>(&self, mut f: F)
    where
        F: FnMut(&syn::Type),
    {
        self.fields.for_each_type(f)
    }

    fn emit(&self, path: &syn::Path) -> TokenStream {
        let name = &self.name;
        let fields = self.fields.emit(path);
        let doc = &self.doc;
        quote!(#path::tree::Variant::new(#name, #fields, #doc))
    }
}

