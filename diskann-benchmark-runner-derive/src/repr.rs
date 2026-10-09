/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::HashSet;

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{parse_quote, spanned::Spanned};

use crate::attributes;

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

        let doc = attributes::Doc::parse(&input.attrs);

        // Parse the type tree for the container.
        let container = Container::parse(&input.data, attrs, doc, input.span())?;
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

struct Generics<'a> {
    generics: &'a syn::Generics,
    params: Vec<&'a syn::Ident>,
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
    fn parse(
        data: &'a syn::Data,
        attrs: attributes::Container,
        doc: attributes::Doc,
        span: Span,
    ) -> syn::Result<Self> {
        match data {
            syn::Data::Struct(s) => {
                Ok(Self::Struct(Struct::parse(s, attrs.try_as_struct()?, doc)?))
            }
            syn::Data::Enum(e) => Ok(Self::Enum(Enum::parse(e, attrs.as_enum(), doc)?)),
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
    pub(crate) fn parse(
        s: &'a syn::DataStruct,
        attrs: attributes::Struct,
        doc: attributes::Doc,
    ) -> syn::Result<Self> {
        let attributes::Struct { rename_all } = attrs;
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
    pub(crate) fn parse(
        enum_: &'a syn::DataEnum,
        attr: attributes::Enum,
        doc: attributes::Doc,
    ) -> syn::Result<Self> {
        let attributes::Enum {
            rename_all,
            enum_repr,
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

        let attributes::Field { rename_field } = attributes::Field::parse(&field.attrs)?;

        // Rejact renames on unnamed fields.
        if let Some(span) = rename_field.span() {
            return Err(syn::Error::new(
                span,
                "rename attributes are not supported on unnamed fields",
            ));
        }

        Ok(Self {
            doc: attributes::Doc::parse(&field.attrs),
            ty: &field.ty,
        })
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
        let attributes::Field { rename_field } = attributes::Field::parse(&field.attrs)?;
        let name = rename_field.apply_to_field(name, rename_all);

        Ok(Self {
            doc: attributes::Doc::parse(&field.attrs),
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
        } = attributes::Variant::parse(&variant.attrs)?;

        let fields = Fields::parse(&variant.fields, rename_variant_fields)?;

        let name = syn::LitStr::new(
            strip_raw_prefix(&variant.ident.to_string()),
            variant.ident.span(),
        );
        let name = rename_variant.apply_to_variant(name, rename_all);

        Ok(Self {
            doc: attributes::Doc::parse(&variant.attrs),
            name,
            fields,
        })
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
