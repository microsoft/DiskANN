/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! These are modeled after the attributes documented in <https://serde.rs/attributes.html>.

#[must_use]
fn is_serde_attr(attr: &syn::Attribute) -> bool {
    attr.path().is_ident("serde")
}

fn identity<T>(x: T) -> T {
    x
}

fn set_unique(opt: &mut Option<syn::LitStr>, value: syn::LitStr, attr: &str) -> syn::Result<()> {
    if opt.is_some() {
        Err(syn::Error::new_spanned(
            value,
            format!("serde attribute `{}` found multiple times", attr),
        ))
    } else {
        *opt = Some(value);
        Ok(())
    }
}

fn set_rename_all(
    opt: &mut Option<RenameAll>,
    rename_all: RenameAll,
    lit: syn::LitStr,
) -> syn::Result<()> {
    if opt.is_some() {
        Err(syn::Error::new_spanned(
            lit,
            "serde attribute `rename_all` found multiple times",
        ))
    } else {
        *opt = Some(rename_all);
        Ok(())
    }
}

pub(crate) struct Struct {
    pub(crate) rename_all: Option<RenameAll>,
}

pub(crate) struct Enum {
    pub(crate) rename_all: Option<RenameAll>,
    pub(crate) enum_repr: EnumRepr,
}

pub(crate) struct Container {
    pub(crate) rename_all: Option<RenameAll>,
    pub(crate) enum_repr: EnumRepr,
}

impl Container {
    pub(crate) fn parse(attrs: &[syn::Attribute]) -> syn::Result<Self> {
        let mut rename_all = Option::None;
        let mut tag = Option::None;
        let mut content = Option::None;

        for attr in attrs.iter().filter(|a| is_serde_attr(*a)) {
            attr.parse_nested_meta(|meta| {
                // serde(rename_all = "...")
                if meta.path.is_ident("rename_all") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    let parsed = RenameAll::parse(&value)?;
                    set_rename_all(&mut rename_all, parsed, value)?;
                    return Ok(());
                }

                // serde(tag = "...")
                if meta.path.is_ident("tag") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    set_unique(&mut tag, value, "tag")?;
                    return Ok(());
                }

                // serde(content = "...")
                if meta.path.is_ident("content") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    set_unique(&mut content, value, "content")?;
                    return Ok(());
                }

                Err(meta.error("unsupported Serde attribute for Reflect"))
            })?;
        }

        Ok(Self {
            rename_all,
            enum_repr: EnumRepr::from_parsed(tag, content)?,
        })
    }

    pub(crate) fn try_as_struct(self) -> syn::Result<Struct> {
        self.enum_repr.assert_struct_compatible()?;
        Ok(Struct {
            rename_all: self.rename_all,
        })
    }

    pub(crate) fn as_enum(self) -> Enum {
        Enum {
            rename_all: self.rename_all,
            enum_repr: self.enum_repr,
        }
    }
}

pub(crate) enum EnumRepr {
    External,
    Internal {
        tag: syn::LitStr,
    },
    Adjacent {
        tag: syn::LitStr,
        content: syn::LitStr,
    },
}

impl EnumRepr {
    pub(crate) fn from_parsed(
        tag: Option<syn::LitStr>,
        content: Option<syn::LitStr>,
    ) -> syn::Result<Self> {
        match (tag, content) {
            (None, None) => Ok(EnumRepr::External),
            (Some(tag), None) => Ok(EnumRepr::Internal { tag }),
            (Some(tag), Some(content)) => Ok(EnumRepr::Adjacent { tag, content }),
            (None, Some(content)) => Err(syn::Error::new_spanned(
                content,
                "serde attribute `content` provided without a `tag`",
            )),
        }
    }

    pub(crate) fn assert_struct_compatible(&self) -> syn::Result<()> {
        match self {
            Self::External => Ok(()),
            Self::Internal { tag } => Err(syn::Error::new_spanned(
                tag,
                "serde attribute `tag` provided on a non-enum",
            )),
            Self::Adjacent { tag, .. } => Err(syn::Error::new_spanned(
                tag,
                "serde attributes `tag` and `content` provided on a non-enum",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RenameAll {
    Lower,
    Snake,
    Kebab,
}

impl RenameAll {
    fn supported() -> &'static str {
        "\"lowercase\", \"snake_case\", or \"kebab-case\""
    }

    fn parse(lit: &syn::LitStr) -> syn::Result<Self> {
        let s = lit.value();

        match &*s {
            "lowercase" => Ok(Self::Lower),
            "snake_case" => Ok(Self::Snake),
            "kebab-case" => Ok(Self::Kebab),
            _ => Err(syn::Error::new_spanned(
                lit,
                format!(
                    "unsupported serde `rename_all` rule \"{}\" - expected one of {}",
                    s,
                    Self::supported()
                ),
            )),
        }
    }

    fn apply(&self, lit: syn::LitStr) -> syn::LitStr {
        let s = lit.value();
        let s = match self {
            Self::Lower => s.to_lowercase(),
            Self::Snake => heck::AsSnakeCase(s).to_string(),
            Self::Kebab => heck::AsKebabCase(s).to_string(),
        };

        syn::LitStr::new(&s, lit.span())
    }

    pub(crate) fn visitor(me: Option<Self>) -> impl Fn(syn::LitStr) -> syn::LitStr {
        move |v| {
            if let Some(rename_all) = me {
                rename_all.apply(v)
            } else {
                v
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct Variant {
    rename: Option<syn::LitStr>,
    rename_all: Option<RenameAll>,
}

impl Variant {
    pub(crate) fn parse(attrs: &[syn::Attribute]) -> syn::Result<Self> {
        let mut me = Self::default();

        for attr in attrs.iter().filter(|a| is_serde_attr(*a)) {
            attr.parse_nested_meta(|meta| {
                // serde(rename_all = "...")
                if meta.path.is_ident("rename_all") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    let rename_all = RenameAll::parse(&value)?;
                    set_rename_all(&mut me.rename_all, rename_all, value)?;
                    return Ok(());
                }

                // serde(rename = "...")
                if meta.path.is_ident("rename") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    set_unique(&mut me.rename, value, "rename")?;
                    return Ok(());
                }

                Err(meta.error("unsupported Serde attribute for Reflect"))
            })?;
        }

        Ok(me)
    }

    /// Replace the name of this variant if directed by `serde(rename = "...")`.
    ///
    /// If the rename attribute exists, it takes precedence. Otherwise, the fallback is used.
    pub(crate) fn rename_variant_or(
        self,
        value: syn::LitStr,
        or_else: &dyn Fn(syn::LitStr) -> syn::LitStr,
    ) -> syn::LitStr {
        self.rename.map_or_else(|| or_else(value), identity)
    }

    pub(crate) fn renamer(&self) -> impl Fn(syn::LitStr) -> syn::LitStr {
        RenameAll::visitor(self.rename_all)
    }
}

#[derive(Default, Clone)]
pub(crate) struct Field {
    rename: Option<syn::LitStr>,
}

impl Field {
    pub(crate) fn parse(attrs: &[syn::Attribute]) -> syn::Result<Self> {
        let mut me = Self::default();

        for attr in attrs.iter().filter(|a| is_serde_attr(*a)) {
            attr.parse_nested_meta(|meta| {
                // serde(rename = "...")
                if meta.path.is_ident("rename") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    set_unique(&mut me.rename, value, "rename")?;
                    return Ok(());
                }

                Err(meta.error("unsupported Serde attribute for Reflect"))
            })?;
        }

        Ok(me)
    }

    /// Apply the renaming rules defined in `self`.
    ///
    /// If no renaming rules are present, instead invoke `or_else`.
    pub(crate) fn rename_field_or(
        self,
        value: syn::LitStr,
        or_else: &dyn Fn(syn::LitStr) -> syn::LitStr,
    ) -> syn::LitStr {
        let Self { rename } = self;
        rename.map_or_else(|| or_else(value), identity)
    }
}
