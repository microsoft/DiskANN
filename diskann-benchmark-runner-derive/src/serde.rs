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

pub(crate) struct Struct {
    pub(crate) rename_all: RenameAll,
}

pub(crate) struct Enum {
    pub(crate) rename_all: RenameAll,
    pub(crate) enum_repr: EnumRepr,
}

pub(crate) struct Container {
    pub(crate) rename_all: RenameAll,
    pub(crate) enum_repr: EnumRepr,
}

impl Container {
    pub(crate) fn parse(attrs: &[syn::Attribute]) -> syn::Result<Self> {
        let mut rename_all = RenameAll::None;
        let mut tag = Option::None;
        let mut content = Option::None;

        for attr in attrs.iter().filter(|a| is_serde_attr(*a)) {
            attr.parse_nested_meta(|meta| {
                // serde(rename_all = "...")
                if meta.path.is_ident("rename_all") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    rename_all.parse_in(value)?;
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

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub(crate) enum RenameAll {
    #[default]
    None,
    Lower,
    Snake,
    Kebab,
}

impl RenameAll {
    fn supported() -> &'static str {
        "\"lowercase\", \"snake_case\", or \"kebab-case\""
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "lowercase" => Some(Self::Lower),
            "snake_case" => Some(Self::Snake),
            "kebab-case" => Some(Self::Kebab),
            _ => None,
        }
    }

    fn parse_in(&mut self, s: syn::LitStr) -> syn::Result<Self> {
        if *self != Self::None {
            Err(syn::Error::new_spanned(
                s,
                "serde attribute `rename_all` found multiple times",
            ))
        } else {
            let value = s.value();
            match Self::parse(&value) {
                Some(me) => Ok(me),
                None => Err(syn::Error::new_spanned(
                    s,
                    format!(
                        "unsupported serde `rename_all` rule \"{}\" - expected one of {}",
                        value,
                        Self::supported()
                    ),
                )),
            }
        }
    }

    /// These methods are taken from the `serde_derive` internals as they need to match.
    ///
    /// See: <https://github.com/serde-rs/serde/blob/master/serde_derive/src/internals/case.rs>
    fn apply_to_variant_str(&self, variant: &str) -> String {
        match self {
            Self::None => variant.to_owned(),
            Self::Lower => variant.to_ascii_lowercase(),
            Self::Snake => {
                let mut snake = String::new();
                for (i, ch) in variant.char_indices() {
                    if i > 0 && ch.is_uppercase() {
                        snake.push('_');
                    }
                    snake.push(ch.to_ascii_lowercase());
                }
                snake
            }
            Self::Kebab => (Self::Snake)
                .apply_to_variant_str(variant)
                .replace('_', "-"),
        }
    }

    pub(crate) fn apply_to_variant(&self, variant: syn::LitStr) -> syn::LitStr {
        if *self == Self::None {
            variant
        } else {
            syn::LitStr::new(&self.apply_to_variant_str(&variant.value()), variant.span())
        }
    }

    /// These methods are taken from the `serde_derive` internals as they need to match.
    ///
    /// Since Rust field are generally in lower snake case, there's less work to do.
    ///
    /// See: <https://github.com/serde-rs/serde/blob/master/serde_derive/src/internals/case.rs>
    fn apply_to_field_str(&self, field: &str) -> String {
        match self {
            Self::None | Self::Lower | Self::Snake => field.to_owned(),
            Self::Kebab => field.replace('_', "-"),
        }
    }

    pub(crate) fn apply_to_field(&self, variant: syn::LitStr) -> syn::LitStr {
        if *self == Self::None {
            variant
        } else {
            syn::LitStr::new(&self.apply_to_field_str(&variant.value()), variant.span())
        }
    }
}

#[derive(Default)]
pub(crate) struct Variant {
    rename: Option<syn::LitStr>,
    rename_all: RenameAll,
}

impl Variant {
    pub(crate) fn parse(attrs: &[syn::Attribute]) -> syn::Result<Self> {
        let mut me = Self::default();

        for attr in attrs.iter().filter(|a| is_serde_attr(*a)) {
            attr.parse_nested_meta(|meta| {
                // serde(rename_all = "...")
                if meta.path.is_ident("rename_all") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    me.rename_all.parse_in(value)?;
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
    pub(crate) fn rename_variant_or(self, variant: syn::LitStr, or_else: RenameAll) -> syn::LitStr {
        self.rename
            .map_or_else(|| or_else.apply_to_variant(variant), identity)
    }

    pub(crate) fn field_rename_all(&self) -> RenameAll {
        self.rename_all
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
    pub(crate) fn rename_field_or(self, field: syn::LitStr, or_else: RenameAll) -> syn::LitStr {
        let Self { rename } = self;
        rename.map_or_else(|| or_else.apply_to_field(field), identity)
    }
}
