/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! These are modeled after the attributes documented in <https://serde.rs/attributes.html>.

#[must_use]
fn is_serde_attr(attr: &syn::Attribute) -> bool {
    attr.path().is_ident("serde")
}

#[must_use]
fn is_reflect_attr(attr: &syn::Attribute) -> bool {
    attr.path().is_ident("reflect")
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

/// Attributes applicable to struct definitions.
pub(crate) struct Struct {
    /// A univeral rename rule for all fields.
    ///
    /// [`Field`] specific renames take precedence.
    pub(crate) rename_all: RenameAll,
}

/// Attributes applicable to enum definitions.
pub(crate) struct Enum {
    /// A universal rename rule for all variants.
    ///
    /// [`Variant`] specific renames take precedence.
    pub(crate) rename_all: RenameAll,

    /// Enum's representation.
    pub(crate) enum_repr: EnumRepr,
}

/// Attributes on the top-level [`syn::DeriveInput`].
///
/// Uses should go through [`Container::as_enum`] or [`Container::try_as_struct`] to ensure
/// the attributes are appropriate for the actual type.
pub(crate) struct Container {
    rename_all: RenameAll,
    enum_repr: EnumRepr,
    type_name: TypeName,
}

impl Container {
    pub(crate) fn parse(attrs: &[syn::Attribute]) -> syn::Result<Self> {
        let mut rename_all = RenameAll::None;
        let mut tag = Option::None;
        let mut content = Option::None;
        let mut type_name = TypeName::None;

        // Parse serde attributes.
        for attr in attrs.iter().filter(|a| is_serde_attr(*a)) {
            attr.parse_nested_meta(|meta| {
                // serde(rename_all = "...")
                if meta.path.is_ident("rename_all") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    rename_all.parse_once(value)?;
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

        // Parse reflect attributes
        for attr in attrs.iter().filter(|a| is_reflect_attr(*a)) {
            attr.parse_nested_meta(|meta| {
                // reflect(prefix = "...")
                if meta.path.is_ident("prefix") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    type_name.set_unique(TypeNameKind::Prefix, value)?;
                    return Ok(());
                }

                // reflect(type_name = "...")
                if meta.path.is_ident("type_name") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    type_name.set_unique(TypeNameKind::Rename, value)?;
                    return Ok(());
                }

                Err(meta.error("unsupported attribute for Reflect"))
            })?;
        }

        Ok(Self {
            rename_all,
            enum_repr: EnumRepr::from_parsed(tag, content)?,
            type_name,
        })
    }

    /// Verify the parsed attributes are compatible with an aggregate definition.
    pub(crate) fn try_as_struct(self) -> syn::Result<Struct> {
        self.enum_repr.assert_struct_compatible()?;
        Ok(Struct {
            rename_all: self.rename_all,
        })
    }

    /// Verify the parsed attributes are compatible with an enum definition.
    pub(crate) fn as_enum(self) -> Enum {
        Enum {
            rename_all: self.rename_all,
            enum_repr: self.enum_repr,
        }
    }

    /// Extract the type-name attributes.
    pub(crate) fn type_name(&self) -> TypeName {
        self.type_name.clone()
    }
}

/// The `serde` representation for an enum.
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
    /// Verify that the parsed `tag` and `content` fields are coherent.
    ///
    /// This checks that `content` cannot be applied without a `tag`.
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

    /// Verify that no `enum` tag attributes are present.
    ///
    /// These do not apply to structs, so we give a compile error with a diagnostic if they
    /// are observed.
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

/// Supported subset of `serde(rename_all = "...")`
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

    /// Attempt to parse `s`, returning an error if `self` is already parsed.
    fn parse_once(&mut self, s: syn::LitStr) -> syn::Result<()> {
        if *self != Self::None {
            Err(syn::Error::new_spanned(
                s,
                "serde attribute `rename_all` found multiple times",
            ))
        } else {
            let value = s.value();
            match Self::parse(&value) {
                Some(me) => {
                    *self = me;
                    Ok(())
                }
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

    /// Apply the rename rule to `variant`.
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

    /// Apply the rename rule to `field`.
    pub(crate) fn apply_to_field(&self, field: syn::LitStr) -> syn::LitStr {
        if *self == Self::None {
            field
        } else {
            syn::LitStr::new(&self.apply_to_field_str(&field.value()), field.span())
        }
    }
}

/// A one-type variant or field renamer.
#[derive(Default)]
pub(crate) struct RenameOnce {
    rename: Option<syn::LitStr>,
}

impl RenameOnce {
    /// Apply the configured rename to `variant`. If no rename is configured, apply `or_else`.
    pub(crate) fn apply_to_variant(self, variant: syn::LitStr, or_else: RenameAll) -> syn::LitStr {
        self.rename
            .map_or_else(|| or_else.apply_to_variant(variant), identity)
    }

    /// Apply the configured rename to `field`. If no rename is configured, apply `or_else`.
    pub(crate) fn apply_to_field(self, field: syn::LitStr, or_else: RenameAll) -> syn::LitStr {
        self.rename
            .map_or_else(|| or_else.apply_to_field(field), identity)
    }
}

/// Strategy for generting type-names.
#[derive(Default, Clone)]
pub(crate) enum TypeName {
    #[default]
    None,
    Prefix(syn::LitStr),
    Rename(syn::LitStr),
}

enum TypeNameKind {
    Prefix,
    Rename,
}

impl TypeName {
    fn set_unique(&mut self, kind: TypeNameKind, value: syn::LitStr) -> syn::Result<()> {
        if !matches!(self, Self::None) {
            Err(syn::Error::new_spanned(
                value,
                "reflect attribute `prefix` found multiple times",
            ))
        } else {
            match kind {
                TypeNameKind::Prefix => *self = Self::Prefix(value),
                TypeNameKind::Rename => *self = Self::Rename(value),
            }
            Ok(())
        }
    }
}

//---------//
// Variant //
//---------//

/// Variant level attributes.
#[derive(Default)]
pub(crate) struct Variant {
    pub(crate) rename_variant: RenameOnce,
    pub(crate) rename_variant_fields: RenameAll,
}

impl Variant {
    pub(crate) fn parse(attrs: &[syn::Attribute]) -> syn::Result<Self> {
        let mut me = Self::default();

        for attr in attrs.iter().filter(|a| is_serde_attr(*a)) {
            attr.parse_nested_meta(|meta| {
                // serde(rename_all = "...")
                if meta.path.is_ident("rename_all") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    me.rename_variant_fields.parse_once(value)?;
                    return Ok(());
                }

                // serde(rename = "...")
                if meta.path.is_ident("rename") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    set_unique(&mut me.rename_variant.rename, value, "rename")?;
                    return Ok(());
                }

                Err(meta.error("unsupported Serde attribute for Reflect"))
            })?;
        }

        Ok(me)
    }
}

//-------//
// Field //
//-------//

/// Field level attributes.
#[derive(Default)]
pub(crate) struct Field {
    pub(crate) rename_field: RenameOnce,
}

impl Field {
    pub(crate) fn parse(attrs: &[syn::Attribute]) -> syn::Result<Self> {
        let mut me = Self::default();

        for attr in attrs.iter().filter(|a| is_serde_attr(*a)) {
            attr.parse_nested_meta(|meta| {
                // serde(rename = "...")
                if meta.path.is_ident("rename") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    set_unique(&mut me.rename_field.rename, value, "rename")?;
                    return Ok(());
                }

                Err(meta.error("unsupported Serde attribute for Reflect"))
            })?;
        }

        Ok(me)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rename_all_parse() {
        assert!(RenameAll::parse("none").is_none());
        assert_eq!(RenameAll::parse("lowercase").unwrap(), RenameAll::Lower);
        assert_eq!(RenameAll::parse("snake_case").unwrap(), RenameAll::Snake);
        assert_eq!(RenameAll::parse("kebab-case").unwrap(), RenameAll::Kebab);

        assert!(RenameAll::parse("foo").is_none());
        assert!(RenameAll::parse("bar").is_none());
    }

    #[test]
    fn test_apply_to_variant() {
        assert_eq!(
            RenameAll::None.apply_to_variant_str("MiXeDUpper_Case"),
            "MiXeDUpper_Case"
        );

        assert_eq!(
            RenameAll::Lower.apply_to_variant_str("MiXeDUpper_Case"),
            "mixedupper_case"
        );
        assert_eq!(
            RenameAll::Lower.apply_to_variant_str("all_lower"),
            "all_lower"
        );

        assert_eq!(
            RenameAll::Snake.apply_to_variant_str("MixedUpperCase"),
            "mixed_upper_case"
        );
        assert_eq!(
            RenameAll::Snake.apply_to_variant_str("X86_64_V4"),
            "x86_64__v4"
        );

        assert_eq!(
            RenameAll::Kebab.apply_to_variant_str("MixedUpperCase"),
            "mixed-upper-case"
        );
        assert_eq!(
            RenameAll::Kebab.apply_to_variant_str("X86_64_V4"),
            "x86-64--v4"
        );
    }

    #[test]
    fn test_apply_to_field() {
        assert_eq!(
            RenameAll::None.apply_to_field_str("a_standard_field"),
            "a_standard_field"
        );
        assert_eq!(
            RenameAll::Lower.apply_to_field_str("a_standard_field"),
            "a_standard_field"
        );
        assert_eq!(
            RenameAll::Snake.apply_to_field_str("a_standard_field"),
            "a_standard_field"
        );
        assert_eq!(
            RenameAll::Kebab.apply_to_field_str("a_standard_field"),
            "a-standard-field"
        );
    }
}
