/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Model of the Rust type system.
//!
//! This is largely based on the structure of types in [`syn`](https://docs.rs/syn/latest/syn/)
//! with some extra entries (e.g. [`Type::Optional`]) for a closer representation with how
//! types are serialized by [`serde`].

use super::{Reflect, Reflection};

pub type Doc = std::borrow::Cow<'static, str>;

/// Classification of types.
#[derive(Debug)]
pub enum Type {
    Primitive(Primitive),
    Aggregate(Aggregate),
    Enum(Enum),
    Sequence(Sequence),
    Optional(Optional),
}

impl Type {
    /// Construc a new [`Type::Primitive`].
    pub fn primitive(kind: PrimitiveKind, doc: Option<Doc>) -> Self {
        Self::Primitive(Primitive::new(kind, doc))
    }

    /// Construc a new [`Type::Aggregate`].
    pub fn aggregate(fields: Fields, doc: Option<Doc>) -> Self {
        Self::Aggregate(Aggregate::new(fields, doc))
    }

    /// Construc a new [`Type::Enum`].
    pub fn enum_(
        repr: EnumRepr,
        variants: impl IntoIterator<Item = Variant>,
        doc: Option<Doc>,
    ) -> Self {
        Self::Enum(Enum::new(repr, variants, doc))
    }

    /// Construc a new [`Type::Sequence`].
    pub fn sequence<T>(doc: Option<Doc>) -> Self
    where
        T: Reflect,
    {
        Self::Sequence(Sequence::new::<T>(doc))
    }

    /// Keep the constructor private since we don't want users constructing the very special
    /// `Optional` type for their own types.
    pub(crate) fn optional<T>(doc: Option<Doc>) -> Self
    where
        T: Reflect,
    {
        Self::Optional(Optional::new::<T>(doc))
    }

    /// Return the struct level documentation if available.
    pub fn doc(&self) -> Option<&str> {
        match self {
            Self::Primitive(p) => p.doc(),
            Self::Aggregate(a) => a.doc(),
            Self::Enum(e) => e.doc(),
            Self::Sequence(s) => s.doc(),
            Self::Optional(o) => o.doc(),
        }
    }

    /// Return the primitive JSON kind (if there is one).
    pub(super) fn json_kind(&self) -> Option<&str> {
        match self {
            Self::Primitive(p) => Some(p.kind().json_kind()),
            Self::Aggregate(_) | Self::Enum(_) | Self::Sequence(_) | Self::Optional(_) => None,
        }
    }

    /// Return `true` if there is field level information of some kind to render.
    pub(super) fn has_body(&self) -> bool {
        match self {
            Self::Primitive(_) => false,
            Self::Aggregate(a) => a.has_body(),
            Self::Enum(e) => e.has_body(),
            Self::Sequence(_) => true,
            Self::Optional(_) => true,
        }
    }

    #[cfg(test)]
    pub(super) fn as_aggregate(&self) -> Option<&Aggregate> {
        if let Self::Aggregate(aggregate) = self {
            Some(aggregate)
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(super) fn as_enum(&self) -> Option<&Enum> {
        if let Self::Enum(enum_) = self {
            Some(enum_)
        } else {
            None
        }
    }
}

//-----------//
// Primitive //
//-----------//

/// The native JSON representation for a primitive.
#[derive(Debug, Clone, Copy)]
pub enum PrimitiveKind {
    Null,
    Boolean,
    Number,
    String,
}

impl PrimitiveKind {
    pub(crate) fn json_kind(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Boolean => "bool",
            Self::Number => "number",
            Self::String => "string",
        }
    }
}

/// A primitive type that maps closely to a native JSON type.
#[derive(Debug)]
pub struct Primitive {
    kind: PrimitiveKind,
    doc: Option<Doc>,
}

impl Primitive {
    pub(crate) fn new(kind: PrimitiveKind, doc: Option<Doc>) -> Self {
        Self { kind, doc }
    }

    pub(super) fn kind(&self) -> PrimitiveKind {
        self.kind
    }

    pub(super) fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }
}

//-----------//
// Aggregate //
//-----------//

/// A representation of aggregates like normal structs, unit structs, and tuple-like structs.
#[derive(Debug)]
pub struct Aggregate {
    fields: Fields,
    doc: Option<Doc>,
}

impl Aggregate {
    pub(crate) fn new(fields: Fields, doc: Option<Doc>) -> Self {
        Self { fields, doc }
    }

    pub(super) fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }

    pub(super) fn fields(&self) -> &Fields {
        &self.fields
    }

    pub(super) fn has_body(&self) -> bool {
        self.fields.has_body()
    }
}

/// Represent the fields of a struct.
#[derive(Debug)]
pub enum Fields {
    /// Standard Rust structs.
    Named(Vec<NamedField>),

    /// Tuple-like structs.
    Unnamed(Vec<UnnamedField>),

    /// Tuple-like structs with a single named field.
    ///
    /// These are treated specially by `serde` and thus get their own variant.
    NewType(UnnamedField),

    /// Unit structs.
    Unit,
}

impl Fields {
    pub(super) fn has_body(&self) -> bool {
        match self {
            Self::Named(fields) => !fields.is_empty(),
            Self::Unnamed(fields) => !fields.is_empty(),
            Self::NewType(_) => true,
            Self::Unit => false,
        }
    }

    /// Construct [`Fields::Named`] from the iterator.
    pub fn named(itr: impl IntoIterator<Item = NamedField>) -> Self {
        Self::Named(itr.into_iter().collect())
    }

    /// Construct [`Fields::Unamed`] from the iterator.
    pub fn unnamed(itr: impl IntoIterator<Item = UnnamedField>) -> Self {
        Self::Unnamed(itr.into_iter().collect())
    }

    /// Construct [`Fields::NewType`] from the iterator.
    pub fn newtype(field: UnnamedField) -> Self {
        Self::NewType(field)
    }

    #[cfg(test)]
    pub(super) fn as_named(&self) -> Option<&[NamedField]> {
        if let Self::Named(fields) = self {
            Some(fields)
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(super) fn as_unnamed(&self) -> Option<&[UnnamedField]> {
        if let Self::Unnamed(fields) = self {
            Some(fields)
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(super) fn as_newtype(&self) -> Option<&UnnamedField> {
        if let Self::NewType(field) = self {
            Some(field)
        } else {
            None
        }
    }

    /// Return `true` if `self` is [`Self::Unit`].
    pub(super) fn is_unit(&self) -> bool {
        matches!(self, Self::Unit)
    }
}

/// A struct field with a name.
#[derive(Debug)]
pub struct NamedField {
    name: &'static str,
    field: Reflection,
    doc: Option<Doc>,
}

impl NamedField {
    /// Construct a new [`NamedField`] for `T`.
    pub fn new<T>(name: &'static str, doc: Option<Doc>) -> Self
    where
        T: Reflect,
    {
        Self {
            name,
            field: Reflection::new::<T>(),
            doc,
        }
    }

    pub(super) fn name(&self) -> &str {
        self.name
    }

    pub(super) fn field(&self) -> Reflection {
        self.field
    }

    pub(super) fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }
}

/// An unnamed field.
#[derive(Debug)]
pub struct UnnamedField {
    field: Reflection,
    doc: Option<Doc>,
}

impl UnnamedField {
    /// Construct a new [`UnnamedField`] for `T`.
    pub fn new<T>(doc: Option<Doc>) -> Self
    where
        T: Reflect,
    {
        Self {
            field: Reflection::new::<T>(),
            doc,
        }
    }

    pub(super) fn field(&self) -> Reflection {
        self.field
    }

    pub(super) fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }
}

//------//
// Enum //
//------//

/// A representation for enums.
#[derive(Debug)]
pub struct Enum {
    repr: EnumRepr,
    variants: Vec<Variant>,
    doc: Option<Doc>,
}

impl Enum {
    /// Construct a new [`Enum`].
    pub fn new(
        repr: EnumRepr,
        variants: impl IntoIterator<Item = Variant>,
        doc: Option<Doc>,
    ) -> Self {
        Self {
            repr,
            variants: variants.into_iter().collect(),
            doc,
        }
    }

    pub(super) fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }

    pub(super) fn repr(&self) -> &EnumRepr {
        &self.repr
    }

    pub(super) fn variants(&self) -> &[Variant] {
        &self.variants
    }

    pub(super) fn has_body(&self) -> bool {
        !self.variants.is_empty()
    }
}

/// Describe how an enum is being represented by `serde`.
#[derive(Debug)]
#[non_exhaustive]
pub enum EnumRepr {
    /// Enums are tagged as the key in a collection.
    External,

    /// Enums are tagged by an internal field.
    ///
    /// This contains the name of the field that is used as the tag.
    ///
    /// ```json
    /// {
    ///   "tag": "some-enum-tag",
    ///   "value": 10,
    ///   "members": [
    ///     1,
    ///     "world"
    ///   ]
    /// }
    /// ```
    Internal { tag: &'static str },

    /// Enums have a separate tag and content payload.
    ///
    /// ```json
    /// {
    ///   "tag": "some-enum-tag",
    ///   "content": {
    ///     "value": 10,
    ///     "members": [
    ///       1,
    ///       "world"
    ///     ]
    ///   }
    /// }
    /// ```
    Adjacent {
        tag: &'static str,
        content: &'static str,
    },
}

/// A variant of an [`Enum`].
#[derive(Debug)]
pub struct Variant {
    name: &'static str,
    fields: Fields,
    doc: Option<Doc>,
}

impl Variant {
    /// Construct a new [`Variant`].
    pub fn new(name: &'static str, fields: Fields, doc: Option<Doc>) -> Self {
        Self { name, fields, doc }
    }

    pub(super) fn name(&self) -> &'static str {
        self.name
    }

    pub(super) fn fields(&self) -> &Fields {
        &self.fields
    }

    pub(super) fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }
}

//----------//
// Sequence //
//----------//

/// A homogeneous sequence of values.
#[derive(Debug)]
pub struct Sequence {
    element: Reflection,
    doc: Option<Doc>,
}

impl Sequence {
    /// Create a new [`Sequence`] containing `T`.
    pub fn new<T>(doc: Option<Doc>) -> Self
    where
        T: Reflect,
    {
        Self {
            element: Reflection::new::<T>(),
            doc,
        }
    }

    pub(super) fn element(&self) -> Reflection {
        self.element
    }

    pub(super) fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }
}

//----------//
// Optional //
//----------//

/// An [`Option`].
#[derive(Debug)]
pub struct Optional {
    element: Reflection,
    doc: Option<Doc>,
}

impl Optional {
    /// Keep the constructor private since we don't want users constructing the very special
    /// `Optional` type for their own types.
    pub(super) fn new<T>(doc: Option<Doc>) -> Self
    where
        T: Reflect,
    {
        Self {
            element: Reflection::new::<T>(),
            doc,
        }
    }

    pub(super) fn value(&self) -> Reflection {
        self.element
    }

    pub(super) fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }
}
