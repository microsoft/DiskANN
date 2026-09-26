/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::{Reflect, Reflection};

pub type Doc = std::borrow::Cow<'static, str>;

#[derive(Debug)]
pub enum Type {
    Primitive(Primitive),
    Aggregate(Aggregate),
    Enum(Enum),
    Sequence(Sequence),
}

impl Type {
    pub fn primitive(doc: Option<Doc>) -> Self {
        Self::from(Primitive::new(doc))
    }

    pub fn aggregate(fields: Fields, doc: Option<Doc>) -> Self {
        Self::from(Aggregate::new(fields, doc))
    }

    pub fn enum_(
        repr: EnumRepr,
        variants: impl IntoIterator<Item = Variant>,
        doc: Option<Doc>,
    ) -> Self {
        Self::from(Enum::new(repr, variants, doc))
    }

    pub fn sequence<T>(doc: Option<Doc>) -> Self
    where
        T: Reflect,
    {
        Self::from(Sequence::new::<T>(doc))
    }

    pub fn doc(&self) -> Option<&str> {
        match self {
            Self::Primitive(p) => p.doc(),
            Self::Aggregate(a) => a.doc(),
            Self::Enum(e) => e.doc(),
            Self::Sequence(s) => s.doc(),
        }
    }

    pub(super) fn has_body(&self) -> bool {
        match self {
            Self::Primitive(_) => false,
            Self::Aggregate(a) => a.has_body(),
            Self::Enum(e) => e.has_body(),
            Self::Sequence(_) => true,
        }
    }

    pub(super) fn as_aggregate(&self) -> Option<&Aggregate> {
        if let Self::Aggregate(aggregate) = self {
            Some(aggregate)
        } else {
            None
        }
    }

    pub(super) fn as_enum(&self) -> Option<&Enum> {
        if let Self::Enum(enum_) = self {
            Some(enum_)
        } else {
            None
        }
    }
}

impl From<Primitive> for Type {
    fn from(primitive: Primitive) -> Self {
        Self::Primitive(primitive)
    }
}

impl From<Aggregate> for Type {
    fn from(aggergate: Aggregate) -> Self {
        Self::Aggregate(aggergate)
    }
}

impl From<Enum> for Type {
    fn from(e: Enum) -> Self {
        Self::Enum(e)
    }
}

impl From<Sequence> for Type {
    fn from(s: Sequence) -> Self {
        Self::Sequence(s)
    }
}

//-----------//
// Primitive //
//-----------//

#[derive(Debug)]
pub struct Primitive {
    doc: Option<Doc>,
}

impl Primitive {
    pub fn new(doc: Option<Doc>) -> Self {
        Self { doc }
    }

    fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }
}

//-----------//
// Aggregate //
//-----------//

#[derive(Debug)]
pub struct Aggregate {
    fields: Fields,
    doc: Option<Doc>,
}

impl Aggregate {
    pub fn new(fields: Fields, doc: Option<Doc>) -> Self {
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

#[derive(Debug)]
pub enum Fields {
    Named(Vec<NamedField>),
    Unnamed(Vec<UnnamedField>),
    Unit,
}

impl Fields {
    pub(super) fn has_body(&self) -> bool {
        match self {
            Self::Named(fields) => !fields.is_empty(),
            Self::Unnamed(fields) => !fields.is_empty(),
            Self::Unit => false,
        }
    }

    pub fn named(itr: impl IntoIterator<Item = NamedField>) -> Self {
        Self::Named(itr.into_iter().collect())
    }

    pub fn unnamed(itr: impl IntoIterator<Item = UnnamedField>) -> Self {
        Self::Unnamed(itr.into_iter().collect())
    }

    pub(super) fn as_named(&self) -> Option<&[NamedField]> {
        if let Self::Named(fields) = self {
            Some(fields)
        } else {
            None
        }
    }

    pub(super) fn as_unnamed(&self) -> Option<&[UnnamedField]> {
        if let Self::Unnamed(fields) = self {
            Some(fields)
        } else {
            None
        }
    }

    pub(super) fn is_unit(&self) -> bool {
        matches!(self, Self::Unit)
    }
}

#[derive(Debug)]
pub struct NamedField {
    name: &'static str,
    field: Reflection,
    doc: Option<Doc>,
}

impl NamedField {
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

#[derive(Debug)]
pub struct UnnamedField {
    field: Reflection,
    doc: Option<Doc>,
}

impl UnnamedField {
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

#[derive(Debug)]
pub struct Enum {
    repr: EnumRepr,
    variants: Vec<Variant>,
    doc: Option<Doc>,
}

impl Enum {
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

#[derive(Debug)]
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

#[derive(Debug)]
pub struct Variant {
    name: &'static str,
    fields: Fields,
    doc: Option<Doc>,
}

impl Variant {
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

#[derive(Debug)]
pub struct Sequence {
    element: Reflection,
    doc: Option<Doc>,
}

impl Sequence {
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
