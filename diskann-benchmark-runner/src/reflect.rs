/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    any::TypeId,
    borrow::Cow,
    fmt::{self, Write},
};

pub use diskann_benchmark_runner_derive::Reflect;

use crate::utils::fmt::{Indent, Quote};

const INDENT: usize = 2;

pub trait Reflect: 'static {
    fn reflect() -> Type;
}

pub fn reflect<T>() -> Reflection
where
    T: Reflect,
{
    Reflection::new::<T>()
}

#[derive(Clone, Copy)]
pub struct Reflection {
    reflection: &'static dyn internal::Reflect,
}

impl Reflection {
    pub const fn new<T>() -> Self
    where
        T: Reflect,
    {
        Self {
            reflection: &internal::Wrapper::<T>::INSTANCE,
        }
    }

    pub fn reflect(&self) -> Type {
        self.reflection.reflect()
    }
}

pub type Doc = Cow<'static, str>;

#[derive(Debug)]
pub enum Query<'a> {
    Field(Cow<'a, str>),
    Index(usize),
}

pub enum Type {
    Primitive(Primitive),
    Aggregate(Aggregate),
    Enum(Enum),
}

impl Type {
    pub fn primitive(type_name: &'static str, type_id: TypeId, doc: &'static str) -> Self {
        Self::from(Primitive::new(type_name, type_id, doc))
    }

    pub fn aggregate(
        type_name: &'static str,
        type_id: TypeId,
        fields: Fields,
        doc: Option<Doc>,
    ) -> Self {
        Self::from(Aggregate::new(type_name, type_id, fields, doc))
    }

    pub fn enum_(
        type_name: &'static str,
        type_id: TypeId,
        variants: impl IntoIterator<Item = Variant>,
        doc: Option<Doc>,
    ) -> Self {
        Self::from(Enum::new(type_name, type_id, variants, doc))
    }

    fn format_into(&self, f: &mut dyn Write) -> fmt::Result {
        match self {
            Self::Primitive(p) => p.format_into(f),
            Self::Aggregate(a) => a.format_into(f),
            Self::Enum(e) => e.format_into(f),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format_into(f)
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

pub struct Primitive {
    type_name: &'static str,
    type_id: TypeId,
    doc: &'static str,
}

impl Primitive {
    pub fn new(type_name: &'static str, type_id: TypeId, doc: &'static str) -> Self {
        Self {
            type_name,
            type_id,
            doc,
        }
    }

    fn format_into(&self, f: &mut dyn std::fmt::Write) -> std::fmt::Result {
        write!(f, "{}: {}", self.type_name, self.doc)
    }
}

pub struct Aggregate {
    type_name: &'static str,
    type_id: TypeId,
    fields: Fields,
    doc: Option<Doc>,
}

impl Aggregate {
    pub fn new(type_name: &'static str, type_id: TypeId, fields: Fields, doc: Option<Doc>) -> Self {
        Self {
            type_name,
            type_id,
            fields,
            doc,
        }
    }

    fn format_into(&self, f: &mut dyn Write) -> fmt::Result {
        f.write_str(self.type_name)?;
        if let Some(doc) = &self.doc {
            write!(f, "\n{}\n", Indent::new(&doc, INDENT))?;
        }

        let mut scratch = String::new();
        self.fields.format_into(&mut scratch)?;
        write!(f, "{}", Indent::new(&scratch, INDENT))
    }
}

pub enum Fields {
    Named(Vec<NamedField>),
    Unnamed(Vec<UnnamedField>),
    Unit,
}

impl Fields {
    fn format_into(&self, f: &mut dyn Write) -> fmt::Result {
        match self {
            Self::Named(fields) => {
                for field in fields.iter() {
                    field.format_into(f)?;
                    f.write_str("\n\n")?;
                }
            }
            Self::Unnamed(fields) => {
                let mut buf = String::new();
                for (i, field) in fields.iter().enumerate() {
                    write!(f, "{}", i)?;
                    buf.clear();
                    field.format_into(&mut buf)?;
                    write!(f, "{}", Indent::new(&buf, INDENT))?;
                }
            }
            Self::Unit => {}
        }
        Ok(())
    }
}

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
            field: reflect::<T>(),
            doc,
        }
    }

    fn format_into(&self, f: &mut dyn Write) -> fmt::Result {
        write!(f, "{}", Quote(self.name))?;
        if let Some(doc) = &self.doc {
            write!(f, "\n{}\n", Indent::new(doc, INDENT))?;
        }

        let mut buf = String::new();
        self.field.reflect().format_into(&mut buf)?;
        write!(f, "{}", Indent::new(&buf, INDENT))
    }
}

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
            field: reflect::<T>(),
            doc,
        }
    }

    fn format_into(&self, f: &mut dyn Write) -> fmt::Result {
        self.field.reflect().format_into(f)
    }
}

pub struct Enum {
    type_name: &'static str,
    type_id: TypeId,
    variants: Vec<Variant>,
    doc: Option<Doc>,
}

impl Enum {
    pub fn new(
        type_name: &'static str,
        type_id: TypeId,
        variants: impl IntoIterator<Item = Variant>,
        doc: Option<Doc>,
    ) -> Self {
        Self {
            type_name,
            type_id: type_id,
            variants: variants.into_iter().collect(),
            doc,
        }
    }

    fn format_into(&self, f: &mut dyn Write) -> fmt::Result {
        f.write_str(self.type_name)?;
        if let Some(doc) = &self.doc {
            write!(f, "{}", Indent::new(&doc, INDENT))?;
        }

        let mut buf = String::new();
        for variant in self.variants.iter() {
            variant.format_into(&mut buf)?;
            write!(f, "{}", Indent::new(&buf, INDENT))?;
        }

        Ok(())
    }
}

pub struct Variant {
    name: &'static str,
    fields: Fields,
    doc: Option<Doc>,
}

impl Variant {
    pub fn new(name: &'static str, fields: Fields, doc: Option<Doc>) -> Self {
        Self { name, fields, doc }
    }

    fn format_into(&self, f: &mut dyn Write) -> fmt::Result {
        f.write_str(self.name)?;
        self.fields.format_into(f)
    }
}

// ////////////////
// // Algorithms //
// ////////////////
//
// pub fn walk<'a, I>(reflection: Reflection, paths: I) -> Result<Reflection, WalkError>
// where
//     I: IntoIterator<Item = &'a str>,
// {
//     let mut current = reflection;
//     for p in paths {
//         current = current.reflect().walk(p)?;
//     }
//     Ok(current)
// }
//
// #[derive(Debug, Clone, Copy)]
// pub struct WalkError;

///////////////
// Bootstrap //
///////////////

impl Reflect for usize {
    fn reflect() -> Type {
        Type::primitive(
            "usize",
            TypeId::of::<usize>(),
            "An system dependent unsigned integer",
        )
    }
}

/// This is a test!
///
/// Hello world!
#[derive(Reflect)]
pub struct Test {
    /// This field affects this value.
    a: usize,

    /// This field does something else.
    b: usize,
}

#[derive(Reflect)]
struct Wrapper<T> {
    /// Inner
    a: T,
}

pub(crate) mod internal {
    use std::marker::PhantomData;

    pub(crate) trait Reflect {
        fn reflect(&self) -> super::Type;
    }

    pub(crate) struct Wrapper<T>(PhantomData<T>);

    impl<T> Wrapper<T> {
        pub(crate) const INSTANCE: Self = Self::new();

        pub(crate) const fn new() -> Self {
            Self(PhantomData)
        }
    }

    impl<T> Reflect for Wrapper<T>
    where
        T: super::Reflect,
    {
        fn reflect(&self) -> super::Type {
            <T as super::Reflect>::reflect()
        }
    }
}
