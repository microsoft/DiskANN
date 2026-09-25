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
    fn type_name(f: &mut dyn Write) -> fmt::Result;
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

    pub fn type_name(&self) -> TypeName {
        TypeName(*self)
    }

    pub fn render(&self) -> Render {
        Render(*self)
    }
}

pub type Doc = Cow<'static, str>;

pub struct TypeName(Reflection);

impl std::fmt::Display for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.reflection.type_name(f)
    }
}

pub struct Render(Reflection);

impl std::fmt::Display for Render {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut r = Renderer::new(f, 2);
        r.render_reflection(self.0)
    }
}

pub enum Type {
    Primitive(Primitive),
    Aggregate(Aggregate),
    Enum(Enum),
    Sequence(Sequence),
}

impl Type {
    pub fn primitive(type_id: TypeId, doc: Option<Doc>) -> Self {
        Self::from(Primitive::new(type_id, doc))
    }

    pub fn aggregate(type_id: TypeId, fields: Fields, doc: Option<Doc>) -> Self {
        Self::from(Aggregate::new(type_id, fields, doc))
    }

    pub fn enum_(
        type_id: TypeId,
        variants: impl IntoIterator<Item = Variant>,
        doc: Option<Doc>,
    ) -> Self {
        Self::from(Enum::new(type_id, variants, doc))
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

pub struct Primitive {
    type_id: TypeId,
    doc: Option<Doc>,
}

impl Primitive {
    pub fn new(type_id: TypeId, doc: Option<Doc>) -> Self {
        Self { type_id, doc }
    }

    fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }
}

//-----------//
// Aggregate //
//-----------//

pub struct Aggregate {
    type_id: TypeId,
    fields: Fields,
    doc: Option<Doc>,
}

impl Aggregate {
    pub fn new(type_id: TypeId, fields: Fields, doc: Option<Doc>) -> Self {
        Self {
            type_id,
            fields,
            doc,
        }
    }

    fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }
}

pub enum Fields {
    Named(Vec<NamedField>),
    Unnamed(Vec<UnnamedField>),
    Unit,
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

    fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
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
}

//------//
// Enum //
//------//

pub struct Enum {
    type_id: TypeId,
    variants: Vec<Variant>,
    doc: Option<Doc>,
}

impl Enum {
    pub fn new(
        type_id: TypeId,
        variants: impl IntoIterator<Item = Variant>,
        doc: Option<Doc>,
    ) -> Self {
        Self {
            type_id: type_id,
            variants: variants.into_iter().collect(),
            doc,
        }
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
}

//----------//
// Sequence //
//----------//

pub struct Sequence {
    type_id: TypeId,
    element: Reflection,
    doc: Option<Doc>,
}

impl Sequence {
    pub fn new<T>(type_id: TypeId, doc: Option<Doc>) -> Self
    where
        T: Reflect,
    {
        Self {
            type_id,
            element: reflect::<T>(),
            doc,
        }
    }
}

//////////////
// Renderer //
//////////////

struct Renderer<'a> {
    output: &'a mut dyn Write,
    indent: usize,
    depth: usize,
    max_depth: usize,
}

impl<'a> Renderer<'a> {
    fn new(output: &'a mut dyn Write, max_depth: usize) -> Self {
        Self {
            output,
            indent: 0,
            depth: 0,
            max_depth,
        }
    }

    fn at_top(&self) -> bool {
        self.indent == 0
    }

    fn at_bottom(&self) -> bool {
        self.depth == self.max_depth
    }

    fn line<D>(&mut self, display: D) -> fmt::Result
    where
        D: std::fmt::Display,
    {
        let indent = INDENT * self.indent;
        write!(self.output, "{: >indent$}{}\n", "", display)
    }

    fn blank(&mut self) -> fmt::Result {
        self.output.write_char('\n')
    }

    fn maybe_indent<F>(&mut self, indent: bool, f: F) -> fmt::Result
    where
        F: FnOnce(&mut Self) -> fmt::Result,
    {
        if indent {
            self.indent += 1;
        }
        let result = f(self);
        if indent {
            self.indent -= 1;
        }
        result
    }

    fn indent<F>(&mut self, f: F) -> fmt::Result
    where
        F: FnOnce(&mut Self) -> fmt::Result,
    {
        self.maybe_indent(true, f)
    }

    fn next<F>(&mut self, f: F) -> fmt::Result
    where
        F: FnOnce(&mut Self) -> fmt::Result,
    {
        if self.at_bottom() {
            Ok(())
        } else {
            self.depth += 1;
            let result = self.indent(f);
            self.depth -= 1;
            result
        }
    }

    fn render_doc(&mut self, s: Option<&str>) -> fmt::Result {
        if let Some(s) = s {
            for ln in s.lines() {
                if ln.is_empty() {
                    self.blank()?;
                } else {
                    self.line(ln)?;
                }
            }
        }

        Ok(())
    }

    //-------//
    // Types //
    //-------//

    fn render_reflection(&mut self, reflection: Reflection) -> fmt::Result {
        // Render the type name if this is the first item in the stack.
        if self.at_top() {
            self.line(reflection.type_name())?;
        }

        match reflection.reflect() {
            Type::Primitive(primitive) => self.render_primitive(&primitive),
            Type::Aggregate(aggregate) => self.render_aggregate(&aggregate),
            Type::Enum(enum_) => todo!(),
            Type::Sequence(sequence) => todo!(),
        }
    }

    fn render_primitive(&mut self, primitive: &Primitive) -> fmt::Result {
        if self.at_top() {
            self.indent(|r| r.render_doc(primitive.doc()))?;
        }

        Ok(())
    }

    fn render_aggregate(&mut self, aggregate: &Aggregate) -> fmt::Result {
        self.maybe_indent(self.at_top(), |r| {
            r.render_doc(aggregate.doc())?;
            r.render_fields(&aggregate.fields)
        })
    }

    //--------//
    // Fields //
    //--------//

    fn render_fields(&mut self, fields: &Fields) -> fmt::Result {
        match fields {
            Fields::Named(named) => {
                for field in named.iter() {
                    self.render_named_field(field)?;
                }
            }
            Fields::Unnamed(_) => todo!(),
            Fields::Unit => todo!(),
        }

        Ok(())
    }

    fn render_named_field(&mut self, field: &NamedField) -> fmt::Result {
        let f = field.field;

        self.blank()?;
        self.line(format_args!("{}: {}", Quote(field.name), TypeName(f)))?;
        self.indent(|r| r.render_doc(field.doc()))?;
        self.next(|r| r.render_reflection(f))
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
            TypeId::of::<usize>(),
            Some("A system dependent unsigned integer".into()),
        )
    }

    fn type_name(f: &mut dyn Write) -> fmt::Result {
        f.write_str("usize")
    }
}

// impl<T> Reflect for Vec<T>
// where
//     T: Reflect,
// {
//     fn reflect() -> Type {
//
//     }
// }

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

/// This is a nother test!
#[derive(Reflect)]
pub struct Test2 {
    /// This field affects this value.
    a: usize,

    other: Test,
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
        fn type_name(&self, f: &mut dyn std::fmt::Write) -> std::fmt::Result;
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

        fn type_name(&self, f: &mut dyn std::fmt::Write) -> std::fmt::Result {
            <T as super::Reflect>::type_name(f)
        }
    }
}
