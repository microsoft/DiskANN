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

use crate::utils::fmt::Quote;

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

    pub fn type_id(&self) -> TypeId {
        self.reflection.type_id()
    }

    pub fn render(&self) -> Render {
        Render(*self)
    }
}

impl fmt::Debug for Reflection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reflection")
            .field("type_name", &self.type_name())
            .finish_non_exhaustive()
    }
}

pub type Doc = Cow<'static, str>;

pub struct TypeName(Reflection);

impl TypeName {
    fn format_into(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.reflection.type_name(f)
    }
}

impl std::fmt::Debug for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format_into(f)
    }
}

impl std::fmt::Display for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format_into(f)
    }
}

pub struct Render(Reflection);

impl std::fmt::Display for Render {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut r = Renderer::new(f, 3);
        r.render_subject(self.0)
    }
}

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

    fn has_body(&self) -> bool {
        match self {
            Self::Primitive(_) => false,
            Self::Aggregate(a) => a.has_body(),
            Self::Enum(e) => e.has_body(),
            Self::Sequence(_) => true,
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

    fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }

    fn has_body(&self) -> bool {
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
    fn has_body(&self) -> bool {
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
            field: reflect::<T>(),
            doc,
        }
    }

    fn doc(&self) -> Option<&str> {
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
            field: reflect::<T>(),
            doc,
        }
    }

    fn doc(&self) -> Option<&str> {
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

    pub fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }

    fn has_body(&self) -> bool {
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
    ///     "world",
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
    ///       "world",
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

    fn doc(&self) -> Option<&str> {
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
            element: reflect::<T>(),
            doc,
        }
    }

    pub fn doc(&self) -> Option<&str> {
        self.doc.as_deref()
    }
}

//////////////
// Renderer //
//////////////

#[derive(Debug)]
struct Tagged {
    ty: Type,
    reflection: Reflection,
}

impl Tagged {
    fn new(reflection: Reflection) -> Self {
        Self {
            ty: reflection.reflect(),
            reflection,
        }
    }

    fn ty(&self) -> &Type {
        &self.ty
    }

    fn reflection(&self) -> Reflection {
        self.reflection
    }
}

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

    fn maybe_indent<F, R>(&mut self, indent: bool, f: F) -> Result<R, fmt::Error>
    where
        F: FnOnce(&mut Self) -> Result<R, fmt::Error>,
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

    fn indent<F, R>(&mut self, f: F) -> Result<R, fmt::Error>
    where
        F: FnOnce(&mut Self) -> Result<R, fmt::Error>,
    {
        self.maybe_indent(true, f)
    }

    fn next_with(
        &mut self,
        pre: impl FnOnce(&mut Self) -> fmt::Result,
        body: impl FnOnce(&mut Self) -> fmt::Result,
        post: impl FnOnce(&mut Self) -> fmt::Result,
    ) -> fmt::Result {
        if self.at_bottom() {
            Ok(())
        } else {
            pre(self)?;
            self.depth += 1;
            let result = self.indent(body);
            self.depth -= 1;
            post(self)?;
            result
        }
    }

    fn next<F>(&mut self, f: F) -> fmt::Result
    where
        F: FnOnce(&mut Self) -> fmt::Result,
    {
        self.next_with(|_| Ok(()), f, |_| Ok(()))
    }

    fn render_doc(&mut self, s: Option<&str>) -> Result<bool, fmt::Error> {
        if let Some(s) = s {
            let mut rendered = false;
            for ln in s.lines() {
                if ln.is_empty() {
                    self.blank()?;
                } else {
                    self.line(ln)?;
                }

                rendered = true;
            }
            Ok(rendered)
        } else {
            Ok(false)
        }
    }

    /// Return `true` if a nested type will be rendered.
    fn will_render(&self, ty: &Type) -> bool {
        !self.at_bottom() && ty.has_body()
    }

    //-------//
    // Types //
    //-------//

    fn render_subject(&mut self, reflection: Reflection) -> fmt::Result {
        self.line(reflection.type_name())?;
        self.indent(|r| {
            let tagged = Tagged::new(reflection);
            let wrote_doc = r.render_doc(tagged.ty().doc())?;

            if wrote_doc && r.will_render(tagged.ty()) {
                r.blank()?;
            }

            r.render_body(&tagged)
        })
    }

    fn render_body(&mut self, tagged: &Tagged) -> fmt::Result {
        match tagged.ty() {
            Type::Primitive(_) => Ok(()),
            Type::Aggregate(aggregate) => self.render_aggregate(&aggregate),
            Type::Enum(enum_) => self.render_enum(enum_),
            Type::Sequence(sequence) => self.render_sequence(&sequence),
        }
    }

    fn render_aggregate(&mut self, aggregate: &Aggregate) -> fmt::Result {
        self.render_fields(&aggregate.fields)
    }

    fn render_enum(&mut self, enum_: &Enum) -> fmt::Result {
        match enum_.repr {
            EnumRepr::External => self.line("Representation: string")?,
            EnumRepr::Internal { tag } => {
                self.line(format_args!("Discriminant field: {}", Quote(tag)))?
            }
            EnumRepr::Adjacent { tag, content } => {
                self.line(format_args!("Discriminant field: {}", Quote(tag)))?;
                self.line(format_args!("Content field: {}", Quote(content)))?;
            }
        }

        self.blank()?;
        self.line("Options:")?;
        self.indent(|r| {
            let mut first = true;
            for variant in enum_.variants.iter() {
                if !first {
                    r.blank()?;
                }

                r.render_variant(variant)?;
                first = false;
            }

            Ok(())
        })
    }

    fn render_sequence(&mut self, sequence: &Sequence) -> fmt::Result {
        self.next_with(
            |r| r.line(format_args!("Elements: {}", sequence.element.type_name())),
            |r| r.render_body(&Tagged::new(sequence.element)),
            |_| Ok(()),
        )
    }

    //--------//
    // Fields //
    //--------//

    fn render_fields(&mut self, fields: &Fields) -> fmt::Result {
        match fields {
            Fields::Named(named) => {
                let mut first = true;
                for field in named.iter() {
                    if !first {
                        self.blank()?;
                    }
                    self.render_named_field(field)?;
                    first = false;
                }
            }
            Fields::Unnamed(unnamed) => {
                for (i, field) in unnamed.iter().enumerate() {
                    if i != 0 {
                        self.blank()?;
                    }
                    self.render_unnamed_field(i, field)?;
                }
            }
            Fields::Unit => {}
        }

        Ok(())
    }

    fn render_named_field(&mut self, field: &NamedField) -> fmt::Result {
        let tagged = Tagged::new(field.field);
        let will_render_body = self.will_render(tagged.ty());

        self.line(format_args!(
            "{}: {}",
            Quote(field.name),
            tagged.reflection().type_name()
        ))?;

        let rendered_doc = self.indent(|r| r.render_doc(field.doc()))?;
        if rendered_doc && will_render_body {
            self.blank()?;
        }

        if will_render_body {
            self.next(|r| r.render_body(&tagged))?;
        }
        Ok(())
    }

    fn render_unnamed_field(&mut self, index: usize, field: &UnnamedField) -> fmt::Result {
        let tagged = Tagged::new(field.field);
        let will_render_body = self.will_render(tagged.ty());

        self.line(format_args!(
            "{}: {}",
            index,
            tagged.reflection().type_name()
        ))?;

        let rendered_doc = self.indent(|r| r.render_doc(field.doc()))?;
        if rendered_doc && will_render_body {
            self.blank()?;
        }

        if will_render_body {
            self.next(|r| r.render_body(&tagged))?;
        }
        Ok(())
    }

    //---------//
    // Variant //
    //---------//

    fn render_variant(&mut self, variant: &Variant) -> fmt::Result {
        self.line(Quote(variant.name))?;

        self.indent(|r| {
            let rendered_doc = r.render_doc(variant.doc())?;

            if rendered_doc && variant.fields.has_body() {
                r.blank()?;
            }

            r.render_fields(&variant.fields)
        })
    }
}

///////////////
// Bootstrap //
///////////////

macro_rules! primitive {
    ($T:ty, $doc:literal, $type_name:literal) => {
        impl Reflect for $T {
            fn reflect() -> Type {
                Type::primitive(Some($doc.into()))
            }

            fn type_name(f: &mut dyn Write) -> fmt::Result {
                f.write_str($type_name)
            }
        }
    }
}

primitive!(usize, "A system dependent unsigned integer", "usize");
primitive!(u32, "A 32-bit unsigned integer", "u32");

primitive!(String, "A string", "string");

impl<T> Reflect for Vec<T>
where
    T: Reflect,
{
    fn reflect() -> Type {
        Type::sequence::<T>(Some("An ordered collection of elements".into()))
    }

    fn type_name(f: &mut dyn Write) -> fmt::Result {
        write!(f, "Vec<{}>", reflect::<T>().type_name())
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

/// This is a nother test!
#[derive(Reflect)]
pub struct Test2 {
    /// This field affects this value.
    a: usize,

    other: Test,

    /// How are we going to compute distances?
    metric: AdjacentEnum,

    /// These control a bunch of parameters.
    seq: Vec<Test>,
}

#[derive(Reflect)]
struct Wrapper<T> {
    /// Inner
    a: T,
}

/// An enum with no payloads.
#[derive(Debug, Clone, Copy)]
pub enum Metric {
    SquaredL2,
    InnerProduct,
    Cosine,
}

impl Reflect for Metric {
    fn reflect() -> Type {
        Type::enum_(
            EnumRepr::External,
            [
                Variant::new("squared-l2", Fields::Unit, Some("Squared Euclidean".into())),
                Variant::new("inner-product", Fields::Unit, Some("Inner Product".into())),
                Variant::new("cosine", Fields::Unit, Some("Cosine Similarity".into())),
            ],
            Some("The similarity measure to use".into()),
        )
    }

    fn type_name(f: &mut dyn Write) -> fmt::Result {
        f.write_str("Metric")
    }
}

/// An enum with no payloads.
#[derive(Debug)]
pub enum AdjacentEnum {
    SquaredL2,
    InnerProduct(u32),
    Cosine { test: String },
}

impl Reflect for AdjacentEnum {
    fn reflect() -> Type {
        Type::enum_(
            EnumRepr::Adjacent {
                tag: "enum-type",
                content: "content",
            },
            [
                Variant::new("squared-l2", Fields::Unit, None),
                Variant::new(
                    "inner-product",
                    Fields::unnamed([UnnamedField::new::<u32>(Some("testing".into()))]),
                    Some("Inner Product with some payload".into()),
                ),
                Variant::new(
                    "cosine",
                    Fields::named([NamedField::new::<String>("test", None)]),
                    Some("Cosine Similarity".into()),
                ),
            ],
            Some("The similarity measure to use".into()),
        )
    }

    fn type_name(f: &mut dyn Write) -> fmt::Result {
        f.write_str("Metric")
    }
}

//////////////
// Internal //
//////////////

pub(crate) mod internal {
    use std::marker::PhantomData;

    pub(crate) trait Reflect {
        fn reflect(&self) -> super::Type;
        fn type_name(&self, f: &mut dyn std::fmt::Write) -> std::fmt::Result;
        fn type_id(&self) -> std::any::TypeId;
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

        fn type_id(&self) -> std::any::TypeId {
            std::any::TypeId::of::<T>()
        }
    }
}
