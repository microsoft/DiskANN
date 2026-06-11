/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::borrow::Cow;

pub use diskann_benchmark_runner_derive::Reflect;

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

pub enum Type {
    Primitive(Primitive),
    Aggregate(Aggregate),
    Enum(Enum),
}

impl Type {
    pub fn primitive(type_name: &'static str, doc: &'static str) -> Self {
        Self::from(Primitive::new(type_name, doc))
    }

    pub fn aggregate(type_name: &'static str, fields: Fields, doc: Option<Doc>) -> Self {
        Self::from(Aggregate::new(type_name, fields, doc))
    }

    pub fn enum_<Itr>(type_name: &'static str, variants: Itr, doc: Option<Doc>) -> Self
    where
        Itr: IntoIterator<Item = Variant>,
    {
        Self::from(Enum::new(type_name, variants, doc))
    }

    // pub(crate) fn walk(&self, name: &str) -> Result<Reflection, WalkError> {
    //     match self {
    //         Self::Primitive(_) => Err(WalkError),
    //         Self::Aggregate(aggregate) => aggregate.walk(name),
    //         Self::Enum(variant) => variant.walk(name),
    //     }
    // }
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
    doc: &'static str,
}

impl Primitive {
    pub fn new(type_name: &'static str, doc: &'static str) -> Self {
        Self { type_name, doc }
    }
}

pub struct Aggregate {
    type_name: &'static str,
    fields: Fields,
    doc: Option<Doc>,
}

impl Aggregate {
    pub fn new(type_name: &'static str, fields: Fields, doc: Option<Doc>) -> Self {
        Self {
            type_name,
            fields,
            doc,
        }
    }

    // fn walk(&self, field: &str) -> Result<Reflection, WalkError> {
    //     match self.fields.iter().find(|f| f.name == field) {
    //         Some(f) => Ok(f.field),
    //         None => Err(WalkError),
    //     }
    // }
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

pub struct Enum {
    type_name: &'static str,
    variants: Vec<Variant>,
    doc: Option<Doc>,
}

impl Enum {
    pub fn new<Itr>(type_name: &'static str, variants: Itr, doc: Option<Doc>) -> Self
    where
        Itr: IntoIterator<Item = Variant>,
    {
        Self {
            type_name,
            variants: variants.into_iter().collect(),
            doc,
        }
    }

    // fn walk(&self, variant: &str) -> Result<Reflection, WalkError> {
    //     match self.variants.iter().find(|(v, _)| *v == variant) {
    //         Some(v) => Ok(v.1.variant),
    //         None => Err(WalkError),
    //     }
    // }
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

////////////////
// Algorithms //
////////////////

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
        Type::primitive("usize", "An system dependent unsigned integer")
    }
}

/// This is a test!
///
/// Hello world!
#[derive(Reflect)]
struct Test {
    /// This field affects this value.
    a: usize,

    /// This field does something else.
    b: usize,
}

// impl Reflect for Test {
//     fn reflect() -> Type {
//         Type::aggregate(
//             "Test",
//             [
//                 Field::new::<usize>("a", Some("this fields does a thing".into())),
//                 Field::new::<usize>("b", Some("this fields does another thing".into())),
//             ],
//             None,
//         )
//     }
// }

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
