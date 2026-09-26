/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    any::TypeId,
    fmt::{self, Write},
};

pub use diskann_benchmark_runner_derive::Reflect;

mod render;
pub mod tree;
pub use tree::Type;

pub trait Reflect: 'static {
    fn ty() -> Type;
    fn format_type_name(f: &mut dyn Write) -> fmt::Result;
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

    pub fn ty(&self) -> Type {
        self.reflection.ty()
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

pub struct TypeName(Reflection);

impl TypeName {
    fn format_type_name(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.reflection.format_type_name(f)
    }
}

impl std::fmt::Debug for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format_type_name(f)
    }
}

impl std::fmt::Display for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format_type_name(f)
    }
}

pub struct Render(Reflection);

impl std::fmt::Display for Render {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut r = render::Renderer::new(f, 3);
        r.render_subject(self.0)
    }
}

///////////////
// Bootstrap //
///////////////

impl<T> Reflect for std::marker::PhantomData<T>
where
    T: Reflect,
{
    fn ty() -> Type {
        Type::primitive(None)
    }

    fn format_type_name(f: &mut dyn Write) -> fmt::Result {
        write!(f, "PhantomData<{}>", Reflection::new::<T>().type_name())
    }
}

macro_rules! primitive {
    ($T:ty, $doc:literal, $type_name:literal) => {
        impl Reflect for $T {
            fn ty() -> Type {
                Type::primitive(Some($doc.into()))
            }

            fn format_type_name(f: &mut dyn Write) -> fmt::Result {
                f.write_str($type_name)
            }
        }
    };
}

primitive!(usize, "A system dependent unsigned integer", "usize");
primitive!(u32, "A 32-bit unsigned integer", "u32");

primitive!(String, "A string", "string");

impl<T> Reflect for Vec<T>
where
    T: Reflect,
{
    fn ty() -> Type {
        Type::sequence::<T>(Some("An ordered collection of elements".into()))
    }

    fn format_type_name(f: &mut dyn Write) -> fmt::Result {
        write!(f, "Vec<{}>", Reflection::new::<T>().type_name())
    }
}

#[derive(Reflect)]
pub struct UnitWithConst<const N: usize> {}

#[derive(Reflect)]
pub struct GenericBoundAdded<T> {
    uses_t: Vec<T>,
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
pub struct TestUnnamed<T>(
    /// Can I document this?
    usize,
    T,
);

/// This is a nother test!
#[derive(Reflect)]
pub struct Test2 {
    /// This field affects this value.
    a: usize,

    /// This field doesn't have any names.
    unnamed: TestUnnamed<u32>,

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
#[derive(Debug, Clone, Copy, Reflect)]
pub enum Metric {
    SquaredL2,
    InnerProduct,
    Cosine,
}

/// An enum with no payloads.
#[derive(Debug, Reflect)]
#[serde(rename_all = "kebab-case")]
pub enum AdjacentEnum {
    SquaredL2,
    /// Let me see if this works
    InnerProduct(u32),

    /// Compute the cosine similarity
    Cosine {
        /// Thos actually doesn't do anything.
        test: String,
    },
}

// impl Reflect for AdjacentEnum {
//     fn ty() -> Type {
//         Type::enum_(
//             EnumRepr::Adjacent {
//                 tag: "enum-type",
//                 content: "content",
//             },
//             [
//                 Variant::new("squared-l2", Fields::Unit, None),
//                 Variant::new(
//                     "inner-product",
//                     Fields::unnamed([UnnamedField::new::<u32>(Some("testing".into()))]),
//                     Some("Inner Product with some payload".into()),
//                 ),
//                 Variant::new(
//                     "cosine",
//                     Fields::named([NamedField::new::<String>("test", None)]),
//                     Some("Cosine Similarity".into()),
//                 ),
//             ],
//             Some("The similarity measure to use".into()),
//         )
//     }
//
//     fn format_type_name(f: &mut dyn Write) -> fmt::Result {
//         f.write_str("Metric")
//     }
// }

//////////////
// Internal //
//////////////

pub(crate) mod internal {
    use std::marker::PhantomData;

    pub(crate) trait Reflect {
        fn ty(&self) -> super::Type;
        fn format_type_name(&self, f: &mut dyn std::fmt::Write) -> std::fmt::Result;
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
        fn ty(&self) -> super::Type {
            <T as super::Reflect>::ty()
        }

        fn format_type_name(&self, f: &mut dyn std::fmt::Write) -> std::fmt::Result {
            <T as super::Reflect>::format_type_name(f)
        }

        fn type_id(&self) -> std::any::TypeId {
            std::any::TypeId::of::<T>()
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::assert_matches;

    #[test]
    fn test_unit() {
        /// A unit struct.
        #[derive(Reflect)]
        struct Unit;

        let r = Reflection::new::<Unit>();
        let ty = r.ty();

        assert_eq!(r.ty().doc().unwrap(), "A unit struct.");
        assert_eq!(r.type_name().to_string(), "Unit");
        assert!(!ty.has_body());
    }

    #[test]
    fn test_unit_const_generic() {
        /// A unit struct.
        #[derive(Reflect)]
        struct Unit<const N: usize>;

        let r = Reflection::new::<Unit<10>>();
        let ty = r.ty();

        assert_eq!(ty.doc().unwrap(), "A unit struct.");
        assert_eq!(r.type_name().to_string(), "Unit<10>");
        assert!(!ty.has_body());
    }

    #[test]
    fn test_unit_const_generic_2() {
        #[derive(Reflect)]
        struct Unit<const N: usize, const M: usize>;

        let r = Reflection::new::<Unit<10, 20>>();
        let ty = r.ty();

        assert!(ty.doc().is_none());
        assert_eq!(r.type_name().to_string(), "Unit<10, 20>");
        assert!(!ty.has_body());
    }

    #[test]
    fn test_empty_tuple_like() {
        /// An empty tuple-like struct.
        #[derive(Reflect)]
        struct Empty();

        let r = Reflection::new::<Empty>();
        let ty = r.ty();

        assert_eq!(ty.doc().unwrap(), "An empty tuple-like struct.");
        assert_eq!(r.type_name().to_string(), "Empty");
        assert!(!ty.has_body());
    }

    #[test]
    fn test_empty_struct_like() {
        /// An empty struct.
        #[derive(Reflect)]
        struct Empty {}

        let r = Reflection::new::<Empty>();
        let ty = r.ty();

        assert_eq!(ty.doc().unwrap(), "An empty struct.");
        assert_eq!(r.type_name().to_string(), "Empty");
        assert!(!ty.has_body());
    }

    #[test]
    fn test_struct() {
        /// A struct with two fields.
        #[expect(unused)]
        #[derive(Reflect)]
        struct Woo {
            /// Foo
            foo: usize,
            /// Bar
            bar: usize,
        }

        let r = Reflection::new::<Woo>();
        let ty = r.ty();
        assert_eq!(ty.doc().unwrap(), "A struct with two fields.");
        assert_eq!(r.type_name().to_string(), "Woo");
        assert!(ty.has_body());

        let f = ty.as_aggregate().unwrap().fields().as_named().unwrap();
        assert_eq!(f.len(), 2);

        assert_eq!(f[0].name, "foo");
        assert_eq!(f[0].doc().unwrap(), "Foo");
        assert_eq!(f[0].field.type_name().to_string(), "usize");

        assert_eq!(f[1].name, "bar");
        assert_eq!(f[1].doc().unwrap(), "Bar");
        assert_eq!(f[1].field.type_name().to_string(), "usize");
    }

    #[test]
    fn test_tuple1() {
        /// A tuple with one field.
        #[expect(unused)]
        #[derive(Reflect)]
        struct Tuple1(
            /// Field 0.
            usize,
        );

        let r = Reflection::new::<Tuple1>();
        let ty = r.ty();
        assert_eq!(ty.doc().unwrap(), "A tuple with one field.");
        assert_eq!(r.type_name().to_string(), "Tuple1");

        assert!(ty.has_body());

        let f = ty.as_aggregate().unwrap().fields().as_unnamed().unwrap();
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].doc().unwrap(), "Field 0.");
        assert_eq!(f[0].field().type_name().to_string(), "usize");
    }

    #[test]
    fn test_tuple2() {
        #[expect(unused)]
        #[derive(Reflect)]
        struct Tuple2<T, U>(
            T,
            /// Field 1.
            Vec<U>,
        );

        let r = Reflection::new::<Tuple2<usize, u32>>();
        let ty = r.ty();
        assert!(ty.doc().is_none());
        assert_eq!(r.type_name().to_string(), "Tuple2<usize, u32>");
        assert!(ty.has_body());

        let f = ty.as_aggregate().unwrap().fields().as_unnamed().unwrap();
        assert_eq!(f.len(), 2);

        assert!(f[0].doc().is_none());
        assert_eq!(f[0].field().type_name().to_string(), "usize");

        assert_eq!(f[1].doc().unwrap(), "Field 1.");
        assert_eq!(f[1].field().type_name().to_string(), "Vec<u32>");
    }

    #[test]
    fn test_empty_enum() {
        /// An empty enum.
        #[derive(Reflect)]
        enum Empty {}

        let r = Reflection::new::<Empty>();
        let ty = r.ty();
        assert_eq!(ty.doc().unwrap(), "An empty enum.");
        assert_eq!(r.type_name().to_string(), "Empty");
        assert!(!ty.has_body());
    }

    #[test]
    fn test_enum_with_generics() {
        /// An enum with generics.
        #[expect(unused)]
        #[derive(Reflect)]
        enum Either<A, B> {
            A(Vec<A>),
            /// It's a bee!
            B(
                /// Buzz buzz
                B,
            ),
        }

        let r = Reflection::new::<Either<u32, String>>();
        let ty = r.ty();
        assert_eq!(ty.doc().unwrap(), "An enum with generics.");
        assert_eq!(r.type_name().to_string(), "Either<u32, string>");
        assert!(ty.has_body());

        let variants = ty.as_enum().unwrap().variants();
        assert_eq!(variants.len(), 2);

        // Variant 0
        assert!(variants[0].doc().is_none());
        assert_eq!(variants[0].name(), "A");
        assert!(variants[0].fields.has_body());

        let f = variants[0].fields.as_unnamed().unwrap();
        assert_eq!(f.len(), 1);
        assert!(f[0].doc().is_none());
        assert_eq!(f[0].field().type_name().to_string(), "Vec<u32>");

        // Variant 1
        assert_eq!(variants[1].doc().unwrap(), "It's a bee!");
        assert_eq!(variants[1].name(), "B");
        assert!(variants[1].fields.has_body());

        let f = variants[1].fields.as_unnamed().unwrap();
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].doc().unwrap(), "Buzz buzz");
        assert_eq!(f[0].field().type_name().to_string(), "string");
    }

    #[test]
    fn test_enum_variants() {
        /// All the enums.
        #[expect(unused)]
        #[derive(Reflect)]
        #[serde(tag = "tag", content = "content")]
        enum All {
            /// A unit variant.
            Unit,
            /// A tuple-like variant.
            Tuple(
                /// Field 0.
                usize,
                String,
            ),
            /// Struct-like.
            Struct {
                foo: usize,
                /// All the bars!
                bar: u32,
            },
        }

        let r = Reflection::new::<All>();
        let ty = r.ty();
        assert_eq!(ty.doc().unwrap(), "All the enums.");
        assert_eq!(r.type_name().to_string(), "All");
        assert!(ty.has_body());

        let variants = ty.as_enum().unwrap().variants();
        assert_eq!(variants.len(), 3);

        // Variant 0
        assert_eq!(variants[0].doc().unwrap(), "A unit variant.");
        assert_eq!(variants[0].name(), "Unit");
        assert!(!variants[0].fields.has_body());
        assert_matches!(variants[0].fields, Fields::Unit);

        // Variant 1
        assert_eq!(variants[1].doc().unwrap(), "A tuple-like variant.");
        assert_eq!(variants[1].name(), "Tuple");
        assert!(variants[1].fields.has_body());

        let f = variants[1].fields.as_unnamed().unwrap();
        assert_eq!(f.len(), 2);

        assert_eq!(f[0].doc().unwrap(), "Field 0.");
        assert_eq!(f[0].field.type_name().to_string(), "usize");

        assert!(f[1].doc().is_none());
        assert_eq!(f[1].field.type_name().to_string(), "string");

        // Variant 2
        assert_eq!(variants[2].name(), "Struct");
        assert_eq!(variants[2].doc().unwrap(), "Struct-like.");
        let f = variants[2].fields.as_named().unwrap();
        assert_eq!(f.len(), 2);

        assert_eq!(f[0].name, "foo");
        assert!(f[0].doc().is_none());
        assert_eq!(f[0].field.type_name().to_string(), "usize");

        assert_eq!(f[1].name, "bar");
        assert_eq!(f[1].doc().unwrap(), "All the bars!");
        assert_eq!(f[1].field.type_name().to_string(), "u32");
    }
}
