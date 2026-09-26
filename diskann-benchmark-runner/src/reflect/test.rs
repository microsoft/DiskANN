/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::{
    tree::{Fields, Type},
    Reflect, Reflection,
};
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
fn test_unit_rename() {
    /// A unit struct.
    #[derive(Reflect)]
    #[reflect(type_name = "something-else")]
    struct Unit;

    let r = Reflection::new::<Unit>();
    let ty = r.ty();

    assert_eq!(r.ty().doc().unwrap(), "A unit struct.");
    assert_eq!(r.type_name().to_string(), "something-else");
    assert!(!ty.has_body());
}

#[test]
fn test_unit_prefix() {
    /// A unit struct.
    #[derive(Reflect)]
    #[reflect(prefix = "module::")]
    struct Unit;

    let r = Reflection::new::<Unit>();
    let ty = r.ty();

    assert_eq!(r.ty().doc().unwrap(), "A unit struct.");
    assert_eq!(r.type_name().to_string(), "module::Unit");
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
fn test_unit_const_generic_prefix() {
    /// A unit struct.
    #[derive(Reflect)]
    #[reflect(prefix = "module::")]
    struct Unit<const N: usize>;

    let r = Reflection::new::<Unit<10>>();
    let ty = r.ty();

    assert_eq!(ty.doc().unwrap(), "A unit struct.");
    assert_eq!(r.type_name().to_string(), "module::Unit<10>");
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

    assert_eq!(f[0].name(), "foo");
    assert_eq!(f[0].doc().unwrap(), "Foo");
    assert_eq!(f[0].field().type_name().to_string(), "usize");

    assert_eq!(f[1].name(), "bar");
    assert_eq!(f[1].doc().unwrap(), "Bar");
    assert_eq!(f[1].field().type_name().to_string(), "usize");
}

#[test]
fn test_struct_rename() {
    /// A struct with three fields.
    #[expect(unused)]
    #[derive(Reflect)]
    #[serde(rename_all = "kebab-case")]
    struct Woo {
        /// Foo
        foo_bar: usize,
        /// Bar
        baz: usize,

        #[serde(rename = "oops")]
        biz: usize,
    }

    let r = Reflection::new::<Woo>();
    let ty = r.ty();
    assert_eq!(ty.doc().unwrap(), "A struct with three fields.");
    assert_eq!(r.type_name().to_string(), "Woo");
    assert!(ty.has_body());

    let f = ty.as_aggregate().unwrap().fields().as_named().unwrap();
    assert_eq!(f.len(), 3);

    assert_eq!(
        f[0].name(),
        "foo-bar",
        "fields should be renamed by `rename_all`"
    );
    assert_eq!(f[0].doc().unwrap(), "Foo");
    assert_eq!(f[0].field().type_name().to_string(), "usize");

    assert_eq!(f[1].name(), "baz");
    assert_eq!(f[1].doc().unwrap(), "Bar");
    assert_eq!(f[1].field().type_name().to_string(), "usize");

    assert_eq!(
        f[2].name(),
        "oops",
        "explicit rename should take precedence"
    );
    assert!(f[2].doc().is_none());
    assert_eq!(f[2].field().type_name().to_string(), "usize");
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
    assert!(variants[0].fields().has_body());

    let f = variants[0].fields().as_unnamed().unwrap();
    assert_eq!(f.len(), 1);
    assert!(f[0].doc().is_none());
    assert_eq!(f[0].field().type_name().to_string(), "Vec<u32>");

    // Variant 1
    assert_eq!(variants[1].doc().unwrap(), "It's a bee!");
    assert_eq!(variants[1].name(), "B");
    assert!(variants[1].fields().has_body());

    let f = variants[1].fields().as_unnamed().unwrap();
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
    assert!(!variants[0].fields().has_body());
    assert_matches!(variants[0].fields(), Fields::Unit);

    // Variant 1
    assert_eq!(variants[1].doc().unwrap(), "A tuple-like variant.");
    assert_eq!(variants[1].name(), "Tuple");
    assert!(variants[1].fields().has_body());

    let f = variants[1].fields().as_unnamed().unwrap();
    assert_eq!(f.len(), 2);

    assert_eq!(f[0].doc().unwrap(), "Field 0.");
    assert_eq!(f[0].field().type_name().to_string(), "usize");

    assert!(f[1].doc().is_none());
    assert_eq!(f[1].field().type_name().to_string(), "string");

    // Variant 2
    assert_eq!(variants[2].name(), "Struct");
    assert_eq!(variants[2].doc().unwrap(), "Struct-like.");
    let f = variants[2].fields().as_named().unwrap();
    assert_eq!(f.len(), 2);

    assert_eq!(f[0].name(), "foo");
    assert!(f[0].doc().is_none());
    assert_eq!(f[0].field().type_name().to_string(), "usize");

    assert_eq!(f[1].name(), "bar");
    assert_eq!(f[1].doc().unwrap(), "All the bars!");
    assert_eq!(f[1].field().type_name().to_string(), "u32");
}
