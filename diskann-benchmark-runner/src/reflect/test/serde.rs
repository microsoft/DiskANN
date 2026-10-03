/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`Reflect`] macro serde-compatibility tests.
//!
//! Serde does allow some differences between serialization and deserialization. We don't
//! care to support such fanciness in our benchmark inputs and outputs, especially since we
//! need example JSONs to be round-trippable back to their serialized representations.
//!
//! As such, the main entry points ([`check_struct`] and [`check_enums`]) just take [`Serialize`]
//! bounds. We do not check round-trippability here. That's the user's problem.

use std::{assert_matches, borrow::Cow};

use hashbrown::HashSet;
use serde::Serialize;
use serde_json::Value;

use crate::reflect::{Reflect, Reflection, Type, tree};

/// Check that the serialized representation of `s` in JSON matches the [`Reflection`]
/// generated for this type.
fn check_struct<T>(s: T)
where
    T: Serialize + Reflect,
{
    let r = Reflection::new::<T>();
    let val = serde_json::to_value(s).unwrap();
    check_reflection(
        r,
        &val,
        Context::new(&val, &val, format_args!("struct: {}", r.type_name())),
    );
}

/// Check that the serialized representations of all the variants of the enum `T` match the
/// [`Reflection`] for `T`.
///
/// Note that `examples` is expected to contain examples of all variants in declaration order
/// and this function will panic if this is not the case.
fn check_enums<T>(examples: &[T], ctx: std::fmt::Arguments<'_>)
where
    T: Serialize + Reflect,
{
    let r = Reflection::new::<T>();
    let serialized: Vec<_> = examples
        .iter()
        .map(|e| serde_json::to_value(e).unwrap())
        .collect();

    let Type::Enum(e) = r.ty() else {
        panic!("expected an enum for type {}", r.type_name());
    };

    check_enum_variants(&e, &serialized, ctx);
}

////////////////////
// Implementation //
////////////////////

/// A context for displaying where we are in the type tree.
///
/// Contains the top level JSON we're working, the current JSON, and a stack of
/// [`std::fmt::Arguments`] that describe the sequence of operations that led us into a mess.
///
/// Use the [`context`] macro for creating nested contexts.
#[derive(Debug, Clone, Copy)]
struct Context<'a> {
    top: &'a Value,
    current: &'a Value,
    stack: std::fmt::Arguments<'a>,
}

impl<'a> Context<'a> {
    fn new(top: &'a Value, current: &'a Value, stack: std::fmt::Arguments<'a>) -> Self {
        Self {
            top,
            current,
            stack,
        }
    }
}

impl std::fmt::Display for Context<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Full JSON:\n\n{}\n\nCurrent JSON:\n\n{}\n\nContext: {}",
            serde_json::to_string_pretty(self.top).unwrap(),
            serde_json::to_string_pretty(self.current).unwrap(),
            self.stack
        )
    }
}

macro_rules! context {
    ($current:ident, $next:ident, $fmt:expr) => {
        Context {
            top: $current.top,
            current: $next,
            stack: format_args!(
                "{}\n- {}",
                $current.stack,
                format_args!($fmt),
            ),
        }
    };
    ($current:ident, $next:ident, $fmt:expr, $($args:tt)*) => {
        Context {
            top: $current.top,
            current: $next,
            stack: format_args!(
                "{}\n- {}",
                $current.stack,
                format_args!($fmt, $($args)*),
            ),
        }
    };
}

fn check_reflection(r: Reflection, s: &Value, ctx: Context<'_>) {
    check_type(
        &r.ty(),
        s,
        context!(ctx, s, "type_name = {}", r.type_name()),
    );
}

fn check_type(ty: &Type, s: &Value, ctx: Context<'_>) {
    match ty {
        Type::Primitive(p) => check_primitive(p, s, context!(ctx, s, "primitive")),
        Type::Aggregate(a) => check_fields(a.fields(), s, context!(ctx, s, "aggregate")),
        Type::Enum(e) => check_enum(e, s, context!(ctx, s, "enum")),
        Type::Sequence(seq) => check_sequence(seq, s, context!(ctx, s, "sequence")),
        Type::Optional(opt) => check_optional(opt, s, context!(ctx, s, "optional")),
    }
}

fn check_primitive(p: &tree::Primitive, s: &Value, ctx: Context<'_>) {
    // Verify that `s` has the correct `PrimitiveKind`.
    match p.kind() {
        tree::PrimitiveKind::Null => {
            if !s.is_null() {
                panic!("expected `Null`\n\n{}", ctx);
            }
        }
        tree::PrimitiveKind::Boolean => {
            if !s.is_boolean() {
                panic!("expected `Bool`\n\n{}", ctx);
            }
        }
        tree::PrimitiveKind::Number => {
            if !s.is_number() {
                panic!("expected `Number`\n\n{}", ctx);
            }
        }
        tree::PrimitiveKind::String => {
            if !s.is_string() {
                panic!("expected `String`\n\n{}", ctx);
            }
        }
    }
}

fn check_fields(fields: &tree::Fields, s: &Value, ctx: Context<'_>) {
    match fields {
        tree::Fields::Named(named_fields) => {
            // All names must be unique.
            assert_all_unique(named_fields.iter().map(|f| f.name()), ctx);
            let map = value_as_map(s, ctx);
            assert_eq!(
                map.len(),
                named_fields.len(),
                "mismatch in number of fields\n\n{}",
                ctx,
            );

            for f in named_fields {
                let val = match map.get(f.name()) {
                    Some(val) => val,
                    None => panic!("Could not match field name \"{}\"\n\n{}", f.name(), ctx),
                };

                check_reflection(
                    f.field(),
                    &val,
                    context!(ctx, val, "named field \"{}\"", f.name()),
                );
            }
        }
        tree::Fields::Unnamed(unnamed_fields) => {
            let array = value_as_array(s, ctx);
            assert_eq!(
                array.len(),
                unnamed_fields.len(),
                "mismatch in number of fields\n\n{}",
                ctx
            );

            for (i, (f, val)) in std::iter::zip(unnamed_fields.iter(), array.iter()).enumerate() {
                check_reflection(f.field(), val, context!(ctx, val, "unnamed field {}", i));
            }
        }
        tree::Fields::NewType(newtype) => {
            check_reflection(newtype.field(), s, context!(ctx, s, "newtype"))
        }
        tree::Fields::Unit => {
            assert_matches!(s, Value::Null, "expected a unit\n{}", ctx,);
        }
    }
}

fn check_enum_variants(e: &tree::Enum, s: &[Value], ctx: std::fmt::Arguments<'_>) {
    let num_variants = e.variants().len();
    assert_eq!(num_variants, s.len(), "expected one example per variant");

    for (i, (variant, example)) in std::iter::zip(e.variants().iter(), s.iter()).enumerate() {
        let ctx = Context::new(example, example, ctx);

        // Verify that the extracted tag matches.
        let (tag, _content) = extract_tag_and_content(e.repr(), example, ctx);
        assert_eq!(variant.name(), tag, "{}", ctx);

        // Use `check_enum`.
        //
        // This is a little wasteful because we've already extracted the tag and the enum,
        // but this is test-only code so we can afford it.
        check_enum(
            e,
            example,
            context!(
                ctx,
                example,
                "variant = {} ({} of {})",
                variant.name(),
                i + 1,
                num_variants
            ),
        );
    }
}

/// To summarize. For a struct like
/// ```
/// enum Foo {
///     Baz {
///         a: usize,
///     },
/// }
/// ```
///
/// External looks like
/// ```json
/// {
///   "baz": {
///     "a": 10
///   }
/// }
/// ```
/// Internal (say "tag = "mytag") looks like
/// ```json
/// {
///   "mytag": "baz",
///   "a": 10,
/// }
/// ```
/// Adjacency (say "tag = "mytag", "content" = "mycontent") looks like
/// ```json
/// {
///   "mytag": "baz",
///   "mycontent": {
///     "a": 10
///   }
/// }
/// ```
fn extract_tag_and_content<'a>(
    repr: &tree::EnumRepr,
    s: &'a Value,
    ctx: Context<'_>,
) -> (&'a str, Option<std::borrow::Cow<'a, Value>>) {
    match repr {
        // When using external tagging, the associated value can look like one of two things:
        //
        // 1. A raw string. This is only valid if the associated payload is a unit type.
        // 2. A map consisting of a single key-value pair. The `key` is the enum tag.
        tree::EnumRepr::External => match s {
            Value::String(s) => (s, None),
            Value::Object(m) => {
                assert_eq!(
                    m.len(),
                    1,
                    "externally tagged enums should only have a single key-value pair\n\n{}",
                    ctx
                );

                let kv = m.iter().next().unwrap();
                (&kv.0, Some(Cow::Borrowed(&kv.1)))
            }
            _ => panic!("invalid representation\n\n{}", ctx),
        },

        // For internally tagged enums - we remove the tag after retrieval.
        //
        // If the remaining dictionary is empty, then we change it to `None`.
        // This is technically a little ambiguous between unit variants and empty struct
        // variants (e.g. `Enum::Unit` and `Enum::Empty {}`.
        //
        // However, we teach the latter case to expect empty results with internal tagging
        // for purposes of the check.
        tree::EnumRepr::Internal { tag } => {
            let map = value_as_map(s, ctx);

            let t = match map.get(*tag) {
                Some(value) => value_as_str(value, context!(ctx, value, "tag extraction")),
                None => panic!("Could not find tag \"{}\"\n\n{}", tag, ctx),
            };

            // Delete the tag field to reuse the rest of the checking infrastructure.
            let mut map = map.clone();
            map.remove(*tag);
            (t, Some(Cow::Owned(Value::Object(map))))
        }

        // For adjacent tagging, the "content" field is omitted when the corresponding
        // variant is a unit variant.
        tree::EnumRepr::Adjacent { tag, content } => {
            let outer = value_as_map(s, ctx);
            assert!(outer.len() == 1 || outer.len() == 2);

            let t = match outer.get(*tag) {
                Some(value) => value_as_str(value, context!(ctx, value, "tag extraction")),
                None => panic!("expected tag field \"{}\"", tag),
            };

            if outer.len() == 1 {
                (t, None)
            } else {
                let c = match outer.get(*content) {
                    Some(c) => c,
                    None => panic!("expected content field \"{}\"", content),
                };

                (t, Some(Cow::Borrowed(c)))
            }
        }
    }
}

fn check_enum(e: &tree::Enum, s: &Value, ctx: Context<'_>) {
    let (tag, content) = extract_tag_and_content(e.repr(), s, ctx);
    assert_all_unique(e.variants().iter().map(|f| f.name()), ctx);

    let variant = match e.variants().iter().find(|v| v.name() == tag) {
        Some(variant) => variant,
        None => panic!("could not find variant \"{}\"\n\n{}", tag, ctx),
    };

    let next: &Value = match (e.repr(), &content) {
        (tree::EnumRepr::External, None) => {
            assert!(
                variant.fields().is_unit(),
                "content may only be excluded for unit variants\n\n{}",
                ctx
            );
            return;
        }
        (tree::EnumRepr::External, Some(c)) => &c,
        (tree::EnumRepr::Internal { .. }, None) => unreachable!("internal always returns content"),
        (tree::EnumRepr::Internal { .. }, Some(c)) => {
            if variant.fields().is_unit() {
                let map = value_as_map(c, ctx);
                assert!(map.is_empty(), "unit enums should have no remaining values");
                return;
            } else {
                c
            }
        }
        (tree::EnumRepr::Adjacent { .. }, None) => {
            assert!(
                variant.fields().is_unit(),
                "content may only be excluded for unit variants\n\n{}",
                ctx
            );
            return;
        }
        (tree::EnumRepr::Adjacent { .. }, Some(c)) => &c,
    };

    check_fields(
        variant.fields(),
        next,
        context!(ctx, next, "variant \"{}\"", variant.name()),
    )
}

//----------//
// sequence //
//----------//

fn check_sequence(seq: &tree::Sequence, s: &Value, ctx: Context<'_>) {
    let a = value_as_array(s, ctx);
    for (i, v) in a.iter().enumerate() {
        check_reflection(
            seq.element(),
            v,
            context!(ctx, v, "element {} of {}", i + 1, a.len()),
        );
    }
}

//----------//
// optional //
//----------//

fn check_optional(opt: &tree::Optional, s: &Value, ctx: Context<'_>) {
    if !s.is_null() {
        check_reflection(opt.value(), s, context!(ctx, s, "present optional"))
    }
}

//---------//
// Helpers //
//---------//

fn assert_all_unique<I>(itr: I, ctx: Context<'_>)
where
    I: IntoIterator,
    I::Item: std::hash::Hash + Eq + std::fmt::Debug + Clone,
{
    let mut seen = HashSet::new();
    for i in itr {
        if !seen.insert(i.clone()) {
            panic!("item {:?} seen multiple times\n\n{}", i, ctx);
        }
    }
}

fn value_as_str<'a>(v: &'a Value, ctx: Context<'_>) -> &'a str {
    if let Value::String(s) = v {
        s
    } else {
        panic!("expected value to be a string\n\n{}", ctx);
    }
}

fn value_as_map<'a>(v: &'a Value, ctx: Context<'_>) -> &'a serde_json::value::Map<String, Value> {
    if let Value::Object(m) = v {
        m
    } else {
        panic!("expected value to be an object\n\n{}", ctx);
    }
}

fn value_as_array<'a>(v: &'a Value, ctx: Context<'_>) -> &'a [Value] {
    if let Value::Array(a) = v {
        a
    } else {
        panic!("expected value to be an array\n\n{}", ctx);
    }
}

///////////
// Tests //
///////////

#[test]
fn test_primitives() {
    check_struct(());

    check_struct(false);

    check_struct(0u8);
    check_struct(0u16);
    check_struct(0u32);
    check_struct(0u64);
    check_struct(0usize);

    check_struct(0i8);
    check_struct(0i16);
    check_struct(0i32);
    check_struct(0i64);
    check_struct(0isize);

    check_struct(0f32);
    check_struct(0f64);

    check_struct(std::marker::PhantomData::<usize>);

    check_struct(String::from("hello"));
}

#[test]
fn simple_aggregate() {
    #[derive(Serialize, Reflect, Clone)]
    struct Simple {
        a: usize,
        b: usize,
        r#type: (),
    }

    let s = Simple {
        a: 10,
        b: 20,
        r#type: (),
    };
    check_struct(s.clone());

    #[derive(Serialize, Reflect)]
    #[serde(rename_all = "kebab-case")]
    struct Nested {
        some_long_field: usize,
        #[serde(rename = "s")]
        manual_rename: Simple,
    }

    check_struct(Nested {
        some_long_field: 4,
        manual_rename: s,
    });
}

#[test]
fn simple_tuple() {
    // A simple newtype tuple.
    #[derive(Serialize, Reflect, Clone)]
    struct NewType(String);

    let newtype = NewType(String::from("hello"));
    check_struct(newtype.clone());

    // A tuple struct of length 2
    #[derive(Serialize, Reflect, Clone)]
    struct Tuple2(usize, NewType);

    let tuple2 = Tuple2(100, newtype.clone());
    check_struct(tuple2.clone());

    // A tuple struct of length 3 with a generic
    #[derive(Serialize, Reflect)]
    struct Tuple3<T>(usize, NewType, T);

    let tuple3 = Tuple3(4, NewType(String::from("foo")), tuple2.clone());
    check_struct(tuple3);

    // A tuple struct of length 3 with a generic and zero-sized type.
    #[derive(Serialize, Reflect)]
    struct Unit;

    #[derive(Serialize, Reflect)]
    struct Tuple4<T>(usize, Unit, NewType, T);

    let tuple3 = Tuple4(4, Unit, newtype, tuple2);
    check_struct(tuple3);
}

#[test]
fn nested_newtype() {
    #[derive(Serialize, Reflect)]
    struct NewType0(usize);

    #[derive(Serialize, Reflect)]
    struct NewType1(NewType0);

    #[derive(Serialize, Reflect)]
    struct NewType2(NewType1);

    check_struct(NewType2(NewType1(NewType0(10))));
}

#[test]
fn enum_externally_tagged() {
    #[derive(Serialize, Reflect)]
    struct Aggregate {
        foo: usize,
        bar: usize,
    }

    #[derive(Serialize, Reflect)]
    #[serde(rename_all = "kebab-case")]
    enum Enum<T> {
        Unit,
        NewType(T),
        Tuple0(),
        Tuple2(usize, usize),
        Tuple3(usize, usize, Aggregate),
        #[serde(rename = "longer-unit-2")]
        Unit2 {},

        #[serde(rename_all = "kebab-case")]
        Struct1 {
            hello_world: usize,
        },

        #[serde(rename_all = "kebab-case")]
        Struct2 {
            hello_world: usize,
            #[serde(rename = "baz")]
            foo: usize,
        },
    }

    check_enums(
        &[
            Enum::Unit,
            Enum::NewType(0usize),
            Enum::Tuple0(),
            Enum::Tuple2(1, 2),
            Enum::Tuple3(1, 2, Aggregate { foo: 10, bar: 20 }),
            Enum::Unit2 {},
            Enum::Struct1 { hello_world: 3 },
            Enum::Struct2 {
                hello_world: 4,
                foo: 5,
            },
        ],
        format_args!("externally tagged enums"),
    );
}

#[test]
fn enum_internally_tagged() {
    #[derive(Serialize, Reflect)]
    struct NewTypePayload {
        value: usize,
    }

    #[derive(Serialize, Reflect)]
    struct NewTypeEmpty {}

    #[derive(Serialize, Reflect)]
    #[serde(rename_all = "kebab-case")]
    #[serde(tag = "fizzle")]
    enum Enum {
        Unit,
        EmptyStruct {},
        NewType(NewTypePayload),
        NewTypeEmpty(NewTypeEmpty),
        Struct1 { a: usize },
        Struct2 { a: usize, b: usize },
        Struct3 { a: usize, b: usize, c: usize },
    }

    check_enums(
        &[
            Enum::Unit,
            Enum::EmptyStruct {},
            Enum::NewType(NewTypePayload { value: 10 }),
            Enum::NewTypeEmpty(NewTypeEmpty {}),
            Enum::Struct1 { a: 0 },
            Enum::Struct2 { a: 0, b: 1 },
            Enum::Struct3 { a: 0, b: 1, c: 3 },
        ],
        format_args!("internally tagged enums"),
    );
}

#[test]
fn enum_adjacently_tagged() {
    #[derive(Serialize, Reflect)]
    struct Nested {
        a: usize,
        b: usize,
    }

    #[derive(Serialize, Reflect)]
    #[serde(rename_all = "kebab-case")]
    #[serde(tag = "fizzle", content = "fuzzle")]
    enum Enum {
        Unit,
        r#RawUnit,
        Empty {},
        NewType(()),
        NewType2(usize),
        NewType3(Nested),
        Tuple0(),
        Tuple1(usize),
        Tuple2(usize, ()),
        Struct1 { a: usize },
        Struct2 { a: usize, b: usize },
        Struct3 { a: usize, b: usize, c: usize },
    }

    check_enums(
        &[
            Enum::Unit,
            Enum::r#RawUnit,
            Enum::Empty {},
            Enum::NewType(()),
            Enum::NewType2(10),
            Enum::NewType3(Nested { a: 0, b: 1 }),
            Enum::Tuple0(),
            Enum::Tuple1(0),
            Enum::Tuple2(0, ()),
            Enum::Struct1 { a: 0 },
            Enum::Struct2 { a: 0, b: 1 },
            Enum::Struct3 { a: 0, b: 1, c: 3 },
        ],
        format_args!("adjacently tagged enums"),
    );
}

#[test]
fn nested_enum() {
    #[derive(Serialize, Reflect)]
    enum Nested {
        Unit,
        NewType(usize),
        Struct { value: usize },
    }

    #[derive(Serialize, Reflect)]
    struct Outer {
        nested: Nested,
    }

    check_struct(Outer {
        nested: Nested::Unit,
    });
    check_struct(Outer {
        nested: Nested::NewType(10),
    });
    check_struct(Outer {
        nested: Nested::Struct { value: 20 },
    });
}

#[test]
fn sequence_of_newtypes() {
    #[derive(Serialize, Reflect)]
    struct NewType(usize);

    #[derive(Serialize, Reflect)]
    struct Outer {
        values: Vec<NewType>,
    }

    check_struct(Outer {
        values: vec![NewType(10), NewType(20)],
    });
}

#[test]
fn test_optionals() {
    #[derive(Serialize, Reflect)]
    #[serde(tag = "tag", content = "content", rename_all = "snake_case")]
    enum CasesAdjacent {
        NewType(Option<usize>),
        Struct { val: Option<usize> },
    }

    #[derive(Serialize, Reflect)]
    struct NewType(Option<usize>);

    #[derive(Serialize, Reflect)]
    struct More {
        a: usize,
        b: usize,
    }

    #[derive(Serialize, Reflect)]
    struct Struct {
        e0: CasesAdjacent,
        e1: CasesAdjacent,
        more: Option<More>,
        opt: Option<usize>,
        newtype: NewType,
    }

    check_struct(Struct {
        e0: CasesAdjacent::NewType(None),
        e1: CasesAdjacent::Struct { val: None },
        more: None,
        opt: None,
        newtype: NewType(None),
    });

    let some = Some(1);

    check_struct(Struct {
        e0: CasesAdjacent::NewType(some),
        e1: CasesAdjacent::Struct { val: some },
        more: Some(More { a: 0, b: 1 }),
        opt: some,
        newtype: NewType(some),
    });
}
