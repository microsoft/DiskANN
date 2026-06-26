/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Baseline Checking
//!
//! The [`Regression`](diskann_benchmark_runner::benchmark::Regression) provides a means
//! of performing before/after comparisons against previously generated results. However,
//! presentation of these results is largely left to the devices of the implementors.
//!
//! This module provides a means of aggregating all match failures (if any) and presenting
//! all failures as a single unit.

use std::{
    borrow::Cow,
    fmt::{Display, Write},
};

use diskann_benchmark_runner::{benchmark::PassFail, utils::fmt::Table};
use serde::{Serialize, Serializer};

/// Perform a baseline check on `self` and a `previous`ly saved result.
pub(crate) trait CheckMatch {
    fn check_match(&self, previous: &Self) -> Match;
}

/// The result of a basline check.
#[must_use = "this is a result type"]
#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Match {
    /// Successful match.
    Ok,

    /// A mismatch on a specific field.
    Mismatch {
        got: String,
        expected: String,
        remark: Option<Cow<'static, str>>,
    },

    /// A collection of mismatches for an aggregate data type or collection.
    ///
    /// Use [`MatchBuilder`] to easier construction.
    Nested {
        children: Vec<(Key, Match)>,
        remark: Option<Cow<'static, str>>,
    },
}

impl Match {
    /// Return `true` if `self` is [`Match::Ok`].
    #[must_use = "this has no side-effects"]
    pub(crate) fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Record a single mismatch between the retrieved value `got` and the `expected` result.
    pub(crate) fn mismatch(got: &dyn Display, expected: &dyn Display) -> Self {
        Self::mismatch_with_remark(got, expected, None)
    }

    /// Record a single mismatch between the retrieved value `got` and the `expected` result
    /// with an additional optional remark.
    ///
    /// The remark can be used for contexts where matches are more complex than simple
    /// equality.
    pub(crate) fn mismatch_with_remark(
        got: &dyn Display,
        expected: &dyn Display,
        remark: Option<Cow<'static, str>>,
    ) -> Self {
        Self::Mismatch {
            expected: expected.to_string(),
            got: got.to_string(),
            remark,
        }
    }

    /// Convert `self` into a [`PassFail`] for regression checks.
    ///
    /// Returns `PassFail::Pass` only if `self.is_ok`.
    pub(crate) fn pass_fail(self) -> PassFail<Self, Self> {
        if self.is_ok() {
            PassFail::Pass(self)
        } else {
            PassFail::Fail(self)
        }
    }
}

impl std::fmt::Display for Match {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ok => f.write_str("ok"),
            Self::Mismatch {
                got,
                expected,
                remark,
            } => {
                let header = ["got", "expected", "remark"];
                let mut table = Table::new(header, 1);
                let mut row = table.row(0);
                row.insert(got.clone(), 0);
                row.insert(expected.clone(), 1);
                if let Some(remark) = remark {
                    row.insert(remark.clone(), 2);
                }

                table.fmt(f)
            }
            Self::Nested { children, remark } => {
                let mut records = Vec::new();
                if let Some(remark) = remark {
                    records.push(Record {
                        path: String::new(),
                        got: "",
                        expected: "",
                        remark,
                    });
                }

                let mut buf = String::new();
                gather_mismatches(children, &mut records, Stack::new(&mut buf));

                let mut table = Table::new(["path", "got", "expected", "remark"], records.len());
                for (i, r) in records.into_iter().enumerate() {
                    let mut row = table.row(i);
                    row.insert(r.path, 0);
                    row.insert(r.got.to_owned(), 1);
                    row.insert(r.expected.to_owned(), 2);
                    row.insert(r.remark.to_owned(), 3);
                }

                table.fmt(f)
            }
        }
    }
}

fn gather_mismatches<'a>(
    mismatches: &'a [(Key, Match)],
    records: &mut Vec<Record<'a>>,
    mut path: Stack<'_>,
) {
    for (k, m) in mismatches.iter() {
        match m {
            Match::Ok => continue,
            Match::Mismatch {
                got,
                expected,
                remark,
            } => {
                let record = Record {
                    path: path.push(k).get(),
                    got,
                    expected,
                    remark: remark.as_deref().unwrap_or(""),
                };
                records.push(record);
            }
            Match::Nested { children, remark } => {
                let path = path.push(k);

                if let Some(remark) = remark {
                    records.push(Record {
                        path: path.get(),
                        got: "",
                        expected: "",
                        remark,
                    })
                }

                gather_mismatches(children, records, path)
            }
        }
    }
}

#[derive(Debug)]
struct Stack<'a> {
    s: &'a mut String,
    len: usize,
}

impl<'a> Stack<'a> {
    fn new(s: &'a mut String) -> Self {
        s.clear();
        Self { s, len: 0 }
    }

    #[expect(clippy::unwrap_used, reason = "formatting shouldn't be failing here")]
    fn push(&mut self, key: &Key) -> Stack<'_> {
        let len = self.s.len();
        if len == 0 {
            write!(self.s, "{}", key).unwrap();
        } else {
            write!(self.s, ".{}", key).unwrap();
        }

        Stack { s: self.s, len }
    }

    fn get(&self) -> String {
        self.s.clone()
    }
}

impl Drop for Stack<'_> {
    fn drop(&mut self) {
        self.s.truncate(self.len)
    }
}

#[derive(Debug)]
struct Record<'a> {
    path: String,
    got: &'a str,
    expected: &'a str,
    remark: &'a str,
}

/////////
// Key //
/////////

/// A key to develop the full hierarchical path for a match.
///
/// Keys can either be strings or positional indices. The latter are used when traversing
/// arrays.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum Key {
    Str(&'static str),
    Position(usize),
    String(String),
}

impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Str(s) => f.write_str(s),
            Self::Position(i) => write!(f, "{}", i),
            Self::String(s) => f.write_str(s),
        }
    }
}

impl Serialize for Key {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Str(s) => serializer.serialize_str(s),
            Self::Position(i) => serializer.serialize_u64(*i as u64),
            Self::String(s) => serializer.serialize_str(s),
        }
    }
}

impl From<&'static str> for Key {
    fn from(s: &'static str) -> Key {
        Key::Str(s)
    }
}

impl From<usize> for Key {
    fn from(i: usize) -> Key {
        Key::Position(i)
    }
}

impl From<String> for Key {
    fn from(s: String) -> Key {
        Key::String(s)
    }
}

/////////////
// Builder //
/////////////

/// A utility for building a nested [`Match`].
#[derive(Debug)]
pub(crate) struct MatchBuilder {
    children: Vec<(Key, Match)>,
}

impl MatchBuilder {
    /// Construct a new empty collection of matches.
    pub(crate) fn new() -> Self {
        Self {
            children: Vec::new(),
        }
    }

    /// Push the [`Match`] into the collection only if [`Match::is_ok`] fails.
    pub(crate) fn push(&mut self, key: Key, child: Match) {
        if !child.is_ok() {
            self.children.push((key, child));
        }
    }

    /// Package the collection of matches into a single [`Match`].
    ///
    /// If no failing matches have been aggregated, returns [`Match::Ok`].
    pub(crate) fn finish(self) -> Match {
        self.finish_with_remark(None)
    }

    /// Package the collection of matches into a single [`Match`] with a remark.
    ///
    /// If no failing matches have been aggregated, returns [`Match::Ok`].
    pub(crate) fn finish_with_remark(self, remark: Option<Cow<'static, str>>) -> Match {
        if self.children.is_empty() {
            Match::Ok
        } else {
            Match::Nested {
                children: self.children,
                remark,
            }
        }
    }
}

macro_rules! check_match_impl {
    ($T:ty) => {
        impl CheckMatch for $T {
            fn check_match(
                &self,
                previous: &Self,
            ) -> Match {
                if self == previous {
                    Match::Ok
                } else {
                    Match::mismatch(self, previous)
                }
            }
        }
    };
    ($($Ts:ty),+ $(,)?) => {
        $(check_match_impl!($Ts);)+
    }
}

check_match_impl!(
    bool, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64, &str, String
);

impl<T> CheckMatch for [T]
where
    T: CheckMatch,
{
    fn check_match(&self, previous: &[T]) -> Match {
        if self.len() != previous.len() {
            return Match::mismatch_with_remark(
                &self.len(),
                &previous.len(),
                Some("number of results is different between runs".into()),
            );
        }

        let mut builder = MatchBuilder::new();
        for (i, (got, expected)) in std::iter::zip(self.iter(), previous.iter()).enumerate() {
            builder.push(Key::from(i), got.check_match(expected));
        }

        builder.finish()
    }
}

impl<T> CheckMatch for Vec<T>
where
    T: CheckMatch,
{
    fn check_match(&self, previous: &Vec<T>) -> Match {
        self.as_slice().check_match(previous.as_slice())
    }
}

//--------//
// Macros //
//--------//

macro_rules! check_all_fields {
    ($self:expr, $prev:expr, { $($field:ident),+ $(,)? } $(,)?) => {{
        let Self { $($field),+ } = $self;
        let mut builder = $crate::support::check::MatchBuilder::new();
        $(
            builder.push(
                stringify!($field).into(),
                <_ as $crate::support::check::CheckMatch>::check_match(
                    $field,
                    &$prev.$field
                ),
            );
        )+
        builder
    }};
}

pub(crate) use check_all_fields;

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    //-------//
    // Match //
    //-------//

    #[test]
    fn match_is_ok() {
        assert!(Match::Ok.is_ok());
        assert!(!Match::mismatch(&1, &2).is_ok());
    }

    #[test]
    fn mismatch_records_got_and_expected() {
        match Match::mismatch(&1, &2) {
            Match::Mismatch {
                got,
                expected,
                remark,
            } => {
                assert_eq!(got, "1");
                assert_eq!(expected, "2");
                assert!(remark.is_none());
            }
            other => panic!("expected Mismatch, got {other:?}"),
        }
    }

    #[test]
    fn mismatch_with_remark_records_remark() {
        match Match::mismatch_with_remark(&"a", &"b", Some("note".into())) {
            Match::Mismatch {
                got,
                expected,
                remark,
            } => {
                assert_eq!(got, "a");
                assert_eq!(expected, "b");
                assert_eq!(remark.as_deref(), Some("note"));
            }
            other => panic!("expected Mismatch, got {other:?}"),
        }
    }

    #[test]
    fn pass_fail_follows_is_ok() {
        assert!(matches!(Match::Ok.pass_fail(), PassFail::Pass(Match::Ok)));

        assert!(matches!(
            Match::mismatch(&1, &2).pass_fail(),
            PassFail::Fail(Match::Mismatch { .. })
        ));

        let mut builder = MatchBuilder::new();
        builder.push(Key::from("test"), Match::mismatch(&1, &2));
        builder.push(Key::from("test2"), Match::mismatch(&2, &3));
        let mismatch = builder.finish();

        assert!(matches!(mismatch, Match::Nested { .. }));
        assert!(matches!(
            mismatch.pass_fail(),
            PassFail::Fail(Match::Nested { .. })
        ));
    }

    //------------//
    // CheckMatch //
    //------------//

    #[test]
    fn primitive_check_match() {
        assert!(1u32.check_match(&1u32).is_ok());
        assert!(!2u32.check_match(&3u32).is_ok());
        assert!("x".check_match(&"x").is_ok());
        assert!(!"x".check_match(&"y").is_ok());
    }

    #[test]
    fn slice_check_match_equal() {
        let a = vec![1u32, 2, 3];
        let b = vec![1u32, 2, 3];
        assert!(a.check_match(&b).is_ok());
    }

    #[test]
    fn slice_check_match_length_mismatch() {
        let a = vec![1u32, 2, 3];
        let b = vec![1u32, 2];
        match a.check_match(&b) {
            Match::Mismatch {
                got,
                expected,
                remark,
            } => {
                assert_eq!(got, "3");
                assert_eq!(expected, "2");
                assert!(remark.is_some());
            }
            other => panic!("expected length Mismatch, got {other:?}"),
        }
    }

    #[test]
    fn slice_check_match_element_mismatch() {
        let a = vec![1u32, 9, 3];
        let b = vec![1u32, 2, 3];
        match a.check_match(&b) {
            Match::Nested { children, .. } => {
                assert_eq!(children.len(), 1);
                assert!(matches!(children[0].0, Key::Position(1)));
            }
            other => panic!("expected Nested, got {other:?}"),
        }
    }

    //--------------//
    // MatchBuilder //
    //--------------//

    #[test]
    fn builder_empty_is_ok() {
        assert!(MatchBuilder::new().finish().is_ok());
    }

    #[test]
    fn builder_skips_ok_matches() {
        let mut builder = MatchBuilder::new();
        builder.push("a".into(), Match::Ok);
        builder.push("b".into(), Match::Ok);
        assert!(builder.finish().is_ok());
    }

    #[test]
    fn builder_collects_failures() {
        let mut builder = MatchBuilder::new();
        builder.push("a".into(), Match::Ok);
        builder.push("b".into(), Match::mismatch(&1, &2));
        match builder.finish() {
            Match::Nested { children, remark } => {
                assert_eq!(children.len(), 1);
                assert!(remark.is_none());
            }
            other => panic!("expected Nested, got {other:?}"),
        }
    }

    #[test]
    fn builder_finish_with_remark() {
        let mut builder = MatchBuilder::new();
        builder.push("b".into(), Match::mismatch(&1, &2));
        match builder.finish_with_remark(Some("ctx".into())) {
            Match::Nested { remark, .. } => assert_eq!(remark.as_deref(), Some("ctx")),
            other => panic!("expected Nested, got {other:?}"),
        }
    }

    //-----//
    // Key //
    //-----//

    #[test]
    fn key_display() {
        assert_eq!(Key::from("field").to_string(), "field");
        assert_eq!(Key::from(7usize).to_string(), "7");
        assert_eq!(Key::from(String::from("owned")).to_string(), "owned");
    }

    #[test]
    fn key_serde() {
        let k = serde_json::to_value(Key::Str("field")).unwrap();
        assert_eq!(k, serde_json::Value::String("field".into()));

        let k = serde_json::to_value(Key::Position(10)).unwrap();
        assert_eq!(k, serde_json::Value::Number(10.into()));

        let k = serde_json::to_value(Key::String("world".into())).unwrap();
        assert_eq!(k, serde_json::Value::String("world".into()));
    }

    //---------//
    // Display //
    //---------//

    #[test]
    fn display_ok() {
        assert_eq!(Match::Ok.to_string(), "ok");
    }

    #[test]
    fn display_nonnested() {
        let mismatch = Match::mismatch_with_remark(&"hello", &1, Some("word".into()));
        let rendered = mismatch.to_string();

        let expected = r#"
  got,   expected,   remark
===========================
hello,          1,     word
"#;
        let expected = expected.strip_prefix('\n').unwrap();

        println!("rendered = {:?}", rendered);

        let mut count = 0;
        for (line, (got, expected)) in
            std::iter::zip(rendered.lines(), expected.lines()).enumerate()
        {
            count += 1;
            assert_eq!(got.trim(), expected.trim(), "failed on line {line}",);
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn display_nested() {
        // Build a nested match and ensure the hierarchical path is rendered.
        let mut inner = MatchBuilder::new();
        inner.push(1usize.into(), Match::mismatch(&9, &2));
        inner.push(
            "test".into(),
            Match::mismatch_with_remark(&9, &2, Some("hello".into())),
        );
        let nested = inner.finish_with_remark(Some("some remark".into()));

        let mut outer = MatchBuilder::new();
        outer.push("results".into(), nested);
        let rendered = outer
            .finish_with_remark(Some("final remarks".into()))
            .to_string();

        let expected = r#"
         path,   got,   expected,          remark
 ================================================
             ,      ,           ,   final remarks
      results,      ,           ,     some remark
    results.1,     9,          2,
 results.test,     9,          2,           hello
 "#;

        let expected = expected.strip_prefix('\n').unwrap();

        println!("rendered = {:?}", rendered);

        let mut count = 0;
        for (line, (got, expected)) in
            std::iter::zip(rendered.lines(), expected.lines()).enumerate()
        {
            count += 1;
            assert_eq!(got.trim(), expected.trim(), "failed on line {line}",);
        }
        assert_eq!(count, 6);
    }

    //-------------------//
    // check_all_fields! //
    //-------------------//

    #[derive(Debug)]
    struct Sample {
        a: u32,
        b: String,
    }

    impl CheckMatch for Sample {
        fn check_match(&self, previous: &Self) -> Match {
            check_all_fields!(self, previous, { a, b }).finish()
        }
    }

    #[test]
    fn check_all_fields_equal() {
        let x = Sample {
            a: 1,
            b: "hi".into(),
        };
        let y = Sample {
            a: 1,
            b: "hi".into(),
        };
        assert!(x.check_match(&y).is_ok());
    }

    #[test]
    fn check_all_fields_reports_changed_field() {
        let x = Sample {
            a: 1,
            b: "hi".into(),
        };
        let y = Sample {
            a: 1,
            b: "bye".into(),
        };
        match x.check_match(&y) {
            Match::Nested { children, .. } => {
                assert_eq!(children.len(), 1);
                assert_eq!(children[0].0.to_string(), "b");
            }
            other => panic!("expected Nested, got {other:?}"),
        }
    }
}
