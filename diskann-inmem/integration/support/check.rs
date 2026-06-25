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

/// Perform a basline check on `self` and a `previous`ly saved result.
pub(crate) trait CheckMatch {
    fn check_match(&self, previous: &Self) -> Match;
}

/// The result of a basline.
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
#[derive(Debug, Clone)]
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

#[derive(Debug)]
pub(crate) struct MatchBuilder {
    children: Vec<(Key, Match)>,
}

impl MatchBuilder {
    pub(crate) fn new() -> Self {
        Self {
            children: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, key: Key, child: Match) {
        if !child.is_ok() {
            self.children.push((key, child));
        }
    }

    pub(crate) fn finish(self) -> Match {
        self.finish_with_remark(None)
    }

    pub(crate) fn finish_with_remark(self, remark: Option<Cow<'static, str>>) -> Match {
        if self.children.is_empty() {
            Match::Ok
        } else {
            Match::Nested {
                children: self.children,
                remark: remark,
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

////////////
// Macros //
////////////

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
