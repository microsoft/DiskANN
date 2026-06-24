/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Write, borrow::Cow};

trait ExactMatch {
    fn exact_match(&self, other: &Self, matcher: Matcher<'_>);
}

struct Mismatch {
    path: String,
    expected: String,
    got: String,
    remark: Option<Cow<'static, str>>,
}

pub(crate) struct Matcher<'a> {
    mismatches: &'a mut Vec<Mismatch>,
    path: &'a mut String,
    len: usize,
}

impl<'a> Matcher<'a> {
    pub(crate) fn push<D>(&mut self, field: &D) -> Matcher<'_>
    where
        D: std::fmt::Display,
    {
        let len = self.path.len();
        if len == 0 {
            write!(self.path, "{}", field).unwrap();
        } else {
            write!(self.path, ".{}", field).unwrap();
        }

        Matcher {
            mismatches: self.mismatches,
            path: self.path,
            len,
        }
    }

    pub(crate) fn mismatch<D, R>(&mut self, expected: &D, got: &D, remark: Option<R>)
    where
        D: std::fmt::Display,
        R: Into<Cow<'static, str>>,
    {
        let mismatch = Mismatch {
            path: self.path.clone(),
            expected: expected.to_string(),
            got: got.to_string(),
            remark: remark.map(|x| x.into())
        };

        self.mismatches.push(mismatch);
    }
}

impl Drop for Matcher<'_> {
    fn drop(&mut self) {
        self.path.truncate(self.len);
    }
}

