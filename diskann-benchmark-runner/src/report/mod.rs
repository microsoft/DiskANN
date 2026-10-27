/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Baseline Checking
//!
//! The [`Regression`](crate::benchmark::Regression) provides a means
//! of performing before/after comparisons against previously generated results. However,
//! presentation of these results is largely left to the devices of the implementors.
//!
//! This module provides a means of aggregating all match failures (if any) and presenting
//! all failures as a single unit.

use std::borrow::Cow;

use serde::{Serialize, Serializer};

pub mod tolerance;

/// A [`Check`], [`Table`], or nested grouping of multiple [`Report`]s used to gather
/// regression checks across arbitrary benchamrks and reporting the overall success or failure.
///
/// On success, all attributes are printed.
/// ```
/// use diskann_benchmark_runner::report::{Report, Check};
///
/// let mut builder = Report::builder();
/// {
///     let mut nested = Report::builder();
///     nested.push("foo", Report::equal(1, 1));
///     nested.push("bar", Report::info("cat"));
///     builder.push("nested", nested.finish());
/// }
///
/// let expected = "
/// nested:
///   foo: 1
///   bar: cat
/// ".trim();
///
/// let report = builder.finish();
/// assert_eq!(report.to_string(), expected);
/// ```
///
/// On failure, only the failing subset will be displayed.
#[derive(Debug, Serialize)]
#[serde(transparent)]
#[must_use = "reports are result types"]
pub struct Report {
    inner: ReportInner,
}

impl Report {
    /// Construct an empty [`Builder`] for constructing hierarchical reports.
    /// ```
    /// use diskann_benchmark_runner::report::Report;
    ///
    /// let mut builder = Report::builder();
    /// builder.push("foo", Report::info(10));
    /// builder.push("baz", Report::equal(10, 10));
    /// builder.push("bar", Report::equal(10, 20));
    /// let report = builder.finish();
    ///
    /// let s = report.to_string();
    /// let expected = "
    /// foo: 10
    /// baz: 10
    /// bar: (old) 10 != (new) 20 | FAIL
    /// ".trim();
    /// assert_eq!(s, expected);
    /// assert!(report.status().is_fail());
    /// ```
    pub fn builder() -> Builder {
        Builder::new()
    }

    /// Construct a [`Table`] using the provided list of headers.
    /// ```
    /// use diskann_benchmark_runner::report::{Report, Check};
    ///
    /// // Convenience for testing two renders.
    /// fn process(x: &str) -> &str {
    ///     let x = x.strip_prefix('\n').unwrap_or(x);
    ///     x.strip_suffix('\n').unwrap_or(x)
    /// }
    ///
    /// let mut table = Report::table(["hello", "world"]);
    ///
    /// // Tables are built one row at a time.
    /// //
    /// // Rows must be `finish`ed to be committed. Otherwise,
    /// // their contents are // dropped.
    /// let mut row = table.new_row();
    /// row.push(Check::equal(10, 20)); // Failing check.
    /// row.push(Check::info("bar"));
    /// table.push_row(row).unwrap();
    /// let report = table.finish();
    ///
    /// let s = report.to_string();
    /// let expected = process("
    ///                  hello,   world,   status
    /// =========================================
    /// (old) 10 != (new) 20 !,     bar,     FAIL
    /// ");
    /// assert_eq!(process(&s), expected);
    /// ```
    pub fn table<I>(headers: I) -> Table
    where
        I: IntoIterator<Item: Into<Cow<'static, str>>>,
    {
        Table::new(headers)
    }

    /// Report `info` for diagnostic purposes.
    ///
    /// The information will be displayed and serialized unchanged.
    pub fn info<T>(info: T) -> Self
    where
        T: std::fmt::Display + Serialize + 'static,
    {
        Check::info(info).into()
    }

    /// Report an `error`.
    ///
    /// Errors are considered a failure for diagnostic purposes.
    ///
    /// The error will be displayed and serialized using its [`ToString`] representation.
    pub fn error<T>(error: T) -> Self
    where
        T: ToString,
    {
        Check::error(error).into()
    }

    /// Assert that `old == new`.
    ///
    /// If the check succeeds, one argument will be retained and displayed for informational
    /// purposes.
    ///
    /// If the check fails, both values will be retained and the report fails.
    pub fn equal<T>(old: T, new: T) -> Self
    where
        T: std::fmt::Display + Serialize + PartialEq + 'static,
    {
        Check::equal(old, new).into()
    }

    /// Construct a new [`Report`] using [`tolerance::HigherIsBetter`].
    pub fn higher_is_better<T, U>(tolerance: &T, old: U, new: U, spec: tolerance::FmtSpec) -> Self
    where
        T: tolerance::HigherIsBetter<U>,
    {
        tolerance.higher_is_better(old, new, spec).into()
    }

    /// Construct a new [`Report`] using [`tolerance::LowerIsBetter`].
    pub fn lower_is_better<T, U>(tolerance: &T, old: U, new: U, spec: tolerance::FmtSpec) -> Self
    where
        T: tolerance::LowerIsBetter<U>,
    {
        tolerance.lower_is_better(old, new, spec).into()
    }

    /// Return the [`Status`] of this report.
    ///
    /// [`Status::Pass`] will only be reported if all nested [`Report`]s, [`Check`]s, and
    /// [`Table`]s report a pass.
    pub fn status(&self) -> Status {
        self.inner.status()
    }

    /// Convert `self` into a [`Finished`].
    ///
    /// Is this benchmark is a fail, then constructed `Finished` will only render the paths
    /// that failed. Otherwise, all paths will be rendered.
    pub fn pass_fail(self) -> crate::benchmark::PassFail<Finished, Finished> {
        let status = self.status();
        let rendering = match status {
            Status::Pass => Rendering::All,
            Status::Fail => Rendering::FailOnly,
        };

        let finished = Finished::new(self, rendering);
        match status {
            Status::Pass => crate::benchmark::PassFail::Pass(finished),
            Status::Fail => crate::benchmark::PassFail::Fail(finished),
        }
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Render::new(self, Rendering::All).fmt(f)
    }
}

impl From<Table> for Report {
    fn from(table: Table) -> Self {
        Self {
            inner: ReportInner::Table(table),
        }
    }
}

impl From<Check> for Report {
    fn from(check: Check) -> Self {
        Self {
            inner: ReportInner::Check(check),
        }
    }
}

/// The success or failure of checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pass,
    Fail,
}

impl Status {
    /// Return `true` is `self == Status::Fail`.
    pub fn is_fail(self) -> bool {
        self == Self::Fail
    }

    /// Return `true` is `self == Status::Success`.
    pub fn is_success(self) -> bool {
        self == Self::Pass
    }

    fn merge(&mut self, status: Self) {
        if status == Status::Fail {
            *self = Status::Fail;
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
        }
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for Status {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// A finished [`Report`].
///
/// This applies a dynamic [`std::fmt::Display`] policy depending on the final status of
/// the report.
#[derive(Debug, Serialize)]
#[serde(transparent)]
pub struct Finished {
    report: Report,
    #[serde(skip)]
    rendering: Rendering,
}

impl Finished {
    fn new(report: Report, rendering: Rendering) -> Self {
        Self { report, rendering }
    }
}

impl std::fmt::Display for Finished {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Render::new(&self.report, self.rendering).fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rendering {
    All,
    FailOnly,
}

fn render(report: &Report, policy: Rendering) -> Option<String> {
    let ctx = FmtContext::Tree;
    match &report.inner {
        ReportInner::Table(table) => match policy {
            Rendering::All => Some(table.to_string()),
            Rendering::FailOnly => {
                if table.status() == Status::Fail {
                    Some(table.to_string())
                } else {
                    None
                }
            }
        },
        ReportInner::Check(check) => match policy {
            Rendering::All => Some(check.render(ctx)),
            Rendering::FailOnly => {
                if check.status() == Status::Fail {
                    Some(check.render(ctx))
                } else {
                    None
                }
            }
        },
        ReportInner::Nested { children, status } => {
            if policy == Rendering::FailOnly && *status == Status::Pass {
                return None;
            }

            let mut kv = crate::utils::fmt::KeyValue::new();

            // Allocate numbuffers to format all the integer keys in `children`.
            let keys: Vec<_> = children.iter().map(|c| c.0.remove_position()).collect();

            for (key, child) in std::iter::zip(keys.iter(), children.iter()) {
                if let Some(rendered) = render(&child.1, policy) {
                    let force_nest = match &child.1.inner {
                        ReportInner::Table(_) => true,
                        ReportInner::Check(_) => false,
                        ReportInner::Nested { .. } => true,
                    };

                    if force_nest {
                        kv.push_eager_nested(key, rendered);
                    } else {
                        kv.push_eager(key, rendered);
                    }
                }
            }

            if kv.is_empty() {
                None
            } else {
                Some(kv.to_string())
            }
        }
    }
}

struct Render<'a> {
    report: &'a Report,
    policy: Rendering,
}

impl<'a> Render<'a> {
    fn new(report: &'a Report, policy: Rendering) -> Self {
        Self { report, policy }
    }
}

impl std::fmt::Display for Render<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match render(self.report, self.policy) {
            Some(s) => f.write_str(&s),
            None => Ok(()),
        }
    }
}

#[derive(Debug, Serialize)]
enum ReportInner {
    /// Report is formatted as a table.
    ///
    /// Table must be inspected for success/failure.
    Table(Table),

    /// An equal check.
    Check(Check),

    /// Report is formatted as a nested report.
    Nested {
        children: Vec<(Key, Report)>,
        status: Status,
    },
}

impl ReportInner {
    fn status(&self) -> Status {
        match self {
            Self::Table(table) => table.status(),
            Self::Check(check) => check.status(),
            Self::Nested { status, .. } => *status,
        }
    }
}

/////////////
// Builder //
/////////////

/// A builder for constructing nested [`Report`]s.
pub struct Builder {
    children: Vec<(Key, Report)>,
    status: Status,
}

impl Builder {
    /// Construct a new empty builder.
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
            status: Status::Pass,
        }
    }

    /// Add `report` to the collection and associate it with `key`.
    ///
    /// Reports are collected in the order they are pushed and no uniqueness requirements
    /// exist for `key`. If building reports positionally, [`Builder::len`] can be used to
    /// get the current number of items.
    pub fn push(&mut self, key: impl Into<Key>, report: Report) {
        self.status.merge(report.status());
        self.children.push((key.into(), report))
    }

    /// Finish building the [`Report`].
    pub fn finish(self) -> Report {
        let Self { children, status } = self;
        Report {
            inner: ReportInner::Nested { children, status },
        }
    }

    /// Return the current number of [`Report`]s.
    pub fn len(&self) -> usize {
        self.children.len()
    }

    /// Return `true` only if [`Builder::len`] return 0.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

/// A key for a nested [`Report`] when given to [`Builder::push`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Key {
    Str(&'static str),
    Position(usize),
    String(String),
}

impl Key {
    fn remove_position(&self) -> Cow<'_, str> {
        match self {
            Self::Str(s) => Cow::Borrowed(s),
            Self::Position(p) => Cow::Owned(p.to_string()),
            Self::String(s) => Cow::Borrowed(s),
        }
    }
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

///////////
// Table //
///////////

/// A more horizontally compact leaf for [`Report`].
#[derive(Debug)]
pub struct Table {
    headers: Vec<Cow<'static, str>>,
    rows: Vec<Row>,
    status: Status,
}

impl Table {
    /// Create a new empty [`Table`] with the specified headers.
    ///
    /// Trying to add a new row via [`Table::push_row`] will fail is the row does not
    /// contain the same number of entries as `headers`.
    pub fn new<I>(headers: I) -> Self
    where
        I: IntoIterator<Item: Into<Cow<'static, str>>>,
    {
        Self {
            headers: headers.into_iter().map(|h| h.into()).collect(),
            rows: Vec::new(),
            status: Status::Pass,
        }
    }

    /// Return the overall status of the table.
    ///
    /// [`Status::Fail`] will be return if any internal check is a failure.
    pub fn status(&self) -> Status {
        self.status
    }

    /// Return the number of columns in the table.
    ///
    /// This is equal to the number of headers provided to [`Table::new`].
    pub fn ncols(&self) -> usize {
        self.headers.len()
    }

    /// Return the number of rows in the table.
    pub fn nrows(&self) -> usize {
        self.rows.len()
    }

    /// Construct a new [`Row`].
    ///
    /// The resturned [`Row`] will not be automatically added to the table. Use
    /// [`Table::push_row`] to add it.
    pub fn new_row(&self) -> Row {
        Row::new()
    }

    /// Add `row` to the `self`.
    ///
    /// Returns an error if `row.len() != self.ncols()`.
    pub fn push_row(&mut self, row: Row) -> anyhow::Result<()> {
        if row.len() != self.ncols() {
            anyhow::bail!(
                "finished row has {} columns when {} were expected",
                row.len(),
                self.ncols(),
            );
        }

        self.status.merge(row.status);
        self.rows.push(row);
        Ok(())
    }

    /// Complete the table.
    pub fn finish(self) -> Report {
        self.into()
    }
}

impl Serialize for Table {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        let columns: Vec<Vec<&Check>> = (0..self.ncols())
            .map(|column| self.rows.iter().map(|row| &row.checks[column]).collect())
            .collect();

        let row_status: Vec<Status> = self.rows.iter().map(|row| row.status).collect();

        let mut state = serializer.serialize_struct("Table", 4)?;
        state.serialize_field("headers", &self.headers)?;
        state.serialize_field("columns", &columns)?;
        state.serialize_field("row_status", &row_status)?;
        state.serialize_field("status", &self.status)?;
        state.end()
    }
}

const EMPTY_TABLE: &str = "<empty table>";

impl std::fmt::Display for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Nothing to write for empty tables.
        if self.ncols() == 0 {
            return f.write_str(EMPTY_TABLE);
        }

        // If this table is the source of a failure, we render an additional column to
        // identify which rows caused the overall failure.
        //
        // To cut down on visual noise:
        //
        // * We don't print out "PASS". This makes "FAIL" easier to identify.
        // * We do not render this column when the entire table is a pass.
        let nrows = self.nrows();
        let mut table = if self.status == Status::Pass {
            crate::utils::fmt::Table::new(self.headers.iter().cloned(), nrows)
        } else {
            let last_column = Cow::Borrowed("status");
            crate::utils::fmt::Table::new(
                self.headers
                    .iter()
                    .cloned()
                    .chain(std::iter::once(last_column)),
                nrows,
            )
        };

        for (row_index, row) in self.rows.iter().enumerate() {
            let mut out = table.row(row_index);
            let mut col_index = 0;
            for check in row.checks.iter() {
                out.insert(check.render(FmtContext::Table), col_index);
                col_index += 1;
            }

            if row.status == Status::Fail {
                out.insert(row.status, col_index);
            }
        }

        write!(f, "{}", table)
    }
}

/// A new row for [`Table`].
///
/// Use [`Table::push_row`] to add a row to a [`Table`].
#[derive(Debug)]
pub struct Row {
    checks: Vec<Check>,
    status: Status,
}

impl Row {
    /// Create a new empty [`Row`].
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            status: Status::Pass,
        }
    }

    /// Add a new [`Check`] to the row. This increases the column count of the row by 1.
    pub fn push(&mut self, check: Check) {
        self.status.merge(check.status());
        self.checks.push(check);
    }

    /// Return the current number of columns.
    pub fn len(&self) -> usize {
        self.checks.len()
    }

    /// Return `true` only if the row currently has no columns.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for Row {
    fn default() -> Self {
        Self::new()
    }
}

///////////
// Check //
///////////

/// A singular regression check.
///
/// This is one of the leaf elements of [`Report`].
#[must_use = "checks are result types"]
pub struct Check {
    inner: Box<dyn Checkable>,
}

impl Check {
    /// Return the [`Status`] of the check.
    pub fn status(&self) -> Status {
        self.inner.status()
    }

    /// Report `info` for diagnostic purposes.
    ///
    /// See also: [`Report::info`].
    pub fn info<T>(info: T) -> Self
    where
        T: std::fmt::Display + Serialize + 'static,
    {
        Self::new(Info(info))
    }

    /// Report an `error`.
    ///
    /// See also: [`Report::error`].
    pub fn error<T>(error: T) -> Self
    where
        T: ToString,
    {
        Self::new(Error(error.to_string()))
    }

    /// Assert that `old == new`.
    ///
    /// See also: [`Report::error`].
    pub fn equal<T>(old: T, new: T) -> Self
    where
        T: std::fmt::Display + Serialize + PartialEq + 'static,
    {
        if old == new {
            Self::new(Equal(old))
        } else {
            Self::new(NotEqual { old, new })
        }
    }

    /// Construct a new [`Check`] using [`tolerance::HigherIsBetter`].
    ///
    /// See also: [`Report::higher_is_better`].
    pub fn higher_is_better<T, U>(tolerance: &T, old: U, new: U, spec: tolerance::FmtSpec) -> Self
    where
        T: tolerance::HigherIsBetter<U>,
    {
        tolerance.higher_is_better(old, new, spec)
    }

    /// Construct a new [`Report`] using [`tolerance::LowerIsBetter`].
    ///
    /// See also: [`Report::lower_is_better`].
    pub fn lower_is_better<T, U>(tolerance: &T, old: U, new: U, spec: tolerance::FmtSpec) -> Self
    where
        T: tolerance::LowerIsBetter<U>,
    {
        tolerance.lower_is_better(old, new, spec)
    }

    fn new<T>(inner: T) -> Self
    where
        T: Checkable + 'static,
    {
        Self {
            inner: Box::new(inner),
        }
    }

    fn render(&self, ctx: FmtContext) -> String {
        let mut s = String::new();

        #[expect(clippy::expect_used)]
        self.inner
            .fmt(&mut s, ctx)
            .expect("rust treats string formatting as infallible");
        s
    }
}

impl std::fmt::Debug for Check {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.render(FmtContext::Tree);
        let status = self.status();
        f.debug_struct("Check")
            .field("kind", &inner)
            .field("status", &status)
            .finish()
    }
}

impl Serialize for Check {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut serializer = Some(serializer);
        let mut result = None;

        self.inner.serialize(&mut |value| {
            #[expect(clippy::expect_used)]
            let serializer = serializer
                .take()
                .expect("serialization callback called more than once");
            result = Some(value.serialize(serializer));
        });

        #[expect(clippy::expect_used)]
        result.expect("serialization callback was not called")
    }
}

//-------//
// Equal //
//-------//

#[derive(Debug)]
struct Equal<T>(T);

impl<T> Checkable for Equal<T>
where
    T: std::fmt::Display + Serialize,
{
    fn status(&self) -> Status {
        Status::Pass
    }

    fn fmt(&self, f: &mut dyn std::fmt::Write, _ctx: FmtContext) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }

    fn serialize(&self, f: &mut dyn FnMut(&dyn erased_serde::Serialize)) {
        f(&SerEqual::pass(&self.0))
    }
}

#[derive(Debug)]
struct NotEqual<T> {
    old: T,
    new: T,
}

impl<T> Checkable for NotEqual<T>
where
    T: std::fmt::Display + Serialize,
{
    fn status(&self) -> Status {
        Status::Fail
    }

    fn fmt(&self, f: &mut dyn std::fmt::Write, ctx: FmtContext) -> std::fmt::Result {
        write!(
            f,
            "(old) {} != (new) {} {}",
            self.old,
            self.new,
            ctx.fail_marker()
        )
    }

    fn serialize(&self, f: &mut dyn FnMut(&dyn erased_serde::Serialize)) {
        f(&SerEqual::fail(&self.old, &self.new))
    }
}

/// Serializer for [`Equal`] and [`NotEqual`].
///
/// If `new` is present, this is assumed to be a failure.
struct SerEqual<'a, T> {
    old: &'a T,
    new: Option<&'a T>,
}

impl<'a, T> SerEqual<'a, T> {
    fn pass(value: &'a T) -> Self {
        Self {
            old: value,
            new: None,
        }
    }

    fn fail(old: &'a T, new: &'a T) -> Self {
        Self {
            old,
            new: Some(new),
        }
    }
}

impl<T> Serialize for SerEqual<'_, T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        let (new, status) = match self.new {
            Some(new) => (new, Status::Fail),
            None => (self.old, Status::Pass),
        };

        let mut state = serializer.serialize_struct("Equal", 3)?;
        state.serialize_field("old", self.old)?;
        state.serialize_field("new", new)?;
        state.serialize_field("status", &status)?;
        state.end()
    }
}

//------//
// Info //
//------//

#[derive(Debug)]
struct Info<T>(T);

impl<T> Checkable for Info<T>
where
    T: std::fmt::Display + Serialize,
{
    fn status(&self) -> Status {
        Status::Pass
    }

    fn fmt(&self, f: &mut dyn std::fmt::Write, _ctx: FmtContext) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }

    fn serialize(&self, f: &mut dyn FnMut(&dyn erased_serde::Serialize)) {
        f(&self.0)
    }
}

//-------//
// Error //
//-------//

#[derive(Debug)]
struct Error(String);

impl Checkable for Error {
    fn status(&self) -> Status {
        Status::Fail
    }

    fn fmt(&self, f: &mut dyn std::fmt::Write, ctx: FmtContext) -> std::fmt::Result {
        write!(f, "{} {}", self.0, ctx.fail_marker())
    }

    fn serialize(&self, f: &mut dyn FnMut(&dyn erased_serde::Serialize)) {
        struct Ser<'a>(&'a str);

        impl Serialize for Ser<'_> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                use serde::ser::SerializeStruct;

                let mut state = serializer.serialize_struct("Error", 2)?;
                state.serialize_field("error", self.0)?;
                state.serialize_field("status", &Status::Fail)?;
                state.end()
            }
        }

        f(&Ser(&self.0))
    }
}

/// Are we rendering in a [`Table`] or as part of the normal tree rendering in [`Report`]?
#[derive(Debug, Clone, Copy)]
enum FmtContext {
    Tree,
    Table,
}

impl FmtContext {
    // Select how to mark failures depending on the context.
    //
    // In a table, we just use `!` since we already attach a row-level `FAIL`.
    //
    // This keeps things a little more compact horizontally.
    fn fail_marker(&self) -> &'static str {
        match self {
            Self::Tree => "| FAIL",
            Self::Table => "!",
        }
    }
}

/// Trait object for checks.
trait Checkable {
    fn status(&self) -> Status;
    fn fmt(&self, f: &mut dyn std::fmt::Write, ctx: FmtContext) -> std::fmt::Result;
    fn serialize(&self, f: &mut dyn FnMut(&dyn erased_serde::Serialize));
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_display_and_serialization() {
        let cases = [
            (Key::Str("name"), "name", serde_json::json!("name")),
            (Key::Position(42), "42", serde_json::json!(42)),
            (
                Key::String(String::from("owned")),
                "owned",
                serde_json::json!("owned"),
            ),
        ];

        for (key, rendered, serialized) in cases {
            assert_eq!(key.to_string(), rendered);
            assert_eq!(serde_json::to_value(&key).unwrap(), serialized);
        }
    }

    use diskann_utils::assert_contains;

    //-------//
    // Check //
    //-------//

    #[test]
    fn test_check_info() {
        let cases = [
            (
                Check::info(u64::MAX),
                "18446744073709551615",
                serde_json::json!(u64::MAX),
            ),
            (Check::info(0.25), "0.25", serde_json::json!(0.25)),
            (Check::info(false), "false", serde_json::json!(false)),
            (
                Check::info(String::from("first\nsecond")),
                "first\nsecond",
                serde_json::json!("first\nsecond"),
            ),
        ];

        for (check, rendered, serialized) in cases {
            assert_eq!(check.status(), Status::Pass);
            assert_eq!(check.render(FmtContext::Tree), rendered);
            assert_eq!(check.render(FmtContext::Table), rendered);
            assert_eq!(serde_json::to_value(&check).unwrap(), serialized);
        }
    }

    #[test]
    fn test_check_equal() {
        let cases = [
            (
                Check::equal(u64::MAX, u64::MAX),
                Status::Pass,
                "18446744073709551615",
                "18446744073709551615",
                serde_json::json!({
                    "old": u64::MAX,
                    "new": u64::MAX,
                    "status": "PASS",
                }),
            ),
            (
                Check::equal(-1, 2),
                Status::Fail,
                "(old) -1 != (new) 2 | FAIL",
                "(old) -1 != (new) 2 !",
                serde_json::json!({ "old": -1, "new": 2, "status": "FAIL" }),
            ),
            (
                Check::equal("same", "same"),
                Status::Pass,
                "same",
                "same",
                serde_json::json!({ "old": "same", "new": "same", "status": "PASS" }),
            ),
            (
                Check::equal("before", "after"),
                Status::Fail,
                "(old) before != (new) after | FAIL",
                "(old) before != (new) after !",
                serde_json::json!({ "old": "before", "new": "after", "status": "FAIL" }),
            ),
        ];

        for (check, status, tree, table, serialized) in cases {
            assert_eq!(check.status(), status);
            assert_eq!(check.render(FmtContext::Tree), tree);
            assert_eq!(check.render(FmtContext::Table), table);
            assert_eq!(serde_json::to_value(&check).unwrap(), serialized);
        }
    }

    #[test]
    fn test_check_error() {
        let cases = [
            (
                Check::error("invalid input"),
                "invalid input | FAIL",
                "invalid input !",
                serde_json::json!({ "error": "invalid input", "status": "FAIL" }),
            ),
            (
                Check::error("invalid \"value\"\ntry again"),
                "invalid \"value\"\ntry again | FAIL",
                "invalid \"value\"\ntry again !",
                serde_json::json!({
                    "error": "invalid \"value\"\ntry again",
                    "status": "FAIL",
                }),
            ),
        ];

        for (check, tree, table, serialized) in cases {
            assert_eq!(check.status(), Status::Fail);
            assert_eq!(check.render(FmtContext::Tree), tree);
            assert_eq!(check.render(FmtContext::Table), table);
            assert_eq!(serde_json::to_value(&check).unwrap(), serialized);
        }
    }

    //-------//
    // Table //
    //-------//

    fn check_render(expected: &str, rendered: &str) {
        // To support `expected` coming from raw strings, we remove leading and trailing
        // newlines.
        let expected = {
            let temp = expected.strip_prefix('\n').unwrap_or(expected);
            temp.strip_suffix('\n').unwrap_or(temp)
        };

        struct Pretty<'a> {
            expected: &'a str,
            rendered: &'a str,
        }

        impl std::fmt::Display for Pretty<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                writeln!(
                    f,
                    "EXPECTED\n\n{}\n\nRENDERED\n\n{}",
                    self.expected, self.rendered
                )
            }
        }

        let pretty = Pretty { expected, rendered };

        // Check number of lines first for better error diagnostics.
        let expected_lines: Vec<&str> = expected.lines().collect();
        let rendered_lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(
            expected_lines.len(),
            rendered_lines.len(),
            "expected {} lines - got {}\n\n{}",
            expected_lines.len(),
            rendered_lines.len(),
            pretty
        );

        for (i, (expected, rendered)) in
            std::iter::zip(expected_lines.iter(), rendered_lines.iter()).enumerate()
        {
            // We trim the end because the expected tables trim trailing whitespace.
            //
            // We do keep leading whitespace, though, for slightly more accurate checks.
            //
            // Also, in the error message, we display the lines horizontally aligned to help
            // find whitespace woes.
            let expected = expected.trim_end();
            let rendered = rendered.trim_end();
            assert_eq!(
                expected,
                rendered,
                "mismatch on line {} of {}\n\n{}",
                i + 1,
                expected_lines.len(),
                pretty
            );
        }
    }

    #[test]
    fn test_table_serialization() {
        let mut table = Table::new(["name", "value"]);
        for (name, check) in [("good", Check::info(3)), ("bad", Check::equal(1, 2))] {
            let mut row = table.new_row();
            row.push(Check::info(name));
            row.push(check);
            table.push_row(row).unwrap();
        }

        assert_eq!(
            serde_json::to_value(&table).unwrap(),
            serde_json::json!({
                "headers": ["name", "value"],
                "columns": [
                    ["good", "bad"],
                    [3, { "old": 1, "new": 2, "status": "FAIL" }],
                ],
                "row_status": ["PASS", "FAIL"],
                "status": "FAIL",
            }),
        );
    }

    #[test]
    fn test_table_rendering() {
        let mut table = Table::new(["foo", "bar"]);

        let mut row = table.new_row();
        assert!(row.is_empty());
        assert_eq!(row.len(), 0);

        row.push(Check::info(10));
        assert!(!row.is_empty());
        assert_eq!(row.len(), 1);

        row.push(Check::info("hello"));
        assert!(!row.is_empty());
        assert_eq!(row.len(), 2);

        table.push_row(row).unwrap();

        let mut row = table.new_row();
        row.push(Check::info("world"));
        row.push(Check::info(20));
        table.push_row(row).unwrap();

        // Check that incorrect table lenghts are rejected.
        assert_contains!(
            table.push_row(Row::new()).unwrap_err().to_string(),
            "finished row has 0 columns when 2 were expected"
        );

        assert!(table.status().is_success());
        let expected = "
  foo,     bar
==============
   10,   hello
world,      20";

        check_render(expected, &table.to_string());
    }

    #[test]
    fn test_table_rendering_mixed_status() {
        let mut table = Table::new(["value"]);
        for check in [Check::equal(1, 2), Check::error("invalid"), Check::info(3)] {
            let mut row = table.new_row();
            row.push(check);
            table.push_row(row).unwrap();
        }

        assert!(table.status().is_fail());
        let expected = "
               value,   status
==============================
(old) 1 != (new) 2 !,     FAIL
           invalid !,     FAIL
                   3,";

        check_render(expected, &table.to_string());
        check_render(expected, &table.finish().to_string());
    }

    #[test]
    fn test_table_rendering_zero_rows() {
        let table = Table::new(["foo", "bar"]);

        assert!(table.status().is_success());
        let expected = "
foo,   bar
==========";

        check_render(expected, &table.to_string());
        check_render(expected, &table.finish().to_string());
    }

    #[test]
    fn test_table_rendering_zero_columns() {
        let mut table = Table::new([] as [&str; 0]);
        table.push_row(table.new_row()).unwrap();

        assert!(table.status().is_success());
        check_render(EMPTY_TABLE, &table.to_string());
        check_render(EMPTY_TABLE, &table.finish().to_string());
    }

    #[test]
    fn test_table_rendering_empty() {
        let table = Table::new([] as [&str; 0]);

        assert!(table.status().is_success());
        check_render(EMPTY_TABLE, &table.to_string());
        check_render(EMPTY_TABLE, &table.finish().to_string());
    }

    #[test]
    fn test_report_rendering() {
        let mut good = Report::builder();
        good.push(0, Report::info("ready"));
        good.push(String::from("count"), Report::equal(2, 2));

        let mut mixed = Report::builder();
        mixed.push("keep", Report::info(7));
        mixed.push("bad", Report::equal(1, 2));
        mixed.push("err", Report::error("invalid"));

        {
            let mut nested = Report::builder();
            nested.push("should be newline", Report::info(true));
            mixed.push("nested", nested.finish());
        }

        let mut table = Report::table(["value"]);
        for check in [Check::equal(3, 3), Check::equal(1, 2)] {
            let mut row = table.new_row();
            row.push(check);
            table.push_row(row).unwrap();
        }

        let mut builder = Report::builder();
        builder.push("info", Report::info("example"));
        builder.push("good", good.finish());
        builder.push("mixed", mixed.finish());
        builder.push("table", table.finish());
        builder.push("omitted", Builder::new().finish());
        let report = builder.finish();

        assert!(report.status().is_fail());
        check_render(
            "
info:  example
good:
  0:     ready
  count: 2
mixed:
  keep:   7
  bad:    (old) 1 != (new) 2 | FAIL
  err:    invalid | FAIL
  nested:
    should be newline: true
table:
                 value,   status
  ==============================
                     3,
  (old) 1 != (new) 2 !,     FAIL",
            &report.to_string(),
        );
        check_render(
            "
mixed:
  bad: (old) 1 != (new) 2 | FAIL
  err: invalid | FAIL
table:
                 value,   status
  ==============================
                     3,
  (old) 1 != (new) 2 !,     FAIL",
            &Finished::new(report, Rendering::FailOnly).to_string(),
        );
    }

    #[test]
    fn test_finished_report_rendering_success() {
        let mut builder = Report::builder();
        builder.push("info", Report::info("ready"));
        builder.push("same", Report::equal(2, 2));
        let report = builder.finish();

        assert!(report.status().is_success());
        let expected = "
info: ready
same: 2";

        check_render(expected, &Finished::new(report, Rendering::All).to_string());
    }
}
