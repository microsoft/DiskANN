/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashMap,
    fmt::{Display, Write},
};

/// A 2-d table for formatting properly spaced values in a table.
pub struct Table {
    // The number of columns is implicitly described by the number of entries in `header`.
    header: Box<[Box<dyn Display>]>,
    body: HashMap<(usize, usize), Box<dyn Display>>,
    nrows: usize,
}

impl Table {
    pub fn new<I>(header: I, nrows: usize) -> Self
    where
        I: IntoIterator<Item: Display + 'static>,
    {
        fn as_dyn_display<T: Display + 'static>(x: T) -> Box<dyn Display> {
            Box::new(x)
        }

        let header: Box<[_]> = header.into_iter().map(as_dyn_display).collect();
        Self {
            header,
            body: HashMap::new(),
            nrows,
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.header.len()
    }

    pub fn insert<T>(&mut self, item: T, row: usize, col: usize) -> bool
    where
        T: Display + 'static,
    {
        self.check_bounds(row, col);
        self.body.insert((row, col), Box::new(item)).is_some()
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&dyn Display> {
        self.check_bounds(row, col);
        self.body.get(&(row, col)).map(|x| &**x)
    }

    pub fn row(&mut self, row: usize) -> Row<'_> {
        self.check_bounds(row, 0);
        Row::new(self, row)
    }

    #[expect(clippy::panic, reason = "table interfaces are bounds checked")]
    fn check_bounds(&self, row: usize, col: usize) {
        if row >= self.nrows() {
            panic!("row {} is out of bounds (max {})", row, self.nrows());
        }
        if col >= self.ncols() {
            panic!("col {} is out of bounds (max {})", col, self.ncols());
        }
    }
}

pub struct Row<'a> {
    table: &'a mut Table,
    row: usize,
}

impl<'a> Row<'a> {
    // A **private** constructor assuming that `row` is inbounds.
    fn new(table: &'a mut Table, row: usize) -> Self {
        Self { table, row }
    }

    /// Insert a value into the specified column of this row.
    pub fn insert<T>(&mut self, item: T, col: usize) -> bool
    where
        T: Display + 'static,
    {
        self.table.insert(item, self.row, col)
    }
}

impl Display for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const SEP: &str = ",   ";

        // Compute the maximum width of each column.
        struct Count(usize);

        impl Write for Count {
            fn write_str(&mut self, s: &str) -> std::fmt::Result {
                self.0 += s.len();
                Ok(())
            }
        }

        fn formatted_size<T>(x: &T) -> usize
        where
            T: Display + ?Sized,
        {
            let mut buf = Count(0);
            match write!(&mut buf, "{}", x) {
                // Return the number of bytes "written",
                Ok(()) => buf.0,
                Err(_) => 0,
            }
        }

        let mut widths: Vec<usize> = self.header.iter().map(formatted_size).collect();
        for row in 0..self.nrows() {
            for (col, width) in widths.iter_mut().enumerate() {
                if let Some(v) = self.body.get(&(row, col)) {
                    *width = (*width).max(formatted_size(v))
                }
            }
        }

        let header_width: usize = widths.iter().sum::<usize>() + (widths.len() - 1) * SEP.len();

        let mut buf = String::new();
        // Print the header.
        std::iter::zip(widths.iter(), self.header.iter())
            .enumerate()
            .try_for_each(|(col, (width, head))| {
                buf.clear();
                write!(buf, "{}", head)?;
                write!(f, "{:>width$}", buf)?;
                if col + 1 != self.ncols() {
                    write!(f, "{}", SEP)?;
                }
                Ok(())
            })?;

        // Banner
        write!(f, "\n{:=>header_width$}\n", "")?;

        // Write out each row.
        for row in 0..self.nrows() {
            for (col, width) in widths.iter_mut().enumerate() {
                match self.body.get(&(row, col)) {
                    Some(v) => {
                        buf.clear();
                        write!(buf, "{}", v)?;
                        write!(f, "{:>width$}", buf)?;
                    }
                    None => write!(f, "{:>width$}", "")?,
                }
                if col + 1 != self.ncols() {
                    write!(f, "{}", SEP)?;
                } else {
                    writeln!(f)?;
                }
            }
        }
        Ok(())
    }
}

////////////
// Banner //
////////////

pub(crate) struct Banner<'a>(&'a str);

impl<'a> Banner<'a> {
    pub(crate) fn new(message: &'a str) -> Self {
        Self(message)
    }
}

impl std::fmt::Display for Banner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = format!("# {} #", self.0);
        let len = st.len();
        writeln!(f, "{:#>len$}", "")?;
        writeln!(f, "{}", st)?;
        writeln!(f, "{:#>len$}", "")?;
        Ok(())
    }
}

////////////
// Indent //
////////////

/// Indents each line of a string by a fixed number of spaces.
///
/// Each line is prefixed with `spaces` spaces and terminated with a newline.
///
/// # Examples
///
/// ```
/// use diskann_benchmark_runner::utils::fmt::Indent;
///
/// let indented = Indent::new("hello\nworld", 4).to_string();
/// assert_eq!(indented, "    hello\n    world\n");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Indent<'a> {
    string: &'a str,
    spaces: usize,
}

impl<'a> Indent<'a> {
    /// Create a new [`Indent`] that will prefix each line of `string` with `spaces` spaces.
    pub fn new(string: &'a str, spaces: usize) -> Self {
        Self { string, spaces }
    }
}

impl std::fmt::Display for Indent<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let spaces = self.spaces;
        self.string
            .lines()
            .try_for_each(|ln| writeln!(f, "{: >spaces$}{}", "", ln))
    }
}

/////////////
// Delimit //
/////////////

/// Formats an iterator with a delimiter between items and an optional distinct last delimiter.
///
/// This is a single-use wrapper: the iterator is consumed on the first call to [`Display::fmt`].
/// Subsequent calls will print `<missing>`.
///
/// The `last` parameter allows a different delimiter before the final item (e.g., `", and "`),
/// which is useful for natural-language lists like `"a, b, and c"`.
///
/// # Examples
///
/// ```
/// use diskann_benchmark_runner::utils::fmt::Delimit;
///
/// let d = Delimit::new(["a", "b", "c"], ", ", Some(", and "));
/// assert_eq!(d.to_string(), "a, b, and c");
/// ```
pub struct Delimit<'a, I> {
    itr: std::cell::Cell<Option<I>>,
    delimiter: &'a str,
    last: Option<&'a str>,
}

impl<'a, I> Delimit<'a, I> {
    /// Create a new [`Delimit`] from an iterable, a delimiter, and an optional last delimiter.
    ///
    /// If `last` is `None`, the regular `delimiter` is used before the final item.
    pub fn new(
        itr: impl IntoIterator<IntoIter = I>,
        delimiter: &'a str,
        last: Option<&'a str>,
    ) -> Self {
        Self {
            itr: std::cell::Cell::new(Some(itr.into_iter())),
            delimiter,
            last,
        }
    }
}

impl<I> std::fmt::Display for Delimit<'_, I>
where
    I: Iterator<Item: std::fmt::Display>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Some(mut itr) = self.itr.take() else {
            return write!(f, "<missing>");
        };

        let mut first = true;
        let mut current = if let Some(item) = itr.next() {
            item
        } else {
            // Empty iterator
            return Ok(());
        };

        loop {
            match itr.next() {
                None => {
                    // "current" is the last item. If it is also the first, we write it
                    // directly. Otherwise, we use the "last" delimiter if available, falling
                    // back to "delimiter".
                    let delimiter = if first {
                        ""
                    } else if let Some(last) = self.last {
                        last
                    } else {
                        self.delimiter
                    };

                    return write!(f, "{}{}", delimiter, current);
                }
                Some(next) => {
                    // There is at least one item next. We print "current" and move on.
                    let delimiter = if first {
                        first = false;
                        ""
                    } else {
                        self.delimiter
                    };

                    write!(f, "{}{}", delimiter, current)?;
                    current = next;
                }
            }
        }
    }
}

///////////
// Quote //
///////////

/// Wraps a value in double quotes when displayed.
///
/// # Examples
///
/// ```
/// use diskann_benchmark_runner::utils::fmt::Quote;
///
/// assert_eq!(Quote("hello").to_string(), "\"hello\"");
/// assert_eq!(Quote(42).to_string(), "\"42\"");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Quote<T>(pub T);

impl<T> std::fmt::Display for Quote<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\"{}\"", self.0)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_banner() {
        let b = Banner::new("hello world");
        let s = b.to_string();

        let expected = "###############\n\
                        # hello world #\n\
                        ###############\n";

        assert_eq!(s, expected);

        let b = Banner::new("");
        let s = b.to_string();

        let expected = "####\n\
                        #  #\n\
                        ####\n";

        assert_eq!(s, expected);

        let b = Banner::new("foo");
        let s = b.to_string();

        let expected = "#######\n\
                        # foo #\n\
                        #######\n";

        assert_eq!(s, expected);
    }

    #[test]
    fn test_format() {
        // One column
        {
            let headers = ["h 0"];
            let mut table = Table::new(headers, 3);
            table.insert("a", 0, 0);
            table.insert("hello world", 1, 0);
            table.insert(62, 2, 0);

            let s = table.to_string();
            let expected = r#"
        h 0
===========
          a
hello world
         62
"#;
            assert_eq!(s, expected.strip_prefix('\n').unwrap());
        }

        // Two columns
        {
            let headers = ["a really really long header", "h1"];
            let mut table = Table::new(headers, 3);
            table.insert("a", 0, 0);
            table.insert("b", 0, 1);

            table.insert("hello world", 1, 0);
            table.insert("hello world version 2", 1, 1);

            table.insert(7, 2, 0);
            table.insert("bar", 2, 1);

            let s = table.to_string();
            let expected = r#"
a really really long header,                      h1
====================================================
                          a,                       b
                hello world,   hello world version 2
                          7,                     bar
"#;
            assert_eq!(s, expected.strip_prefix('\n').unwrap());
        }
    }

    #[test]
    fn test_row_api() {
        let mut table = Table::new(["a", "b", "c"], 2);
        let mut row = table.row(0);
        row.insert(1, 0);
        row.insert("long", 1);
        row.insert("s", 2);

        let mut row = table.row(1);
        row.insert("string", 0);
        row.insert(2, 1);
        row.insert(3, 2);

        let s = table.to_string();

        let expected = r#"
     a,      b,   c
===================
     1,   long,   s
string,      2,   3
"#;
        assert_eq!(s, expected.strip_prefix('\n').unwrap());
    }

    #[test]
    fn missing_values() {
        let mut table = Table::new(["a", "loong", "c"], 1);
        let mut row = table.row(0);
        row.insert("string", 0);
        row.insert("string", 2);

        let s = table.to_string();
        let expected = r#"
     a,   loong,        c
=========================
string,        ,   string
"#;
        assert_eq!(s, expected.strip_prefix('\n').unwrap());
    }

    #[test]
    #[should_panic(expected = "row 3 is out of bounds (max 2)")]
    fn test_panic_row() {
        let mut table = Table::new([1, 2, 3], 2);
        let _ = table.row(3);
    }

    #[test]
    #[should_panic(expected = "col 3 is out of bounds (max 2)")]
    fn test_panic_col() {
        let mut table = Table::new([1, 2], 1);
        let mut row = table.row(0);
        row.insert(1, 3);
    }

    #[test]
    fn test_indent_single_line() {
        let s = Indent::new("hello", 4).to_string();
        assert_eq!(s, "    hello\n");
    }

    #[test]
    fn test_indent_multi_line() {
        let s = Indent::new("hello\nworld\nfoo", 2).to_string();
        assert_eq!(s, "  hello\n  world\n  foo\n");
    }

    #[test]
    fn test_indent_zero_spaces() {
        let s = Indent::new("hello\nworld", 0).to_string();
        assert_eq!(s, "hello\nworld\n");
    }

    #[test]
    fn test_indent_empty_string() {
        let s = Indent::new("", 4).to_string();
        assert_eq!(s, "");
    }

    #[test]
    fn test_delimit_empty() {
        let d = Delimit::new(std::iter::empty::<&str>(), ", ", None);
        assert_eq!(d.to_string(), "");
    }

    #[test]
    fn test_delimit_single_item() {
        let d = Delimit::new(["a"], ", ", Some(", and "));
        assert_eq!(d.to_string(), "a");
    }

    #[test]
    fn test_delimit_two_items_with_last() {
        let d = Delimit::new(["a", "b"], ", ", Some(", and "));
        assert_eq!(d.to_string(), "a, and b");
    }

    #[test]
    fn test_delimit_three_items_with_last() {
        let d = Delimit::new(["a", "b", "c"], ", ", Some(", and "));
        assert_eq!(d.to_string(), "a, b, and c");
    }

    #[test]
    fn test_delimit_without_last() {
        let d = Delimit::new(["x", "y", "z"], " | ", None);
        assert_eq!(d.to_string(), "x | y | z");
    }

    #[test]
    fn test_delimit_second_display_prints_missing() {
        let d = Delimit::new(["a", "b"], ", ", None);
        assert_eq!(d.to_string(), "a, b");
        assert_eq!(d.to_string(), "<missing>");
    }

    #[test]
    fn test_quote() {
        assert_eq!(Quote("hello").to_string(), "\"hello\"");
    }

    #[test]
    fn test_quote_with_integer() {
        assert_eq!(Quote(42).to_string(), "\"42\"");
    }

    #[test]
    fn test_delimit_with_quote() {
        let d = Delimit::new(["topk", "range"].iter().map(Quote), ", ", Some(", and "));
        assert_eq!(d.to_string(), "\"topk\", and \"range\"");
    }
}
