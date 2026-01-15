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
}
