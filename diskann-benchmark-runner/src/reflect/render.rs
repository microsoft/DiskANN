/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::{self, Write};

use crate::utils::fmt::Quote;

use super::{
    tree::{
        Aggregate, Enum, EnumRepr, Fields, NamedField, Optional, Sequence, Type, UnnamedField,
        Variant,
    },
    Reflection,
};

const INDENT: usize = 2;

#[derive(Debug)]
struct Tagged {
    ty: Type,
    reflection: Reflection,
}

impl Tagged {
    fn new(reflection: Reflection) -> Self {
        Self {
            ty: reflection.ty(),
            reflection,
        }
    }

    fn ty(&self) -> &Type {
        &self.ty
    }

    fn reflection(&self) -> Reflection {
        self.reflection
    }
}

pub(super) struct Renderer<'a> {
    output: &'a mut dyn Write,
    indent: usize,
    depth: usize,
    max_depth: usize,
}

impl<'a> Renderer<'a> {
    pub(super) fn new(output: &'a mut dyn Write, max_depth: usize) -> Self {
        Self {
            output,
            indent: 0,
            depth: 0,
            max_depth,
        }
    }

    fn at_bottom(&self) -> bool {
        self.depth >= self.max_depth
    }

    fn line<D>(&mut self, display: D) -> fmt::Result
    where
        D: std::fmt::Display,
    {
        let indent = INDENT * self.indent;
        write!(self.output, "{: >indent$}{}\n", "", display)
    }

    fn blank(&mut self) -> fmt::Result {
        self.output.write_char('\n')
    }

    fn maybe_indent<F, R>(&mut self, indent: bool, f: F) -> Result<R, fmt::Error>
    where
        F: FnOnce(&mut Self) -> Result<R, fmt::Error>,
    {
        if indent {
            self.indent += 1;
        }
        let result = f(self);
        if indent {
            self.indent -= 1;
        }
        result
    }

    fn indent<F, R>(&mut self, f: F) -> Result<R, fmt::Error>
    where
        F: FnOnce(&mut Self) -> Result<R, fmt::Error>,
    {
        self.maybe_indent(true, f)
    }

    fn next_with(
        &mut self,
        pre: impl FnOnce(&mut Self) -> fmt::Result,
        body: impl FnOnce(&mut Self) -> fmt::Result,
        post: impl FnOnce(&mut Self) -> fmt::Result,
    ) -> fmt::Result {
        if self.at_bottom() {
            Ok(())
        } else {
            pre(self)?;
            self.depth += 1;
            let result = self.indent(body);
            self.depth -= 1;
            post(self)?;
            result
        }
    }

    fn next<F>(&mut self, f: F) -> fmt::Result
    where
        F: FnOnce(&mut Self) -> fmt::Result,
    {
        self.next_with(|_| Ok(()), f, |_| Ok(()))
    }

    fn render_doc(&mut self, s: Option<&str>) -> Result<bool, fmt::Error> {
        if let Some(s) = s {
            let mut rendered = false;
            for ln in s.lines() {
                if ln.is_empty() {
                    self.blank()?;
                } else {
                    self.line(ln)?;
                }

                rendered = true;
            }
            Ok(rendered)
        } else {
            Ok(false)
        }
    }

    /// Return `true` if a nested type will be rendered.
    fn will_render(&self, ty: &Type) -> bool {
        !self.at_bottom() && ty.has_body()
    }

    //-------//
    // Types //
    //-------//

    pub(super) fn render_subject(&mut self, reflection: Reflection) -> fmt::Result {
        self.line(reflection.type_name())?;
        self.indent(|r| {
            let tagged = Tagged::new(reflection);
            let wrote_doc = r.render_doc(tagged.ty().doc())?;

            if wrote_doc && r.will_render(tagged.ty()) {
                r.blank()?;
            }

            r.render_body(&tagged)
        })
    }

    fn render_body(&mut self, tagged: &Tagged) -> fmt::Result {
        match tagged.ty() {
            Type::Primitive(_) => Ok(()),
            Type::Aggregate(aggregate) => self.render_aggregate(aggregate),
            Type::Enum(enum_) => self.render_enum(enum_),
            Type::Sequence(sequence) => self.render_sequence(sequence),
            Type::Optional(opt) => self.render_optional(opt),
        }
    }

    fn render_aggregate(&mut self, aggregate: &Aggregate) -> fmt::Result {
        self.render_fields(aggregate.fields())
    }

    fn render_enum(&mut self, enum_: &Enum) -> fmt::Result {
        match enum_.repr() {
            EnumRepr::External => self.line("Representation: externally tagged")?,
            EnumRepr::Internal { tag } => {
                self.line(format_args!("Discriminant field: {}", Quote(tag)))?
            }
            EnumRepr::Adjacent { tag, content } => {
                self.line(format_args!("Discriminant field: {}", Quote(tag)))?;
                self.line(format_args!("Content field: {}", Quote(content)))?;
            }
        }

        self.blank()?;
        self.line("Options:")?;
        self.indent(|r| {
            let mut previous: Option<&Variant> = None;
            for variant in enum_.variants().iter() {
                // Decide whether or not to put a space before this variant.
                // Spaces and be skipped if the previous one was a `Unit` with no docs.
                if let Some(previous) = previous {
                    let skip = previous.fields().is_unit() && previous.doc().is_none();
                    if !skip {
                        r.blank()?;
                    }
                }

                r.render_variant(variant)?;
                previous = Some(variant)
            }

            Ok(())
        })
    }

    fn render_sequence(&mut self, sequence: &Sequence) -> fmt::Result {
        self.next_with(
            |r| r.line(format_args!("Elements: {}", sequence.element().type_name())),
            |r| r.render_body(&Tagged::new(sequence.element())),
            |_| Ok(()),
        )
    }

    fn render_optional(&mut self, op: &Optional) -> fmt::Result {
        self.line("May be `null`.")?;
        self.next(|r| r.render_body(&Tagged::new(op.value())))
    }

    //--------//
    // Fields //
    //--------//

    fn render_fields(&mut self, fields: &Fields) -> fmt::Result {
        match fields {
            Fields::Named(named) => {
                let mut first = true;
                for field in named.iter() {
                    if !first {
                        self.blank()?;
                    }
                    self.render_named_field(field)?;
                    first = false;
                }
            }
            Fields::Unnamed(unnamed) => {
                for (i, field) in unnamed.iter().enumerate() {
                    if i != 0 {
                        self.blank()?;
                    }
                    self.render_unnamed_field(Some(i), field)?;
                }
            }
            Fields::NewType(newtype) => self.render_unnamed_field(None, newtype)?,
            Fields::Unit => {}
        }

        Ok(())
    }

    fn render_named_field(&mut self, field: &NamedField) -> fmt::Result {
        let tagged = Tagged::new(field.field());
        let will_render_body = self.will_render(tagged.ty());

        self.line(format_args!(
            "{}: {}",
            Quote(field.name()),
            tagged.reflection().type_name()
        ))?;

        let rendered_doc = self.indent(|r| r.render_doc(field.doc()))?;
        if rendered_doc && will_render_body {
            self.blank()?;
        }

        if will_render_body {
            self.next(|r| r.render_body(&tagged))?;
        }
        Ok(())
    }

    fn render_unnamed_field(&mut self, index: Option<usize>, field: &UnnamedField) -> fmt::Result {
        let tagged = Tagged::new(field.field());
        let will_render_body = self.will_render(tagged.ty());

        if let Some(index) = index {
            self.line(format_args!(
                "{}: {}",
                index,
                tagged.reflection().type_name()
            ))?;
        }

        let rendered_doc = self.indent(|r| r.render_doc(field.doc()))?;
        if rendered_doc && will_render_body {
            self.blank()?;
        }

        if will_render_body {
            self.next(|r| r.render_body(&tagged))?;
        }
        Ok(())
    }

    //---------//
    // Variant //
    //---------//

    fn render_variant(&mut self, variant: &Variant) -> fmt::Result {
        self.line(Quote(variant.name()))?;

        self.indent(|r| {
            let rendered_doc = r.render_doc(variant.doc())?;

            if rendered_doc && variant.fields().has_body() {
                r.blank()?;
            }

            r.render_fields(&variant.fields())
        })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::{
        fs::File,
        io::{BufRead, BufReader, Write},
        path::{Path, PathBuf},
    };

    use serde::{Deserialize, Serialize};

    use crate::{ux, Reflect};

    // For these tests, we use a variation of baseline tests where all the expected results
    // are put into a single file, mainly to keep from generating a bunch of files for the
    // relatively small tests.

    fn baseline_path() -> PathBuf {
        format!(
            "{}/tests/rendered_reflections.txt",
            env!("CARGO_MANIFEST_DIR")
        )
        .into()
    }

    fn overwrite_hint() -> &'static str {
        "Baselines can be regenerated by running tests with `DISKANN_TEST=overwrite`"
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Content {
        name: String,
        description: String,
    }

    const CASE_SEPARATOR: &'static str = "========";
    const OUTPUT_SEPARATOR: &'static str = "--------";

    #[derive(Debug)]
    struct Case {
        content: Content,
        r: Reflection,
    }

    impl Case {
        fn new<T>(name: &str, description: &str) -> Self
        where
            T: Reflect,
        {
            Self {
                content: Content {
                    name: name.into(),
                    description: description.into(),
                },
                r: Reflection::new::<T>(),
            }
        }
    }

    #[derive(Debug)]
    struct Rendered {
        content: Content,
        rendered: String,
    }

    #[derive(Default, Debug, PartialEq)]
    enum State {
        #[default]
        Content,
        Rendered,
        Done,
    }

    impl State {
        fn to_rendered(&mut self) {
            assert_eq!(*self, Self::Content);
            *self = Self::Rendered;
        }

        fn to_done(&mut self) {
            assert_eq!(*self, Self::Rendered);
            *self = Self::Done;
        }

        fn assert_is_done(self) {
            assert_eq!(self, Self::Done);
        }
    }

    impl Rendered {
        fn parse_if_present<B>(lines: &mut std::io::Lines<B>) -> Option<Self>
        where
            B: std::io::BufRead,
        {
            match lines.next() {
                Some(ln) => assert_eq!(ln.unwrap(), CASE_SEPARATOR),
                None => return None,
            };

            let mut content = String::new();
            let mut rendered = String::new();

            let mut state = State::default();

            while let Some(ln) = lines.next() {
                let ln: &str = &ln.unwrap();
                match ln {
                    CASE_SEPARATOR => {
                        state.to_done();
                        break;
                    }
                    OUTPUT_SEPARATOR => {
                        state.to_rendered();
                    }
                    ln => match state {
                        State::Content => content.push_str(ln),
                        State::Rendered => {
                            rendered.push('\n');
                            rendered.push_str(ln);
                        }
                        State::Done => panic!("invalid state"),
                    },
                }
            }

            state.assert_is_done();

            Some(Self {
                content: serde_json::from_str(&content).unwrap(),
                rendered: rendered.trim().to_string(),
            })
        }
    }

    fn parse_baselines(path: &Path) -> Vec<Rendered> {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(err) => panic!(
                "Opening path \"{}\" failed with {}. {}",
                path.display(),
                err,
                overwrite_hint()
            ),
        };
        let mut lines = BufReader::new(file).lines();
        let mut baselines = Vec::new();
        while let Some(baseline) = Rendered::parse_if_present(&mut lines) {
            baselines.push(baseline);
        }

        baselines
    }

    fn write_baselines(rendered: &[Rendered], path: &Path) {
        let mut io = File::create(path).unwrap();

        for r in rendered.iter() {
            let Rendered { content, rendered } = r;
            writeln!(io, "{}", CASE_SEPARATOR).unwrap();
            writeln!(io, "{}", serde_json::to_string_pretty(content).unwrap()).unwrap();
            writeln!(io, "{}", OUTPUT_SEPARATOR).unwrap();
            writeln!(io, "{}", ux::normalize(rendered.clone())).unwrap();
            writeln!(io, "{}", CASE_SEPARATOR).unwrap();
        }
    }

    fn run_tests_inner(cases: &[Case], path: &Path, overwrite: bool) {
        // Generate the current set of baseline.
        let current: Vec<_> = cases
            .iter()
            .map(|case| {
                let rendered = ux::normalize(case.r.render().to_string());
                Rendered {
                    content: case.content.clone(),
                    rendered,
                }
            })
            .collect();

        if overwrite {
            write_baselines(&current, path);
        } else {
            let expected = parse_baselines(path);
            assert_eq!(
                current.len(),
                expected.len(),
                "Number of baseline cases differs. {}",
                overwrite_hint(),
            );

            for (current, expected) in std::iter::zip(current.iter(), expected.iter()) {
                assert_eq!(
                    current.content,
                    expected.content,
                    "Baseline headers differ. {}",
                    overwrite_hint(),
                );

                if current.rendered != expected.rendered {
                    panic!(
                        "Difference for case name {}\n\nEXPECTED\n\n{}\n\nGOT\n\n{}\n\n{}",
                        expected.content.name,
                        expected.rendered,
                        current.rendered,
                        overwrite_hint()
                    );
                }
            }
        }
    }

    /// Select the data type please.
    #[derive(Reflect)]
    #[serde(rename_all = "kebab-case")]
    #[reflect(prefix = "render::")]
    #[expect(unused, reason = "testing")]
    enum SimpleDataType {
        Float32,
        Float16,
    }

    /// Select the data type please.
    #[derive(Reflect)]
    #[serde(rename_all = "kebab-case")]
    #[reflect(prefix = "render::")]
    #[expect(unused, reason = "testing")]
    enum AnnotatedDataType {
        /// Use high-precision.
        Float32,
        /// Use lower precision.
        Float16,
        Int8,
    }

    #[derive(Reflect)]
    #[serde(rename_all = "snake_case")]
    #[reflect(prefix = "render::")]
    #[expect(unused, reason = "testing")]
    enum Source {
        /// Build from scratch.
        Build {
            /// The type of the input data.
            data_type: SimpleDataType,

            /// Input data in the `.bin` binary format.
            file: String,

            /// Output file.
            ///
            /// If provided, saved data will go here.
            output: Option<String>,
        },
        /// Run from a previously generated output.
        FromPrevious {
            data_type: AnnotatedDataType,
            /// The previously generated output.
            file: String,
        },
    }

    /// A new-type wrapper around `Source`.
    #[derive(Reflect)]
    #[expect(unused, reason = "testing")]
    struct SourceWrapper(Source);

    /// A top level config.
    #[derive(Reflect)]
    #[expect(unused, reason = "testing")]
    struct Config {
        source: SourceWrapper,
        /// This does one thing.
        param1: String,
        /// This does another.
        param2: Vec<AnnotatedDataType>,
    }

    #[test]
    fn run_tests() {
        let cases = [
            Case::new::<usize>("usize", "a simple test"),
            Case::new::<Source>("source", "a configurable source enum"),
            Case::new::<SourceWrapper>("source wrapper", "render a newtype wrapper"),
            Case::new::<Config>("config", "a sample struct config."),
        ];

        run_tests_inner(&cases, &baseline_path(), ux::overwrite());
    }
}
