/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::{self, Write};

use crate::utils::fmt::Quote;

use super::{
    tree::{Aggregate, Enum, EnumRepr, Fields, NamedField, Sequence, Type, UnnamedField, Variant},
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
            Type::Aggregate(aggregate) => self.render_aggregate(&aggregate),
            Type::Enum(enum_) => self.render_enum(enum_),
            Type::Sequence(sequence) => self.render_sequence(&sequence),
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
                    self.render_unnamed_field(i, field)?;
                }
            }
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

    fn render_unnamed_field(&mut self, index: usize, field: &UnnamedField) -> fmt::Result {
        let tagged = Tagged::new(field.field());
        let will_render_body = self.will_render(tagged.ty());

        self.line(format_args!(
            "{}: {}",
            index,
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
