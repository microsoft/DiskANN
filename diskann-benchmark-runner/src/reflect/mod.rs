/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    any::TypeId,
    fmt::{self, Write},
};

pub use diskann_benchmark_runner_derive::Reflect;

mod render;
pub mod tree;
pub use tree::Type;

#[cfg(test)]
mod test;

pub trait Reflect: 'static {
    fn ty() -> Type;
    fn format_type_name(f: &mut dyn Write) -> fmt::Result;
}

#[derive(Clone, Copy)]
pub struct Reflection {
    reflection: &'static internal::VTable,
}

impl Reflection {
    pub const fn new<T>() -> Self
    where
        T: Reflect,
    {
        Self {
            reflection: internal::VTable::new::<T>(),
        }
    }

    pub fn ty(&self) -> Type {
        (self.reflection.ty)()
    }

    pub fn type_name(&self) -> TypeName {
        TypeName(*self)
    }

    pub fn type_id(&self) -> TypeId {
        (self.reflection.type_id)()
    }

    pub fn render(&self) -> Render {
        Render(*self)
    }

    pub(crate) fn visit_with<F, E>(&self, f: F) -> Result<(), E>
    where
        F: FnMut(Reflection) -> Result<bool, E>,
    {
        visit_with(*self, f)
    }
}

impl fmt::Debug for Reflection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reflection")
            .field("type_name", &self.type_name())
            .finish_non_exhaustive()
    }
}

pub struct TypeName(Reflection);

impl TypeName {
    fn format_type_name(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.0.reflection.format_type_name)(f)
    }
}

impl std::fmt::Debug for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format_type_name(f)
    }
}

impl std::fmt::Display for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format_type_name(f)
    }
}

pub struct Render(Reflection);

impl std::fmt::Display for Render {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut r = render::Renderer::new(f, 2);
        r.render_subject(self.0)
    }
}

//------------//
// Algorithms //
//------------//

fn visit_with<F, E>(mut reflection: Reflection, mut f: F) -> Result<(), E>
where
    F: FnMut(Reflection) -> Result<bool, E>,
{
    use tree::Fields;

    let mut stack = Vec::new();
    loop {
        // visit: expand this node if the closure returns `true`.
        if f(reflection)? {
            let mut push = |r: Reflection| stack.push(r);
            let mut push_fields = |fields: &Fields| match fields {
                Fields::Named(named) => named.iter().for_each(|field| push(field.field())),
                Fields::Unnamed(unnamed) => unnamed.iter().for_each(|field| push(field.field())),
                Fields::NewType(newtype) => push(newtype.field()),
                Fields::Unit => {}
            };

            // explore
            match reflection.ty() {
                // Nothing to do for primitives as there is no other object that can be reached.
                Type::Primitive(_) => {}
                Type::Aggregate(aggregate) => push_fields(aggregate.fields()),
                Type::Enum(enum_) => enum_
                    .variants()
                    .iter()
                    .for_each(|variant| push_fields(variant.fields())),
                Type::Sequence(seq) => push(seq.element()),
                Type::Optional(opt) => push(opt.value()),
            }
        }

        // loop
        if let Some(r) = stack.pop() {
            reflection = r;
        } else {
            break Ok(());
        }
    }
}

///////////////
// Bootstrap //
///////////////

impl<T> Reflect for std::marker::PhantomData<T>
where
    T: Reflect,
{
    fn ty() -> Type {
        Type::primitive(tree::PrimitiveKind::Null, None)
    }

    fn format_type_name(f: &mut dyn Write) -> fmt::Result {
        write!(f, "PhantomData<{}>", Reflection::new::<T>().type_name())
    }
}

macro_rules! primitive {
    ($T:ty, $kind:ident, $doc:literal, $type_name:literal) => {
        impl Reflect for $T {
            fn ty() -> Type {
                Type::primitive(tree::PrimitiveKind::$kind, Some($doc.into()))
            }

            fn format_type_name(f: &mut dyn Write) -> fmt::Result {
                f.write_str($type_name)
            }
        }
    };
}

primitive!((), Null, "empty", "()");
primitive!(
    usize,
    Number,
    "A system dependent unsigned integer",
    "usize"
);
primitive!(isize, Number, "A system dependent signed integer", "isize");

primitive!(u8, Number, "An 8-bit unsigned integer", "u8");
primitive!(u16, Number, "A 16-bit unsigned integer", "u16");
primitive!(u32, Number, "A 32-bit unsigned integer", "u32");
primitive!(u64, Number, "A 64-bit unsigned integer", "u64");

primitive!(i8, Number, "An 8-bit signed integer", "i8");
primitive!(i16, Number, "A 16-bit signed integer", "i16");
primitive!(i32, Number, "A 32-bit signed integer", "i32");
primitive!(i64, Number, "A 64-bit signed integer", "i64");

primitive!(f32, Number, "An 32-bit floating-point number", "f32");
primitive!(f64, Number, "An 64-bit floating-point number", "f64");

primitive!(
    std::num::NonZeroUsize,
    Number,
    "A system dependent, non-zero, unsigned integer",
    "NonZero<usize>"
);

primitive!(bool, Boolean, "A value of \"true\" or \"false\"", "bool");
primitive!(String, String, "A string", "string");

impl<T> Reflect for Option<T>
where
    T: Reflect,
{
    fn ty() -> Type {
        Type::optional::<T>(Some("An optional type".into()))
    }

    fn format_type_name(f: &mut dyn Write) -> fmt::Result {
        f.write_str("Option<")?;
        T::format_type_name(f)?;
        f.write_str(">")
    }
}

impl<T> Reflect for Vec<T>
where
    T: Reflect,
{
    fn ty() -> Type {
        Type::sequence::<T>(Some("An ordered collection of elements".into()))
    }

    fn format_type_name(f: &mut dyn Write) -> fmt::Result {
        write!(f, "Vec<{}>", Reflection::new::<T>().type_name())
    }
}

//////////////
// Internal //
//////////////

pub mod internal {
    pub(super) struct VTable {
        pub(super) ty: fn() -> super::Type,
        pub(super) format_type_name: fn(&mut dyn std::fmt::Write) -> std::fmt::Result,
        pub(super) type_id: fn() -> std::any::TypeId,
    }

    impl VTable {
        pub(super) const fn new<T>() -> &'static Self
        where
            T: super::Reflect,
        {
            &Self {
                ty: ty::<T>,
                format_type_name: format_type_name::<T>,
                type_id: type_id::<T>,
            }
        }
    }

    fn ty<T>() -> super::Type
    where
        T: super::Reflect,
    {
        <T as super::Reflect>::ty()
    }

    fn format_type_name<T>(f: &mut dyn std::fmt::Write) -> std::fmt::Result
    where
        T: super::Reflect,
    {
        <T as super::Reflect>::format_type_name(f)
    }

    fn type_id<T>() -> std::any::TypeId
    where
        T: super::Reflect,
    {
        std::any::TypeId::of::<T>()
    }
}
