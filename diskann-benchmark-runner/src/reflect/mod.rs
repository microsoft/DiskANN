/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    any::TypeId,
    collections::HashSet,
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
    reflection: &'static dyn internal::Reflect,
}

impl Reflection {
    pub const fn new<T>() -> Self
    where
        T: Reflect,
    {
        Self {
            reflection: &internal::Wrapper::<T>::INSTANCE,
        }
    }

    pub fn ty(&self) -> Type {
        self.reflection.ty()
    }

    pub fn type_name(&self) -> TypeName {
        TypeName(*self)
    }

    pub fn type_id(&self) -> TypeId {
        self.reflection.type_id()
    }

    pub fn render(&self) -> Render {
        Render(*self)
    }

    // pub(crate) fn visit_all_reachable<F, E>(&self, f: F) -> Result<(), E>
    // where
    //     F: FnMut(Reflection) -> Result<(), E>,
    // {
    //     let mut id_map = HashSet::new();
    //     visit_all_reachable(*self, &mut id_map, f)
    // }

    pub(crate) fn visit_unique<F, E>(&self, mut f: F) -> Result<(), E>
    where
        F: FnMut(Reflection) -> Result<bool, E>,
    {
        let mut seen = HashSet::new();
        self.visit_with(|r: Reflection| {
            if seen.insert(r.type_id()) {
                f(r)?;
                Ok(true)
            } else {
                Ok(false)
            }
        })
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
        self.0.reflection.format_type_name(f)
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
        let mut r = render::Renderer::new(f, 3);
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
        Type::primitive(None)
    }

    fn format_type_name(f: &mut dyn Write) -> fmt::Result {
        write!(f, "PhantomData<{}>", Reflection::new::<T>().type_name())
    }
}

macro_rules! primitive {
    ($T:ty, $doc:literal, $type_name:literal) => {
        impl Reflect for $T {
            fn ty() -> Type {
                Type::primitive(Some($doc.into()))
            }

            fn format_type_name(f: &mut dyn Write) -> fmt::Result {
                f.write_str($type_name)
            }
        }
    };
}

primitive!((), "empty", "()");
primitive!(usize, "A system dependent unsigned integer", "usize");
primitive!(u32, "A 32-bit unsigned integer", "u32");
primitive!(bool, "A value of \"true\" or \"false\"", "bool");

primitive!(String, "A string", "string");

impl<T> Reflect for Option<T>
where
    T: Reflect,
{
    fn ty() -> Type {
        Type::enum_(
            // TODO: Untagged
            tree::EnumRepr::External,
            [
                tree::Variant::new(
                    "<null>",
                    tree::Fields::Unit,
                    Some("Use `null` to indicate that this value does not exist".into()),
                ),
                tree::Variant::new(
                    "<present>",
                    tree::Fields::unnamed([tree::UnnamedField::new::<T>(None)]),
                    Some("Presence indicates the value is present".into()),
                ),
            ],
            Some("An optional configuration".into()),
        )
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

pub(crate) mod internal {
    use std::marker::PhantomData;

    pub(crate) trait Reflect {
        fn ty(&self) -> super::Type;
        fn format_type_name(&self, f: &mut dyn std::fmt::Write) -> std::fmt::Result;
        fn type_id(&self) -> std::any::TypeId;
    }

    pub(crate) struct Wrapper<T>(PhantomData<T>);

    impl<T> Wrapper<T> {
        pub(crate) const INSTANCE: Self = Self::new();

        pub(crate) const fn new() -> Self {
            Self(PhantomData)
        }
    }

    impl<T> Reflect for Wrapper<T>
    where
        T: super::Reflect,
    {
        fn ty(&self) -> super::Type {
            <T as super::Reflect>::ty()
        }

        fn format_type_name(&self, f: &mut dyn std::fmt::Write) -> std::fmt::Result {
            <T as super::Reflect>::format_type_name(f)
        }

        fn type_id(&self) -> std::any::TypeId {
            std::any::TypeId::of::<T>()
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
}
