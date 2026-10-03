/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Run time inspection of types.

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

/// Provide run-time information about compile-time types, including documentation.
///
/// This trait is derivable and supports a subset of `serde` attributes.
///
/// ```
/// use diskann_benchmark_runner::{Reflect, Reflection};
///
/// /// An example struct.
/// #[derive(Reflect)]
/// struct Foo {
///     /// An awesome field.
///     #[serde(rename = "bar")]
///     foo: usize,
///     baz: usize,
/// }
///
/// // Get information about `Foo`.
/// let ty = Foo::ty();
///
/// // The documentation of `Foo` will automatically be extracted from the docstrings.
/// assert_eq!(ty.doc().unwrap(),  "An example struct.");
///
/// // The type name can be extracted as well.
/// let mut name = String::new();
/// Foo::format_type_name(&mut name);
/// assert_eq!(name, "Foo");
///
/// // To make typenames easier to extract, a `Reflection` can be use.
/// let reflection = Reflection::new::<Foo>();
/// assert_eq!(reflection.type_name().to_string(), "Foo");
/// ```
///
/// # Attributes
///
/// For derivation, a combination of `serde` attributes and custom `reflect` attributes
/// are supported.
///
/// ## Serde Attributes
///
/// Like the [serde crate](https://serde.rs/attributes.html), attributes are categorized by
/// container, variant, or field.
///
/// ### Container Attributes
///
/// * `#[serde(rename_all = "...")]`: Rename all fields (if a struct) or variants (if enum)
///   arrocding to the given case.
///
///   Possible values are "lowercase", "snake_case", and "kebab-case".
///
/// * `#[serde(tag = "type")]`: Used for internally tagged enums.
///
/// * `#[serde(tag = "t", content = "c")]`: Used for adjacently tagged enums.
///
/// ### Variant Attributes
///
/// * `#[serde(rename = "name")]`: Describe with the given name instead of its Rust name.
///
/// * `#[serde(rename_all = "...)]`: Rename all fiels of this struct variant with the given
///   case convention.
///
///   Possible values are "lowercase", "snake_case", and "kebab-case".
///
/// ### Field Attributes
///
/// * `#[serde(rename = "name")]`: Describe with the given name instead of its Rust name.
///
/// ## Reflect Attributes
///
/// ### Container Attributes
///
/// * `#[reflect(type_name = "name")]`: Use the given name to describe a struct instead of
///   its Rust name. This is used to avoid naming conflicts within a [`Registry`], which
///   enforces that type names are unique.
///
///   Because of this, this attribute cannot be used on generic structs.
///
///   This is mutually exclusive with the `prefix` attribute.
///
///   ```
///   use diskann_benchmark_runner::{Reflect, Reflection};
///
///   #[derive(Reflect)]
///   #[reflect(type_name = "Bar")]
///   struct Foo;
///
///   assert_eq!(Reflection::new::<Foo>().type_name().to_string(), "Bar");
///   ```
///
/// * `#[reflect(prefix = "...")]`: Prefix the Rust name with the provided prefix. Like the
///   `type_name` attribute, this can be used to create name spaces to help generate unique
///   type names.
///
///   ```
///   use diskann_benchmark_runner::{Reflect, Reflection};
///
///   #[derive(Reflect)]
///   #[reflect(prefix = "mod::")]
///   struct Foo;
///
///   assert_eq!(Reflection::new::<Foo>().type_name().to_string(), "mod::Foo");
///   ```
pub trait Reflect: 'static {
    /// Return the [`Type`] containing the information about `self`.
    fn ty() -> Type;

    /// Write the type-name for `self` into the buffer.
    fn format_type_name(f: &mut dyn Write) -> fmt::Result;
}

/// A [`Reflect`]ed type.
#[derive(Clone, Copy)]
pub struct Reflection {
    reflection: &'static internal::VTable,
}

impl Reflection {
    /// Construct a new [`Reflection`] for `T`.
    pub const fn new<T>() -> Self
    where
        T: Reflect,
    {
        Self {
            reflection: internal::VTable::new::<T>(),
        }
    }

    /// Return the [`Type`] for the type being reflected.
    pub fn ty(&self) -> Type {
        (self.reflection.ty)()
    }

    /// Return a [`std::fmt::Display`] compatible struct for rendering the name of the type
    /// being reflected.
    pub fn type_name(&self) -> TypeName {
        TypeName(*self)
    }

    /// Return the [`TypeId`] of the
    pub fn type_id(&self) -> TypeId {
        (self.reflection.type_id)()
    }

    /// Return a [`std::fmt::Display`] comaptible struct for rendering the reflected type.
    pub(crate) fn render(&self) -> Render {
        Render(*self)
    }

    /// Visit all types reachable from the reflected type.
    ///
    /// This will traverse through all structs, enum variants, container types etc.
    ///
    /// The closure `f` can be used to direct the exploration by returning the following values:
    ///
    /// * `Ok(true)`: Continue exploring through the argument [`Reflection`].
    /// * `Ok(false)`: Do not continue exploring through the argument [`Reflection`].
    /// * `Err(E)`: Immediately stop exploring and return the error `E`.
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

/// A [`std::fmt::Display`] compatible type for [`Reflection`].
///
/// See: [`Reflection::type_name`].
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

/// A [`std::fmt::Display`] compatible type for rendering a [`Reflection`].
///
/// See: [`Reflection::render`].
pub(crate) struct Render(Reflection);

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
    std::num::NonZeroU32,
    Number,
    "A system dependent, 32-bit unsigned integer",
    "NonZero<u32>"
);
primitive!(
    std::num::NonZeroUsize,
    Number,
    "A system dependent, non-zero, unsigned integer",
    "NonZero<usize>"
);

primitive!(bool, Boolean, "A value of \"true\" or \"false\"", "bool");
primitive!(String, String, "A string", "string");
primitive!(std::path::PathBuf, String, "A file path", "PathBuf");

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
