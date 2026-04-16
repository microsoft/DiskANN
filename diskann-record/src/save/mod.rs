/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod value;
pub use value::{Handle, Record, Value, Versioned};

mod context;
pub use context::Context;

mod error;
pub use error::{Error, Result};

use crate::Version;

pub fn save_to_disk<T>(x: &T, dir: impl AsRef<std::path::Path>, metadata: impl AsRef<std::path::Path>) -> Result<()>
where
    T: Saveable,
{
    let inner = context::ContextInner::new(dir.as_ref().into(), metadata.as_ref().into());
    let value = x.save(inner.context())?;
    inner.finish(value)
}

/// Save objects!
pub trait Save {
    const VERSION: Version;
    fn save(&self, context: Context<'_>) -> Result<Record<'_>>;
}

/// Save anything!
pub trait Saveable {
    fn save(&self, context: Context<'_>) -> Result<Value<'_>>;
}

impl<T> Saveable for T
where
    T: Save,
{
    fn save(&self, context: Context<'_>) -> Result<Value<'_>> {
        let record = self.save(context)?;
        let versioned = Versioned::new(record, T::VERSION);
        Ok(Value::Object(versioned))
    }
}

//////////////////
// Random Stuff //
//////////////////

impl<T> Saveable for [T]
where
    T: Saveable,
{
    fn save(&self, mut context: Context<'_>) -> Result<Value<'_>> {
        let values: Result<Vec<_>> = self.iter().map(|t| t.save(context.clone())).collect();
        values.map(Value::Array)
    }
}

impl<T> Saveable for Vec<T>
where
    T: Saveable,
{
    fn save(&self, context: Context<'_>) -> Result<Value<'_>> {
        self.as_slice().save(context)
    }
}

impl Saveable for str {
    fn save(&self, _: Context<'_>) -> Result<Value<'_>> {
        Ok(Value::String(self.into()))
    }
}

impl Saveable for String {
    fn save(&self, _: Context<'_>) -> Result<Value<'_>> {
        Ok(Value::String(self.as_str().into()))
    }
}

impl Saveable for Handle {
    fn save(&self, _: Context<'_>) -> Result<Value<'_>> {
        Ok(Value::Handle(self.clone()))
    }
}

macro_rules! save_number {
    ($T:ty) => {
        impl Saveable for $T {
            fn save(&self, _: Context<'_>) -> Result<Value<'_>> {
                Ok(Value::Number((*self).into()))
            }
        }
    };
    ($($Ts:ty),+ $(,)?) => {
        $(save_number!($Ts);)+
    }
}

save_number!(usize, u64, u32, u16, u8, i64, i32, i16, i8, f32, f64);

#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct Serializing(pub &'static str);

impl std::fmt::Display for Serializing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "while serializing field \"{}\"", self.0)
    }
}

#[macro_export]
macro_rules! save_fields {
    ($me:ident, $context:ident, [$($field:ident),+ $(,)?]) => {{
        $crate::save::Record::from_iter(
            [
                $(
                    (
                        ::std::borrow::Cow::Borrowed(stringify!($field)),
                        <_ as $crate::save::Saveable>::save(
                            &$me.$field,
                            $context.clone()
                        ).map_err(|err| {
                            err.context($crate::save::Serializing(stringify!($field)))
                        })?
                    ),
                )+
            ]
        )
    }};
}
