/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod value;
pub use value::{Handle, Record, Value, Versioned};

mod context;
pub use context::Context;

use crate::Version;

/// Save objects!
pub trait Save {
    const VERSION: Version;
    fn save<'a>(&'a self, context: Context<'a>) -> Record<'a>;
}

/// Save anything!
pub trait Saveable {
    fn save<'a>(&'a self, context: Context<'a>) -> Value<'a>;
}

impl<T> Saveable for T
where
    T: Save,
{
    fn save<'a>(&'a self, context: Context<'a>) -> Value<'a> {
        let record = self.save(context);
        let versioned = Versioned::new(record, T::VERSION);
        Value::Object(versioned)
    }
}

//////////////////
// Random Stuff //
//////////////////

impl<T> Saveable for [T]
where
    T: Saveable,
{
    fn save<'a>(&'a self, mut context: Context<'a>) -> Value<'a> {
        let values = self.iter().map(|t| t.save(context.clone())).collect();
        Value::Array(values)
    }
}

impl<T> Saveable for Vec<T>
where
    T: Saveable,
{
    fn save<'a>(&'a self, context: Context<'a>) -> Value<'a> {
        self.as_slice().save(context)
    }
}

impl Saveable for str {
    fn save<'a>(&'a self, _: Context<'a>) -> Value<'a> {
        Value::String(self.into())
    }
}

impl Saveable for String {
    fn save<'a>(&'a self, _: Context<'a>) -> Value<'a> {
        Value::String(self.as_str().into())
    }
}

impl Saveable for Handle {
    fn save<'a>(&'a self, _: Context<'a>) -> Value<'a> {
        Value::Handle(self.clone())
    }
}

macro_rules! save_number {
    ($T:ty) => {
        impl $crate::save::Saveable for $T {
            fn save<'a>(
                &'a self,
                _: $crate::save::Context<'a>
            ) -> $crate::save::Value<'a> {
                $crate::save::Value::Number((*self).into())
            }
        }
    };
    ($($Ts:ty),+ $(,)?) => {
        $(save_number!($Ts);)+
    }
}

save_number!(usize, u64, u32, u16, u8, i64, i32, i16, i8, f32, f64);

#[macro_export]
macro_rules! save_fields {
    ($me:ident, $context:ident, [$($field:ident),+ $(,)?]) => {{
        // // Check for reserved keywords.
        // $(
        //     const {
        //         assert!(
        //             !$crate::is_reserved(stringify!($field)),
        //             concat!("field \"", stringify!($field), "\" cannot start with \"$\""),
        //         );
        //     }
        // )+

        $crate::save::Record::from_iter(
            [
                $(
                    (
                        ::std::borrow::Cow::Borrowed(stringify!($field)),
                        <_ as $crate::save::Saveable>::save(&$me.$field, $context.clone()),
                    ),
                )+
            ]
        )
    }};
}
