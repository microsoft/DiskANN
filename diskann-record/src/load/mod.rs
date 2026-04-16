/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod error;
pub use error::{Error, Result};

mod context;
pub use context::{Context, Object};

use crate::{Version, save};

pub trait Load<'a>: Sized {
    const VERSION: Version;
    fn load(object: Object<'a>) -> Result<Self>;
    fn load_legacy(object: Object<'a>) -> Result<Self>;
}

pub trait Loadable<'a>: Sized {
    fn load(context: Context<'a>) -> Result<Self>;
}

impl<'a, T> Loadable<'a> for T
where
    T: Load<'a>,
{
    fn load(context: Context<'a>) -> Result<Self> {
        let object = context
            .as_object()
            .ok_or_else(|| error::Kind::TypeMismatch)?;
        let version = object.version();
        if version == T::VERSION {
            T::load(object)
        } else {
            T::load_legacy(object)
        }
    }
}

////////////
// Macros //
////////////

#[macro_export]
macro_rules! load_fields {
    (@field $object:ident, $field:ident: $T:ty) => {
        let $field: $T = $object.field(stringify!($field))?;
    };
    (@field $object:ident, $field:ident) => {
        let $field = $object.field(stringify!($field))?;
    };
    ($object:ident, [$($field:ident $(: $ty:ty)?),+ $(,)?]) => {
        $(
            $crate::load_fields!(@field $object, $field $(: $ty)?);
        )+
    };
}

///////////////
// Bootstrap //
///////////////

impl<'a> Loadable<'a> for &'a str {
    fn load(context: Context<'a>) -> Result<Self> {
        context
            .as_str()
            .map(|s| s.into())
            .ok_or_else(|| error::Kind::TypeMismatch.into())
    }
}

impl Loadable<'_> for String {
    fn load(context: Context<'_>) -> Result<Self> {
        context.load::<&str>().map(|s| s.into())
    }
}

impl<'a, T> Loadable<'a> for Vec<T>
where
    T: Loadable<'a>,
{
    fn load(context: Context<'a>) -> Result<Self> {
        match context.as_array() {
            Some(array) => array.iter().map(T::load).collect(),
            None => Err((error::Kind::TypeMismatch).into()),
        }
    }
}

macro_rules! load_number {
    ($T:ty) => {
        impl Loadable<'_> for $T {
            fn load(context: Context<'_>) -> Result<Self> {
                match context.as_number() {
                    Some(n) => Ok(n.try_into().unwrap()),
                    None => Err((error::Kind::TypeMismatch).into()),
                }
            }
        }
    };
    ($($Ts:ty),+ $(,)?) => {
        $(load_number!($Ts);)+
    }
}

load_number!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);
