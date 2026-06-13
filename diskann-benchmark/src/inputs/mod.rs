/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) mod disk;
pub(crate) mod exhaustive;
pub(crate) mod filters;
pub(crate) mod flat;
pub(crate) mod graph_index;
pub(crate) mod multi_vector;
pub(crate) mod save_and_load;

#[cfg(feature = "bftree")]
pub(crate) mod bftree;

/// Construct an example input of type `Self`.
pub(crate) trait Example {
    fn example() -> Self;
}

/// Implement [`diskann_benchmark_runner::Input`] for `$T` using `Raw = $T`.
///
/// Requires `$T` to:
/// - implement [`Example`];
/// - provide an inherent `fn tag() -> &'static str` method;
/// - provide a
///   `fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()>` method; and
/// - implement the serde traits required by
///   [`diskann_benchmark_runner::Input`] and `serde_json::to_value(self)`.
macro_rules! as_input {
    ($T:ty) => {
        impl diskann_benchmark_runner::Input for $T {
            type Raw = $T;

            fn tag() -> &'static str {
                <$T>::tag()
            }

            fn from_raw(
                mut raw: Self::Raw,
                checker: &mut diskann_benchmark_runner::Checker,
            ) -> anyhow::Result<Self> {
                raw.validate(checker)?;
                Ok(raw)
            }

            fn serialize(&self) -> anyhow::Result<serde_json::Value> {
                Ok(serde_json::to_value(self)?)
            }

            fn example() -> Self {
                <$T as $crate::inputs::Example>::example()
            }
        }
    };
}

// This constant is used to ensure that summaries of graph-index related jobs properly have
// their field descriptions aligned.
const PRINT_WIDTH: usize = 18;

macro_rules! write_field {
    ($f:ident, $field:tt, $($expr:tt)*) => {
        writeln!($f, "{:>PRINT_WIDTH$}: {}", $field, $($expr)*)
    }
}

use as_input;
use write_field;
