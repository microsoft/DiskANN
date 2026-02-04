/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A moderately functional utility for making simple benchmarking CLI applications.

mod checker;
mod jobs;
mod result;

pub mod any;
pub mod app;
pub mod dispatcher;
pub mod files;
pub mod output;
pub mod registry;
pub mod utils;

pub use any::Any;
pub use app::App;
pub use checker::{CheckDeserialization, Checker};
pub use output::Output;
pub use result::Checkpoint;

#[cfg(any(test, feature = "test-app"))]
pub mod test;

#[cfg(any(test, feature = "ux-tools"))]
#[doc(hidden)]
pub mod ux;

//-------//
// Input //
//-------//

pub trait Input {
    /// Return the discriminant associated with this type.
    ///
    /// This is used to map inputs types to their respective parsers.
    ///
    /// Well formed implementations should always return the same result.
    fn tag(&self) -> &'static str;

    /// Attempt to deserialize an opaque object from the raw `serialized` representation.
    ///
    /// Deserialized values can be constructed and returned via [`Checker::any`],
    /// [`Any::new`] or [`Any::raw`].
    ///
    /// If using the [`Any`] constructors directly, implementations should associate
    /// [`Self::tag`] with the returned `Any`. If [`Checker::any`] is used - this will
    /// happen automatically.
    ///
    /// Implementations are **strongly** encouraged to implement [`CheckDeserialization`]
    /// and use this API to ensure shared resources (like input files or output files)
    /// are correctly resolved and properly shared among all jobs in a benchmark run.
    fn try_deserialize(
        &self,
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any>;

    /// Print an example JSON representation of objects this input is expected to parse.
    ///
    /// Well formed implementations should passing the returned [`serde_json::Value`] back
    /// to [`Self::try_deserialize`] correctly deserializes, though it need not necessarily
    /// pass [`CheckDeserialization`].
    fn example(&self) -> anyhow::Result<serde_json::Value>;
}
