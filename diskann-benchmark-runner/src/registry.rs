/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::HashMap;

use thiserror::Error;

use crate::{
    dispatcher::{DispatchRule, Map},
    output::Sink,
    Any, Checkpoint, Input, Output,
};

/// A collection of [`crate::Input`].
pub struct Inputs {
    // Inputs keyed by their tag type.
    inputs: HashMap<&'static str, Box<dyn Input>>,
}

impl Inputs {
    /// Construct a new empty [`Inputs`] registry.
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
        }
    }

    /// Return the input with the registerd `tag` if present. Otherwise, return `None`.
    pub fn get(&self, tag: &str) -> Option<&dyn Input> {
        match self.inputs.get(tag) {
            Some(v) => Some(&**v),
            None => None,
        }
    }

    /// Register `input` in the registry.
    ///
    /// Returns an error if any other input with the same [`Input::tag()`] has been registered
    /// while leaving the underlying registry unchanged.
    pub fn register<T>(&mut self, input: T) -> anyhow::Result<()>
    where
        T: Input + 'static,
    {
        use std::collections::hash_map::Entry;

        let tag = input.tag();
        match self.inputs.entry(tag) {
            Entry::Vacant(entry) => {
                entry.insert(Box::new(input));
                Ok(())
            }
            Entry::Occupied(_) => {
                #[derive(Debug, Error)]
                #[error("An input with the tag \"{}\" already exists", self.0)]
                struct AlreadyExists(&'static str);

                Err(anyhow::anyhow!(AlreadyExists(tag)))
            }
        }
    }

    /// Return an iterator over all registered input tags in an unspecified order.
    pub fn tags(&self) -> impl ExactSizeIterator<Item = &'static str> + use<'_> {
        self.inputs.keys().copied()
    }
}

impl Default for Inputs {
    fn default() -> Self {
        Self::new()
    }
}

/// A collection of registerd benchmarks.
pub struct Benchmarks {
    dispatcher: Dispatcher,
}

impl Benchmarks {
    /// Return a new empty registry.
    pub fn new() -> Self {
        Self {
            dispatcher: Dispatcher::new(),
        }
    }

    /// Register a new benchmark with the given name.
    ///
    /// The type parameter `T` is used to match this benchmark with a registered
    /// [`crate::Any`], which is determined using by `<T as DispatchRule<&Any>>`.
    pub fn register<T>(
        &mut self,
        name: impl Into<String>,
        benchmark: impl Fn(T::Type<'_>, Checkpoint<'_>, &mut dyn Output) -> anyhow::Result<serde_json::Value>
            + 'static,
    ) where
        T: for<'a> Map<Type<'a>: DispatchRule<&'a Any>>,
    {
        self.dispatcher
            .register::<_, T, CheckpointRef, DynOutput>(name.into(), benchmark)
    }

    pub(crate) fn methods(&self) -> impl ExactSizeIterator<Item = &(String, Method)> {
        self.dispatcher.methods()
    }

    /// Return `true` if `job` matches with any registerd benchmark. Otherwise, return `false`.
    pub fn has_match(&self, job: &Any) -> bool {
        let sink: &mut dyn Output = &mut Sink::new();
        self.dispatcher.has_match(&job, &Checkpoint::empty(), &sink)
    }

    /// Attempt to the best matching benchmark for `job` - forwarding the `checkpoint` and
    /// `output` to the benchmark.
    ///
    /// Returns the results of the benchmark if successful.
    ///
    /// Errors if a suitable method could not be found or if the invoked benchmark failed.
    pub fn call(
        &self,
        job: &Any,
        checkpoint: Checkpoint<'_>,
        output: &mut dyn Output,
    ) -> anyhow::Result<serde_json::Value> {
        self.dispatcher.call(job, checkpoint, output).unwrap()
    }

    /// Attempt to debug reasons for a missed dispatch, returning at most `methods` reasons.
    ///
    /// This implementation works by invoking [`DispatchRule::try_match`] with
    /// `job` on all registered benchmarks. If no successful matches are found, the lowest
    /// ranking [`crate::dispatcher::FailureScore`]s are collected and used to report details
    /// of the nearest misses using [`DispatchRule::description`].
    ///
    /// Returns `Ok(())` is a match was found.
    pub fn debug(&self, job: &Any, methods: usize) -> Result<(), Vec<Mismatch>> {
        let checkpoint = Checkpoint::empty();
        let sink: &mut dyn Output = &mut Sink::new();
        let mismatches = match self.dispatcher.debug(methods, &job, &checkpoint, &sink) {
            Ok(()) => return Ok(()),
            Err(mismatches) => mismatches,
        };

        // Just retrieve the mismatch information for the first argument since that is the
        // one that does all the heavy lifting.
        Err(mismatches
            .into_iter()
            .map(|m| {
                let reason = m.mismatches()[0]
                    .as_ref()
                    .map(|opt| opt.to_string())
                    .unwrap_or("<missing>".into());
                Mismatch {
                    method: m.method().to_string(),
                    reason,
                }
            })
            .collect())
    }
}

impl Default for Benchmarks {
    fn default() -> Self {
        Self::new()
    }
}

/// Document the reason for a method mathing failure.
pub struct Mismatch {
    method: String,
    reason: String,
}

impl Mismatch {
    /// Return the name of the benchmark that we failed to match.
    pub fn method(&self) -> &str {
        &self.method
    }

    /// Return the reason why this method was not a match.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

//------------------//
// Dispatch Helpers //
//------------------//

/// A [`Map`] for `&mut dyn Output`.
pub(crate) struct DynOutput;

impl Map for DynOutput {
    type Type<'a> = &'a mut dyn Output;
}

/// A dispatcher compatible mapper for [`Checkpoint`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct CheckpointRef;

impl Map for CheckpointRef {
    type Type<'a> = Checkpoint<'a>;
}

/// The internal `Dispatcher` used for method resolution.
type Dispatcher = crate::dispatcher::Dispatcher3<
    anyhow::Result<serde_json::Value>,
    crate::dispatcher::Ref<Any>,
    CheckpointRef,
    DynOutput,
>;

/// The concrete type of a method.
type Method = Box<
    dyn crate::dispatcher::Dispatch3<
        anyhow::Result<serde_json::Value>,
        crate::dispatcher::Ref<Any>,
        CheckpointRef,
        DynOutput,
    >,
>;
