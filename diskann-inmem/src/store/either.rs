// /*
//  * Copyright (c) Microsoft Corporation.
//  * Licensed under the MIT license.
//  */
//
// use thiserror::Error;
//
// use crate::{
//     epoch,
//     num::IdLimit,
//     store::{Lifecycle, slots},
//     tag,
// };
//
// /// A [`slots::SlotsConfig`] for [`Either`].
// #[derive(Debug)]
// pub(crate) struct Config<F, S> {
//     inner: ConfigInner<F, S>,
// }
//
// #[derive(Debug)]
// enum ConfigInner<F, S> {
//     First(F),
//     Second(S),
// }
//
// impl<F, S> Config<F, S> {
//     pub(crate) fn first(first: F) -> Self {
//         Self {
//             inner: ConfigInner::First(first),
//         }
//     }
//
//     pub(crate) fn second(second: S) -> Self {
//         Self {
//             inner: ConfigInner::Second(second),
//         }
//     }
// }
//
// impl<F, S> slots::SlotsConfig for Config<F, S>
// where
//     F: slots::SlotsConfig,
//     S: slots::SlotsConfig,
// {
//     type Slots = Either<F::Slots, S::Slots>;
//     type Error = EitherError<F::Error, S::Error>;
//
//     unsafe fn build(
//         self,
//         handle: epoch::RegistryHandle,
//         tags: &tag::Authoritative,
//     ) -> Result<Self::Slots, Self::Error> {
//         match self.inner {
//             ConfigInner::First(first) => {
//                 // SAFETY: Inherited from caller.
//                 let first = unsafe { first.build(handle, tags) }.map_err(EitherError::First)?;
//                 Ok(Either::First(first))
//             }
//             ConfigInner::Second(second) => {
//                 // SAFETY: Inherited from caller.
//                 let second = unsafe { second.build(handle, tags) }.map_err(EitherError::Second)?;
//                 Ok(Either::Second(second))
//             }
//         }
//     }
// }
//
// #[derive(Debug, Error)]
// pub(crate) enum EitherError<F, S> {
//     #[error("couldn't construct first slots")]
//     First(#[source] F),
//     #[error("couldn't construct second slots")]
//     Second(#[source] S),
// }
//
// #[derive(Debug)]
// pub(crate) enum Either<F, S> {
//     First(F),
//     Second(S),
// }
//
// impl<F, S> slots::Slots for Either<F, S>
// where
//     F: slots::Slots,
//     S: slots::Slots,
// {
//     type Exclusive<'a> = Exclusive<F::Exclusive<'a>, S::Exclusive<'a>>;
//
//     fn id_limit(&self) -> IdLimit {
//         match self {
//             Self::First(first) => first.id_limit(),
//             Self::Second(second) => second.id_limit(),
//         }
//     }
//
//     unsafe fn acquire(&self, i: u32, _: Lifecycle) -> Self::Exclusive<'_> {
//         // SAFETY: Inherited from caller.
//         unsafe {
//             match self {
//                 Self::First(first) => Exclusive::First(first.acquire(i, Lifecycle::new())),
//                 Self::Second(second) => Exclusive::Second(second.acquire(i, Lifecycle::new())),
//             }
//         }
//     }
//
//     unsafe fn retire(&self, i: u32, _: Lifecycle) {
//         // SAFETY: Inherited from caller.
//         unsafe {
//             match self {
//                 Self::First(first) => first.retire(i, Lifecycle::new()),
//                 Self::Second(second) => second.retire(i, Lifecycle::new()),
//             }
//         }
//     }
//
//     unsafe fn reclaim(&self, i: u32, _: Lifecycle) {
//         // SAFETY: Inherited from caller.
//         unsafe {
//             match self {
//                 Self::First(first) => first.reclaim(i, Lifecycle::new()),
//                 Self::Second(second) => second.reclaim(i, Lifecycle::new()),
//             }
//         }
//     }
// }
//
// #[derive(Debug)]
// pub(crate) enum Exclusive<F, S> {
//     First(F),
//     Second(S),
// }
//
// impl<F, S> slots::Exclusive for Exclusive<F, S>
// where
//     F: slots::Exclusive,
//     S: slots::Exclusive,
// {
//     fn publish(self, _: Lifecycle) {
//         match self {
//             Self::First(first) => first.publish(Lifecycle::new()),
//             Self::Second(second) => second.publish(Lifecycle::new()),
//         }
//     }
//
//     fn freeze(self, _: Lifecycle) {
//         match self {
//             Self::First(first) => first.freeze(Lifecycle::new()),
//             Self::Second(second) => second.freeze(Lifecycle::new()),
//         }
//     }
//
//     fn abort(self, _: Lifecycle) {
//         match self {
//             Self::First(first) => first.abort(Lifecycle::new()),
//             Self::Second(second) => second.abort(Lifecycle::new()),
//         }
//     }
// }
