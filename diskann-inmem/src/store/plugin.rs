/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # EBR lifecycle hooks for [`super::Store`]
//!
//! Please read this section carefully - the protocol is not difficult, but it *is* subtle.
//!
//! The transitions are a simplified version of the protocol described in [`crate::tag`]
//! that storage plugins need to implement to be compatible. A state diagram is shown below:
//!
//! ```text
//!         +--------------- `reclaim` ----------------+
//!         |                                          |
//!         V                                          |
//!   +-----------+                               +----------+
//!   | Available |<---+                          | Retiring |
//!   +-----------+    |                          +----------+
//!         |          |                               ^
//!         |          |                               |
//!         |       `abort`                         `retire`
//!     `acquire`      |                               |
//!         |          |                               |
//!         |      +----------+                   +-----------+
//!         +----->| Slot<'_> |---- `publish` --->| Published |
//!                +----------+                   +-----------+
//!                    |
//!                 `freeze`
//!                    |
//!        +-----------+
//!        |
//!        V
//!    +--------+
//!    | Frozen |
//!    +--------+
//! ```
//!
//! ## Readable States
//!
//! * `published`: **New** references to slots may be given out in the "published" state. It is
//!   possible for a transition to go from "published" to "retiring" while references are lent
//!   out. This is fine as long as the lifetime of these references is bounded by a
//!   [`crate::epoch::Guard`]. Using [`super::Store::guard`] will provide such a guard.
//!
//! * `frozen`: Since "frozen" is a terminal state, it is safe to give out references to
//!   frozen slots.
//!
//! ## Writable States
//!
//! * [`Slot`]: Slots are a little spooky. Plugins can assume that a [`Slot`] for an index
//!   `i` is exclusive for its duration. This means that [`Slot`] implementations can lend
//!   out mutable references to its contents (for example, [`invasive::Slot::as_mut_slice`]).
//!
//!   Code in [`super`] is very careful to maintain this invariant and all users of [`Slot`]
//!   must carefully maintain this as well.
//!
//! * `reclaim`: On a call to [`Plugin::reclaim`], it can be assumed that the plugin has
//!   exclusive access to the indicated slot for the duration of the function call.
//!
//! ## Contracts
//!
//! Users of [`Plugin`] must ensure that the lifecycle shown above is strictly observed.
//! Furthermore, for [`Slot`]s, exactly one of the terminal methods **must** be called.
//!
//! State transitions are driven by the authoritative [`super::Store`]. Before invoking a
//! plugin transition, the store ensures the slot is not externally available in its previous
//! state. Further, the store commits the destination state only after the plugin API call
//! completes.

use std::fmt::Debug;

use crate::num::IdLimit;

use super::Lifecycle;

/// A configuration for a [`Plugin`].
pub(crate) trait PluginConfig: Debug {
    /// The type of the resulting [`Plugin`].
    type Plugin: Plugin;

    /// Construction errors.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Build the associated [`Plugin`] from self with the [`IdLimit`].
    fn build(self, id_limit: IdLimit) -> Result<Self::Plugin, Self::Error>;
}

/// A lifecycle backend for [`super::Store`]'s EBR scheme.
///
/// See the [module level documentation](self) for details.
pub(crate) trait Plugin: Debug + 'static {
    /// The writable [`Slot`] for this plugin.
    type Slot<'a>: Slot;

    /// Return the exclusive upper bound for indices provided to this API.
    ///
    /// Callers should ensure that indices are in the range `[0..self.id_limit())`.
    fn id_limit(&self) -> IdLimit;

    /// Immediately transition slot `i` from the "available" state to the "slot" state.
    ///
    /// Implementations may panic when `i` is out-of-bounds, but must not rely on
    /// `i < Self::id_limit` for memory safety.
    ///
    /// # Safety
    ///
    /// Callers must ensure **all** of the following:
    ///
    /// 1. The plugin is in the implicit "available" state according to the [module docs](self).
    ///
    /// 2. Access to slot `i` is exclusive before invoking this method and that exclusivity
    ///    is maintained until the returned [`Slot`] is consumed by a terminal method.
    ///
    /// 3. Exactly one of the [`Slot`] terminal methods is called. The [`Slot`] **may not**
    ///    be dropped or forgotten without one of these methods being called.
    unsafe fn acquire(&self, i: u32, _: Lifecycle) -> Self::Slot<'_>;

    /// Transition slot `i` from the "published" state to the "retiring" state.
    ///
    /// Implementations may panic when `i` is out-of-bounds, but must not rely on
    /// `i < Self::id_limit` for memory safety.
    ///
    /// # Safety
    ///
    /// The plugin is in the implicit "published" state.
    unsafe fn retire(&self, i: u32, _: Lifecycle);

    /// Transition slot `i` from the "retiring" state to the "available" state.
    ///
    /// Implementations may panic when `i` is out-of-bounds, but must not rely on
    /// `i < Self::id_limit` for memory safety.
    ///
    /// # Safety
    ///
    /// Callers must ensure **all** of the following:
    ///
    /// 1. The plugin is in the implicit "retiring" state.
    ///
    /// 2. All [`crate::epoch::Guard`]s for this [`Plugin`] that could have obtained a
    ///    reference while this slot was in the "published" state have been dropped.
    unsafe fn reclaim(&self, i: u32, _: Lifecycle);
}

/// A writable slot for [`Plugin`].
///
/// [`Slot`]s may assume that they have exclusive ownership of their plugin slots for their
/// duration in accordance with [`Plugin::acquire`].
pub(crate) trait Slot: Debug {
    /// Mark this slot as readable, transition it to the "published" state.
    fn publish(self, _: Lifecycle);

    /// Mark this slot as "frozen".
    fn freeze(self, _: Lifecycle);

    /// Abort any action, returning the slot to "available".
    fn abort(self, _: Lifecycle);
}
