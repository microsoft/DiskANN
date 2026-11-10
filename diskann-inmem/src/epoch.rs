/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The core logic for the epoch-based reclamation algorithm.
//!
//! ## What Problem is Being Solved?
//!
//! Epoch-based reclamation (EBR) can be used to safely implement read-heavy algorithms with
//! a moderate level of concurrent writes. In this context, we want readers to be able to ask
//! the question: "can I safely read some data" in a way that generates only read traffic to
//! the CPU caches.
//!
//! The crux is that after the safety check, a reader can hold a reference to the associated
//! data for an arbitrary period of time. Any actor trying to *write* to that data needs
//! to figure out when it is safe to do so.
//!
//! EBR solves this problem by separating when data is "retired" versus "reclaimed".
//! Retirement involves disabling the safety check. When an item is retired, concurrent
//! readers will fail the safety check and no longer try to read the associated data.
//! However, we still need to wait until we can prove that readers who passed the safety
//! check before retirement are no longer accessing the data. At this point, the data can be
//! "reclaimed" and written to safely.
//!
//! We can prove this by using a monotonically increasing epoch: if an item was "retired"
//! at epoch `N` its associated data could be in use by any reader belonging to any epoch
//! `N` or lower. Therefore, it is only safe to "reclaim" when all readers belong to epoch
//! `N+1` or higher.
//!
//! One consequence of this design is that misbehaving (e.g. long-lived) readers can delay
//! reclamation indefinitely. As such, this system must be used with care and in situations
//! where there is enough slack in the system to accommodate the lifetime of any readers.
//!
//! ## Primitives
//!
//! Actors call one of [`Registry::read_guard`] or [`Registry::write_guard`] to receive a
//! [`ReadGuard`] or [`WriteGuard`] respectively, depending on the requested capabilities
//! of the worker. This guard protects items at its creation epoch. Any items "retired" while
//! the guard is held will be buffered until the guard is destroyed.
//!
//! The difference between a [`ReadGuard`] and a [`WriteGuard`] is solely for controlling
//! what workers get excluded by [`Registry::try_exclude_workers`] and
//! [`Registry::try_exclude_writers`]. More on that later.
//!
//! To retire an item, [`Registry::retire`] is used to receive a [`Retire`]. This behaves
//! exactly like a [`ReadGuard`] with the extra capability of retiring items. All items
//! retired through [`Retire::retire`] will be buffered until the [`Registry`] can prove that
//! all guards (correctly using the data structure) that could have observed the retired item
//! have been destroyed.
//!
//! Items can be reclaimed via [`Registry::try_advance`]. If successful, a [`Drain`] of
//! such items will be returned for processing.
//!
//! Note that retired payloads are fixed to `u32` ids (typically interpreted by the caller
//! as indices into some external storage); this is not a general-purpose deferred-drop EBR
//! system.
//!
//! Additionally, [`ReadGuard`]s are cheaply cloneable using [`ReadGuard::share`]. These
//! behave like [`std::sync::Arc`] with atomic reference counting, but do not require an
//! initial allocation as the reference count lives inside the [`Registry`]. This enables
//! uses where an initial [`ReadGuard`] is acquired, then shared among multiple dataset
//! readers.
//!
//! Finally, [`RegistryHandle`]s can be created with [`Registry::handle`]. Such a handle may
//! check that a [`ReadGuard`] actually belongs to registry with
//! [`RegistryHandle::assert_guard_belongs`].
//!
//! ## Worker Exclusion
//!
//! To facilitate cases for a coherent shapshot of data controlled by a [`Registry`] is
//! needed the functions [`Registry::try_exclude_workers`] and
//! [`Registry::try_exclude_writers`] can be used. These will wait for all current workers
//! of the requested type to complete and new worker registration while their respective
//! guards are held.
//!
//! Note that this is merely provided as a means of waiting and blocking. It's the
//! responsibility of the user of the [`Registry`] to register read-only workers and writers
//! in a way that takes advantage of this functionality.

mod internal_docs {
    //! The EBR implementation is separated into two logical interfaces:
    //!
    //! * Worker Interface: This is interface used by threads that are accessing the EBR
    //!   protected data and consists of the functions:
    //!
    //!   - [`Registry::read_guard`]: Construct a [`ReadGuard`] to protect data.
    //!
    //!   - [`Registry::write_guard`]: Construct a [`WriteGuard`] to protect data and declare
    //!     the worker as one that is mutating data.
    //!
    //!   - [`Registry::retire`]: A specialized [`WriteGuard`] that also allows items to be
    //!     retired to wait until it is safe to access their corresponding slots again.
    //!
    //!   Many threads can use this interface concurrently.
    //!
    //! * Control Interface: Used to perform "registry maintenance" style tasks like:
    //!
    //!   - [`Registry::try_advance`]: Advance the registry's epoch if all registered workers
    //!     belong to the current epoch.
    //!
    //!   - [`Registry::try_exclude_writers`]: Block writers from registering and wait for
    //!     all current writers to finish.
    //!
    //!   - [`Registry::try_exclude_workers`]: Block all workers from registering and wait
    //!     for all current workers to finish.
    //!
    //!   Importantly, **all** controls side operations use a single shared mutex to ensure
    //!   mutual exclusion of these methods. This is crucial for the correctness of the
    //!   algorithm.
    //!
    //! # Registration Handshake and Epoch Advancement
    //!
    //! The interaction between the worker and control interface is done through a
    //! registry's central [`Epoch`] and a collection of [`GuardSlot`]s.
    //!
    //! Worker threads read the central epoch and claim an unoccupied [`GuardSlot`].
    //!
    //! Control threads wishing to advance the epoch must scan all occupied guard slots to
    //! check if all worker threads belong to the current epoch. Only if that is the case is
    //! it safe to advance the epoch.
    //!
    //! Care is needed to protect against the situation where a worker thread reads the epoch,
    //! gets suspended while a control thread scans and advances the epoch, and then resumes
    //! by claiming a slot in the older epoch.
    //!
    //! To do this, we use two sequentially consistent fences, one on the control side
    //! (called "CONTROL FENCE" or "CF") and one on the worker wide (called "WORKER FENCE"
    //! or "WF"). Pseudo code for registration is shown below (with some details omitted
    //! for clarity):
    //! ```text
    //! use std::sync::atomic::{fence, Ordering};
    //!
    //! // Control Side
    //! fn scan(...) -> bool {
    //!     // CONTROL FENCE (CF)
    //!     fence(Ordering::SeqCst);
    //!
    //!     // CONTROL CHECK
    //!     let current = central_epoch.load(Ordering::Relaxed);
    //!     let mut min = current;
    //!     for slot in guard_slots.iter() {
    //!         // SLOT CHECK
    //!         let registered = slot.registered_epoch.load(Ordering::Relaxed);
    //!         if registered != 0 {
    //!             min = min.min(registered);
    //!         }
    //!     }
    //!
    //!     // Synchronize with slot releases
    //!     fence(Ordering::Acquire);
    //!
    //!     // We can advance if `min == current`.
    //!     min == current
    //! }
    //!
    //! fn advance_epoch(...) {
    //!     let exclusive = //* acquire control side mutex guard *//;
    //!
    //!     // If we can scan - increment the epoch.
    //!     if scan(...) {
    //!         // RELEASE STORE
    //!         central_epoch.fetch_add(1, Ordering::Release);
    //!     }
    //! }
    //!
    //! // Register Side
    //! fn register(...) {
    //!     // WORKER CHECK
    //!     let epoch = central_epoch.load(Ordering::Acquire);
    //!
    //!     // Claim an unoccupied slot using a CAS
    //!     if let Some(slot) = guard_slots.claim(epoch, Ordering::Relaxed) {
    //!         // A slot has been claimed - but we need to protect against a scan occurring
    //!         // between the epoch read and slot claim.
    //!         let mut final_epoch = epoch;
    //!         loop {
    //!             // WORKER FENCE (WF)
    //!             fence(Ordering::SeqCst);
    //!
    //!             // WORKER RECHECK
    //!             let reread = central_epoch.load(Ordering::Acquire);
    //!             if reread == final_epoch {
    //!                 break;
    //!             }
    //!             final_epoch = reread;
    //!         }
    //!
    //!         // If there was a race and the epoch changed between the first read and the
    //!         // reread, we have to update the slot to the quiesced point.
    //!         if epoch != final_epoch {
    //!             slot.update_to(final_epoch, Ordering::Relaxed);
    //!         }
    //!
    //!         // Continue
    //!     }
    //! }
    //! ```
    //! To see how this works, consider the following.
    //!
    //! 1. The worker thread invokes WF after a successful CAS and then rechecks the epoch
    //!    at "RECHECK".
    //!
    //! 2. A control thread invokes CF the reads all the registered epochs for all guard slots.
    //!
    //! With the total order induced by sequential consistency, exactly one of these fences
    //! precedes the other in the sequentially consistent total order.
    //!
    //! * If the worker thread's WF precedes CF in the total order, then the control thread
    //!   will observe a successful CAS with the result of "WORKER CHECK" or a later
    //!   modification of that same slot. The initial publication of "WORKER CHECK" is no
    //!   greater than the value read by "CONTROL CHECK".
    //!
    //!   As proof, consider the following:
    //!
    //!   - The epoch is monotonically increasing.
    //!   - All modifications to the central epoch are done under the control-side mutex.
    //!   - The mutex orders previous controller modifications before the "CONTROL CHECK".
    //!     Subsequent modifications follow that load through sequencing or mutex
    //!     synchronization.
    //!
    //!   If "WORKER CHECK" were to read a future hypothetical modification to the epoch `M`,
    //!   we would have:
    //!   - CF is sequenced before "CONTROL CHECK".
    //!   - "CONTROL CHECK" happens before this later modification `M`.
    //!   - `M` synchronizes with "WORKER CHECK" (since all modifications are done with
    //!     release ordering).
    //!   - "WORKER CHECK" is sequenced before WF.
    //!
    //!   Therefore CF strongly happens before WF and thus must precede WF in the total
    //!   order, contradicting our assumption that WF precedes CF.
    //!
    //!   If the observed slot is from an older epoch, then advancement is blocked. Since
    //!   "WORKER RECHECK" only move the registration forward and clearing remove
    //!   registration, these later transitions do not invalidate the protection established
    //!   by publication.
    //!
    //! * If CF precedes WF in the total order, then "WORKER RECHECK" will see *at least*
    //!   the result of "CONTROL CHECK".
    //!
    //!   To see this, let `U` be the modification to the epoch observed in "CONTROL CHECK"
    //!   and let `R` be acquire load in "WORKER RECHECK". Suppose `R` reads a modification
    //!   preceding `U` in the modification order of the epoch. Then `R` would be
    //!   coherence-ordered before `U`. Since:
    //!
    //!   - `U` happens before `CF` on the control side due to the mutex.
    //!   - `WF` sequenced before `R` on the worker side.
    //!
    //!   This situation would imply
    //!
    //!   - `WF` is a seqcst fence that happens-before `R`.
    //!   - `R` is coherence ordered before `U`.
    //!   - `U` happens-before a seqcst fence `CF`.
    //!
    //!   Therefore, `WF` must precede `CF`, contradicting our assumption.
    //!
    //!   Thus, "WORKER RECHECK" must observe at least the result of "CONTROL CHECK" and it
    //!   will update its slot accordingly. Until the recheck completes, the slot will stay
    //!   at its initial epoch, which is no greater than the final reconciled epoch. This
    //!   can conservatively delay advancement.
    //!
    //! # Registry Locks
    //!
    //! Since the [`Registry`] already tracks registered workers, we can modify the above
    //! protocol slightly to allow tracking of current readers and writers, to temporarily
    //! block registration of new readers/writers, and to wait for all current readers and/or
    //! writers to finish. These are the registry locks and can be used to:
    //!
    //! * Provide a consistent serialization snapshot of data protected by the registry by
    //!   ensuring no writers are present.
    //!
    //! * Provide graceful shutdown mechanisms by blocking all workers.
    //!
    //! Not that safe uses of this interface rely on properly respecting read/write controls
    //! as managed by the [`Registry`]. The [`Registry`] merely provides a way of tracking
    //! this information.
    //!
    //! To support this, we steal the upper two bits from the central epoch and the
    //! [`GuardSlot`] registered epochs. For the central epoch, these bits correspond to
    //! advertising acquisition of the locks. For the [`GuardSlot`], a non-zero epoch already
    //! indicates the presence of a worker. The second MSB is used to indicate that the
    //! worker is a writer. The MSB is used to indicate registry notification as described
    //! below.
    //!
    //! ## Lock Acquisition
    //!
    //! The protocol is similar to epoch advancement. Acquiring a lock is done under the
    //! same mutex to ensure mutual exclusion on the control side and a similar use of
    //! sequentially consistent fences is used in the handshake.
    //!
    //! Lock acquisition begins with setting the appropriate bit in the central epoch in
    //! a modification `M` with is sequenced before a sequentially-consistent "CONTROL FENCE".
    //! Following the proof of epoch advancement, "WORKER FENCE" either:
    //!
    //! * Occurs before CF in the total order, meaning the control thread will observe the
    //!   modification to the worker's guard slot or some later modification.
    //!
    //! * Occurs after CF in the total order, meaning the "WORKER RECHECK" will observe the
    //!   acquisition bit, or some future modification, and correct itself accordingly.
    //!
    //! The main additional complexity comes from needing to wait for current workers to
    //! finish before the lock can be fully acquired. Here, the control thread (while still
    //! under the mutex), scans the [`GuardSlot`]s. Every time an appropriate worker is
    //! discovered, it attempts to set the notify bit in [`SlotEpoch`]. Each worker that
    //! gets its notified bit set becomes a "pending" worker that must notify the
    //! [`Registry`] when it is finished. If the control thread observes any pending workers,
    //! it sleeps on a [`Condvar`] as it waits for workers to finish.
    //!
    //! Workers, on the other hand, need to detect if their "notify" bit was set and if so,
    //! access the [`Registry`], lock the control mutex, and decrement the "pending" count.
    //!
    //! Only when there are no pending workers left can the lock be finally acquired.
    //!
    //! Note once more that all control-side operations are mutually exclusive. When a lock
    //! is being acquired, epoch advancement or acquisition of another lock will not proceed.
    //! Similarly, when epoch advancement is running, a lock cannot be acquired.
    //!
    //! This means that lock acquisition must block all other control side operations even
    //! if the control thread is sleeping on the condition variable with the mutex unheld.
    //! We implement that using [`Waiters`] in the [`Waiters::Pending`] state. Only the thread
    //! attempting to acquire the lock may move [`Waiters`] out of the pending state.

    #[expect(unused_imports, reason = "this makes cross-reference nicer")]
    use super::*;
}

use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use crossbeam_queue::SegQueue;
use crossbeam_utils::CachePadded;
use diskann::utils::IntoUsize;
use parking_lot::{Condvar, Mutex, MutexGuard};

/// The default number of guard slots in a registry.
///
/// This should be relatively high to avoid contention, but not so that scans take
/// unnecessarily long.
const DEFAULT_GUARD_SLOTS: NonZeroUsize = NonZeroUsize::new(256).unwrap();

/// A 62-bit unsigned integer.
///
/// When converted to a 64-bit integer via [`Self::value`], this guarantees that bits
/// 62 and 63 are zero.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct U62(u64);

impl U62 {
    /// The maximum value of [`U62`].
    const MAX: u64 = 0x3fff_ffff_ffff_ffff;

    /// Create a new [`U62`] by clearing the top two bits of `v`.
    pub(crate) fn new_masked(v: u64) -> Self {
        Self(v & Self::MAX)
    }

    /// Return the value of `self` as a `u64`.
    ///
    /// The result is guaranteed to have bits 62 and 63 set to 0.
    pub(crate) fn value(self) -> u64 {
        self.0
    }

    // Test //

    /// Construct a new [`U62`] if `v` fits within 62-bits.
    #[cfg(test)]
    pub(crate) fn new(v: u64) -> Option<Self> {
        if v > Self::MAX { None } else { Some(Self(v)) }
    }

    /// Add `v` to `self`, returning the result if it fits in 62-bits.
    #[cfg(test)]
    pub(crate) fn checked_add(&self, v: u64) -> Option<Self> {
        self.0
            .checked_add(v)
            .and_then(|r| if r > Self::MAX { None } else { Some(Self(r)) })
    }
}

impl std::fmt::Display for U62 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "U62({})", self.0)
    }
}

//-------------//
// Worker Side //
//-------------//

/// Slot registration.
///
/// In addition to the published epoch, this also contains a reference count field to enable
/// sharing of [`ReadGuard`]s and [`WriteGuard`]s.
#[derive(Debug)]
struct GuardSlot {
    /// The epoch guard registered at this slot.
    ///
    /// A value of [`SlotEpoch::new`] means unused.
    epoch: SlotEpoch,

    /// The number of guards sharing this slot. This is used to implement intrusive
    /// reference counting for [`ReadGuard`].
    references: AtomicUsize,
}

impl GuardSlot {
    /// Like [`std::sync::Arc`], we limit the number of [`ReadGuard`] clones to `isize::MAX`.
    ///
    /// This is to give a `fetch_add` to increase reference counts some head room to detect
    /// overflow.
    ///
    /// We abort if we overflow this reference count, which can only happen if a thread is
    /// creating shared references and [`std::mem::forget`]ting them. In the words of the
    /// standard library, we don't care to support such degenerate programs.
    #[cfg_attr(
        not(any(feature = "quantization", test)),
        expect(dead_code, reason = "this is used when features are enabled")
    )]
    const MAX_REFCOUNT: usize = (isize::MAX) as usize;
}

impl Default for GuardSlot {
    fn default() -> Self {
        GuardSlot {
            epoch: SlotEpoch::new(),
            references: AtomicUsize::new(0),
        }
    }
}

/// Indicate whether a worker is a reader or writer.
#[derive(Debug, Clone, Copy)]
enum Worker {
    Reader,
    Writer,
}

/// The epoch occupying a [`GuardSlot`].
///
/// * Bits 0-61: The actual epoch value.
/// * Bit 62 ("writer"): Set to 1 if the owner is a writer.
/// * Bit 63 ("notify"): Set by registry scanners to inform the owner that it must notify
///   the registry when dropped.
///
/// Worker threads are responsible for setting bits 0 to 62 (first 63). A control thread
/// attempting to acquire a [`WorkerExclusionGuard`] or [`WriterExclusionGuard`] will set
/// bit 63 if the worker thread owning the slot must notify the [`Inner`] when it is done.
///
/// The following is a state diagram of the **only** allowed transitions.
/// ```text
///
/// +------------------------+
/// |       "empty"          |
/// | Notify: 0 (bit 63)     |  Worker (W2): Transition back to "empty".
/// | Writer: 0 (bit 62)     |  If the "notify" bit was set, notify the registry.
/// | Epoch: 0               |<--------------+-------------------------+
/// |                        |               |                         |
/// | The slot is not        |               |                         |
/// | claimed by any thread. |               |                         |
/// +------------------------+               |                         |
///    | Worker (W0): Must CAS away from     |                         |
///    | "empty" to this state.              |                         |
///    |                                     |                         |
///    | Writer bit must be set if           |                         |
///    | and only if worker is a writer.     |                         |
///    V                                     |                         |
/// +------------------------------+         |          +------------------------------+
/// |     "occupied - no notify"   |---------+          |     "occupied - notify"      |
/// | Notify: 0 (bit 63)           | Control (C0):      | Notify: 1 (bit 63)           |
/// | Writer: X (bit 62)           | worker must notify | Writer: X (bit 62)           |
/// | Epoch: V                     | registry when it   | Epoch: V                     |
/// |                              | is finished.       |                              |
/// | Slot is claimed by a worker. |                    | Slot is claimed by a worker. |
/// | No notification needed.      | Only valid if V!=0 | Notification needed.         |
/// |                              |------------------->|                              |
/// +------------------------------+                    +------------------------------+
///       |               ^                                     |               ^
///       |               |                                     |               |
///       +---------------+                                     +---------------+
///    Worker (W1A): Increase epoch                            Worker (W1B): Increase epoch
///     due to resampling protocol                              due to resampling protocol
/// ```
/// The following techniques are needed for each transition:
/// * `W0`: A CAS must be used since another thread may attempt to acquire the slot concurrently.
///
/// * `C0`: Must be done with a CAS since it can race with W1A or W2. If it races with W2
///   and the slot gets set back to zero, attempts to set the "notify" bit should cease.
///
/// * `W1A`: Must be done with a CAS since it can race with `C0`. It can be assumed that
///   the only non-worker transition from this stats is due to `C0`.
///
/// * `W1B`: May be done with a swap since it should not race with any external modifications.
///
/// * `W2`: May be done with a swap as long as the old value is inspected. This can race with
///   `C0`, but:
///   - If `W2` wins, the worker is finished and control need not wait on it.
///   - If `C0` wins, the value returned from the swap will have the notification bit set
///     and thus control will be notified.
///
/// There are some additional things to note about C0. The [`Registry`] ensures that only
/// a single control thread is scanning/waiting for notifications. And that this thread
/// waits for all pending notifications before continuing.
///
/// Further, the [`Registry`] prevents new readers/writers (whatever the control thread
/// is waiting for) from making new registrations. This means that:
///
/// * The transition `C0` may assume that the "notify" bit for a slot it hasn't visited
///   yet is unset.
///
/// * Only a single pass over all `SlotEpoch`s is required for a particular signal.
///
/// * It is possible for `C0` to race in a full cycle through `W2` + `W0` with the same
///   kind of worker that is being excluded by the lock, but this situation is transient
///   during lock acquisition. The worker thread would have corrected the issue upon
///   epoch re-read, but we won the race.
///
/// Finally, the presence of the "writer" bit implies a non-zero epoch.
#[derive(Debug)]
struct SlotEpoch(AtomicU64);

impl SlotEpoch {
    /// The mask isolating the "notify" bit.
    const NOTIFY_MASK: u64 = 0x8000_0000_0000_0000;

    /// Mask for isolating the "writer" bit.
    const WRITE_MASK: u64 = 0x4000_0000_0000_0000;

    /// Construct a new [`SlotEpoch`] in the "unoccupied" state.
    fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    //-------------//
    // Worker Side //
    //-------------//

    /// Try to acquire the slot with the given `epoch` for the [`Worker`] of worker.
    ///
    /// Return `true` on success.
    ///
    /// On success, the worker thread is obligated to use [`Self::clear_release`].
    ///
    /// This is transition `W0`.
    #[inline]
    #[must_use]
    fn try_acquire_relaxed(&self, epoch: U62, kind: Worker) -> bool {
        let mask = match kind {
            Worker::Reader => 0,
            Worker::Writer => Self::WRITE_MASK,
        };

        self.0.load(Ordering::Relaxed) == 0
            && self
                .0
                .compare_exchange(
                    0,
                    epoch.value() | mask,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
    }

    /// Update the registered epoch from `current` to `new` while preserving the notify bit.
    ///
    /// The only supported transition is from
    /// ```text
    /// current (| WRITE_MASK) -> new (| WRITE_MASK)
    /// ```
    /// or
    /// ```text
    /// current | NOTIFY_MASK (| WRITE_MASK) -> new | NOTIFY_MASK (| WRITE_MASK)
    /// ```
    ///
    /// This may only be used after "GUARD RECHECK" to update a slot to a resampled epoch.
    ///
    /// This is transition `W1A` or `W1B`.
    #[inline]
    fn update_relaxed(&self, current: U62, new: U62, kind: Worker) {
        // Add the correct writer bit.
        let mask = match kind {
            Worker::Reader => 0,
            Worker::Writer => Self::WRITE_MASK,
        };

        let current = current.value() | mask;
        let new = new.value() | mask;

        // Try once - this works for both `W1A` and `W1B` and will only fail if we race with
        // `C0`. In which case, the later swap will implement `W1B`.
        let found =
            match self
                .0
                .compare_exchange(current, new, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return,
                Err(found) => found,
            };

        assert_eq!(
            found,
            current | Self::NOTIFY_MASK,
            "concurrency violation - CAS can only fail due to a notify bit being set"
        );

        // This is transition `W1B`.
        let out = self.0.swap(new | Self::NOTIFY_MASK, Ordering::Relaxed);
        assert_eq!(
            out, found,
            "concurrency violation - setting the notify bit must be monotonic",
        );
    }

    /// Mark this slot as no longer reserved.
    ///
    /// Returns a [`Obligation`] informing the caller of additional steps that must be taken,
    /// if any.
    ///
    /// This is transition `W2`.
    fn clear_release(&self) -> Obligation {
        let old = self.0.swap(0, Ordering::Release);
        if old & Self::NOTIFY_MASK != 0 {
            Obligation::Notify
        } else {
            Obligation::None
        }
    }

    //--------------//
    // Control Side //
    //--------------//

    /// Return whether or not the slot is occupied.
    #[must_use = "this function has not side-effects"]
    fn current_value_relaxed(&self) -> U62 {
        U62::new_masked(self.0.load(Ordering::Relaxed))
    }

    /// Inform the worker using this slot (if there is one) that it must notify the
    /// [`Registry`] when it is finished.
    ///
    /// Returns `true` if the worker was notified. Returning `false` means this slot does
    /// not need to be waited on.
    ///
    /// This is transition `C0`.
    #[must_use]
    fn must_notify_relaxed(&self, kind: Exclusion) -> bool {
        self.must_notify_relaxed_inner(kind, NoDelay)
    }

    #[inline]
    fn must_notify_relaxed_inner(&self, kind: Exclusion, mut delay: impl SlotEpochDelay) -> bool {
        let mask = match kind {
            Exclusion::Workers => u64::MAX,
            Exclusion::Writers => Self::WRITE_MASK,
        };

        let mut current = self.0.load(Ordering::Relaxed);
        assert_eq!(
            current & Self::NOTIFY_MASK,
            0,
            "only one control thread can be setting notify bits at a time",
        );

        delay.pre_must_notify_relaxed_cas();

        // We can potentially race with `W1A` or even a complete cycle through `W2 -> W0`.
        //
        // Keep doing a CAS while the worker is in "occupied - no notify" and is the kind
        // of worker we are waiting for.
        //
        // If we succeed, then the worker will notify the registry. If the predicate fails,
        // the registry will prevent new workers of this kind from being registered so
        // we can move on.
        while current & mask != 0 {
            match self.0.compare_exchange(
                current,
                current | Self::NOTIFY_MASK,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(found) => {
                    assert_eq!(found & Self::NOTIFY_MASK, 0);
                    current = found;
                    delay.post_must_notify_relaxed_cas();
                }
            }
        }

        // We exited the loop because the slot is unused.
        false
    }

    // Test Utils //

    #[cfg(test)]
    fn raw_relaxed(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Obligation for threads clearing a [`SlotEpoch`].
#[derive(Debug, PartialEq)]
#[must_use]
enum Obligation {
    /// No obligation.
    None,
    /// The thread must notify the [`Registry`] that it has been retired.
    Notify,
}

//--------------//
// Control Side //
//--------------//

/// The epoch inside the [`Registry`].
///
/// Bits 0 to 61 store the actual epoch value, beginning at "1".
/// Bit 62 is set to block out all workers.
/// Bit 63 is set to block out writers.
#[derive(Debug)]
struct Epoch {
    epoch: AtomicU64,
}

impl Epoch {
    // Everything but the top two bits.
    const MASK: u64 = 0x3fff_ffff_ffff_ffff;

    const READ_LOCK: u64 = 0x4000_0000_0000_0000;
    const WRITE_LOCK: u64 = 0x8000_0000_0000_0000;

    /// Create a new [`Epoch`] with value "1".
    fn new() -> Self {
        Self {
            epoch: AtomicU64::new(1),
        }
    }

    /// Read the current epoch, returning an [`EpochRead`] describing the current state.
    ///
    /// This function uses [`Ordering::Acquire`].
    fn read_acquire(&self) -> EpochRead {
        let value = self.epoch.load(Ordering::Acquire);
        match value >> 62 {
            0 => EpochRead::Ready(U62::new_masked(value)),
            1 => EpochRead::WorkersExcluded,
            _ => EpochRead::WritersExcluded(U62::new_masked(value)),
        }
    }

    /// Return the current value of the epoch regardless of the lock acquisition state.
    ///
    /// This function uses [`Ordering::Relaxed`].
    fn current_value_relaxed(&self) -> U62 {
        U62::new_masked(self.epoch.load(Ordering::Relaxed))
    }

    /// Advance the epoch by 1.
    ///
    /// This function uses [`Ordering::Release`].
    ///
    /// # Panics
    ///
    /// Panics if `current = U62::MAX` or if the actual value of `self` is not exactly
    /// `current` extended to 64-bits.
    #[expect(
        clippy::panic,
        reason = "the panic is exceedingly unlikely to happen and if it does, we can't continue"
    )]
    fn advance_release(&self, current: U62) {
        let current = current.value();
        if current == Self::MASK {
            panic!(
                "we've managed to go through nearly `u64::MAX / 4` ids - this is unlikely in a real program"
            );
        }

        let previous = self.epoch.swap(current + 1, Ordering::Release);
        if previous != current {
            panic!(
                "invalid advancement from {} - current value is {}",
                current, previous
            );
        }
    }

    /// Set the requested lock bit.
    ///
    /// This function uses [`Ordering::Release`].
    ///
    /// # Panics
    ///
    /// Panics if either lock-bit is already set.
    #[inline]
    fn set_lock_release(&self, kind: Exclusion) {
        let mask = match kind {
            Exclusion::Workers => Self::READ_LOCK,
            Exclusion::Writers => Self::WRITE_LOCK,
        };

        let previous = self.epoch.fetch_or(mask, Ordering::Release);
        assert_eq!(previous >> 62, 0);
    }

    /// Clear the requested lock bit.
    ///
    /// This function uses [`Ordering::Release`].
    ///
    /// # Panics
    ///
    /// Panics if the requested lock bit was not set.
    #[inline]
    fn clear_lock_release(&self, kind: Exclusion) {
        let previous = self.epoch.fetch_and(Self::MASK, Ordering::Release);

        let mask = match kind {
            Exclusion::Workers => Self::READ_LOCK,
            Exclusion::Writers => Self::WRITE_LOCK,
        };

        assert_ne!(previous & mask, 0);
    }
}

/// The current state of [`Epoch`].
#[derive(Debug, Clone, Copy, PartialEq)]
enum EpochRead {
    Ready(U62),
    WritersExcluded(U62),
    WorkersExcluded,
}

impl EpochRead {
    #[cfg(test)]
    fn assert_ready(self) -> U62 {
        match self {
            Self::Ready(v) => v,
            _ => panic!("epoch is not ready"),
        }
    }
}

/// The kind of exclusion being requested.
#[derive(Debug, Clone, Copy)]
enum Exclusion {
    Workers,
    Writers,
}

/// Number of pending workers to be retired.
#[derive(Debug, Clone, Copy)]
enum Waiters {
    None,
    Pending(usize),
}

/// A registry of epoch-based [`ReadGuard`]s. See the [module-level docs](self).
#[derive(Debug)]
pub(crate) struct Registry {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    // A record of the active guards.
    //
    // Acquisition state depends on [`GuardSlot::epoch`]:
    //
    // * 0 = "available".
    // * Anything else = "guarded".
    guards: Box<[GuardSlot]>,

    // We use four queues for storing retiring items. The rationale is documented below.
    //
    // ```
    //
    //                                1. Safe to drain
    //                           +--------------------------
    //  Items retired at N-1 can |    2. Epoch N-1
    //  be observed by guards at |  +-----------------------
    //  N. If we transition to   |  | 3. Epoch N
    //  N+1, guards at N can be  +--------------------------
    //  active still. This it is    | 4. Epoch N+1
    //  not safe to reclaim items   +-----------------------
    //  from this queue until all     5. Epoch N+2 (reuse #1 queue)
    //  guards are at least N+1.
    // ```
    //
    // We cycle among the queues in a round-robin manner.
    retiring: Box<[SegQueue<u32>; 4]>,

    // We can only retire a single generation at a time.
    //
    // This guard avoids situations where two threads concurrently advance the epoch and
    // hand out overlapping `Drain`s referring to the same retiring queue.
    exclusive: CachePadded<Mutex<Waiters>>,

    // A hint for the next available registration slot.
    hint: CachePadded<AtomicUsize>,

    // Waiters attempting to acquire a read or a write lock.
    waiter: CachePadded<Condvar>,

    // The current epoch. This begins at 1 (to not be conflated with the 0 state in `guards`)
    // and increments over time.
    //
    // NOTE: This can **only** be mutated in `try_advance`.
    //
    // Additionally, the logic in the module ensures that there are at most two active epochs
    // at any given time. It is only safe to advance an epoch if *all* readers belong to the
    // current `epoch`.
    epoch: CachePadded<Epoch>,
}

impl Inner {
    #[must_use = "this function has no side-effects and is used to justify unsafe code"]
    fn guard_belongs(&self, guard: &ReadGuard<'_>) -> bool {
        std::ptr::eq(self, guard.registry)
    }

    /// Notify a waiter that a worker has finished.
    ///
    /// This function may block.
    ///
    /// # Panics
    ///
    /// Panics if a control thread is not waiting for workers or if the pending count is
    /// already zero.
    #[expect(
        clippy::panic,
        reason = "the panic is the result of a bug and if we trigger it - we cannot continue"
    )]
    fn notify(&self) {
        let mut exclusive = self.exclusive.lock();
        match &mut *exclusive {
            Waiters::None => panic!("notify called when there are no waiters"),
            Waiters::Pending(pending) => match pending.checked_sub(1) {
                Some(remaining) => {
                    *pending = remaining;
                    if remaining == 0 {
                        // There should only ever be a single waiter on the condition
                        // variable, so we just use `notify_one`.
                        self.waiter.notify_one();
                    }
                }
                None => panic!("notify called when there are no pending workers"),
            },
        }
    }
}

// Return the queue index for the `epoch`.
fn queue(epoch: u64) -> usize {
    epoch.into_usize() % 4
}

fn last_queue(epoch: u64) -> usize {
    queue(epoch.wrapping_sub(2))
}

impl Registry {
    /// Return the default number of guard slots.
    pub(crate) const fn default_guard_slots() -> NonZeroUsize {
        DEFAULT_GUARD_SLOTS
    }

    /// Construct a new [`Registry`] with the default number of guard slots (256).
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_capacity(DEFAULT_GUARD_SLOTS)
    }

    /// Construct a new [`Registry`] with `capacity` guard slots.
    ///
    /// This is the number of [`ReadGuard`]s that can be registered concurrently.
    pub(crate) fn with_capacity(capacity: NonZeroUsize) -> Self {
        let inner = Inner {
            guards: std::iter::repeat_with(Default::default)
                .take(capacity.get())
                .collect(),
            retiring: Box::new(core::array::from_fn(|_| SegQueue::new())),
            exclusive: CachePadded::new(Mutex::new(Waiters::None)),
            hint: CachePadded::new(AtomicUsize::new(0)),
            waiter: CachePadded::new(Condvar::new()),
            epoch: CachePadded::new(Epoch::new()),
        };

        Self {
            inner: Arc::new(inner),
        }
    }

    /// Create a read-only handle to the registry.
    ///
    /// The [`RegistryHandle`] can be used to validate whether or not [`ReadGuard`]s
    /// belong to this [`Registry`] or not.
    ///
    /// Unsafe code may rely on these checks.
    pub(crate) fn handle(&self) -> RegistryHandle {
        RegistryHandle {
            inner: self.inner.clone(),
        }
    }

    /// Register the caller with `self`.
    ///
    /// Any items retired while [`ReadGuard`] is held will be protected.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of currently active guards exceeds the number of
    /// internal guard slots and thus a new guard cannot be made or if all workers are
    /// currently being excluded from the registry.
    pub(crate) fn read_guard(&self) -> Result<ReadGuard<'_>, Unavailable> {
        self.register(Worker::Reader, NoDelay)
    }

    /// Register the caller with `self` as a writer.
    ///
    /// Any items retired while [`WriteGuard`] is held will be protected.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of currently active guards exceeds the number of
    /// internal guard slots and thus a new guard cannot be made or if writers are currently
    /// being excluded from the registry.
    pub(crate) fn write_guard(&self) -> Result<WriteGuard<'_>, Unavailable> {
        let inner = self.register(Worker::Writer, NoDelay)?;
        Ok(WriteGuard { inner })
    }

    /// Register the caller with `self` to construct a [`Retire`].
    ///
    /// Items passed to the returned [`Retire`] will be queued and held until any registry
    /// guard that would have observed the retirement has been dropped.
    ///
    /// # Errors
    ///
    /// Return an error if a guard cannot be acquired or if writers are currently being
    /// excluded from the registry.
    pub(crate) fn retire(&self) -> Result<Retire<'_>, Unavailable> {
        let guard = self.write_guard()?;
        Ok(Retire { guard })
    }

    #[inline]
    fn register<T>(&self, kind: Worker, mut delay: T) -> Result<ReadGuard<'_>, Unavailable>
    where
        T: GuardDelay,
    {
        let inner = &self.inner;

        // WORKER CHECK
        let epoch = match inner.epoch.read_acquire() {
            EpochRead::Ready(epoch) => epoch,
            EpochRead::WorkersExcluded => return Err(Unavailable),
            EpochRead::WritersExcluded(epoch) => match kind {
                Worker::Reader => epoch,
                Worker::Writer => return Err(Unavailable),
            },
        };

        let hint = inner.hint.fetch_add(1, Ordering::Relaxed);
        delay.post_guard_check();
        let nguards = inner.guards.len();
        for i in 0..nguards {
            let slot = hint.wrapping_add(i) % nguards;

            let guard_slot = &inner.guards[slot];
            delay.pre_cas();
            if guard_slot.epoch.try_acquire_relaxed(epoch, kind) {
                delay.post_cas();

                let roll_back_and_exit = || -> Unavailable {
                    match guard_slot.epoch.clear_release() {
                        Obligation::None => {}
                        Obligation::Notify => inner.notify(),
                    };
                    Unavailable
                };

                let mut final_epoch = epoch;
                loop {
                    // WORKER FENCE: See internal docs for meaning.
                    delay.pre_fence();
                    std::sync::atomic::fence(Ordering::SeqCst);
                    delay.post_fence();

                    // WORKER RECHECK
                    let current = match inner.epoch.read_acquire() {
                        EpochRead::Ready(current) => current,
                        EpochRead::WorkersExcluded => return Err(roll_back_and_exit()),
                        EpochRead::WritersExcluded(current) => match kind {
                            Worker::Reader => current,
                            Worker::Writer => return Err(roll_back_and_exit()),
                        },
                    };

                    if current == final_epoch {
                        break;
                    }

                    final_epoch = current;
                }

                if epoch != final_epoch {
                    guard_slot.epoch.update_relaxed(epoch, final_epoch, kind);
                }

                if cfg!(any(test, feature = "integration-test")) {
                    let old = guard_slot.references.swap(1, Ordering::Relaxed);
                    assert_eq!(
                        old, 0,
                        "unclaimed guard slots should have zero reference count"
                    );
                }

                // Under `test/integration-test` - this redoes work. However, we do want
                // to make sure we hit it.
                guard_slot.references.store(1, Ordering::Relaxed);

                return Ok(ReadGuard {
                    slot: guard_slot,
                    registry: inner,
                    #[cfg(test)]
                    test_epoch: final_epoch,
                    #[cfg(test)]
                    test_slot_index: slot,
                });
            }
        }

        Err(Unavailable)
    }

    //------------------//
    // Registry Control //
    //------------------//

    /// Attempt to acquire the exclusive mutex.
    ///
    /// This function returns `None` if:
    ///
    /// * The lock cannot be acquired.
    /// * The lock can be acquired but another thread is waiting for pending workers.
    ///
    /// If this function returns, `Some`, the inner [`Waiters`] is guaranteed to be
    /// [`Waiters::None`].
    fn try_lock_if_no_waiters(&self) -> Option<MutexGuard<'_, Waiters>> {
        let guard = self.inner.exclusive.try_lock()?;
        match &*guard {
            Waiters::None => Some(guard),
            Waiters::Pending(_) => None,
        }
    }

    // Epoch Advancement //

    /// Try to advance the current epoch.
    ///
    /// If successful, returns a [`Drain`]. All items in the drain can be reclaimed.
    ///
    /// Returns `None` if the epoch cannot yet be advanced (some [`ReadGuard`] still belongs
    /// to a prior epoch) or if another [`Drain`] is currently active.
    ///
    /// # Panics
    ///
    /// Panics if the epoch counter is about to overflow `u64::MAX`. In practice this is
    /// effectively unreachable.
    pub(crate) fn try_advance(&self) -> Option<Drain<'_>> {
        self.try_advance_inner(NoDelay)
    }

    fn try_advance_inner<T>(&self, mut delay: T) -> Option<Drain<'_>>
    where
        T: TryAdvanceDelay,
    {
        // We first try to acquire the `exclusive` lock.
        //
        // It can only fail if someone else is holding the exclusive lock, which means we
        // can't proceed anyways.
        //
        // This can help save an expensive slot scan.
        let exclusive = self.try_lock_if_no_waiters()?;
        let (can_advance, current) = self.can_advance(&mut delay);

        // All waiters belong to the current epoch. Therefore, it is safe to release the old
        // array queue
        if can_advance {
            // We are safe to use a `fetch_add` here because `exclusive` is ensuring exclusivity
            // of the access. Release ordering is sufficient. See internal docs for proof.
            self.inner.epoch.advance_release(current);
            let queue = &self.inner.retiring[last_queue(current.value())];
            Some(Drain {
                queue,
                _exclusive: exclusive,
            })
        } else {
            // Previous generation has not completely retired.
            None
        }
    }

    #[inline]
    fn can_advance<T>(&self, delay: &mut T) -> (bool, U62)
    where
        T: CanAdvanceDelay,
    {
        // CONTROL FENCE: See the internal docs for why this is important.
        delay.pre_fence();
        std::sync::atomic::fence(Ordering::SeqCst);
        delay.post_fence();

        // CONTROL CHECK
        let current = self.inner.epoch.current_value_relaxed();
        let mut min = current;

        for s in self.inner.guards.iter() {
            // SLOT CHECK
            let guarded = s.epoch.current_value_relaxed();
            if guarded != U62::default() {
                min = min.min(guarded);
            }
        }

        // This synchronizes with all the guard's `Release`s.
        std::sync::atomic::fence(Ordering::Acquire);
        (min == current, min)
    }

    // Registry Locks //

    /// Attempt to exclude all workers. If successful, the resulting [`WorkerExclusionGuard`]
    ///
    /// * Attests that no workers are registered with `self`.
    /// * Blocks new workers from being registered.
    /// * Blocks all other control-side operations.
    ///
    /// This operation fails if another control-side operation is running.
    ///
    /// This function may block as it waits for all current workers to finish.
    ///
    /// # Deadlock Potential
    ///
    /// This function can deadlock if the calling thread is holding onto a [`WriteGuard`],
    /// [`ReadGuard`], or a [`Retire`].
    ///
    /// See also: [`Self::try_exclude_writers`], [`Self::try_advance`].
    pub(crate) fn try_exclude_workers(&self) -> Option<WorkerExclusionGuard<'_>> {
        self.try_exclude_workers_inner(NoDelay)
    }

    fn try_exclude_workers_inner(&self, delay: impl LockDelay) -> Option<WorkerExclusionGuard<'_>> {
        let exclusive = self.try_lock(Exclusion::Workers, delay)?;
        Some(WorkerExclusionGuard {
            registry: self,
            _exclusive: exclusive,
        })
    }

    /// Attempt to exclude writers. If successful, the resulting [`WriterExclusionGuard`]
    ///
    /// * Attests that no writers are registered with `self`.
    /// * Blocks new writers from being registered.
    /// * Blocks all other control-side operations.
    ///
    /// This operation fails if another control-side operation is running.
    ///
    /// This function may block as it waits for all current writers to finish.
    ///
    /// # Deadlock Potential
    ///
    /// This function can deadlock if the calling thread is holding onto a [`WriteGuard`],
    /// [`Retire`], or a [`ReadGuard`] obtained from [`WriteGuard::into_read_guard`].
    ///
    /// See also: [`Self::try_exclude_workers`], [`Self::try_advance`].
    pub(crate) fn try_exclude_writers(&self) -> Option<WriterExclusionGuard<'_>> {
        self.try_exclude_writers_inner(NoDelay)
    }

    fn try_exclude_writers_inner(&self, delay: impl LockDelay) -> Option<WriterExclusionGuard<'_>> {
        let exclusive = self.try_lock(Exclusion::Writers, delay)?;
        Some(WriterExclusionGuard {
            registry: self,
            _exclusive: exclusive,
        })
    }

    fn try_lock(
        &self,
        kind: Exclusion,
        mut delay: impl LockDelay,
    ) -> Option<MutexGuard<'_, Waiters>> {
        let mut exclusive = self.try_lock_if_no_waiters()?;

        // WAITING FENCE LOCK
        self.inner.epoch.set_lock_release(kind);
        std::sync::atomic::fence(Ordering::SeqCst);

        let mut waiters = 0;
        for s in self.inner.guards.iter() {
            if s.epoch.must_notify_relaxed(kind) {
                waiters += 1;
            }
        }

        if waiters != 0 {
            *exclusive = Waiters::Pending(waiters);
            delay.pre_wait();
            self.inner.waiter.wait_while(&mut exclusive, |waiting| {
                if let Waiters::Pending(pending) = waiting
                    && *pending == 0
                {
                    // All waiters done.
                    *waiting = Waiters::None;
                    false
                } else {
                    true
                }
            });
        }

        std::sync::atomic::fence(Ordering::Acquire);
        Some(exclusive)
    }

    // Test //

    #[cfg(test)]
    fn test_epoch(&self) -> EpochRead {
        self.inner.epoch.read_acquire()
    }

    #[cfg(test)]
    pub(super) fn ready_epoch(&self) -> U62 {
        self.test_epoch().assert_ready()
    }

    #[cfg(test)]
    fn assert_no_workers(&self) {
        for s in self.inner.guards.iter() {
            assert_eq!(s.epoch.current_value_relaxed(), U62::default());
        }
    }

    #[cfg(test)]
    fn waiting(&self) -> U62 {
        self.can_advance(&mut NoDelay).1
    }

    #[cfg(test)]
    fn try_snoop_waiters(&self) -> Option<Waiters> {
        self.inner.exclusive.try_lock().map(|waiters| *waiters)
    }

    #[cfg(test)]
    fn snoop_waiters(&self) -> Waiters {
        *self.inner.exclusive.lock()
    }
}

/// A read-only handle to a [`Registry`] used to validate that [`ReadGuard`]s belong to
/// the said registry.
#[derive(Debug, Clone)]
pub(crate) struct RegistryHandle {
    inner: Arc<Inner>,
}

impl RegistryHandle {
    /// Assert that [`ReadGuard`] belongs to the handle's [`Registry`].
    ///
    /// # Panics
    ///
    /// Panics if [`ReadGuard`] belongs to a different registry.
    #[inline]
    pub(crate) fn assert_guard_belongs(&self, guard: &ReadGuard<'_>) {
        if !self.inner.guard_belongs(guard) {
            guard_does_not_belong(self, guard)
        }
    }
}

#[cold]
#[expect(clippy::panic, reason = "internals should not provide invalid guards")]
fn guard_does_not_belong(_handle: &RegistryHandle, _guard: &ReadGuard<'_>) -> ! {
    panic!("invalid epoch guard");
}

//-----------//
// ReadGuard //
//-----------//

/// A handle registering the caller as a reader at a particular epoch.
///
/// While this guard is held, the [`Registry`] will ensure that any item that the guard can
/// observe being retired by `self` is not reclaimed.
///
/// Obtained via [`Registry::read_guard`], [`WriteGuard::into_read_guard`].
///
/// See also: [`WriteGuard`], [`Retire`]
#[derive(Debug)]
pub(crate) struct ReadGuard<'a> {
    slot: &'a GuardSlot,
    registry: &'a Inner,

    #[cfg(test)]
    test_epoch: U62,

    #[cfg(test)]
    test_slot_index: usize,
}

impl<'a> ReadGuard<'a> {
    /// Cheaply clone `self` into another [`ReadGuard`] with the same lifetime.
    ///
    /// The epoch associated with the original [`ReadGuard`] will be released only when all
    /// transitively shared guards are dropped.
    #[cfg(any(test, feature = "quantization"))]
    #[inline]
    pub(crate) fn share(&self) -> ReadGuard<'a> {
        let ref_count = self.slot.references.fetch_add(1, Ordering::Relaxed);

        // See the note on `GuardSlot::MAX_REFCOUNT`.
        if ref_count > GuardSlot::MAX_REFCOUNT {
            std::process::abort();
        }

        ReadGuard {
            slot: self.slot,
            registry: self.registry,
            #[cfg(test)]
            test_epoch: self.test_epoch,
            #[cfg(test)]
            test_slot_index: self.test_slot_index,
        }
    }

    fn epoch(&self) -> U62 {
        self.slot.epoch.current_value_relaxed()
    }

    fn registry(&self) -> &Inner {
        self.registry
    }

    #[cfg(test)]
    pub(super) fn test_epoch(&self) -> U62 {
        self.test_epoch
    }
}

impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        // Decrement the reference count. If this is the last clone, then we can release our
        // registration on the guard slot.
        //
        // For the ordering, we use `Ordering::Release` so all preceeding stores (and uses
        // of data protected by the guard) are ordered before this release.
        //
        // This pairs with the acquire fence to ensure that all releases are made prior
        // to resetting the epoch.
        if self.slot.references.fetch_sub(1, Ordering::Release) == 1 {
            std::sync::atomic::fence(Ordering::Acquire);
            match self.slot.epoch.clear_release() {
                Obligation::None => {}
                Obligation::Notify => self.registry.notify(),
            };
        }
    }
}

//------------//
// WriteGuard //
//------------//

/// A handle registering the caller as a writers at a particular epoch.
///
/// While this guard is held, the [`Registry`] will ensure that any item that the guard can
/// observe being retired by `self` is not reclaimed.
///
/// Obtained via [`Registry::write_guard`].
///
/// See also: [`ReadGuard`], [`Retire`]
#[derive(Debug)]
pub(crate) struct WriteGuard<'a> {
    inner: ReadGuard<'a>,
}

impl<'a> WriteGuard<'a> {
    /// Return a [`ReadGuard`] with the same lifetime as `self`.
    ///
    /// The returned [`ReadGuard`] will still be classified internally as a writer and will
    /// thus delay acquisition of a [`WriterExclusionGuard`] while alive.
    #[cfg(test)]
    pub(crate) fn read_guard(&self) -> &ReadGuard<'a> {
        &self.inner
    }

    /// Consume `self`, returning a [`ReadGuard`] with the same lifetime as `self`.
    ///
    /// The returned [`ReadGuard`] will still be classified internally as a writer and will
    /// thus delay acquisition of a [`WriterExclusionGuard`] while alive.
    pub(crate) fn into_read_guard(self) -> ReadGuard<'a> {
        self.inner
    }

    fn epoch(&self) -> U62 {
        self.inner.epoch()
    }

    fn registry(&self) -> &Inner {
        self.inner.registry()
    }

    #[cfg(test)]
    fn test_epoch(&self) -> U62 {
        self.inner.test_epoch()
    }
}

//--------//
// Retire //
//--------//

/// A handle registering the caller as a reader at a particular epoch.
///
/// While this guard is held, the [`Registry`] will ensure that any item that the guard can
/// observe being retired by `self` is not reclaimed.
///
/// Obtained via [`Registry::retire`].
///
/// See also: [`Retire`]
#[derive(Debug)]
pub(crate) struct Retire<'a> {
    guard: WriteGuard<'a>,
}

impl<'a> Retire<'a> {
    /// Retire the id `i` at this guard's epoch.
    ///
    /// `i` is a caller-defined id (typically an index into external storage). It will be
    /// returned from a future [`Drain`] once the registry has advanced far enough that no
    /// reader could observe it.
    #[inline]
    pub(crate) fn retire(&self, i: u32) {
        self.queue().push(i)
    }

    fn queue(&self) -> &SegQueue<u32> {
        &self.guard.registry().retiring[queue(self.guard.epoch().value())]
    }

    #[cfg(test)]
    pub(super) fn test_epoch(&self) -> U62 {
        self.guard.test_epoch()
    }
}

//-------//
// Drain //
//-------//

/// An iterator over ids that are safe to reclaim, returned from [`Registry::try_advance`].
///
/// While this drain is alive, no other thread can advance the [`Registry`]'s epoch. Drop
/// it promptly after processing.
#[derive(Debug)]
pub(crate) struct Drain<'a> {
    queue: &'a SegQueue<u32>,
    _exclusive: MutexGuard<'a, Waiters>,
}

impl Drain<'_> {
    /// Pop the next id ready for reclamation, or `None` if the drain is empty.
    #[must_use = "reclaimed ids must be reclaimed"]
    pub(crate) fn pop(&self) -> Option<u32> {
        self.queue.pop()
    }

    /// Return the number of ids remaining in this drain.
    pub(crate) fn len(&self) -> usize {
        self.queue.len()
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Iterator for Drain<'_> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        self.pop()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

// NOTE: This relies on `Drain` holding the `exclusive` guard. In this state, we are guaranteed
// that no-one is writing into the queue, which would otherwise invalidate the exact-size
// iterator guarantee.
impl ExactSizeIterator for Drain<'_> {}

//----------------------//
// WorkerExclusionGuard //
//----------------------//

/// An exclusive guard on [`Registry`] that:
///
/// * Attests that there are no workers registered in the [`Registry`].
/// * Blocks new workers from being registered.
/// * Block all control side operations from the parent [`Registry`].
///
/// Created by [`Registry::try_exclude_workers`].
#[derive(Debug)]
pub(crate) struct WorkerExclusionGuard<'a> {
    registry: &'a Registry,
    _exclusive: MutexGuard<'a, Waiters>,
}

impl Drop for WorkerExclusionGuard<'_> {
    fn drop(&mut self) {
        self.registry
            .inner
            .epoch
            .clear_lock_release(Exclusion::Workers)
    }
}

//----------------------//
// WriterExclusionGuard //
//----------------------//

/// An exclusive guard on [`Registry`] that:
///
/// * Attests that there are no writers registered in the [`Registry`] (read-only workers
///   are allowed).
/// * Blocks new writers from being registered while allowing read-only workers.
/// * Block all control side operations from the parent [`Registry`].
///
/// Created by [`Registry::try_exclude_writers`].
#[derive(Debug)]
pub(crate) struct WriterExclusionGuard<'a> {
    registry: &'a Registry,
    _exclusive: MutexGuard<'a, Waiters>,
}

impl Drop for WriterExclusionGuard<'_> {
    fn drop(&mut self) {
        self.registry
            .inner
            .epoch
            .clear_lock_release(Exclusion::Writers)
    }
}

//-------//
// Error //
//-------//

/// Returned by [`Registry::read_guard`] when all guard slots are occupied.
#[derive(Debug)]
#[non_exhaustive]
pub(crate) struct Unavailable;

impl std::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("could not claim a guard slot")
    }
}

impl std::error::Error for Unavailable {}

diskann::convert_error!(Unavailable);

// Delays
//
// To help test standard race scenarios without advanced tooling, we use optional delays
// that our tests can introduce to ensure threads are in various intermediate points.
//
// This does not necessarily test that the memory orderings are correct, but at least
// is a smoke test that various (known) races are handled properly.

#[derive(Debug)]
struct NoDelay;

trait SlotEpochDelay {
    // In `must_notify_relaxed`.
    fn pre_must_notify_relaxed_cas(&mut self) {}
    fn post_must_notify_relaxed_cas(&mut self) {}
}

impl SlotEpochDelay for NoDelay {}

trait GuardDelay {
    fn post_guard_check(&mut self) {}
    fn pre_cas(&mut self) {}
    fn post_cas(&mut self) {}
    fn pre_fence(&mut self) {}
    fn post_fence(&mut self) {}
}

impl GuardDelay for NoDelay {}

trait CanAdvanceDelay {
    fn pre_fence(&mut self) {}
    fn post_fence(&mut self) {}
}

impl CanAdvanceDelay for NoDelay {}

trait TryAdvanceDelay: CanAdvanceDelay {}

impl TryAdvanceDelay for NoDelay {}

trait LockDelay {
    fn pre_wait(&mut self) {}
}

impl LockDelay for NoDelay {}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::assert_matches;

    use diskann_utils::lazy_format;

    use crate::test::Sequencer;

    //-----------//
    // SlotEpoch //
    //-----------//

    /// This test runs though the possible state transitions for [`SlotEpoch`] with a single
    /// thread. A small table of possible cycles and expected results is used.
    ///
    /// Dedicated race condition tests are performed in other tests.
    #[test]
    fn test_slot_guard() {
        let guard = SlotEpoch::new();
        assert_eq!(guard.raw_relaxed(), 0);
        assert_eq!(guard.current_value_relaxed(), U62::default());

        #[derive(Debug, Clone, Copy)]
        enum NotifyAt {
            None,
            BeforeUpdate(Exclusion),
            AfterUpdate(Exclusion),
        }

        let cases = [
            // Reader worker - no notification
            (Worker::Reader, NotifyAt::None, Obligation::None),
            // Reader worker - reader notification
            (
                Worker::Reader,
                NotifyAt::BeforeUpdate(Exclusion::Workers),
                Obligation::Notify,
            ),
            (
                Worker::Reader,
                NotifyAt::AfterUpdate(Exclusion::Workers),
                Obligation::Notify,
            ),
            // Reader worker - writer notification
            (
                Worker::Reader,
                NotifyAt::BeforeUpdate(Exclusion::Writers),
                Obligation::None,
            ),
            (
                Worker::Reader,
                NotifyAt::AfterUpdate(Exclusion::Writers),
                Obligation::None,
            ),
            // Writer worker - no notification
            (Worker::Writer, NotifyAt::None, Obligation::None),
            // Reader worker - reader notification
            (
                Worker::Writer,
                NotifyAt::BeforeUpdate(Exclusion::Workers),
                Obligation::Notify,
            ),
            (
                Worker::Writer,
                NotifyAt::AfterUpdate(Exclusion::Workers),
                Obligation::Notify,
            ),
            // Reader worker - writer notification
            (
                Worker::Writer,
                NotifyAt::BeforeUpdate(Exclusion::Writers),
                Obligation::Notify,
            ),
            (
                Worker::Writer,
                NotifyAt::AfterUpdate(Exclusion::Writers),
                Obligation::Notify,
            ),
        ];

        for (worker, notify, obligation) in cases {
            let ctx = lazy_format!(
                "Worker Kind = {:?}, Notify = {:?}, Obligation: {:?}",
                worker,
                notify,
                obligation,
            );

            let first = U62::new(10).unwrap();
            assert!(
                guard.try_acquire_relaxed(first, worker),
                "uncontended acquisition of an empty GuardSlot must succeed -- {ctx}",
            );

            assert_eq!(guard.current_value_relaxed(), first);
            let raw = guard.raw_relaxed();
            match worker {
                Worker::Reader => {
                    assert_eq!(raw & SlotEpoch::WRITE_MASK, 0);
                    assert_eq!(raw & SlotEpoch::NOTIFY_MASK, 0);
                }
                Worker::Writer => {
                    assert_ne!(raw & SlotEpoch::WRITE_MASK, 0);
                    assert_eq!(raw & SlotEpoch::NOTIFY_MASK, 0);
                }
            }

            // Attempting to acquire as either a reader or a writer should fail.
            assert!(
                !guard.try_acquire_relaxed(U62::new(11).unwrap(), Worker::Reader),
                "an occupied GuardSlot` cannot be reacquired -- {ctx}",
            );
            assert!(
                !guard.try_acquire_relaxed(U62::new(11).unwrap(), Worker::Writer),
                "an occupied GuardSlot` cannot be reacquired -- {ctx}",
            );

            let second = U62::new(20).unwrap();
            match notify {
                NotifyAt::None => {
                    guard.update_relaxed(first, second, worker);
                    assert_eq!(guard.current_value_relaxed(), second);
                }
                NotifyAt::BeforeUpdate(update_kind) => {
                    let will_notify = guard.must_notify_relaxed(update_kind);

                    match obligation {
                        Obligation::None => assert!(!will_notify),
                        Obligation::Notify => assert!(will_notify),
                    }

                    guard.update_relaxed(first, second, worker);
                    assert_eq!(guard.current_value_relaxed(), second);
                }
                NotifyAt::AfterUpdate(update_kind) => {
                    guard.update_relaxed(first, second, worker);
                    assert_eq!(guard.current_value_relaxed(), second);

                    let will_notify = guard.must_notify_relaxed(update_kind);
                    match obligation {
                        Obligation::None => assert!(!will_notify),
                        Obligation::Notify => assert!(will_notify),
                    }

                    assert_eq!(guard.current_value_relaxed(), second);
                }
            }

            assert_eq!(guard.clear_release(), obligation);
            assert_eq!(guard.0.load(Ordering::Relaxed), 0);
        }
    }

    /// Here, we test the case where transition `C0` loses the race with `W2`.
    ///
    /// We ensure that `C0` gracefully backs out without  setting the "notify" bit.
    #[test]
    fn test_slot_notify_races_with_clear() {
        for (worker, exclusion) in [
            (Worker::Reader, Exclusion::Workers),
            (Worker::Writer, Exclusion::Workers),
            (Worker::Writer, Exclusion::Writers),
        ] {
            let slot = SlotEpoch::new();
            assert!(slot.try_acquire_relaxed(U62::new(10).unwrap(), worker));

            let seq = Sequencer::new();
            let mut retries = 0;
            let delay = TestSlotEpochDelay::default()
                .with_pre_must_notify_relaxed_cas(|| seq.wait_for(0))
                .with_post_must_notify_relaxed_cas(|| retries += 1);

            std::thread::scope(|s| {
                let scanner = s.spawn(|| slot.must_notify_relaxed_inner(exclusion, delay));

                // Control thread has read the slot state and is preparing to set the
                // "notify" bit. Exectue transition `W2` so the control thread instead
                // finds an cleared slot.
                seq.until_waiting_for(0);
                assert_eq!(slot.clear_release(), Obligation::None);
                seq.advance_past(0);

                let will_notify = scanner.join().unwrap();
                assert!(
                    !will_notify,
                    "a released {worker:?} must not be counted by {exclusion:?}",
                );
            });

            assert_eq!(retries, 1);
            assert_eq!(slot.raw_relaxed(), 0);
        }
    }

    /// Here, we test the case where transition `C0` loses the race with `W1A`.
    ///
    /// We ensure that `C0` gracefully recovers and still applies the "notify" bit.
    #[test]
    fn test_slot_notify_races_with_epoch_updates() {
        for (worker, exclusion) in [
            (Worker::Reader, Exclusion::Workers),
            (Worker::Writer, Exclusion::Workers),
            (Worker::Writer, Exclusion::Writers),
        ] {
            let slot = SlotEpoch::new();
            let first = U62::new(10).unwrap();
            let second = U62::new(20).unwrap();
            let third = U62::new(30).unwrap();
            assert!(
                slot.try_acquire_relaxed(first, worker),
                "empty slots should be acquireable",
            );

            let seq = Sequencer::new();
            let mut retries = 0;
            let delay = TestSlotEpochDelay::default()
                .with_pre_must_notify_relaxed_cas(|| seq.wait_for(0))
                .with_post_must_notify_relaxed_cas(|| {
                    retries += 1;
                    if retries == 1 {
                        seq.wait_for(1);
                    }
                });

            std::thread::scope(|s| {
                let scanner = s.spawn(|| slot.must_notify_relaxed_inner(exclusion, delay));

                // Control thread has read the state and is preparing to set the "notify"
                // bit. Update the epoch so application of the "notify" bit fails.
                seq.until_waiting_for(0);
                slot.update_relaxed(first, second, worker);
                seq.advance_past(0);
                assert_eq!(slot.current_value_relaxed(), second);

                // Control thread has resampled the epoch value. Update the epoch yet again
                // so the CAS fails and the control thread tries again.
                seq.until_waiting_for(1);
                slot.update_relaxed(second, third, worker);
                seq.advance_past(1);

                assert_eq!(slot.current_value_relaxed(), third);

                let will_notify = scanner.join().unwrap();
                assert!(
                    will_notify,
                    "an updated {worker:?} must still be counted by {exclusion:?}",
                );
            });

            assert_eq!(retries, 2);
            assert_eq!(slot.current_value_relaxed(), third);
            assert_eq!(slot.clear_release(), Obligation::Notify);
            assert_eq!(slot.raw_relaxed(), 0);
        }
    }

    /// This models a more complex interaction between transition `C0` and the other
    /// transitions where a slot goes from being used, to being unused, to being recycled
    /// by another worker (of a type compatible with the lock being acquired).
    ///
    /// Basically, `C0` is attempted but races with both `W2` and `W0`.
    #[test]
    fn test_slot_notify_races_with_reuse() {
        let first = U62::new(10).unwrap();
        let second = U62::new(20).unwrap();

        // 1. The kind of the original worker.
        // 2. The kind of lock being acquired.
        // 3. The kind of the replacement worker.
        // 4. Expected obligation for the replacement worker.
        // 5. Retry attempts for the notification CAS loop.

        // A writer retires and is replaced with a reader.
        //
        // `C0` should not tel the worker to notify the registry.
        let case1 = (
            Worker::Writer,
            Exclusion::Writers,
            Worker::Reader,
            second,
            Obligation::None,
            1,
        );

        // A writer retires and is replaced by another writer.
        //
        // This is transiently possible during registration.
        let case2 = (
            Worker::Writer,
            Exclusion::Writers,
            Worker::Writer,
            second,
            Obligation::Notify,
            1,
        );

        // A reader retires and is replaced by another writer.
        //
        // This is transiently possible during registration.
        let case3 = (
            Worker::Reader,
            Exclusion::Workers,
            Worker::Writer,
            second,
            Obligation::Notify,
            1,
        );

        // A writer retires and is replaced by a reader.
        let case4 = (
            Worker::Writer,
            Exclusion::Workers,
            Worker::Reader,
            second,
            Obligation::Notify,
            1,
        );

        // A writer retires and is replaced by a writer in the same epoch.
        let case5 = (
            Worker::Writer,
            Exclusion::Writers,
            Worker::Writer,
            first,
            Obligation::Notify,
            0,
        );

        for case in [case1, case2, case3, case4, case5] {
            let (worker, exclusion, replacement_kind, replacement_epoch, obligation, retries) =
                case;

            let slot = SlotEpoch::new();
            assert!(
                slot.try_acquire_relaxed(first, worker),
                "empty slots should be acquirable",
            );

            let seq = Sequencer::new();
            let mut actual_retries = 0;
            let delay = TestSlotEpochDelay::default()
                .with_pre_must_notify_relaxed_cas(|| seq.wait_for(0))
                .with_post_must_notify_relaxed_cas(|| actual_retries += 1);

            std::thread::scope(|s| {
                let scanner = s.spawn(|| slot.must_notify_relaxed_inner(exclusion, delay));

                seq.until_waiting_for(0);

                // Control thread has read the slot and thinks it needs to inform the worker
                // that it must be notified.
                //
                // In this interval, release the slot and acquire it again with the new
                // worker.
                assert_eq!(slot.clear_release(), Obligation::None);
                assert!(slot.try_acquire_relaxed(replacement_epoch, replacement_kind));
                seq.advance_past(0);

                let will_notify = scanner.join().unwrap();
                assert_eq!(
                    will_notify,
                    obligation == Obligation::Notify,
                    "{exclusion:?} must re-evaluate a slot reused by {replacement_kind:?}",
                );
            });

            assert_eq!(actual_retries, retries);
            assert_eq!(slot.current_value_relaxed(), replacement_epoch);
            assert_eq!(slot.clear_release(), obligation);
            assert_eq!(slot.raw_relaxed(), 0);
        }
    }

    //----------//
    // Registry //
    //----------//

    #[test]
    fn test_guard_belongs() {
        let other = Registry::with_capacity(NonZeroUsize::new(5).unwrap());

        for capacity in 1..10 {
            let registry = Registry::with_capacity(NonZeroUsize::new(capacity).unwrap());
            let mut guards = Vec::new();
            while let Ok(guard) = registry.read_guard() {
                guards.push(guard);
            }

            let handle = registry.handle();
            for g in guards.iter() {
                assert!(handle.inner.guard_belongs(g));
                assert!(registry.inner.guard_belongs(g));

                handle.assert_guard_belongs(g);

                // We should correctly detect that `g` does not belong to the other registry.
                assert!(!other.inner.guard_belongs(g));
            }
        }
    }

    #[test]
    #[should_panic = "invalid epoch guard"]
    fn test_guard_belongs_panic() {
        let a = Registry::with_capacity(NonZeroUsize::new(5).unwrap());
        let b = Registry::with_capacity(NonZeroUsize::new(5).unwrap());

        let guard = b.read_guard().unwrap();

        // This function should panic since the guard does not belong.
        a.handle().assert_guard_belongs(&guard);
    }

    #[test]
    fn shared_guards_prevent_advancement() {
        let registry = Registry::with_capacity(NonZeroUsize::new(5).unwrap());

        let guard = registry.read_guard().unwrap();

        // Push an item into the retire queue.
        //
        // This should be prevented from being released until all clones get dropped.
        {
            let retire = registry.retire().unwrap();
            retire.retire(10);
        }

        // With one guard active at the current - we should be able to advance the epoch.
        {
            let drain = registry.try_advance().unwrap();
            assert!(drain.is_empty());
        }

        // Push another item in the next epoch.
        //
        // This should be prevented from being released until all clones get dropped.
        {
            let retire = registry.retire().unwrap();
            retire.retire(20);
        }

        // Now that we have advanced the epoch, the old guard should prevent epoch advandement.
        assert!(
            registry.try_advance().is_none(),
            "guard should prevent epoch advancement"
        );

        // Clone the guard and drop the original one.
        let clone = guard.share();
        std::mem::drop(guard);

        // The epoch should still not advance.
        assert!(
            registry.try_advance().is_none(),
            "clone should prevent epoch advancement"
        );

        // Retire into `clone` and then drop it.
        std::mem::drop(clone);

        registry.assert_no_workers();

        {
            let drain = registry.try_advance().unwrap();
            assert!(drain.is_empty());
        }

        // Retrieve the item that was retired while the guards were active.
        {
            let drain = registry.try_advance().unwrap();
            let ids: Vec<_> = drain.collect();
            assert_eq!(&*ids, &[10]);
        }

        // Retrieve the item that was retired while the guards were active but in the epoch
        // **after** the guards.
        {
            let drain = registry.try_advance().unwrap();
            let ids: Vec<_> = drain.collect();
            assert_eq!(&*ids, &[20]);
        }
    }

    // This test ensures that two threads racing on `hint` will correctly resolve themselves
    // when claiming a slot.
    #[test]
    fn test_cas_race() {
        let seq = Sequencer::new();

        let mut thread_a_loop_count = 0;
        let mut thread_b_loop_count = 0;
        let delay = TestGuardDelay::default()
            .post_guard_check(|| seq.wait_for(0))
            .with_post_fence(|| thread_a_loop_count += 1);

        let registry = Registry::with_capacity(NonZeroUsize::new(2).unwrap());
        std::thread::scope(|s| {
            // Thread A
            s.spawn(|| {
                let g = registry.register(Worker::Reader, delay).unwrap();
                assert_eq!(g.test_slot_index, 1);
                seq.wait_for(1);
            });

            // Thread B
            s.spawn(|| {
                // wait for Thread A to reach the delay point.
                seq.until_waiting_for(0);
                {
                    let delay =
                        TestGuardDelay::default().with_post_fence(|| thread_b_loop_count += 1);
                    let g = registry.register(Worker::Reader, delay).unwrap();
                    assert_eq!(g.test_slot_index, 1);
                }
                let g = registry.register(Worker::Reader, NoDelay).unwrap();
                assert_eq!(g.test_slot_index, 0);
                seq.advance_past(0);
                seq.advance_past(1);
            });
        });

        assert_eq!(thread_a_loop_count, 1);
        assert_eq!(thread_b_loop_count, 1);

        registry.assert_no_workers();
    }

    //------------------------------//
    // Registration and Advancement //
    //------------------------------//

    /// This tests the case where a thread enters registration, reads a generation, then
    /// sleeps for several generation advvances.
    ///
    /// It ensures the thread recovers properly by updating its slot to the newest stable
    /// epoch.
    #[test]
    fn test_register_wait() {
        let seq = Sequencer::new();

        let mut loop_count = 0;
        let delay = TestGuardDelay::default()
            .post_guard_check(|| seq.wait_for(0))
            .with_post_cas(|| seq.wait_for(1))
            .with_pre_fence(|| loop_count += 1);

        let registry = Registry::with_capacity(NonZeroUsize::new(2).unwrap());

        std::thread::scope(|s| {
            let handle = s.spawn(|| {
                let guard = registry.register(Worker::Writer, delay).unwrap();
                let guard = Retire {
                    guard: WriteGuard { inner: guard },
                };

                // Since we hit the CAS loop - this serves as a sanity check that we have
                // the correct drain buffer.
                guard.retire(10);
                guard.retire(1);
                guard.retire(2);
                guard.retire(3);
                guard
            });

            // Wait for the spawned thread to reach the critical section.
            seq.until_waiting_for(0);

            assert_eq!(registry.waiting(), U62::new(1).unwrap());
            {
                let drain = registry.try_advance().unwrap();
                assert!(drain.is_empty());
                assert_eq!(
                    registry.test_epoch(),
                    EpochRead::Ready(U62::new(2).unwrap())
                );
            }

            {
                let drain = registry.try_advance().unwrap();
                assert!(drain.is_empty());
                assert_eq!(
                    registry.test_epoch(),
                    EpochRead::Ready(U62::new(3).unwrap())
                );
            }

            // We allow the registering thread to make it past the CAS.
            //
            // We pause it again because we want to verify that it registers an old generation.
            seq.advance_past(0);
            seq.until_waiting_for(1);
            let (can_advance, waiter) = registry.can_advance(&mut NoDelay);
            assert!(!can_advance);
            assert_eq!(
                waiter,
                U62::new(1).unwrap(),
                "waiting thread registers an older generation before observing the change"
            );
            seq.advance_past(1);

            let expected = U62::new(3).unwrap();

            // The generation should be the last set one - even though this thread was
            // parked during the transition.
            let r = handle.join().unwrap();
            assert_eq!(r.guard.test_epoch(), expected);
            assert_eq!(registry.waiting(), expected);
        });

        assert_eq!(
            loop_count, 2,
            "the registering thread should have looped to update its generation"
        );

        registry.assert_no_workers();

        // Verify that we reclaim the ID flushed by the registering thread.
        //
        // This requires three epoch advancements.
        {
            let drain = registry.try_advance().unwrap();
            assert!(drain.is_empty());
        }

        {
            let drain = registry.try_advance().unwrap();
            assert!(drain.is_empty());
        }

        {
            let drain = registry.try_advance().unwrap();
            let ids: Vec<_> = drain.collect();
            assert_eq!(ids, &[10, 1, 2, 3]);
        }
    }

    // Verifies that filling every slot causes `register` to return `Unavailable`, and that
    // dropping an existing guard frees up its slot for a subsequent registration.
    #[test]
    fn test_slot_exhaustion() {
        let registry = Registry::with_capacity(NonZeroUsize::new(2).unwrap());

        let g0 = registry.read_guard().unwrap();
        let g1 = registry.read_guard().unwrap();

        // All guard slots are now occupied. The next registration must fail.
        assert!(matches!(registry.read_guard(), Err(Unavailable)));
        assert!(matches!(registry.read_guard(), Err(Unavailable)));

        // Dropping a guard releases its slot.
        let freed_slot = g0.test_slot_index;
        drop(g0);

        let g2 = registry.read_guard().unwrap();
        assert_eq!(
            g2.test_slot_index, freed_slot,
            "newly freed slot should be reclaimed"
        );

        // Registry is full again.
        assert!(matches!(registry.read_guard(), Err(Unavailable)));

        drop(g1);
        drop(g2);

        registry.assert_no_workers();
    }

    #[test]
    fn test_slot_wrap_around() {
        let registry = Registry::with_capacity(NonZeroUsize::new(4).unwrap());

        let (g2, g3) = {
            let _g0 = registry.read_guard().unwrap();
            let _g1 = registry.read_guard().unwrap();

            let g2 = registry.read_guard().unwrap();
            let g3 = registry.read_guard().unwrap();
            (g2, g3)
        };

        assert_eq!(g2.test_slot_index, 2);
        assert_eq!(g3.test_slot_index, 3);

        let f = || {
            // Keep wrapping and hitting the first two guard slots.
            for _ in 0..10 {
                let g0 = registry.read_guard().unwrap();
                let g1 = registry.read_guard().unwrap();

                let s0 = g0.test_slot_index;
                let s1 = g1.test_slot_index;

                // Due to how the hint works, the slots could be acquired in either order.
                if s0 < s1 {
                    assert_eq!((s0, s1), (0, 1));
                } else {
                    assert_eq!((s0, s1), (1, 0));
                };

                assert!(matches!(registry.read_guard(), Err(Unavailable)));
            }
        };

        // Run with the default hint.
        f();

        // Set the hint to `usize::MAX`.
        registry
            .inner
            .hint
            .store(usize::MAX - 10, Ordering::Relaxed);

        // Run tests again to ensure we can properly handle wrap-around.
        f();

        drop((g2, g3));
        registry.assert_no_workers();
    }

    // Verifies that `try_advance` short-circuits to `None` when another thread already holds
    // the `drain` mutex, even if `can_advance` would otherwise succeed. This guards the
    // early `try_lock` that avoids a redundant slot scan.
    #[test]
    fn test_concurrent_try_advance() {
        let registry = Registry::with_capacity(NonZeroUsize::new(2).unwrap());

        // No outstanding registrations, so `can_advance` would succeed for any caller.
        let drain = registry
            .try_advance()
            .expect("first try_advance must succeed");
        let gen_after_first = registry.test_epoch();
        assert_eq!(gen_after_first, EpochRead::Ready(U62::new(2).unwrap()));

        // While the first `Drain` is alive (holding the drain mutex), a concurrent
        // `try_advance` must return `None` without advancing the generation.
        std::thread::scope(|s| {
            s.spawn(|| {
                assert!(
                    registry.try_advance().is_none(),
                    "try_advance must fail while another holds the drain mutex"
                );
                assert_eq!(
                    registry.test_epoch(),
                    gen_after_first,
                    "generation must not advance when drain is contended"
                );
            });
        });

        // Releasing the drain unblocks subsequent advances.
        drop(drain);

        let _drain2 = registry
            .try_advance()
            .expect("try_advance must succeed once drain is released");
        assert_eq!(
            registry.test_epoch(),
            EpochRead::Ready(U62::new(3).unwrap())
        );
    }

    // Verifies the 4-queue rotation invariant: items retired at generation `G` are drained
    // on the third `try_advance` after `G`. The first advance returns the queue from
    // `(G - 2) % 4` (two cycles older). The second advance returns the queue from one previous
    // cycle. Neither may contain items from `G`.
    #[test]
    fn test_drain_rotation() {
        let registry = Registry::with_capacity(NonZeroUsize::new(1).unwrap());

        // Helper: register, retire one item, drop. Returns the generation we retired at.
        let retire_at = |id: u32| {
            let g = registry.retire().unwrap();
            let epoch = g.guard.test_epoch();
            g.retire(id);
            epoch
        };

        // Retire 100 at generation A (= 1).
        let gen_a = retire_at(100);
        assert_eq!(gen_a, U62::new(1).unwrap());

        // 1st advance after A: must NOT drain item 100.
        {
            let drain = registry.try_advance().unwrap();
            assert!(
                drain.is_empty(),
                "100 must not drain on 1st advance after A"
            );
        }

        // Retire 200 at generation B (= A - 1).
        let gen_b = retire_at(200);
        assert_eq!(gen_b, gen_a.checked_add(1).unwrap());

        // 2nd advance after A: must NOT drain item 100.
        {
            let drain = registry.try_advance().unwrap();
            assert!(
                drain.is_empty(),
                "100 must not drain on 2nd advance after A"
            );
        }

        // Retire 300 at generation C.
        let _gen_c = retire_at(300);

        // 3rd advance after A (1st after B): drains A's queue → [100].
        {
            let drained: Vec<_> = registry.try_advance().unwrap().collect();
            assert_eq!(drained, &[100]);
        }

        // 3rd advance after B: drains B's queue → [200].
        {
            let drained: Vec<_> = registry.try_advance().unwrap().collect();
            assert_eq!(drained, &[200]);
        }

        // 3rd advance after C: drains C's queue → [300].
        {
            let drained: Vec<_> = registry.try_advance().unwrap().collect();
            assert_eq!(drained, &[300]);
        }

        // Rotation has cycled back to where A's queue used to live — must be empty,
        // proving the queue slot was drained cleanly and is reusable.
        {
            let drain = registry.try_advance().unwrap();
            assert!(
                drain.is_empty(),
                "rotation should leave queues empty after one cycle"
            );
        }

        registry.assert_no_workers();
    }

    //----------------//
    // Registry Locks //
    //----------------//

    #[test]
    fn test_basic_lock_acquisition() {
        let registry = Registry::with_capacity(NonZeroUsize::new(2).unwrap());

        // Read Lock
        {
            let _worker_exclusion = registry
                .try_exclude_workers()
                .expect("with no workers - lock acquisition should succeed");

            // With the read lock held, readers and writers should not succeed.
            assert_matches!(registry.read_guard(), Err(Unavailable));
            assert_matches!(registry.write_guard(), Err(Unavailable));
        }

        // With the read lock dropped - we should be able to acquire guards again.
        let _ = registry.read_guard().unwrap();
        let _ = registry.write_guard().unwrap();

        // We can reacquire a read guard.
        {
            let _worker_exclusion = registry
                .try_exclude_workers()
                .expect("with no workers - lock acquisition should succeed");

            assert_matches!(registry.read_guard(), Err(Unavailable));
            assert_matches!(registry.write_guard(), Err(Unavailable));
        }

        // We can acquire a write guard.
        {
            let _write_lock = registry
                .try_exclude_writers()
                .expect("with no workers - lock acquisition should succeed");

            let _guard = registry.read_guard().expect("write lock allows readers");
            assert_matches!(registry.write_guard(), Err(Unavailable));
        }

        // With lock dropped - we can acquire guards again.
        let _ = registry.read_guard().unwrap();
        let _ = registry.write_guard().unwrap();

        // With only readers - an write lock can be acquired immediatly.
        {
            let _g0 = registry.read_guard().unwrap();
            let _g1 = registry.read_guard().unwrap();
            let _write_lock = registry
                .try_exclude_writers()
                .expect("with only readers - a write lock can be acquired");
        }
    }

    /// Test that with a read and a write guard held, acquisition of a
    /// [`WorkerExclusionGuard`] blocks until all guards are dropped.
    #[test]
    fn test_worker_exclusion_waits() {
        let registry = Registry::with_capacity(NonZeroUsize::new(4).unwrap());

        let seq = Sequencer::new();
        let delay = TestLockDelay::default().with_pre_wait(|| seq.wait_for(0));

        std::thread::scope(|s| {
            // Acquire a read, a write guard, and a write guard that gets downgraded to a
            // read guard.
            let read_guard = registry.read_guard().unwrap();
            let write_guard = registry.write_guard().unwrap();
            let downgraded_write_guard = registry.write_guard().unwrap().read_guard().share();

            let handle = s.spawn(|| {
                let _lock = registry.try_exclude_workers_inner(delay).unwrap();
                seq.wait_for(1);
            });

            // Wait until the control thread has finished doing the scan and is waiting
            // on our two guards.
            seq.until_waiting_for(0);
            assert!(registry.try_snoop_waiters().is_none());
            seq.advance_past(0);
            assert_matches!(registry.snoop_waiters(), Waiters::Pending(3));

            // `parking_lot`'s `CondVar` guarantees no spurious wakeups.
            //
            // Nudge the condition variable anyways to make sure that even if there was a
            // spurious wakeup that we handle it.
            registry.inner.waiter.notify_one();

            drop(read_guard);
            assert_matches!(registry.snoop_waiters(), Waiters::Pending(2));

            drop(write_guard);
            assert_matches!(registry.snoop_waiters(), Waiters::Pending(1));

            drop(downgraded_write_guard);

            // Wait for the control thread to completely acquire the read lock.
            seq.until_waiting_for(1);

            // With the lock held, we can't snoop.
            assert!(registry.try_snoop_waiters().is_none());

            seq.advance_past(1);
            handle.join().unwrap();

            assert_matches!(registry.try_snoop_waiters(), Some(Waiters::None));
        });
    }

    /// Test that with a read and a write guard held, acquisition of a
    /// [`WriterExclusionGuard`] blocks until just the write guards are dropped.
    #[test]
    fn test_write_lock_waits() {
        let registry = Registry::with_capacity(NonZeroUsize::new(4).unwrap());

        let seq = Sequencer::new();
        let delay = TestLockDelay::default().with_pre_wait(|| seq.wait_for(0));

        std::thread::scope(|s| {
            // Acquire a read and a write guard.
            let read_guard0 = registry.read_guard().unwrap();
            let _read_guard1 = registry.read_guard().unwrap();
            let write_guard = registry.write_guard().unwrap();
            let downgraded_write_guard = registry.write_guard().unwrap().read_guard().share();

            let handle = s.spawn(|| {
                let _lock = registry.try_exclude_writers_inner(delay).unwrap();
                seq.wait_for(1);
            });

            // Wait until the control thread has finished doing the scan and is waiting
            // on our two guards.
            seq.until_waiting_for(0);
            assert!(registry.try_snoop_waiters().is_none());
            seq.advance_past(0);
            assert_matches!(registry.snoop_waiters(), Waiters::Pending(2));

            drop(write_guard);
            assert_matches!(registry.snoop_waiters(), Waiters::Pending(1));

            drop(read_guard0);
            assert_matches!(
                registry.snoop_waiters(),
                Waiters::Pending(1),
                "read guard should not notify the write-lock waiter",
            );

            drop(downgraded_write_guard);

            // Wait for the control thread to completely acquire the read lock.
            seq.until_waiting_for(1);

            // With the lock held, we can't snoop.
            assert!(registry.try_snoop_waiters().is_none());

            seq.advance_past(1);
            handle.join().unwrap();
            assert_matches!(registry.try_snoop_waiters(), Some(Waiters::None));
        });
    }

    /// This tests that if a registering thread is pre-empted after "GUARD CHECK" and during
    /// that time, an attempt is made to acquire a registry lock, that the registering thread
    /// correctly either proceeds (if allowed) or rolls back its change and exits.
    #[test]
    fn test_registration_after_lock_scan() {
        for (worker, exclusion, allowed) in [
            (Worker::Reader, Exclusion::Workers, false),
            (Worker::Writer, Exclusion::Workers, false),
            (Worker::Writer, Exclusion::Writers, false),
            (Worker::Reader, Exclusion::Writers, true),
        ] {
            let registry = Registry::with_capacity(NonZeroUsize::new(1).unwrap());
            let epoch = registry.ready_epoch();
            let seq = Sequencer::new();
            let delay = TestGuardDelay::default().post_guard_check(|| seq.wait_for(0));

            std::thread::scope(|s| {
                // Pause after reading the initial epoch to set up the race.
                let handle = s.spawn(|| registry.register(worker, delay));
                seq.until_waiting_for(0);

                let finish_registration = || {
                    seq.advance_past(0);
                    let result = handle.join().unwrap();
                    assert_eq!(
                        result.is_ok(),
                        allowed,
                        "{worker:?} registering after the {exclusion:?} scan",
                    );
                    drop(result);
                    registry.assert_no_workers();
                };

                match exclusion {
                    Exclusion::Workers => {
                        let _lock = registry.try_exclude_workers().unwrap();
                        finish_registration();
                    }
                    Exclusion::Writers => {
                        let _lock = registry.try_exclude_writers().unwrap();
                        finish_registration();
                    }
                }
            });

            assert_eq!(registry.ready_epoch(), epoch);
            assert_matches!(registry.try_snoop_waiters(), Some(Waiters::None));
            let _guard = registry.register(worker, NoDelay).unwrap();
        }
    }

    /// This tests the case where a worker succeeds in the initial registration step, but
    /// is pre-empted before the epoch re-check. In that time, a registry lock is acquired
    /// and the slot claimed by the worker is marked as needing to notify.
    ///
    /// We test that the worker correctly rolls back the slot *and* notifies the registry.
    #[test]
    fn test_registration_rollback_notifies_lock() {
        for (worker, exclusion) in [
            (Worker::Reader, Exclusion::Workers),
            (Worker::Writer, Exclusion::Workers),
            (Worker::Writer, Exclusion::Writers),
        ] {
            let registry = Registry::with_capacity(NonZeroUsize::new(1).unwrap());
            let epoch = registry.ready_epoch();
            let worker_seq = Sequencer::new();
            let control_seq = Sequencer::new();
            let worker_delay = TestGuardDelay::default().with_post_cas(|| worker_seq.wait_for(0));
            let control_delay = TestLockDelay::default().with_pre_wait(|| control_seq.wait_for(0));

            std::thread::scope(|s| {
                // Wait until the worker has claimed a slot.
                let worker = s.spawn(|| registry.register(worker, worker_delay));
                worker_seq.until_waiting_for(0);

                // Acquire the lock.
                //
                // At synchronization point 0, the control thread will have completed the
                // scan and marked the worker thread's slot as needing notification.
                //
                // At synchronization point 1, the lock will be held.
                let controller = s.spawn(|| match exclusion {
                    Exclusion::Workers => {
                        let _lock = registry.try_exclude_workers_inner(control_delay).unwrap();
                        control_seq.wait_for(1);
                    }
                    Exclusion::Writers => {
                        let _lock = registry.try_exclude_writers_inner(control_delay).unwrap();
                        control_seq.wait_for(1);
                    }
                });

                control_seq.until_waiting_for(0);
                assert_ne!(
                    registry.inner.guards[0].epoch.raw_relaxed() & SlotEpoch::NOTIFY_MASK,
                    0,
                    "the provisional {worker:?} must be marked by {exclusion:?}",
                );
                control_seq.advance_past(0);
                assert_matches!(registry.snoop_waiters(), Waiters::Pending(1));

                // Verify that the worker rolls back its change.
                worker_seq.advance_past(0);
                assert_matches!(worker.join().unwrap(), Err(Unavailable));

                // Waiting for synchronization point 1 means that worker thread notified the
                // registry of completion.
                control_seq.until_waiting_for(1);

                registry.assert_no_workers();
                assert_eq!(
                    registry.inner.guards[0].references.load(Ordering::Relaxed),
                    0,
                    "the notification bit should be cleared",
                );
                assert!(registry.try_snoop_waiters().is_none());
                control_seq.advance_past(1);
                controller.join().unwrap();
            });

            assert_eq!(registry.ready_epoch(), epoch);
            assert_matches!(registry.try_snoop_waiters(), Some(Waiters::None));
            let _guard = registry.register(worker, NoDelay).unwrap();
        }
    }

    /// In this test, a write lock is in between a writer checking the epoch and attempting
    /// its slot acquisition. However, there is only one slot and it is used by a reader.
    ///
    /// After the lock is acquired, the reader goes away. The write guards slot acquisition
    /// succeeds, but then it observes that the write lock is held.
    ///
    /// This ensures that the writer correctly rolls back its change.
    #[test]
    fn test_registration_reuses_reader_after_write_lock_scan() {
        let registry = Registry::with_capacity(NonZeroUsize::new(1).unwrap());
        let epoch = registry.ready_epoch();
        let reader = registry.read_guard().unwrap();
        let seq = Sequencer::new();
        let delay = TestGuardDelay::default().post_guard_check(|| seq.wait_for(0));

        std::thread::scope(|s| {
            // As synchronization point 0, the worker thread will have read the epoch
            // as being ready but it will not have attempted to claim a slot.
            let worker = s.spawn(|| registry.register(Worker::Writer, delay));
            seq.until_waiting_for(0);

            // We expect this to succeed and not set a write mask because the current
            // worker is just a reader.
            let lock = registry.try_exclude_writers().unwrap();
            assert_eq!(
                registry.inner.guards[0].epoch.raw_relaxed() & SlotEpoch::NOTIFY_MASK,
                0,
                "writer exclusion must leave the reader unmarked",
            );

            // Dropping the reader will allow the writer to continue and observe that the
            // write lock is held.
            drop(reader);
            seq.advance_past(0);

            // Make sure the writer backs out.
            assert_matches!(worker.join().unwrap(), Err(Unavailable));
            assert_eq!(registry.inner.guards[0].epoch.raw_relaxed(), 0);

            let reader = registry.read_guard().unwrap();
            drop(reader);
            drop(lock);
        });

        assert_eq!(registry.ready_epoch(), epoch);
        assert_matches!(registry.try_snoop_waiters(), Some(Waiters::None));
        let _writer = registry.write_guard().unwrap();
    }

    // Assert that all control side operations are locked out.
    fn assert_controllers_excluded(registry: &Registry) {
        let epoch = registry.test_epoch();
        assert!(
            registry.try_advance().is_none(),
            "advancement must be excluded"
        );
        assert!(
            registry.try_exclude_workers().is_none(),
            "all-workers exclusion must be excluded",
        );
        assert!(
            registry.try_exclude_writers().is_none(),
            "writer exclusion must be excluded",
        );
        assert_eq!(
            registry.test_epoch(),
            epoch,
            "failed control operations must not change the epoch or admission flags",
        );
    }

    /// Test that holding a [`Drain`] blocks out all control side operations.
    #[test]
    fn test_control_exclusion_with_drain() {
        let registry = Registry::with_capacity(NonZeroUsize::new(1).unwrap());
        let drain = registry.try_advance().unwrap();
        let epoch = registry.ready_epoch();

        std::thread::scope(|s| {
            s.spawn(|| assert_controllers_excluded(&registry))
                .join()
                .unwrap();
        });

        drop(drain);
        assert!(registry.try_exclude_workers().is_some());
        assert!(registry.try_exclude_writers().is_some());
        let _drain = registry.try_advance().unwrap();
        assert_eq!(registry.ready_epoch(), epoch.checked_add(1).unwrap());
    }

    /// Test that [`WriterExclusionGuard`] and [`WorkerExclusionGuard`] lock out control
    /// side operations.
    #[test]
    fn test_control_exclusion_with_held_lock() {
        for kind in [Exclusion::Workers, Exclusion::Writers] {
            let registry = Registry::with_capacity(NonZeroUsize::new(1).unwrap());
            let epoch = registry.ready_epoch();
            let check_exclusion = || {
                std::thread::scope(|s| {
                    s.spawn(|| assert_controllers_excluded(&registry))
                        .join()
                        .unwrap();
                });
            };

            match kind {
                Exclusion::Workers => {
                    let _lock = registry.try_exclude_workers().unwrap();
                    check_exclusion();
                }
                Exclusion::Writers => {
                    let _lock = registry.try_exclude_writers().unwrap();
                    check_exclusion();
                }
            }

            assert_eq!(registry.ready_epoch(), epoch);
            assert_matches!(registry.try_snoop_waiters(), Some(Waiters::None));
            assert!(registry.try_exclude_workers().is_some());
            assert!(registry.try_exclude_writers().is_some());
            let _drain = registry.try_advance().unwrap();
            assert_eq!(registry.ready_epoch(), epoch.checked_add(1).unwrap());
        }
    }

    /// Test that control side operations are locked out when there is a pending lock
    /// acquisition.
    #[test]
    fn test_control_exclusion_with_waiting_lock() {
        for kind in [Exclusion::Workers, Exclusion::Writers] {
            let registry = Registry::with_capacity(NonZeroUsize::new(1).unwrap());
            let epoch = registry.ready_epoch();
            let worker = registry.write_guard().unwrap();
            let seq = Sequencer::new();
            let delay = TestLockDelay::default().with_pre_wait(|| seq.wait_for(0));

            std::thread::scope(|s| {
                // Synchonization point 0 ensures that the control thread holds the mutex.
                //
                // Synchronization point 1 ensures that the lock itself is held.
                let controller = s.spawn(|| match kind {
                    Exclusion::Workers => {
                        let _lock = registry.try_exclude_workers_inner(delay).unwrap();
                        seq.wait_for(1);
                    }
                    Exclusion::Writers => {
                        let _lock = registry.try_exclude_writers_inner(delay).unwrap();
                        seq.wait_for(1);
                    }
                });

                seq.until_waiting_for(0);
                seq.advance_past(0);
                // Acquiring this mutex proves the controller has released it to wait.
                assert_matches!(registry.snoop_waiters(), Waiters::Pending(1));
                assert_controllers_excluded(&registry);
                assert_matches!(registry.snoop_waiters(), Waiters::Pending(1));

                drop(worker);
                seq.until_waiting_for(1);

                // The lock is held - so controllers should still be excluded.
                assert_controllers_excluded(&registry);
                seq.advance_past(1);
                controller.join().unwrap();
            });

            // Verify that once the locks are dropped - the registry functions as expected.
            assert_eq!(registry.ready_epoch(), epoch);
            assert_matches!(registry.try_snoop_waiters(), Some(Waiters::None));
            assert!(registry.try_exclude_workers().is_some());
            assert!(registry.try_exclude_writers().is_some());
            let _drain = registry.try_advance().unwrap();
            assert_eq!(registry.ready_epoch(), epoch.checked_add(1).unwrap());
        }
    }

    /// This tests the corner case where all pending waiters have completed but the control
    /// thread acquiring the [`WorkerExclusionGuard`]/[`WriterExclusionGuard`] has not fully
    /// woken up yet.
    ///
    /// Because notification happens in a destructor, we cannot easily insert a delay to
    /// exercise this case.
    ///
    /// Instead, we cheat and set up [`Inner::exclusive`] manually.
    #[test]
    fn test_control_exclusion_with_completed_waiters() {
        for kind in [Exclusion::Workers, Exclusion::Writers] {
            let registry = Registry::with_capacity(NonZeroUsize::new(1).unwrap());
            let epoch = registry.ready_epoch();
            {
                let mut exclusive = registry.inner.exclusive.lock();
                registry.inner.epoch.set_lock_release(kind);
                // Construct the window before the waiting controller consumes Pending(0).
                *exclusive = Waiters::Pending(0);
            }

            std::thread::scope(|s| {
                s.spawn(|| assert_controllers_excluded(&registry))
                    .join()
                    .unwrap();
            });
            assert_matches!(registry.snoop_waiters(), Waiters::Pending(0));

            {
                let mut exclusive = registry.inner.exclusive.lock();
                registry.inner.epoch.clear_lock_release(kind);
                *exclusive = Waiters::None;
            }

            // Make sure that everything works fine once the registry is back in a sensible
            // state.
            assert_eq!(registry.ready_epoch(), epoch);
            assert!(registry.try_exclude_workers().is_some());
            assert!(registry.try_exclude_writers().is_some());
            let _drain = registry.try_advance().unwrap();
            assert_eq!(registry.ready_epoch(), epoch.checked_add(1).unwrap());
        }
    }

    #[test]
    fn test_unavailable_display() {
        assert_eq!(Unavailable.to_string(), "could not claim a guard slot",);
    }

    //-------------//
    // Test Delays //
    //-------------//

    macro_rules! tester {
        ($struct:ident, $trait:ident, $($with:ident => $f:ident),* $(,)?) => {
            #[derive(Default)]
            struct $struct<'a> {
                $($f: Option<Box<dyn FnMut() + Send + 'a>>,)*
            }

            impl<'a> $struct<'a> {
                $(
                    fn $with<F>(mut self, f: F) -> Self
                    where
                        F: FnMut() + Send + 'a
                    {
                        self.$f = Some(Box::new(f));
                        self
                    }
                )*
            }

            impl $trait for $struct<'_> {
                $(
                    fn $f(&mut self) {
                        if let Some(f) = self.$f.as_mut() {
                            f()
                        }
                    }
                )*
            }
        }
    }

    tester! {
        TestGuardDelay,
        GuardDelay,
        post_guard_check => post_guard_check,
        with_post_cas => post_cas,
        with_pre_fence => pre_fence,
        with_post_fence => post_fence,
    }

    tester! {
        TestSlotEpochDelay,
        SlotEpochDelay,
        with_pre_must_notify_relaxed_cas => pre_must_notify_relaxed_cas,
        with_post_must_notify_relaxed_cas => post_must_notify_relaxed_cas,
    }

    tester! {
        TestLockDelay,
        LockDelay,
        with_pre_wait => pre_wait,
    }
}
