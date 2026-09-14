use std::sync::atomic::Ordering;

use diskann::utils::IntoUsize;
use thiserror::Error;

use crate::{
    buffer::{Buffer, BufferError, RawSlice},
    epoch,
    num::{Align, Bytes, IdLimit},
    store::{Lifecycle, Store, slots},
    tag::{AtomicTag, Tag},
};

#[derive(Debug, Clone)]
pub(crate) struct Config {
    bytes: Bytes,
}

impl Config {
    pub(crate) fn new(bytes: Bytes) -> Self {
        Self { bytes }
    }

    pub(crate) fn build(self, id_limit: IdLimit) -> Result<Simple, diskann::error::Infallible> {
        let Self { bytes } = self;
        Ok(Simple::new(id_limit, bytes))
    }
}

impl slots::SlotsConfig for Config {
    type Slots = Simple;
    type Error = diskann::error::Infallible;
    fn build(self, id_limit: IdLimit) -> Result<Simple, diskann::error::Infallible> {
        <Config>::build(self, id_limit)
    }
}

#[derive(Debug)]
pub(crate) struct Simple {
    buffer: Buffer,
    bytes: Bytes,
}

impl Simple {
    pub(crate) fn config(bytes: Bytes) -> Config {
        Config::new(bytes)
    }

    pub(crate) fn new(id_limit: IdLimit, bytes: Bytes) -> Self {
        let bytes = bytes.checked_next_multiple_of(Bytes::CACHELINE).unwrap();
        let buffer = Buffer::new(id_limit.as_usize(), bytes, Align::_128).unwrap();

        Self { buffer, bytes }
    }

    /// Return the [`IdLimit`] for this store.
    pub(crate) fn id_limit(&self) -> IdLimit {
        IdLimit::new(self.buffer.len() as u32)
    }

    /// Return the number of bytes for each entry.
    pub(crate) fn bytes(&self) -> Bytes {
        self.bytes
    }

    pub(crate) fn raw_reader(&self) -> RawReader<'_> {
        RawReader {
            buffer: &self.buffer,
            bytes: self.bytes,
        }
    }

    // /// Return a [`Reader`] over [`Self`] inside `store`.
    // pub(crate) fn reader(store: &Store<Self>) -> Result<Reader<'_>, epoch::Unavailable> {
    //     store.guard(|this, guard: epoch::Guard<'_>| Reader {
    //         buffer: &this.buffer,
    //         unpadded: this.unpadded,
    //         _guard: guard,
    //     })
    // }

    /// Return the data at position `i` without bound-checking.
    ///
    /// # Safety
    ///
    /// The index `i` must be less than `self.buffer.len()`.
    unsafe fn data_unchecked(&self, i: usize) -> RawSlice<'_> {
        // SAFETY: inherited from caller.
        unsafe { self.buffer.get_unchecked(i) }.truncate(self.bytes)
    }

    fn data(&self, i: usize) -> Option<RawSlice<'_>> {
        if i >= self.buffer.len() {
            None
        } else {
            // SAFETY: We've checked that `i` is in-bounds.
            Some(unsafe { self.data_unchecked(i) })
        }
    }
}

// #[derive(Debug, Error)]
// #[error(transparent)]
// pub(crate) struct IntrusiveError(IntrusiveErrorInner);
//
// impl IntrusiveError {
//     fn bytes_overflowed() -> Self {
//         Self(IntrusiveErrorInner::BytesOverflowed)
//     }
//
//     fn buffer_error(err: BufferError) -> Self {
//         Self(IntrusiveErrorInner::BufferError(err))
//     }
// }
//
// #[derive(Debug, Error)]
// enum IntrusiveErrorInner {
//     #[error("computation of the bytes per slot overflowed")]
//     BytesOverflowed,
//     #[error(transparent)]
//     BufferError(BufferError),
// }

impl slots::Slots for Simple {
    type Exclusive<'a> = Exclusive<'a>;

    fn id_limit(&self) -> IdLimit {
        <Simple>::id_limit(self)
    }

    unsafe fn acquire(&self, i: u32, _: Lifecycle) -> Exclusive<'_> {
        let Some(data) = self.data(i.into_usize()) else {
            panic!("index {i} is out-of-bounds");
        };

        Exclusive { data }
    }

    unsafe fn reclaim(&self, i: u32, _: Lifecycle) {}
    unsafe fn retire(&self, i: u32, _: Lifecycle) {}
}

#[derive(Debug)]
pub(crate) struct RawReader<'a> {
    buffer: &'a Buffer,
    bytes: Bytes,
}

impl RawReader<'_> {
    /// Return `true` if the index `i` is in-bounds.
    #[inline]
    #[must_use = "this function has no side-effects"]
    pub(crate) fn is_in_bounds(&self, i: usize) -> bool {
        i < self.buffer.len()
    }

    /// Read the data as position `i` if it is guaranteed to be race-free without bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// The index `i` must satisfy [`Self::is_in_bounds`].
    #[inline]
    pub(crate) unsafe fn read_in_bounds<'a>(&'a self, i: usize, _guard: &'a epoch::Guard<'_>) -> &'a [u8] {
        debug_assert!(self.is_in_bounds(i));

        unsafe { self.buffer.get_unchecked(i).truncate_unchecked(self.bytes).as_slice() }
    }
}

// /// A reader into an [`Intrusive`] store.
// #[derive(Debug)]
// pub(crate) struct Reader<'a> {
//     buffer: &'a Buffer,
//     unpadded: Bytes,
//     _guard: epoch::Guard<'a>,
// }
//
// impl<'a> Reader<'a> {
//     /// Attempt to read the value at index `i`. This can fail for any of the
//     /// following reasons:
//     ///
//     /// 1. Index `i` is out-of-bounds.
//     /// 2. The read cannot be guaranteed to be race-free.
//     #[inline]
//     pub(crate) fn read(&self, i: usize) -> Option<&[u8]> {
//         if self.is_in_bounds(i) {
//             // SAFETY: `i` is in-bounds.
//             unsafe { self.read_in_bounds(i) }
//         } else {
//             None
//         }
//     }
//
//     /// Return `true` if the index `i` is in-bounds.
//     #[inline]
//     #[must_use = "this function has no side-effects"]
//     pub(crate) fn is_in_bounds(&self, i: usize) -> bool {
//         i < self.buffer.len()
//     }
//
//     /// Return the [`IdLimit`] for this collection.
//     #[inline]
//     #[must_use = "this function has no side-effects"]
//     pub(crate) fn id_limit(&self) -> IdLimit {
//         // Like `Intrusive::id_limit`, the numeric cast is safe because by construction,
//         // the underlying buffer is limited to `u32::MAX`.
//         IdLimit::new(self.buffer.len() as u32)
//     }
//
//     /// Return `true` if it is safe to read the data at position `i`.
//     ///
//     /// This guarantee only holds while `self` is alive. Construction of a new [`Reader`]
//     /// requires a separate check.
//     #[cfg_attr(
//         not(test),
//         expect(
//             dead_code,
//             reason = "this is non-trivial method that is likely to be used in the future"
//         )
//     )]
//     pub(crate) fn can_read(&self, i: usize) -> Option<bool> {
//         if !self.is_in_bounds(i) {
//             return None;
//         }
//
//         // SAFETY: We've checked that `i` is in-bounds.
//         //
//         // Further, we guarantee that `self.unpadded >= AtomicTag::SIZE`, so the pointer
//         // arithmetic is in-bounds.
//         let tag_ptr = unsafe {
//             self.buffer
//                 .get_unchecked(i)
//                 .as_mut_ptr()
//                 .add(self.unpadded.unchecked_sub(AtomicTag::SIZE).value())
//         };
//
//         // SAFETY: We only access tag pointers atomically.
//         let can_read = unsafe { AtomicTag::from_ptr(tag_ptr.cast()) }
//             .load(Ordering::Acquire)
//             .can_read();
//
//         Some(can_read)
//     }
//
//     /// Read the data as position `i` if it is guaranteed to be race-free without bounds
//     /// checking.
//     ///
//     /// # Safety
//     ///
//     /// The index `i` must satisfy [`Self::is_in_bounds`].
//     #[inline]
//     pub(crate) unsafe fn read_in_bounds(&self, i: usize) -> Option<&[u8]> {
//         debug_assert!(self.is_in_bounds(i));
//
//         // SAFETY:
//         //
//         // * The caller asserts `i` is in-bounds.
//         // * We maintain the internal invariant that `self.unpadded <= self.buffer.stride()`.
//         // * Further, we maintain that `self.unpadded >= AtomicTag::SIZE`.
//         let (data, tag_ptr) = unsafe {
//             self.buffer
//                 .get_unchecked(i)
//                 .truncate_unchecked(self.unpadded)
//                 .split_unchecked(self.unpadded.unchecked_sub(AtomicTag::SIZE))
//         };
//
//         // NOTE: Must be `Acquire` to correctly synchronize with writes.
//         //
//         // SAFETY: We are careful in this module to ensure that inline tags are only accessed
//         // atomically.
//         let can_read = unsafe { AtomicTag::from_ptr(tag_ptr.as_mut_ptr().cast()) }
//             .load(Ordering::Acquire)
//             .can_read();
//
//         if can_read {
//             // SAFETY: We've passed the `can_read` check - `_guard` will ensure the read
//             // slice is valid and race-free.
//             Some(unsafe { data.as_slice() })
//         } else {
//             None
//         }
//     }
//
//     /// Return the raw data slice for index `i` without any race guarantees.
//     ///
//     /// This includes both the data **and** the intrusive tag.
//     ///
//     /// # Safety
//     ///
//     /// The index `i` must satisfy [`Self::is_in_bounds`].
//     ///
//     /// The returned [`RawSlice`] may only be used for prefetching. Callers must never
//     /// materialize it as a proper slice or reference.
//     #[inline]
//     pub(crate) unsafe fn read_raw_unchecked(&self, i: usize) -> RawSlice<'_> {
//         // SAFETY: Inherited from caller: `i` is in bounds.
//         unsafe { self.buffer.get_unchecked(i) }.truncate(self.unpadded)
//     }
//
//     /// Return the number of bytes for each entry.
//     pub(crate) fn bytes(&self) -> Bytes {
//         self.bytes_plus_tag().unchecked_sub(AtomicTag::SIZE)
//     }
//
//     /// Return the number of bytes plus the atomic tag.
//     pub(crate) fn bytes_plus_tag(&self) -> Bytes {
//         self.unpadded
//     }
// }

/// A [`slots::Exclusive`] for [`Intrusive`].
#[derive(Debug)]
pub(crate) struct Exclusive<'a> {
    data: RawSlice<'a>,
}

impl<'a> Exclusive<'a> {
    /// Return the data within this slot as a mutable slice.
    ///
    /// The length of this slice is guaranteed to be the number of bytes passed to
    /// [`Intrusive::new`] or [`Config::new`].
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: Users of the `slots::Exclusive` are obligated to ensure exclusivity.
        //
        // Since `Reader` obeys the slots life-cycle requirements, a concurrent reader
        // of this data should not be possible.
        unsafe { self.data.as_mut_slice() }
    }
}

impl slots::Exclusive for Exclusive<'_> {
    fn publish(self, _: Lifecycle) {}
    fn freeze(self, _: Lifecycle) {}
    fn abort(self, _: Lifecycle) {}
}
