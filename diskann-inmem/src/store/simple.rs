/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::utils::IntoUsize;
use thiserror::Error;

use crate::{
    buffer::{Buffer, RawSlice},
    epoch,
    num::{Align, Bytes, IdLimit},
    store::{Lifecycle, slots},
    tag,
};

#[derive(Debug, Clone)]
pub(crate) struct Config {
    bytes: Bytes,
}

impl Config {
    pub(crate) fn new(bytes: Bytes) -> Self {
        Self { bytes }
    }
}

impl slots::SlotsConfig for Config {
    type Slots = Simple;
    type Error = diskann::error::Infallible;
    unsafe fn build(
        self,
        tags: &tag::Authoritative,
    ) -> Result<Simple, diskann::error::Infallible> {
        let Self { bytes } = self;
        Ok(unsafe { Simple::new(bytes, tags.read_only()) })
    }
}

#[derive(Debug)]
pub(crate) struct Simple {
    buffer: Buffer,
    bytes: Bytes,
    tags: tag::ReadOnly,
}

impl Simple {
    pub(crate) fn config(bytes: Bytes) -> Config {
        Config::new(bytes)
    }

    unsafe fn new(bytes: Bytes, tags: tag::ReadOnly) -> Self {
        let bytes = bytes.checked_next_multiple_of(Bytes::CACHELINE).unwrap();
        let buffer = Buffer::new(tags.id_limit().as_usize(), bytes, Align::_128).unwrap();

        Self {
            buffer,
            bytes,
            tags,
        }
    }

    /// Return the [`IdLimit`] for this store.
    pub(crate) fn id_limit(&self) -> IdLimit {
        IdLimit::new(self.buffer.len() as u32)
    }

    /// Return the number of bytes for each entry.
    pub(crate) fn bytes(&self) -> Bytes {
        self.bytes
    }

    pub(crate) unsafe fn reader_unchecked<'a>(&'a self, guard: epoch::Guard<'a>) -> Reader<'a> {
        Reader {
            buffer: &self.buffer,
            bytes: self.bytes,
            tags: &self.tags,
            _guard: guard,
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

// #[derive(Debug)]
// pub(crate) struct RawReader<'a> {
//     buffer: &'a Buffer,
//     bytes: Bytes,
// }
//
// impl RawReader<'_> {
//     /// Return `true` if the index `i` is in-bounds.
//     #[inline]
//     #[must_use = "this function has no side-effects"]
//     pub(crate) fn is_in_bounds(&self, i: usize) -> bool {
//         i < self.buffer.len()
//     }
//
//     /// Read the data as position `i` if it is guaranteed to be race-free without bounds
//     /// checking.
//     ///
//     /// # Safety
//     ///
//     /// The index `i` must satisfy [`Self::is_in_bounds`].
//     #[inline]
//     pub(crate) unsafe fn read_in_bounds<'a>(
//         &'a self,
//         i: usize,
//         _guard: &'a epoch::Guard<'_>,
//     ) -> &'a [u8] {
//         debug_assert!(self.is_in_bounds(i));
//
//         unsafe {
//             self.buffer
//                 .get_unchecked(i)
//                 .truncate_unchecked(self.bytes)
//                 .as_slice()
//         }
//     }
// }

/// A reader into an [`Intrusive`] store.
#[derive(Debug)]
pub(crate) struct Reader<'a> {
    buffer: &'a Buffer,
    bytes: Bytes,
    tags: &'a tag::ReadOnly,
    _guard: epoch::Guard<'a>,
}

impl<'a> Reader<'a> {
    /// Attempt to read the value at index `i`. This can fail for any of the
    /// following reasons:
    ///
    /// 1. Index `i` is out-of-bounds.
    /// 2. The read cannot be guaranteed to be race-free.
    #[inline]
    pub(crate) fn read(&self, i: usize) -> Option<&[u8]> {
        if self.is_in_bounds(i) {
            // SAFETY: `i` is in-bounds.
            unsafe { self.read_in_bounds(i) }
        } else {
            None
        }
    }

    /// Return `true` if the index `i` is in-bounds.
    #[inline]
    #[must_use = "this function has no side-effects"]
    pub(crate) fn is_in_bounds(&self, i: usize) -> bool {
        i < self.buffer.len()
    }

    /// Return the [`IdLimit`] for this collection.
    #[inline]
    #[must_use = "this function has no side-effects"]
    pub(crate) fn id_limit(&self) -> IdLimit {
        // Like `Intrusive::id_limit`, the numeric cast is safe because by construction,
        // the underlying buffer is limited to `u32::MAX`.
        IdLimit::new(self.buffer.len() as u32)
    }

    /// Read the data as position `i` if it is guaranteed to be race-free without bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// The index `i` must satisfy [`Self::is_in_bounds`].
    #[inline]
    pub(crate) unsafe fn read_in_bounds(&self, i: usize) -> Option<&[u8]> {
        debug_assert!(self.is_in_bounds(i));

        if self.tags.readable(i) {
            Some(unsafe {
                self.buffer
                    .get_unchecked(i)
                    .truncate_unchecked(self.bytes)
                    .as_slice()
            })
        } else {
            None
        }
    }

    // /// Return the raw data slice for index `i` without any race guarantees.
    // ///
    // /// This includes both the data **and** the intrusive tag.
    // ///
    // /// # Safety
    // ///
    // /// The index `i` must satisfy [`Self::is_in_bounds`].
    // ///
    // /// The returned [`RawSlice`] may only be used for prefetching. Callers must never
    // /// materialize it as a proper slice or reference.
    // #[inline]
    // pub(crate) unsafe fn read_raw_unchecked(&self, i: usize) -> RawSlice<'_> {
    //     // SAFETY: Inherited from caller: `i` is in bounds.
    //     unsafe { self.buffer.get_unchecked(i) }.truncate(self.unpadded)
    // }

    // /// Return the number of bytes for each entry.
    // pub(crate) fn bytes(&self) -> Bytes {
    //     self.bytes_plus_tag().unchecked_sub(AtomicTag::SIZE)
    // }

    // /// Return the number of bytes plus the atomic tag.
    // pub(crate) fn bytes_plus_tag(&self) -> Bytes {
    //     self.unpadded
    // }
}

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
