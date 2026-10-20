/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    ffi::c_void,
    fmt, mem,
    ops::Deref,
    slice,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use diskann::provider::ExecutionContext;
use thiserror::Error;

/// Bitmask for extracting the Term bits from a Context.
/// Must have enough bits to represent all Term variants (max value is 6, needs 3 bits).
pub(crate) const TERM_BITMASK: u64 = (1 << 3) - 1;

#[derive(Debug, Error)]
#[error("Invalid term {0}")]
pub(crate) struct InvalidTerm(u32);

#[derive(Copy, Clone, Debug, strum::VariantArray)]
pub(crate) enum Term {
    Vector = 0,
    Neighbors = 1,
    Quantized = 2,
    Attributes = 3,
    Metadata = 4,
    IntMap = 5,
    ExtMap = 6,
}

impl TryFrom<u32> for Term {
    type Error = InvalidTerm;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Term::Vector),
            1 => Ok(Term::Neighbors),
            2 => Ok(Term::Quantized),
            3 => Ok(Term::Attributes),
            4 => Ok(Term::Metadata),
            5 => Ok(Term::IntMap),
            6 => Ok(Term::ExtMap),
            _ => Err(InvalidTerm(value)),
        }
    }
}

#[derive(Debug, Default)]
struct ContextState {
    quantizer_ready: AtomicBool,
    insert_is_update: AtomicBool,
}

#[derive(Clone, Debug)]
pub(crate) struct Context {
    inner: u64,
    state: Arc<ContextState>,
}

impl Context {
    pub(crate) fn new(inner: u64) -> Self {
        Self {
            inner,
            state: Arc::new(ContextState::default()),
        }
    }

    #[cfg(test)]
    pub(crate) fn get(&self) -> u64 {
        self.inner
    }

    pub(crate) fn term(&self, kind: Term) -> Self {
        let Context { inner, state } = self;
        let inner = *inner | (kind as u64 & TERM_BITMASK);
        let state = state.clone();

        Self { inner, state }
    }

    pub(crate) fn quantizer_ready(&self) -> bool {
        self.state.quantizer_ready.load(Ordering::Acquire)
    }

    pub(crate) fn set_quantizer_ready(&self) {
        self.state.quantizer_ready.store(true, Ordering::Release);
    }

    pub(crate) fn insert_is_update(&self) -> bool {
        self.state.insert_is_update.load(Ordering::Acquire)
    }

    pub(crate) fn set_insert_is_update(&self) {
        self.state.insert_is_update.store(true, Ordering::Release);
    }
}

impl ExecutionContext for Context {}

pub(crate) type ReadCallback =
    unsafe extern "C" fn(u64, u32, u32, *const u8, usize, ReadDataCallback, *mut c_void);
pub(crate) type WriteCallback =
    unsafe extern "C" fn(u64, *const u8, usize, *const u8, usize) -> bool;
pub(crate) type DeleteCallback = unsafe extern "C" fn(u64, *const u8, usize) -> bool;
pub(crate) type ReadModifyWriteCallback =
    unsafe extern "C" fn(u64, *const u8, usize, usize, RmwDataCallback, *mut c_void) -> bool;
pub(crate) type ReadDataCallback = unsafe extern "C" fn(u32, *mut c_void, *const u8, usize);
pub(crate) type RmwDataCallback = unsafe extern "C" fn(*mut c_void, *mut u8, usize);
pub(crate) type FilterCallback = unsafe extern "C" fn(u64, *const u8, usize) -> bool;
pub(crate) type LogCallback = unsafe extern "C" fn(u64, *const u8, usize);

#[derive(Copy, Clone)]
pub(crate) struct Callbacks {
    read_callback: ReadCallback,
    write_callback: WriteCallback,
    delete_callback: DeleteCallback,
    rmw_callback: ReadModifyWriteCallback,
    filter_callback: FilterCallback,
    log_callback: LogCallback,
}

impl Callbacks {
    pub(crate) fn new(
        read_callback: ReadCallback,
        write_callback: WriteCallback,
        delete_callback: DeleteCallback,
        rmw_callback: ReadModifyWriteCallback,
        filter_callback: FilterCallback,
        log_callback: LogCallback,
    ) -> Self {
        Self {
            read_callback,
            write_callback,
            delete_callback,
            rmw_callback,
            filter_callback,
            log_callback,
        }
    }

    #[cfg(test)]
    pub(crate) fn read_callback(&self) -> ReadCallback {
        self.read_callback
    }

    #[cfg(test)]
    pub(crate) fn write_callback(&self) -> WriteCallback {
        self.write_callback
    }

    #[cfg(test)]
    pub(crate) fn delete_callback(&self) -> DeleteCallback {
        self.delete_callback
    }

    #[cfg(test)]
    pub(crate) fn rmw_callback(&self) -> ReadModifyWriteCallback {
        self.rmw_callback
    }

    #[cfg(test)]
    pub(crate) fn filter_callback(&self) -> FilterCallback {
        self.filter_callback
    }

    #[cfg(test)]
    pub(crate) fn log_callback(&self) -> LogCallback {
        self.log_callback
    }

    pub(crate) fn exists_iid(&self, ctx: &Context, id: u32, length_hint: usize) -> bool {
        let key = [4, id];
        // SAFETY: Key bytes are preceded by 4 bytes of space.
        unsafe { self.exists_raw(ctx, bytemuck::bytes_of(&key), length_hint) }
    }

    pub(crate) fn exists_wid(&self, ctx: &Context, key: u64, length_hint: usize) -> bool {
        // NOTE: the length is bit-shifted so that we have a u32 in the lower half of the u64.
        let mut key = [8 << 32, key];
        let key_bytes = bytemuck::bytes_of_mut(&mut key);
        // SAFETY: Key bytes are preceded by 8 bytes of extra space.
        unsafe { self.exists_raw(ctx, &key_bytes[4..], length_hint) }
    }

    pub(crate) fn exists_eid(&self, ctx: &Context, id: &GarnetId, length_hint: usize) -> bool {
        // SAFETY: GarnetId ensures there are 4 bytes preceding the key bytes.
        unsafe { self.exists_raw(ctx, id.as_prefixed_key_bytes(), length_hint) }
    }

    /// Check for a key's existance in Garnet.
    ///
    /// The key must be prefixed by a four byte length.
    unsafe fn exists_raw(&self, ctx: &Context, key: &[u8], length_hint: usize) -> bool {
        let mut called = false;
        let mut cb = |_, _: &[u8]| {
            called = true;
        };

        unsafe {
            (self.read_callback)(
                ctx.inner,
                1,
                length_hint as u32,
                key.as_ptr(),
                key.len(),
                make_read_call(&cb),
                &mut cb as *mut _ as *mut c_void,
            );
        }

        called
    }

    #[must_use]
    pub(crate) fn read_single_iid<D: bytemuck::Pod>(
        &self,
        ctx: &Context,
        id: u32,
        value: &mut [D],
    ) -> bool {
        let key = [4, id];
        // SAFETY: Key bytes are preceded by 4 bytes of extra space.
        unsafe {
            self.read_single_raw(
                ctx,
                bytemuck::bytes_of(&key),
                bytemuck::must_cast_slice_mut::<D, u8>(value),
            )
        }
    }

    #[must_use]
    pub(crate) fn read_single_wid<D: bytemuck::Pod>(
        &self,
        ctx: &Context,
        key: u64,
        value: &mut [D],
    ) -> bool {
        let mut key = [0, key];
        let key_bytes = bytemuck::bytes_of_mut(&mut key);
        key_bytes[4..8].copy_from_slice(bytemuck::bytes_of(&8u32));
        // SAFETY: Key bytes are preceded by 8 bytes of extra space.
        unsafe {
            self.read_single_raw(
                ctx,
                &key_bytes[4..],
                bytemuck::must_cast_slice_mut::<D, u8>(value),
            )
        }
    }

    #[must_use]
    pub(crate) fn read_single_eid<D: bytemuck::Pod>(
        &self,
        ctx: &Context,
        id: &GarnetId,
        value: &mut [D],
    ) -> bool {
        // SAFETY: GarnetId ensures there are 4 bytes preceding the key bytes.
        unsafe {
            self.read_single_raw(
                ctx,
                id.as_prefixed_key_bytes(),
                bytemuck::must_cast_slice_mut::<D, u8>(value),
            )
        }
    }

    /// Read a single key from Garnet.
    ///
    /// The key must be prefixed by a four byte length.
    #[must_use]
    unsafe fn read_single_raw(&self, ctx: &Context, key: &[u8], value: &mut [u8]) -> bool {
        let length_hint = value.len() as u32;
        let mut found = false;
        let mut cb = |_, data: &[u8]| {
            found = true;
            value.copy_from_slice(data);
        };

        unsafe {
            (self.read_callback)(
                ctx.inner,
                1,
                length_hint,
                key.as_ptr(),
                key.len(),
                make_read_call(&cb),
                &mut cb as *mut _ as *mut c_void,
            );
        }

        found
    }

    // ids must be passed as 4-byte length prefixed u32s. so [4, I1_u32, 4, I2_u32, ...]
    pub(crate) fn read_multi_lpiid<'a, F, T: bytemuck::Pod>(
        &self,
        ctx: &Context,
        ids: &[u32],
        length_hint: usize,
        mut f: F,
    ) where
        F: FnMut(u32, &'a [T]),
    {
        if ids.is_empty() {
            return;
        }

        unsafe {
            (self.read_callback)(
                ctx.inner,
                ids.len() as u32 / 2,
                length_hint as u32,
                bytemuck::must_cast_slice::<_, u8>(ids).as_ptr(),
                mem::size_of_val(ids),
                make_read_call(&f),
                &mut f as *mut _ as *mut c_void,
            );
        }
    }

    /// Read a variable size value from Garnet.
    ///
    /// This function allocations inside the read callback since it can't know the size
    /// of the value up front.
    #[must_use]
    pub(crate) fn read_varsize_iid<T: bytemuck::Pod>(
        &self,
        ctx: &Context,
        id: u32,
    ) -> Option<Vec<T>> {
        let key = [4, id];
        let mut result = None;
        let mut cb = |_, data: &[u8]| {
            // NOTE: Values in Garnet are stored aligned at least to 8 bytes. This cast will succeed as long as
            // mem::align_of::<T>() <= 8.
            const {
                assert!(
                    std::mem::align_of::<T>() <= 8,
                    "garnet only guarantees 8-byte alignment",
                )
            }
            result = Some(bytemuck::cast_slice::<u8, T>(data).to_owned());
        };

        // NOTE: We hint the length as 8192 bytes, which will often overestimate. The only varsize
        // things to read are the quant state and the external ID map. Quant state is
        // maximum `117 + 6 * dim` bytes, which is several kilobytes in practice.
        // SAFETY: Key bytes are preceded by 4 bytes of extra space.
        unsafe {
            (self.read_callback)(
                ctx.inner,
                1,
                8192,
                bytemuck::bytes_of(&key).as_ptr(),
                mem::size_of_val(&key),
                make_read_call(&cb),
                &mut cb as *mut _ as *mut c_void,
            );
        }

        result
    }

    #[must_use]
    pub(crate) fn write_iid<D: bytemuck::Pod>(&self, ctx: &Context, id: u32, value: &[D]) -> bool {
        let key = [0, id];
        // SAFETY: Key bytes are preceded by 4 bytes of extra space.
        unsafe {
            self.write_raw(
                ctx,
                bytemuck::bytes_of(&key[1]),
                bytemuck::must_cast_slice::<D, u8>(value),
            )
        }
    }

    #[must_use]
    pub(crate) fn write_wid<D: bytemuck::Pod>(&self, ctx: &Context, key: u64, value: &[D]) -> bool {
        let key = [0, key];
        // SAFETY: Key bytes are preceded by 8 bytes of extra space.
        unsafe {
            self.write_raw(
                ctx,
                bytemuck::bytes_of(&key[1]),
                bytemuck::must_cast_slice::<D, u8>(value),
            )
        }
    }

    #[must_use]
    pub(crate) fn write_eid<D: bytemuck::Pod>(
        &self,
        ctx: &Context,
        id: &GarnetId,
        value: &[D],
    ) -> bool {
        // SAFETY: GarnetId ensures there are 4 bytes preceding the key bytes.
        unsafe { self.write_raw(ctx, id, bytemuck::must_cast_slice::<D, u8>(value)) }
    }

    /// Write a value for a key in Garnet.
    ///
    /// The key is passed without a length prefix.
    #[must_use]
    unsafe fn write_raw(&self, ctx: &Context, key: &[u8], value: &[u8]) -> bool {
        let value_ptr = value.as_ptr();
        let value_len = value.len();
        unsafe { (self.write_callback)(ctx.inner, key.as_ptr(), key.len(), value_ptr, value_len) }
    }

    #[must_use]
    pub(crate) fn delete_iid(&self, ctx: &Context, id: u32) -> bool {
        let key = [0, id];
        unsafe { (self.delete_callback)(ctx.inner, bytemuck::bytes_of(&key[1]).as_ptr(), 4) }
    }

    #[must_use]
    pub(crate) fn delete_eid(&self, ctx: &Context, id: &GarnetId) -> bool {
        let id: &[u8] = id;
        unsafe { (self.delete_callback)(ctx.inner, id.as_ptr(), id.len()) }
    }

    /// Modify a value in Garnet by internal ID.
    ///
    /// The provided function `f` will receive the current value, which it can then modify. If no
    /// value exists, zero-initialized value of length `write_len` will be passed in.
    ///
    /// `f` should not panic.
    #[must_use]
    pub(crate) fn rmw_iid<'a, F, T>(
        &self,
        ctx: &Context,
        id: u32,
        write_len: usize,
        mut f: F,
    ) -> bool
    where
        F: FnMut(&'a mut [T]),
        T: bytemuck::Pod,
    {
        let key = [0, id];
        // SAFETY: Key bytes are preceded by 4 bytes of extra space.
        unsafe {
            self.rmw_raw(ctx, bytemuck::bytes_of(&key[1]), write_len, |d| {
                // NOTE: Values in Garnet are stored aligned at least to 8 bytes. This cast will succeed as long as
                // mem::align_of::<T>() <= 8.
                const {
                    assert!(
                        std::mem::align_of::<T>() <= 8,
                        "garnet only guarantees 8-byte alignment",
                    )
                }
                f(bytemuck::cast_slice_mut::<u8, T>(d))
            })
        }
    }

    /// Modify a value in Garnet by wide ID.
    ///
    /// The provided function `f` will receive the current value, which it can then modify. If no
    /// value exists, zero-initialized value of length `write_len` will be passed in.
    ///
    /// `f` should not panic.
    #[must_use]
    pub(crate) fn rmw_wid<'a, F, T>(
        &self,
        ctx: &Context,
        key: u64,
        write_len: usize,
        mut f: F,
    ) -> bool
    where
        F: FnMut(&'a mut [T]),
        T: bytemuck::Pod,
    {
        let key = [0, key];
        // SAFETY: Key bytes are preceded by 8 bytes of extra space.
        unsafe {
            self.rmw_raw(ctx, bytemuck::bytes_of(&key[1]), write_len, |d| {
                // NOTE: Values in Garnet are stored aligned at least to 8 bytes. This cast will succeed as long as
                // mem::align_of::<T>() <= 8.
                const {
                    assert!(
                        std::mem::align_of::<T>() <= 8,
                        "garnet only guarantees 8-byte alignment",
                    )
                }
                f(bytemuck::cast_slice_mut::<u8, T>(d))
            })
        }
    }

    /// Modify a value in Garnet.
    ///
    /// The provided function `f` will receive the current value, which it can then modify. If no
    /// value exists, zero-initialized value of length `write_len` will be passed in.
    ///
    /// The key is passed without a length prefix.
    ///
    /// `f` should not panic.
    #[must_use]
    unsafe fn rmw_raw<'a, F>(&self, ctx: &Context, key: &[u8], write_len: usize, mut f: F) -> bool
    where
        F: FnMut(&'a mut [u8]),
    {
        unsafe {
            (self.rmw_callback)(
                ctx.inner,
                key.as_ptr(),
                key.len(),
                write_len,
                make_rmw_call(&f),
                &mut f as *mut _ as *mut c_void,
            )
        }
    }

    /// Evaluate the filter callback on an ID.
    #[must_use]
    pub(crate) fn matches_filter(&self, ctx: &Context, data: &[u8]) -> bool {
        unsafe {
            (self.filter_callback)(
                ctx.inner,
                if data.is_empty() {
                    std::ptr::null()
                } else {
                    data.as_ptr()
                },
                data.len(),
            )
        }
    }

    /// Log a message to Garnet.
    ///
    /// The context bits can be set with appropriate `Term` to flag which area the log message concerns.
    pub(crate) fn log(&self, ctx: &Context, msg: &str) {
        unsafe {
            (self.log_callback)(ctx.inner, msg.as_ptr(), msg.len());
        }
    }
}

unsafe extern "C" fn read_call<'a, F, T>(index: u32, ptr: *mut c_void, data: *const u8, len: usize)
where
    F: FnMut(u32, &'a [T]),
    T: bytemuck::Pod,
{
    let data_slice = unsafe { slice::from_raw_parts(data, len) };
    // NOTE: Values in Garnet are stored aligned at least to 8 bytes. This cast will succeed as long as
    // mem::align_of::<T>() <= 8.
    const {
        assert!(
            std::mem::align_of::<T>() <= 8,
            "garnet only guarantees 8-byte alignment",
        )
    }
    let data_slice = bytemuck::cast_slice::<u8, T>(data_slice);
    unsafe { (&mut *ptr.cast::<F>())(index, data_slice) }
}

fn make_read_call<'a, F, T>(_: &F) -> ReadDataCallback
where
    F: FnMut(u32, &'a [T]),
    T: bytemuck::Pod,
{
    read_call::<F, T>
}

unsafe extern "C" fn rmw_call<'a, F, T>(ptr: *mut c_void, data: *mut u8, len: usize)
where
    F: FnMut(&'a mut [T]),
    T: bytemuck::Pod,
{
    let data_slice = unsafe { slice::from_raw_parts_mut(data, len) };
    // NOTE: Values in Garnet are stored aligned at least to 8 bytes. This cast will succeed as long as
    // mem::align_of::<T>() <= 8.
    const {
        assert!(
            std::mem::align_of::<T>() <= 8,
            "garnet only guarantees 8-byte alignment",
        )
    }
    let data_slice = bytemuck::cast_slice_mut::<u8, T>(data_slice);
    unsafe { (&mut *ptr.cast::<F>())(data_slice) }
}

fn make_rmw_call<'a, F, T>(_: &F) -> RmwDataCallback
where
    F: FnMut(&'a mut [T]),
    T: bytemuck::Pod,
{
    rmw_call::<F, T>
}

#[derive(Debug, Error, PartialEq)]
pub(crate) enum GarnetError {
    #[error("garnet read failed")]
    Read,
    #[error("garnet write failed")]
    Write,
    #[error("garnet delete failed")]
    Delete,
}

/// A variable length byte string used as the vector ID in a Garnet vector set.
///
/// This is cheap to clone as it uses `Arc` internally, and prefixes the data with a 4-byte length
/// appropriate for use with the read callbacks.
///
/// Dereferencing returns only the ID bytes.
#[derive(Clone, PartialEq)]
pub(crate) struct GarnetId {
    inner: Arc<[u8]>,
}

impl GarnetId {
    pub(crate) fn as_prefixed_key_bytes(&self) -> &[u8] {
        &self.inner
    }
}

impl fmt::Debug for GarnetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GarnetId({:?})", &self.inner[4..])
    }
}

impl From<&[u8]> for GarnetId {
    fn from(value: &[u8]) -> Self {
        let mut inner = Arc::<[u8]>::new_uninit_slice(value.len() + 4);
        let buffer = Arc::get_mut(&mut inner).unwrap();
        let len = value.len() as u32;
        buffer[..4].write_copy_of_slice(bytemuck::bytes_of(&len));
        buffer[4..].write_copy_of_slice(value);
        // SAFETY: The prefix and ID copies initialize every byte of the allocation.
        let inner = unsafe { inner.assume_init() };

        Self { inner }
    }
}

impl From<Vec<u8>> for GarnetId {
    fn from(value: Vec<u8>) -> Self {
        Self::from(&*value)
    }
}

impl Deref for GarnetId {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.inner[4..]
    }
}
