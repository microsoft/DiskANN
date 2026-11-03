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

/// A four-byte length prefix followed immediately by a native-endian internal ID.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LengthPrefixedId {
    length: u32,
    bytes: [u8; 8],
}

impl Default for LengthPrefixedId {
    fn default() -> Self {
        Self::new(0)
    }
}

impl LengthPrefixedId {
    pub(crate) fn new(id: u64) -> Self {
        Self {
            length: 8,
            bytes: id.to_ne_bytes(),
        }
    }

    pub(crate) fn id(&self) -> u64 {
        u64::from_ne_bytes(self.bytes)
    }

    fn key_bytes(&self) -> &[u8] {
        &bytemuck::bytes_of(self)[4..]
    }
}

/// FSM keys have a four-byte tag and an unpadded native-endian block number.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MetadataBlockKey {
    length: u32,
    tag: u32,
    block: [u8; 8],
}

impl MetadataBlockKey {
    pub(crate) fn new(tag: u32, block: u64) -> Self {
        Self {
            length: 12,
            tag,
            block: block.to_ne_bytes(),
        }
    }

    pub(crate) fn key_bytes(&self) -> &[u8] {
        &bytemuck::bytes_of(self)[4..]
    }
}

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

    pub(crate) fn exists_iid(&self, ctx: &Context, id: u64, length_hint: usize) -> bool {
        let key = LengthPrefixedId::new(id);
        // SAFETY: The record includes the four-byte key length prefix.
        unsafe { self.exists_raw(ctx, bytemuck::bytes_of(&key), length_hint) }
    }

    pub(crate) fn exists_block(
        &self,
        ctx: &Context,
        key: &MetadataBlockKey,
        length_hint: usize,
    ) -> bool {
        // SAFETY: The record includes the four-byte key length prefix.
        unsafe { self.exists_raw(ctx, bytemuck::bytes_of(key), length_hint) }
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
        id: u64,
        value: &mut [D],
    ) -> bool {
        let key = LengthPrefixedId::new(id);
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
    pub(crate) fn read_single_meta<D: bytemuck::Pod>(
        &self,
        ctx: &Context,
        key: u32,
        value: &mut [D],
    ) -> bool {
        let key = [4, key];
        // SAFETY: The key includes its four-byte length prefix.
        unsafe {
            self.read_single_raw(
                ctx,
                bytemuck::bytes_of(&key),
                bytemuck::must_cast_slice_mut(value),
            )
        }
    }

    #[must_use]
    pub(crate) fn read_single_block<D: bytemuck::Pod>(
        &self,
        ctx: &Context,
        key: &MetadataBlockKey,
        value: &mut [D],
    ) -> bool {
        // SAFETY: The record includes the four-byte key length prefix.
        unsafe {
            self.read_single_raw(
                ctx,
                bytemuck::bytes_of(key),
                bytemuck::must_cast_slice_mut(value),
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
            if data.len() == value.len() {
                value.copy_from_slice(data);
                found = true;
            }
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

    /// Read one key per typed length-prefixed internal-ID entry.
    pub(crate) fn read_multi_lpiid<'a, F, T: bytemuck::Pod>(
        &self,
        ctx: &Context,
        ids: &[LengthPrefixedId],
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
                ids.len() as u32,
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
        id: u64,
    ) -> Option<Vec<T>> {
        let key = LengthPrefixedId::new(id);
        // SAFETY: The record includes the four-byte key length prefix.
        unsafe { self.read_varsize_raw(ctx, bytemuck::bytes_of(&key)) }
    }

    #[must_use]
    pub(crate) fn read_varsize_meta<T: bytemuck::Pod>(
        &self,
        ctx: &Context,
        key: u32,
    ) -> Option<Vec<T>> {
        let key = [4, key];
        // SAFETY: The key includes its four-byte length prefix.
        unsafe { self.read_varsize_raw(ctx, bytemuck::bytes_of(&key)) }
    }

    unsafe fn read_varsize_raw<T: bytemuck::Pod>(
        &self,
        ctx: &Context,
        key: &[u8],
    ) -> Option<Vec<T>> {
        let mut result = None;
        let mut cb = |_, data: &[u8]| {
            // Typed readers must respect the stored value's alignment. Use bytes when
            // it is unknown; extended namespaces do not guarantee u64 alignment.
            const {
                assert!(
                    std::mem::align_of::<T>() <= 8,
                    "callback read types cannot require more than eight-byte alignment",
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
                key.as_ptr(),
                key.len(),
                make_read_call(&cb),
                &mut cb as *mut _ as *mut c_void,
            );
        }

        result
    }

    #[must_use]
    pub(crate) fn write_iid<D: bytemuck::Pod>(&self, ctx: &Context, id: u64, value: &[D]) -> bool {
        let key = LengthPrefixedId::new(id);
        // SAFETY: Key bytes are preceded by 4 bytes of extra space.
        unsafe {
            self.write_raw(
                ctx,
                key.key_bytes(),
                bytemuck::must_cast_slice::<D, u8>(value),
            )
        }
    }

    #[must_use]
    pub(crate) fn write_meta<D: bytemuck::Pod>(
        &self,
        ctx: &Context,
        key: u32,
        value: &[D],
    ) -> bool {
        let key = [0, key];
        // SAFETY: The key is preceded by four bytes of scratch space.
        unsafe {
            self.write_raw(
                ctx,
                bytemuck::bytes_of(&key[1]),
                bytemuck::must_cast_slice(value),
            )
        }
    }

    #[must_use]
    pub(crate) fn write_block<D: bytemuck::Pod>(
        &self,
        ctx: &Context,
        key: &MetadataBlockKey,
        value: &[D],
    ) -> bool {
        // SAFETY: The record provides four bytes of scratch space before the key.
        unsafe { self.write_raw(ctx, key.key_bytes(), bytemuck::must_cast_slice(value)) }
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
    pub(crate) fn delete_iid(&self, ctx: &Context, id: u64) -> bool {
        let key = LengthPrefixedId::new(id);
        unsafe { (self.delete_callback)(ctx.inner, key.key_bytes().as_ptr(), 8) }
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
    pub(crate) fn rmw_iid<'a, F>(&self, ctx: &Context, id: u64, write_len: usize, f: F) -> bool
    where
        F: FnMut(&'a mut [u8]),
    {
        let key = LengthPrefixedId::new(id);
        // SAFETY: The record provides four bytes of scratch space before the key.
        unsafe { self.rmw_raw(ctx, key.key_bytes(), write_len, f) }
    }

    /// Atomically modify a value stored under a four-byte metadata tag.
    #[must_use]
    pub(crate) fn rmw_meta<'a, F>(&self, ctx: &Context, key: u32, write_len: usize, f: F) -> bool
    where
        F: FnMut(&'a mut [u8]),
    {
        let key = [0, key];
        // SAFETY: The key is preceded by four bytes of scratch space.
        unsafe { self.rmw_raw(ctx, bytemuck::bytes_of(&key[1]), write_len, f) }
    }

    /// Modify an FSM block in Garnet.
    ///
    /// The provided function `f` will receive the current value, which it can then modify. If no
    /// value exists, zero-initialized value of length `write_len` will be passed in.
    ///
    /// `f` should not panic.
    #[must_use]
    pub(crate) fn rmw_block<'a, F>(
        &self,
        ctx: &Context,
        key: &MetadataBlockKey,
        write_len: usize,
        f: F,
    ) -> bool
    where
        F: FnMut(&'a mut [u8]),
    {
        // SAFETY: The record provides four bytes of scratch space before the key.
        unsafe { self.rmw_raw(ctx, key.key_bytes(), write_len, f) }
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
    // Typed readers must respect the stored value's alignment. Use bytes when it
    // is unknown; extended namespaces do not guarantee u64 alignment.
    const {
        assert!(
            std::mem::align_of::<T>() <= 8,
            "callback read types cannot require more than eight-byte alignment",
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

unsafe extern "C" fn rmw_call<'a, F>(ptr: *mut c_void, data: *mut u8, len: usize)
where
    F: FnMut(&'a mut [u8]),
{
    let data_slice = unsafe { slice::from_raw_parts_mut(data, len) };
    unsafe { (&mut *ptr.cast::<F>())(data_slice) }
}

fn make_rmw_call<'a, F>(_: &F) -> RmwDataCallback
where
    F: FnMut(&'a mut [u8]),
{
    rmw_call::<F>
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
