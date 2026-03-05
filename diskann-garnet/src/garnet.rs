/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{ffi::c_void, fmt, mem, ops::Deref, slice};

use diskann::provider::ExecutionContext;
use thiserror::Error;

/// Bitmask for extracting the Term bits from a Context.
/// Must have enough bits to represent all Term variants (max value is 6, needs 3 bits).
pub const TERM_BITMASK: u64 = (1 << 3) - 1;

#[derive(Debug)]
pub enum Term {
    Vector = 0,
    Neighbors = 1,
    Quantized = 2,
    Attributes = 3,
    Metadata = 4,
    IntMap = 5,
    ExtMap = 6,
}

#[derive(Copy, Clone)]
pub struct Context(pub u64);

impl Context {
    pub fn term(&self, kind: Term) -> Self {
        Context(self.0 | (kind as u64 & TERM_BITMASK))
    }
}

impl ExecutionContext for Context {}

pub type ReadCallback =
    unsafe extern "C" fn(u64, u32, *const u8, usize, ReadDataCallback, *mut c_void);
pub type WriteCallback = unsafe extern "C" fn(u64, *const u8, usize, *const u8, usize) -> bool;
pub type DeleteCallback = unsafe extern "C" fn(u64, *const u8, usize) -> bool;
pub type ReadModifyWriteCallback =
    unsafe extern "C" fn(u64, *const u8, usize, usize, RmwDataCallback, *mut c_void) -> bool;
pub type ReadDataCallback = unsafe extern "C" fn(u32, *mut c_void, *const u8, usize);
pub type RmwDataCallback = unsafe extern "C" fn(*mut c_void, *mut u8, usize);

#[derive(Copy, Clone)]
pub struct Callbacks {
    read_callback: ReadCallback,
    write_callback: WriteCallback,
    delete_callback: DeleteCallback,
    rmw_callback: ReadModifyWriteCallback,
}

impl Callbacks {
    pub fn new(
        read_callback: ReadCallback,
        write_callback: WriteCallback,
        delete_callback: DeleteCallback,
        rmw_callback: ReadModifyWriteCallback,
    ) -> Self {
        Self {
            read_callback,
            write_callback,
            delete_callback,
            rmw_callback,
        }
    }

    #[cfg(test)]
    pub fn read_callback(&self) -> ReadCallback {
        self.read_callback
    }

    #[cfg(test)]
    pub fn write_callback(&self) -> WriteCallback {
        self.write_callback
    }

    #[cfg(test)]
    pub fn delete_callback(&self) -> DeleteCallback {
        self.delete_callback
    }

    #[cfg(test)]
    pub fn rmw_callback(&self) -> ReadModifyWriteCallback {
        self.rmw_callback
    }

    pub fn exists_iid(&self, ctx: Context, id: u32) -> bool {
        let key = [4, id];
        // SAFETY: Key bytes are preceded by 4 bytes of space.
        unsafe { self.exists_raw(ctx, bytemuck::bytes_of(&key)) }
    }

    pub fn exists_wid(&self, ctx: Context, key: u64) -> bool {
        // NOTE: the length is bit-shifted so that we have a u32 in the lower half of the u64.
        let mut key = [8 << 32, key];
        let key_bytes = bytemuck::bytes_of_mut(&mut key);
        // SAFETY: Key bytes are preceded by 8 bytes of extra space.
        unsafe { self.exists_raw(ctx, &key_bytes[4..]) }
    }

    pub fn exists_eid(&self, ctx: Context, id: &GarnetId) -> bool {
        // SAFETY: GarnetId ensures there are 4 bytes preceding the key bytes.
        unsafe { self.exists_raw(ctx, id) }
    }

    /// Check for a key's existance in Garnet.
    ///
    /// NOTE: The key bytes must be preceded by 4 valid bytes that Garnet can write into.
    /// This invariant must be checked by the caller.
    unsafe fn exists_raw(&self, ctx: Context, key: &[u8]) -> bool {
        let mut called = false;
        let mut cb = |_, _: &[u8]| {
            called = true;
        };

        unsafe {
            (self.read_callback)(
                ctx.0,
                1,
                key.as_ptr(),
                key.len(),
                make_read_call(&cb),
                &mut cb as *mut _ as *mut c_void,
            );
        }

        called
    }

    #[must_use]
    pub fn read_single_iid<D: bytemuck::Pod>(
        &self,
        ctx: Context,
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
    pub fn read_single_wid<D: bytemuck::Pod>(
        &self,
        ctx: Context,
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
    pub fn read_single_eid<D: bytemuck::Pod>(
        &self,
        ctx: Context,
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
    /// NOTE: The key bytes must be preceded by 4 valid bytes that Garnet can write into.
    /// This invariant must be checked by the caller.
    #[must_use]
    unsafe fn read_single_raw(&self, ctx: Context, key: &[u8], value: &mut [u8]) -> bool {
        let mut found = false;
        let mut cb = |_, data: &[u8]| {
            found = true;
            value.copy_from_slice(data);
        };

        unsafe {
            (self.read_callback)(
                ctx.0,
                1,
                key.as_ptr(),
                key.len(),
                make_read_call(&cb),
                &mut cb as *mut _ as *mut c_void,
            );
        }

        found
    }

    // ids must be passed as 4-byte length prefixed u32s. so [4, I1_u32, 4, I2_u32, ...]
    pub fn read_multi_lpiid<'a, F, T: bytemuck::Pod>(&self, ctx: Context, ids: &[u32], mut f: F)
    where
        F: FnMut(u32, &'a [T]),
    {
        if ids.is_empty() {
            return;
        }

        unsafe {
            (self.read_callback)(
                ctx.0,
                ids.len() as u32 / 2,
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
    pub fn read_varsize_iid<T: bytemuck::Pod>(&self, ctx: Context, id: u32) -> Option<Vec<T>> {
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

        // SAFETY: Key bytes are preceded by 4 bytes of extra space.
        unsafe {
            (self.read_callback)(
                ctx.0,
                1,
                bytemuck::bytes_of(&key).as_ptr(),
                mem::size_of_val(&key),
                make_read_call(&cb),
                &mut cb as *mut _ as *mut c_void,
            );
        }

        result
    }

    #[must_use]
    pub fn write_iid<D: bytemuck::Pod>(&self, ctx: Context, id: u32, value: &[D]) -> bool {
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
    pub fn write_wid<D: bytemuck::Pod>(&self, ctx: Context, key: u64, value: &[D]) -> bool {
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
    pub fn write_eid<D: bytemuck::Pod>(&self, ctx: Context, id: &GarnetId, value: &[D]) -> bool {
        // SAFETY: GarnetId ensures there are 4 bytes preceding the key bytes.
        unsafe { self.write_raw(ctx, id, bytemuck::must_cast_slice::<D, u8>(value)) }
    }

    /// Write a value for a key in Garnet.
    ///
    /// NOTE: The key bytes must be preceded by 4 valid bytes that Garnet can write into.
    /// This invariant must be checked by the caller.
    #[must_use]
    unsafe fn write_raw(&self, ctx: Context, key: &[u8], value: &[u8]) -> bool {
        let value_ptr = value.as_ptr();
        let value_len = value.len();
        unsafe { (self.write_callback)(ctx.0, key.as_ptr(), key.len(), value_ptr, value_len) }
    }

    #[must_use]
    pub fn delete_iid(&self, ctx: Context, id: u32) -> bool {
        let key = [0, id];
        unsafe { (self.delete_callback)(ctx.0, bytemuck::bytes_of(&key[1]).as_ptr(), 4) }
    }

    #[must_use]
    pub fn delete_eid(&self, ctx: Context, id: &GarnetId) -> bool {
        let id: &[u8] = id;
        unsafe { (self.delete_callback)(ctx.0, id.as_ptr(), id.len()) }
    }

    /// Modify a value in Garnet by internal ID.
    ///
    /// The provided function `f` will receive the current value, which it can then modify. If no
    /// value exists, zero-initialized value of length `write_len` will be passed in.
    ///
    /// `f` should not panic.
    #[must_use]
    pub fn rmw_iid<'a, F, T>(&self, ctx: Context, id: u32, write_len: usize, mut f: F) -> bool
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
    pub fn rmw_wid<'a, F, T>(&self, ctx: Context, key: u64, write_len: usize, mut f: F) -> bool
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
    /// The key bytes must be preceded by 4 valid bytes that Garnet can write into.
    /// This invariant must be checked by the caller.
    ///
    /// `f` should not panic.
    #[must_use]
    unsafe fn rmw_raw<'a, F>(&self, ctx: Context, key: &[u8], write_len: usize, mut f: F) -> bool
    where
        F: FnMut(&'a mut [u8]),
    {
        unsafe {
            (self.rmw_callback)(
                ctx.0,
                key.as_ptr(),
                key.len(),
                write_len,
                make_rmw_call(&f),
                &mut f as *mut _ as *mut c_void,
            )
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
pub enum GarnetError {
    #[error("garnet read failed")]
    Read,
    #[error("garnet write failed")]
    Write,
    #[error("garnet delete failed")]
    Delete,
}

/// A variable length byte string used as the vector ID in a Garnet vector set.
///
/// A wrapped type is used because the Garnet callbacks expect some padding bytes it can
/// use to avoid allocation, and this type ensures those bytes exist without interfering
/// with the "real" ID bytes.
///
/// Dereferencing this type will return a slice to the actual ID bytes, without the padding,
/// which makes this interchangeable in most respects with using a raw `Box<[u8]>`.
#[derive(Clone, PartialEq)]
pub struct GarnetId {
    inner: Box<[u8]>,
}

impl GarnetId {
    pub fn as_prefixed_key_bytes(&self) -> &[u8] {
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
        let mut id = Vec::with_capacity(value.len() + 4);
        let len = value.len() as u32;
        id.extend_from_slice(bytemuck::bytes_of(&len));
        id.extend_from_slice(value);
        let inner = id.into();

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
