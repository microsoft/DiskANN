/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::Debug;

use bf_tree::BfTree;
use diskann::{graph::AdjacencyList, utils::IntoUsize};
use diskann_quantization::num::PowerOfTwo;
use diskann_utils::future::AsyncFriendly;
use thiserror::Error;

/// A cache capable of holding values with a configurable maximum capacity.
pub struct Cache {
    cache: BfTree,
    bytes: PowerOfTwo,
}

impl Cache {
    /// Construct a new `BFTree`-based cache with the specified capacity.
    pub fn new(bytes: PowerOfTwo) -> Self {
        let mut config = bf_tree::Config::default();

        // N.B.: When `https://github.com/gim-home/Bf-Tree/issues/59` is resolved, set
        // `cache_only(true)`. But until then, keep this as false to prevent hangs.
        config
            .storage_backend(bf_tree::StorageBackend::Memory)
            .cache_only(false)
            .cb_size_byte(bytes.raw());

        Self {
            cache: bf_tree::BfTree::with_config(config, None),
            bytes,
        }
    }

    /// Return the capacity of the cache in bytes.
    pub fn capacity(&self) -> PowerOfTwo {
        self.bytes
    }

    /// Return the utilization of the cache.
    ///
    /// # NOTE
    ///
    /// This is an expensive function and essentially blocks cache operations for its
    /// duration.
    ///
    /// For a timing reference, this can take 73ms to run in release mode with a cache size
    /// of 1 GiB.
    pub fn estimate_utilization(&self) -> Utilization {
        let metrics = self.cache.get_buffer_metrics();

        let capacity = metrics.capacity;

        // The `size_cnt` map in `CircularBufferMetrics` provides a histogram of the size
        // buckets within the circular buffer.
        //
        // We estimate the capacity by summing the product of bucket counts and bucket sizes.
        let used = metrics.size_cnt.iter().map(|(k, v)| k * v).sum();

        Utilization { used, capacity }
    }

    /// Attempt to retrieve the value associated with `key` from the cache, using `cacher`
    /// to process the raw bytes stored in the cache.
    ///
    /// Returns:
    ///
    /// * `Ok(Some(v))`: If the key is present and corresponding value is successfully
    ///   deserialized. The value `v` is the [`Deserialize`] proxy type for `cacher`.
    /// * `Ok(None)` If the key is not present (corresponding to [`CacheReadError::Deleted`]
    ///   and [`CacheReadError::NotFound`]).
    /// * `Err` is the key is invalid or if the key is present but the corresponding value
    ///   could not be successfully deserialized. For example, the corresponding value could
    ///   be corrupted.
    ///
    /// This function return type borrows `cacher` by mutable reference to avoid an allow
    /// implementations that do not allocate even if the key is not present in the cache.
    ///
    /// See also: [`Self::get_raw`], [`Self::get_into`] and [`Self::get_into_raw`].
    pub fn get<'a, T, K, C>(
        &self,
        key: K,
        cacher: &'a mut C,
    ) -> Result<Option<T>, CacheError<CacheReadError, C::Error>>
    where
        K: bytemuck::Pod,
        C: Deserialize<'a, T>,
    {
        match self.get_raw(key, cacher) {
            Ok(container) => Ok(Some(container)),
            Err(CacheError::Access(inner)) => match inner.suppress_not_present() {
                Ok(()) => Ok(None),
                Err(critical) => Err(CacheError::Access(critical)),
            },
            Err(CacheError::Serde(err)) => Err(CacheError::Serde(err)),
        }
    }

    /// Attempt to retrieve the value associated with `key` from the cache, using `cacher`
    /// to process the raw bytes stored in the cache.
    ///
    /// Unlike [`Self::get`], this method does special case the error conditions
    /// [`CacheReadError::Deleted`] and [`CacheReadError::NotFound`] and instead returns any
    /// error encountered.
    ///
    /// See also: [`Self::get`], [`Self::get_into`], and [`Self::get_into_raw`].
    pub fn get_raw<'a, T, K, C>(
        &self,
        key: K,
        cacher: &'a mut C,
    ) -> Result<T, CacheError<CacheReadError, C::Error>>
    where
        K: bytemuck::Pod,
        C: Deserialize<'a, T>,
    {
        cacher.deserialize(|buffer: &mut [u8], spec: ReadSpec| self.read(key, buffer, spec))
    }

    /// Attempt to retrieve the value associated with `key` and populate the contents of
    /// `value` with the result. Returns:
    ///
    /// * `Ok(true)`: If the key is present and corresponding value is successfully deserialized.
    /// * `Ok(false)` If the key is not present (corresponding to [`CacheReadError::Deleted`]
    ///   and [`CacheReadError::NotFound`]).
    /// * `Err` is the key is invalid or if the key is present but the corresponding value
    ///   could not be successfully deserialized. For example, the corresponding value could
    ///   be corrupted.
    ///
    /// See also: [`Self::get`], [`Self::get_into_raw`], and [`Self::get_raw`].
    pub fn get_into<K, C, T>(
        &self,
        key: K,
        cacher: &mut C,
        value: T,
    ) -> Result<Bool, CacheError<CacheReadError, C::Error>>
    where
        K: bytemuck::Pod,
        C: DeserializeInto<T>,
    {
        match self.get_into_raw(key, cacher, value) {
            Ok(()) => Ok(Bool(true)),
            Err(CacheError::Access(inner)) => match inner.suppress_not_present() {
                Ok(()) => Ok(Bool(false)),
                Err(critical) => Err(CacheError::Access(critical)),
            },
            Err(CacheError::Serde(err)) => Err(CacheError::Serde(err)),
        }
    }

    /// Attempt to retrieve the value associated with `key` and populate the contents of
    /// `value` with the result. Argument `cacher` is used to handle any scratch space needed
    /// to interpret the raw bytes stored in the cache.
    ///
    /// Unlike [`Self::get_into`], this method does special case the error conditions
    /// [`CacheReadError::Deleted`] and [`CacheReadError::NotFound`] and instead returns any
    /// error encountered.
    ///
    /// See also: [`Self::get_into`], [`Self::get`], and [`Self::get_raw`].
    pub fn get_into_raw<K, C, T>(
        &self,
        key: K,
        cacher: &mut C,
        value: T,
    ) -> Result<(), CacheError<CacheReadError, C::Error>>
    where
        K: bytemuck::Pod,
        C: DeserializeInto<T>,
    {
        cacher.deserialize_into(value, |buffer: &mut [u8], spec: ReadSpec| {
            self.read(key, buffer, spec)
        })
    }

    /// Attempt to correlate `key` with `value`, returning any error encountered during the
    /// process. Argument `cacher` is used to handle any scratch space needed for the
    /// serialization of `key` into the cache.
    ///
    /// # Notes
    ///
    /// The underlying `BfTree` does not provide an indication on whether `key` was
    /// already present or not, so that information cannot in turn be relayed by this function.
    pub fn set<K, C, T>(
        &self,
        key: K,
        cacher: &mut C,
        value: T,
    ) -> Result<(), CacheError<CacheWriteError, C::Error>>
    where
        K: bytemuck::Pod,
        C: Serialize<T>,
    {
        cacher.serialize(value, |buffer: &[u8]| {
            use bf_tree::LeafInsertResult::{InvalidKV, Success};

            match self.cache.insert(bytemuck::bytes_of(&key), buffer) {
                Success => Ok(()),
                InvalidKV(message) => Err(CacheWriteError { message }),
            }
        })
    }

    /// Attempt to delete `key` from the underlying cache, providing no information
    /// whatsoever regarding the success or failure of this operation.
    ///
    /// TODO: If `BfTree` offered a prefix scan over keys, we could instead use that to
    /// retrieve just a prefix and gather all different entries in the cache.
    ///
    /// Unfortunately, the `BfTree` scan iterator does not provide such functionality.
    pub fn delete<K>(&self, key: K)
    where
        K: bytemuck::Pod,
    {
        self.cache.delete(bytemuck::bytes_of(&key))
    }

    // Internal common-path for `get_into` and `get`.
    fn read<K>(&self, key: K, buffer: &mut [u8], spec: ReadSpec) -> Result<usize, CacheReadError>
    where
        K: bytemuck::Pod,
    {
        use bf_tree::LeafReadResult::{Deleted, Found, InvalidKey, NotFound};

        let len = buffer.len();
        match self.cache.read(bytemuck::bytes_of(&key), buffer) {
            Deleted => Err(CacheReadError::Deleted),
            NotFound => Err(CacheReadError::NotFound),
            InvalidKey => Err(CacheReadError::InvalidKey),
            Found(bytes) => {
                let bytes = bytes.into_usize();
                spec.check(bytes, len)?;
                Ok(bytes)
            }
        }
    }
}

/// The cache utilization.
#[derive(Debug, Clone, Copy)]
pub struct Utilization {
    /// The memory being used in bytes.
    pub used: usize,
    /// The total capacity of the cache in bytes.
    pub capacity: usize,
}

/// A `#[must_use]` wrapper around `bool` - ensuring `Result<Bool, _>` is checked.
#[derive(Debug, PartialEq)]
#[must_use = "this is used to ensure the value in a `Result<Bool, _>` is checked"]
pub struct Bool(bool);

impl Bool {
    /// Return the inner contained boolean value.
    #[must_use = "this is used to ensure the value in a `Result<Bool, _>` is checked"]
    pub fn into_inner(self) -> bool {
        self.0
    }
}

impl From<bool> for Bool {
    fn from(value: bool) -> Self {
        Self(value)
    }
}

impl PartialEq<bool> for Bool {
    fn eq(&self, other: &bool) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Bool> for bool {
    fn eq(&self, other: &Bool) -> bool {
        *self == other.0
    }
}

// NOTE: The below function is used to check the generated code for various caching
// implementations. It is left as a convenience for future inspection but is not strictly
// necessary.
//
// pub fn test_function<'a>(
//     x: &Cache,
//     key: usize,
//     v: &'a mut AdjacencyListCacher<u32>,
// ) -> Result<AdjacencyListRead<'a, u32>, CacheError<CacheReadError, AdjacencyListReadError<u32>>> {
//     x.get_raw(key, v)
// }

impl Debug for Cache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cache")
            .field("cache", &"BfTree")
            .field("bytes", &self.bytes)
            .finish()
    }
}

#[derive(Debug, Clone, Error, PartialEq)]
pub enum CacheReadError {
    #[error("key was deleted")]
    Deleted,
    #[error("key was not found")]
    NotFound,
    #[error("key was invalid")]
    InvalidKey,
    #[error("incorrect buffer size - expected {1}, got {0}")]
    BufferSize(usize, usize),
}

impl CacheReadError {
    pub fn suppress_not_present(self) -> Result<(), Self> {
        match self {
            Self::Deleted | Self::NotFound => Ok(()),
            others => Err(others),
        }
    }
}

#[derive(Debug, Error)]
#[error("key-value insertion failed with error: {message}")]
pub struct CacheWriteError {
    message: String,
}

/// Callback argument for [`Deserialize`] and [`DeserializeInto`] specifying whether reads
/// filling only a portion of the provided buffer are acceptable, or if the implementor
/// requires the number of bytes read to exactly fill the provided buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReadSpec {
    /// Implementors of [`Deserialize`] and [`DeserializeInto`] expect the number of bytes
    /// read from a cache to be exactly equal to their provided buffer capacity.
    Exact,

    /// Implementors of [`Deserialize`] and [`DeserializeInto`] allow the number of bytes
    /// read from a cache to be fewer than the supplied buffer.
    Partial,
}

impl ReadSpec {
    const fn check(&self, bytes_read: usize, expected: usize) -> Result<(), CacheReadError> {
        match self {
            Self::Exact => {
                if bytes_read != expected {
                    Err(CacheReadError::BufferSize(bytes_read, expected))
                } else {
                    Ok(())
                }
            }
            Self::Partial => {
                if bytes_read > expected {
                    Err(CacheReadError::BufferSize(bytes_read, expected))
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// A compound error type for disambiguating between errors yielded while mechanically
/// accessing the underlying [`Cache`] and errors encountered during object serialization
/// and deserialization.
#[derive(Debug, Error)]
pub enum CacheError<T, U> {
    #[error("encountered while accessing the underlying cache")]
    Access(#[source] T),
    #[error("encountered (de)serializing a cached object")]
    Serde(#[source] U),
}

impl<T, U> CacheError<T, U>
where
    T: Debug,
    U: Debug,
{
    #[cfg(test)]
    fn expect_access(self) -> T {
        match self {
            Self::Access(t) => t,
            Self::Serde(u) => panic!("expected access error - instead found serde error {:?}", u),
        }
    }

    #[cfg(test)]
    fn expect_serde(self) -> U {
        match self {
            Self::Access(t) => panic!("expected serde error - instead found access error {:?}", t),
            Self::Serde(u) => u,
        }
    }
}

/// Serialize an element of type `T` into a buffer of bytes, passed as the argument to `f`.
///
/// If `f` returns an error, propagate that error under the [`CacheError:Access`] variant.
pub trait Serialize<T> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn serialize<F, R>(&mut self, element: T, f: F) -> Result<(), CacheError<R, Self::Error>>
    where
        F: FnOnce(&[u8]) -> Result<(), R>;
}

/// Attempt to scoped deserialization of raw bytes into an object of type `T`.
///
/// The implementation will supply the callback `f` with a mutable byte buffer and a
/// [`ReadSpec`]. The callback, in turn, will populate the buffer and if successful, return
/// the number of bytes written into the buffer (starting from the beginning). The argument
/// [`ReadSpec`] can be used to customize the error behavior of `f` according to the following
/// values:
///
/// * [`ReadSpec::Exact`]: The caller expects `f` to populate the entire buffer exactly.
///   Under fills should be treated as an error, returning `Err`.
///
/// * [`ReadSpec::Partial`]: The caller allows the buffer to be under filled.
///
/// See also: [`Serialize`] and [`DeserializeInto`].
pub trait Deserialize<'a, T> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn deserialize<F, R>(&'a mut self, f: F) -> Result<T, CacheError<R, Self::Error>>
    where
        F: FnOnce(&mut [u8], ReadSpec) -> Result<usize, R>;
}

/// Attempt to deserialize an object of type `T` into a preexisting value.
///
/// The implementation will supply the callback `f` with a mutable byte buffer and a
/// [`ReadSpec`]. The callback, in turn, will populate the buffer and if successful, return
/// the number of bytes written into the buffer (starting from the beginning). The argument
/// [`ReadSpec`] can be used to customize the error behavior of `f` according to the following
/// values:
///
/// * [`ReadSpec::Exact`]: The caller expects `f` to populate the entire buffer exactly.
///   Under fills should be treated as an error, returning `Err`.
///
/// * [`ReadSpec::Partial`]: The caller allows the buffer to be under filled.
///
/// See also: [`Serialize`] and [`DeserializeInto`].
pub trait DeserializeInto<T> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn deserialize_into<F, R>(
        &mut self,
        element: T,
        f: F,
    ) -> Result<(), CacheError<R, Self::Error>>
    where
        F: FnOnce(&mut [u8], ReadSpec) -> Result<usize, R>;
}

//----------------//
// Adjacency List //
//----------------//

/// A [`Serialize`]r and [`Deserialize`]r for [`AdjacencyList`]s.
///
/// The serialized representation consists of `std::mem::size_of::<I>()` bytes containing
/// the length `L` of the slice and then `L * std::mem::size_of::<I>()` bytes containing
/// the values themselves.
///
/// This implementation is not portable across machines as it does not perform and endianness
/// checks.
#[derive(Debug)]
pub struct AdjacencyListCacher<I> {
    // Class invariant: `buffer.len()` is at least 1.
    buffer: Box<[I]>,
}

impl<I> AdjacencyListCacher<I>
where
    I: Default,
{
    /// Construct a new `AdjacencyListCacher` configured to adjacency lists of the specified
    /// `max_degree`.
    pub fn new(max_degree: usize) -> Self {
        // LINT: It is exceedingly unlikely that someone will provider `usize::MAX` as an
        // argument and if they did - it's also likely that they will have crashed the
        // program long before now.
        #[allow(clippy::expect_used)]
        Self {
            buffer: (0..max_degree.checked_add(1).expect("integer overflow"))
                .map(|_| I::default())
                .collect(),
        }
    }

    /// Return the maximum degree this cacher can retrieve.
    pub fn max_degree(&self) -> usize {
        self.buffer.len() - 1
    }
}

#[derive(Debug, Clone, Copy, Error)]
pub enum AdjacencyListReadError<I>
where
    I: Debug,
{
    #[error("bytes read ({}) is not a multiple of the element size ({})", .0, std::mem::size_of::<I>())]
    TornRead(usize),
    #[error("header length {0:?} disagrees with bytes read {1}")]
    Corrupted(I, usize),
    #[error("degree {0:?} exceeds max degree {1}")]
    TooLong(I, usize),
}

/// This trait shadows [`diskann::utils::VectorId`], but only pulls in a sub-set of
/// the full functionality required by that trait.
///
/// This is intentionally done to limit the scope of its use here to what is actually needed
/// for the caching implementation.
pub trait CacheableId:
    bytemuck::Pod
    + TryFrom<usize, Error: Debug>
    + IntoUsize
    + Debug
    + Default
    + diskann_vector::contains::ContainsSimd
    + AsyncFriendly
{
}

impl<T> CacheableId for T where
    T: bytemuck::Pod
        + TryFrom<usize, Error: Debug>
        + IntoUsize
        + Debug
        + Default
        + diskann_vector::contains::ContainsSimd
        + AsyncFriendly
{
}

#[derive(Debug, Clone, Copy, Error)]
pub enum AdjacencyListWriteError {
    #[error("length {0} could not be represented")]
    LengthCannotBeRepresented(usize),
    #[error("length {0} exceeds the maximum degree")]
    MaxDegreeExceeded(usize),
}

impl<I> Serialize<&[I]> for AdjacencyListCacher<I>
where
    I: CacheableId,
{
    type Error = AdjacencyListWriteError;

    fn serialize<F, R>(&mut self, element: &[I], f: F) -> Result<(), CacheError<R, Self::Error>>
    where
        F: FnOnce(&[u8]) -> Result<(), R>,
    {
        let len = element.len();
        self.buffer[0] = len.try_into().map_err(|_| {
            CacheError::Serde(AdjacencyListWriteError::LengthCannotBeRepresented(len))
        })?;

        if len > self.max_degree() {
            return Err(CacheError::Serde(
                AdjacencyListWriteError::MaxDegreeExceeded(len),
            ));
        }

        // We've checked that `len <= self.max_degree()`, so all these accesses are
        // in-bounds.
        self.buffer[1..len + 1].copy_from_slice(element);
        f(bytemuck::must_cast_slice::<I, u8>(&self.buffer[..len + 1])).map_err(CacheError::Access)
    }
}

impl<'a, I> DeserializeInto<&'a mut AdjacencyList<I>> for AdjacencyListCacher<I>
where
    I: CacheableId,
{
    type Error = AdjacencyListReadError<I>;

    fn deserialize_into<F, R>(
        &mut self,
        element: &'a mut AdjacencyList<I>,
        f: F,
    ) -> Result<(), CacheError<R, Self::Error>>
    where
        F: FnOnce(&mut [u8], ReadSpec) -> Result<usize, R>,
    {
        let mut guard = element.resize(self.buffer.len());

        // Adjacency lists are allowed to be variable sized, so provide
        // `ReadSpec::Partial` to indicate this.
        let bytes_read = f(
            bytemuck::must_cast_slice_mut::<I, u8>(&mut guard),
            ReadSpec::Partial,
        )
        .map_err(CacheError::Access)?;

        // Check for torn read. We expect at least `std::mem::size_of::<I>()` bytes to be
        // read for the length.
        if bytes_read == 0 || !bytes_read.is_multiple_of(std::mem::size_of::<I>()) {
            Err(CacheError::Serde(AdjacencyListReadError::TornRead(
                bytes_read,
            )))
        } else {
            // Access to position 0 is always valid because we resized to `max_degree + 1`.
            let degree = guard[0].into_usize();
            let required_bytes = (degree + 1) * std::mem::size_of::<I>();
            if bytes_read < required_bytes {
                Err(CacheError::Serde(AdjacencyListReadError::Corrupted(
                    guard[0], bytes_read,
                )))
            } else if degree > self.max_degree() {
                Err(CacheError::Serde(AdjacencyListReadError::TooLong(
                    guard[0],
                    self.max_degree(),
                )))
            } else {
                // Access is within-bounds because `degree <= max_degree` and the buffer
                // has length `max_degree + 1`.
                guard.copy_within(1..degree + 1, 0);
                guard.finish(degree);
                Ok(())
            }
        }
    }
}

/////////////
// Vectors //
/////////////

/// A [`Serialize`]r and [`Deserialize`]r for vectors of a fixed length.
///
/// The serialized representation is simply [`bytemuck::must_cast_slice`] on the underlying
/// data.
#[derive(Debug, Clone)]
pub struct VecCacher<T> {
    buffer: Box<[T]>,
}

impl<T> VecCacher<T>
where
    T: Default,
{
    /// Construct a new `VecCacher` for vectors of the specified length.
    pub fn new(len: usize) -> Self {
        Self {
            buffer: (0..len).map(|_| T::default()).collect(),
        }
    }

    /// Return the vector length serialized by this object.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Return `true` if `self.len() == 0`. Otherwise, return `false`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("expected a vector with length {expected}, instead got {got}")]
pub struct VecWriteError {
    expected: usize,
    got: usize,
}

/// Serialize the data. Returns a `Serde` error `element.len() != self.len()`.
impl<T> Serialize<&[T]> for VecCacher<T>
where
    T: bytemuck::Pod + Default + Debug,
{
    type Error = VecWriteError;

    fn serialize<F, R>(&mut self, element: &[T], f: F) -> Result<(), CacheError<R, Self::Error>>
    where
        F: FnOnce(&[u8]) -> Result<(), R>,
    {
        let got = element.len();
        let expected = self.len();
        if got != expected {
            Err(CacheError::Serde(VecWriteError { expected, got }))
        } else {
            f(bytemuck::must_cast_slice::<T, u8>(element)).map_err(CacheError::Access)
        }
    }
}

impl<'a, T> Deserialize<'a, &'a [T]> for VecCacher<T>
where
    T: bytemuck::Pod + Default + Debug,
{
    /// The serde layer is infallible because we detect improper read amounts in the callback.
    type Error = diskann::error::Infallible;

    fn deserialize<F, R>(&'a mut self, f: F) -> Result<&'a [T], CacheError<R, Self::Error>>
    where
        F: FnOnce(&mut [u8], ReadSpec) -> Result<usize, R>,
    {
        let bytes_read = f(
            bytemuck::must_cast_slice_mut::<T, u8>(&mut self.buffer),
            ReadSpec::Exact,
        )
        .map_err(CacheError::Access)?;

        debug_assert_eq!(
            bytes_read,
            std::mem::size_of::<T>() * self.buffer.len(),
            "`f` should not return OK unless this were true"
        );

        Ok(&*self.buffer)
    }
}

/// An [`Serialize`] and [`Deserialize`] implementation for working with plain-old data.
#[derive(Debug)]
pub struct PodCacher<T> {
    _marker: std::marker::PhantomData<T>,
}

impl<T> PodCacher<T> {
    pub const fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> Default for PodCacher<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Serialize<T> for PodCacher<T>
where
    T: bytemuck::Pod,
{
    type Error = diskann::error::Infallible;

    fn serialize<F, R>(&mut self, element: T, f: F) -> Result<(), CacheError<R, Self::Error>>
    where
        F: FnOnce(&[u8]) -> Result<(), R>,
    {
        f(bytemuck::bytes_of(&element)).map_err(CacheError::Access)
    }
}

impl<T> Deserialize<'_, T> for PodCacher<T>
where
    T: bytemuck::Pod + Default + Debug,
{
    type Error = diskann::error::Infallible;
    fn deserialize<F, R>(&mut self, f: F) -> Result<T, CacheError<R, Self::Error>>
    where
        F: FnOnce(&mut [u8], ReadSpec) -> Result<usize, R>,
    {
        let mut x = T::default();
        let bytes =
            f(bytemuck::bytes_of_mut(&mut x), ReadSpec::Exact).map_err(CacheError::Access)?;
        debug_assert_eq!(bytes, std::mem::size_of::<T>());
        Ok(x)
    }
}

impl<T> DeserializeInto<&mut T> for PodCacher<T>
where
    T: bytemuck::Pod + Default,
{
    type Error = diskann::error::Infallible;

    fn deserialize_into<F, R>(&mut self, x: &mut T, f: F) -> Result<(), CacheError<R, Self::Error>>
    where
        F: FnOnce(&mut [u8], ReadSpec) -> Result<usize, R>,
    {
        let bytes = f(bytemuck::bytes_of_mut(x), ReadSpec::Exact).map_err(CacheError::Access)?;
        debug_assert_eq!(bytes, std::mem::size_of::<T>());
        Ok(())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_CACHE_SIZE: PowerOfTwo = match PowerOfTwo::new(128 * 1024) {
        Ok(v) => v,
        Err(_) => panic!("not a power of two!"),
    };

    //-------//
    // Cache //
    //-------//

    #[test]
    fn test_cache() {
        let cache = Cache::new(TEST_CACHE_SIZE);
        assert_eq!(cache.capacity(), TEST_CACHE_SIZE);

        let debug = format!("{:?}", cache);
        assert_eq!(
            debug,
            format!(
                "Cache {{ cache: \"BfTree\", bytes: {:?} }}",
                TEST_CACHE_SIZE
            )
        );

        let mut cacher = PodCacher::<usize>::new();
        for k in 0..10 {
            cache.set(k, &mut cacher, k).unwrap();
        }

        for k in 0..10 {
            let v: usize = cache.get(k, &mut cacher).unwrap().unwrap();
            assert_eq!(v, k);

            let v: usize = cache.get_raw(k, &mut cacher).unwrap();
            assert_eq!(v, k);

            let mut u = 0usize;
            assert!(cache.get_into(k, &mut cacher, &mut u).unwrap().into_inner());
            assert_eq!(u, k);

            cache.get_into_raw(k, &mut cacher, &mut u).unwrap();
            assert_eq!(u, k);
        }

        // Make sure we can over write values.
        let offset = 10;
        for k in 0..10 {
            cache.set(k, &mut cacher, k + offset).unwrap();
        }

        for k in 0..10 {
            let v: usize = cache.get(k, &mut cacher).unwrap().unwrap();
            assert_eq!(v, k + offset);

            let v: usize = cache.get_raw(k, &mut cacher).unwrap();
            assert_eq!(v, k + offset);

            let mut u = 0usize;
            assert!(cache.get_into(k, &mut cacher, &mut u).unwrap().into_inner());
            assert_eq!(u, k + offset);

            cache.get_into_raw(k, &mut cacher, &mut u).unwrap();
            assert_eq!(u, k + offset);
        }

        // Behavior is sane if we access not-present values.
        for k in 10..20 {
            assert!(cache.get(k, &mut cacher).unwrap().is_none());

            let err = cache.get_raw(k, &mut cacher).unwrap_err().expect_access();
            assert_eq!(err, CacheReadError::NotFound);

            let mut u = 0usize;
            assert!(!cache.get_into(k, &mut cacher, &mut u).unwrap().into_inner());

            let err = cache
                .get_into_raw(k, &mut cacher, &mut u)
                .unwrap_err()
                .expect_access();
            assert_eq!(err, CacheReadError::NotFound);
        }

        // If we delete values - that is also correctly detected.
        for k in 0..10 {
            cache.delete(k);
        }

        // Behavior is sane if we access deleted values.
        for k in 0..10 {
            assert!(cache.get(k, &mut cacher).unwrap().is_none());

            let err = cache.get_raw(k, &mut cacher).unwrap_err().expect_access();
            if err != CacheReadError::Deleted && err != CacheReadError::NotFound {
                panic!(
                    "expected \"deleted\" or \"not found\", instead got {:?}",
                    err,
                );
            }

            let mut u = 0usize;
            assert!(!cache.get_into(k, &mut cacher, &mut u).unwrap().into_inner());

            let err = cache
                .get_into_raw(k, &mut cacher, &mut u)
                .unwrap_err()
                .expect_access();
            if err != CacheReadError::Deleted && err != CacheReadError::NotFound {
                panic!(
                    "expected \"deleted\" or \"not found\", instead got {:?}",
                    err,
                );
            }
        }

        // We can replace with new values.
        let mut cacher_u32 = PodCacher::<u32>::new();
        for k in 0..10 {
            cache
                .set(k, &mut cacher_u32, k.try_into().unwrap())
                .unwrap();
        }

        for k in 0..10 {
            let expected: u32 = k.try_into().unwrap();
            let v: u32 = cache.get(k, &mut cacher_u32).unwrap().unwrap();
            assert_eq!(v, expected);

            let v: u32 = cache.get_raw(k, &mut cacher_u32).unwrap();
            assert_eq!(v, expected);

            let mut u = 0u32;
            assert!(
                cache
                    .get_into(k, &mut cacher_u32, &mut u)
                    .unwrap()
                    .into_inner()
            );
            assert_eq!(u, expected);

            cache.get_into_raw(k, &mut cacher_u32, &mut u).unwrap();
            assert_eq!(u, expected);
        }

        // If we try to access the data as `usize` - we should get a size error.
        for k in 0..10 {
            let err = cache.get(k, &mut cacher).unwrap_err().expect_access();
            assert_eq!(err, CacheReadError::BufferSize(4, 8));

            let err = cache.get_raw(k, &mut cacher).unwrap_err().expect_access();
            assert_eq!(err, CacheReadError::BufferSize(4, 8));

            let mut u = 0usize;
            let err = cache
                .get_into(k, &mut cacher, &mut u)
                .unwrap_err()
                .expect_access();
            assert_eq!(err, CacheReadError::BufferSize(4, 8));

            let err = cache
                .get_into_raw(k, &mut cacher, &mut u)
                .unwrap_err()
                .expect_access();
            assert_eq!(err, CacheReadError::BufferSize(4, 8));
        }
    }

    // This tests some less than ideal behavior of `BfTree` where it panics when the provided
    // buffer is smaller than the value.
    //
    // If this stops panicking, we can introduce better diagnostics when the supplied buffer
    // is too small.
    #[test]
    #[should_panic]
    fn test_cache_panics_buffer_too_small() {
        let cache = Cache::new(TEST_CACHE_SIZE);
        assert_eq!(cache.capacity(), TEST_CACHE_SIZE);

        let mut cacher = PodCacher::<usize>::new();
        cache.set(0usize, &mut cacher, 10usize).unwrap();

        let mut cacher_u32 = PodCacher::<u32>::new();

        // Panics - there's nothing we can do.
        cache.get(0usize, &mut cacher_u32).unwrap().unwrap();
    }

    // These tests ensure that the underlying cache is correctly propagating errors returned
    // by the [`Deserialize`] and [`DeserializeInto`] methods.
    #[test]
    fn test_cache_error() {
        let cache = Cache::new(TEST_CACHE_SIZE);
        let key: u32 = 5;

        // To work around BfTree panicking on a too-small read, we introduce a malformed
        // adjacency list manually by writing a byte-array with a length not equal to a
        // multiple of `std::mem::size_of::<u32>()`, then try to read an adjacency list
        // from that value.
        let mut v = VecCacher::<u8>::new(9);
        let slice: &[u8] = &[0, 0, 0, 2, 3, 4, 5, 6, 7];
        cache.set(key, &mut v, slice).unwrap();

        let mut cacher = AdjacencyListCacher::<u32>::new(3);
        let mut u = AdjacencyList::<u32>::new();
        let err = cache
            .get_into(key, &mut cacher, &mut u)
            .unwrap_err()
            .expect_serde();

        let expected = "bytes read (9) is not a multiple of the element size (4)";
        assert_eq!(err.to_string(), expected);
    }

    // We don't have a way to test this situation from BfTree, so we test it manually.
    #[test]
    fn test_read_spec() {
        assert_eq!(
            (ReadSpec::Partial).check(10, 9).unwrap_err(),
            CacheReadError::BufferSize(10, 9)
        );
    }

    //---------//
    // Cachers //
    //---------//

    #[derive(Debug, Error)]
    #[error("test error: {0}")]
    struct TestError(usize);

    fn make_set(bytes: &mut Vec<u8>) -> impl FnMut(&[u8]) -> Result<(), TestError> + use<'_> {
        |buf| {
            bytes.clear();
            bytes.extend_from_slice(buf);
            Ok(())
        }
    }

    /// A custom callback to provide to `Serialize` implementations that copies out
    /// the contents of `bytes` and validates that the provided `ReadSpec` matches.
    fn make_get(
        bytes: &[u8],
        expected_spec: ReadSpec,
    ) -> impl Fn(&mut [u8], ReadSpec) -> Result<usize, TestError> + use<'_> {
        move |buf, spec| {
            assert_eq!(spec, expected_spec);
            let m = bytes.len().min(buf.len());
            buf[..m].copy_from_slice(&bytes[..m]);
            Ok(bytes.len())
        }
    }

    #[test]
    fn test_adjacency_list_serialization() {
        use std::error::Error;

        let mut bytes = Vec::new();
        let mut cacher = AdjacencyListCacher::<u32>::new(3);

        // Happy path cases: These all fit within the specified max degree.
        let test_cases: [&[u32]; _] = [&[], &[1u32], &[2u32], &[1u32, 10], &[1u32, 10, 4]];
        for case in test_cases {
            cacher.serialize(case, make_set(&mut bytes)).unwrap();
            assert_eq!(
                bytes.len(),
                (case.len() + 1) * std::mem::size_of::<u32>(),
                "case = {:?}",
                case
            );

            // DeserializeInto - happy path.
            let mut a = AdjacencyList::new();
            cacher
                .deserialize_into(&mut a, make_get(&bytes, ReadSpec::Partial))
                .unwrap();
            assert_eq!(&*a, case, "case = {:?}", case);
        }

        //-------------//
        // Error Tests //
        //-------------//

        let expected = "encountered (de)serializing a cached object";

        // Cacher should error when the bytes read is not a multiple of the element size.
        for len in [0, 1, 2, 5, 6, 9] {
            bytes.resize(len, 0u8);
            let source_expected = format!(
                "bytes read ({}) is not a multiple of the element size (4)",
                len
            );

            // DeserializeInto
            let mut a = AdjacencyList::new();
            let err = cacher
                .deserialize_into(&mut a, make_get(&bytes, ReadSpec::Partial))
                .unwrap_err();
            let msg = err.to_string();
            assert_eq!(msg, expected, "len = {}", len);
            assert_eq!(
                err.source().unwrap().to_string(),
                &*source_expected,
                "len = {}",
                len
            );
        }

        // Cacher should error if the length stored in the header does not match the
        // number of bytes read.
        {
            // Resize the buffer to be 12 bytes long, using 4 bytes for the header and
            // two valid values. But we state in the header that there are three elements.
            bytes.resize(12, 0u8);
            bytes[..4].copy_from_slice(bytemuck::bytes_of(&3u32));

            let source_expected = "header length 3 disagrees with bytes read 12";

            // DeserializeInto
            let mut a = AdjacencyList::new();
            let err = cacher
                .deserialize_into(&mut a, make_get(&bytes, ReadSpec::Partial))
                .unwrap_err();

            let msg = err.to_string();
            assert_eq!(msg, expected);
            assert_eq!(err.source().unwrap().to_string(), source_expected);
        }

        // Stored distance is too long. Here, we intentionally have a faulty implementation
        // of the callback so the bytes-read is consistent, but the header length is too long.
        {
            bytes.resize(32, 0u8);
            bytes[..4].copy_from_slice(bytemuck::bytes_of(&100u32));

            let source_expected = "degree 100 exceeds max degree 3";

            let cb = |buf: &mut [u8], spec| -> Result<usize, TestError> {
                assert_eq!(spec, ReadSpec::Partial);
                let m = bytes.len().min(buf.len());
                buf[..m].copy_from_slice(&bytes[..m]);
                Ok(101 * std::mem::size_of::<u32>()) // This is incorrect
            };

            let mut a = AdjacencyList::new();
            let err = cacher.deserialize_into(&mut a, cb).unwrap_err();
            let msg = err.to_string();
            assert_eq!(msg, expected);
            assert_eq!(err.source().unwrap().to_string(), source_expected);
        }

        // Error Propagation
        {
            let err = cacher
                .serialize(&[1, 2, 3], |_| -> Result<(), TestError> {
                    Err(TestError(10))
                })
                .unwrap_err();
            assert!(matches!(err, CacheError::Access(TestError(_))));

            let mut a = AdjacencyList::new();
            let err = cacher
                .deserialize_into(&mut a, |_, _| -> Result<usize, TestError> {
                    Err(TestError(10))
                })
                .unwrap_err();
            assert!(matches!(err, CacheError::Access(TestError(_))));
        }
    }

    #[test]
    fn test_vec_serialization() {
        let mut bytes = Vec::new();
        for len in 0..10 {
            let mut cacher = VecCacher::<f32>::new(len);
            let mut case: Vec<f32> = (0..len).map(|i| i as f32).collect();

            cacher.serialize(&case, make_set(&mut bytes)).unwrap();
            assert_eq!(bytes.len(), len * std::mem::size_of::<f32>());

            // Deserialize - happy path.
            let v: &[f32] = cacher
                .deserialize(make_get(&bytes, ReadSpec::Exact))
                .unwrap();

            assert_eq!(v, &*case, "case = {:?}", case);

            // Serialization should fail if the length is incorrect.
            case.push(0.0);
            bytes.clear();
            let err = cacher.serialize(&case, make_set(&mut bytes)).unwrap_err();
            assert!(matches!(err, CacheError::Serde(VecWriteError { .. })))
        }
    }

    //---------//
    // Metrics //
    //---------//

    #[test]
    fn test_metrics() {
        let capacity = 2usize.pow(20);
        let cache = Cache::new(PowerOfTwo::new(capacity).unwrap());
        assert_eq!(cache.capacity().raw(), capacity);

        let utilization = cache.estimate_utilization();

        assert_eq!(utilization.used, 0);
        assert_eq!(utilization.capacity, capacity);

        let mut v = VecCacher::<f32>::new(128);
        let data = vec![1.0f32; v.len()];
        for i in 0usize..4096 {
            cache.set(i, &mut v, &*data).unwrap();
        }

        let utilization = cache.estimate_utilization();
        assert_eq!(utilization.capacity, capacity);

        let lower_bound: usize = (0.95 * (capacity as f64)).trunc() as usize;
        assert!(
            utilization.used >= lower_bound,
            "got utilization {} of {} - expected a 95% lower bound of {}",
            utilization.used,
            utilization.capacity,
            lower_bound,
        );
    }
}
