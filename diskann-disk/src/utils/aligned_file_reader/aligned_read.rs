/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::marker::PhantomData;

use diskann::{ANNError, ANNResult};
use diskann_quantization::num::PowerOfTwo;

/// Type-level memory-alignment witness for [`AlignedRead`]. Each implementor is
/// a unit type carrying a single `PowerOfTwo` value.
///
/// Custom readers can define their own marker (e.g. `A4096`) by adding a unit
/// type and an `impl Alignment` with the desired `VALUE`.
pub trait Alignment {
    /// The alignment, in bytes.
    const VALUE: PowerOfTwo;
}

macro_rules! alignment_marker {
    ($name:ident, $value:expr) => {
        #[doc = concat!("Alignment witness for ", stringify!($value), " bytes.")]
        #[derive(Debug, Clone, Copy)]
        pub struct $name;
        impl Alignment for $name {
            const VALUE: PowerOfTwo = $value;
        }
    };
}

alignment_marker!(A1, PowerOfTwo::V1);
alignment_marker!(A512, PowerOfTwo::V512);

/// Disk-IO read request, parameterized by its required memory alignment `A`.
///
/// Three constraints govern a read:
/// 1. Disk offset alignment.
/// 2. Buffer length alignment.
/// 3. Buffer pointer alignment in memory.
///
/// All three are checked against `A::VALUE` at construction time by
/// [`AlignedRead::new`]. A typed `AlignedRead<T, A>` is therefore a witness
/// that the request satisfies its declared alignment, and the file reader's
/// `read` method can rely on it without re-checking.
#[derive(Debug)]
pub struct AlignedRead<'a, T, A: Alignment = A1> {
    offset: u64,
    aligned_buf: &'a mut [T],
    _alignment: PhantomData<A>,
}

impl<'a, T, A: Alignment> AlignedRead<'a, T, A> {
    /// Build an `AlignedRead` after validating that `offset`, the buffer
    /// length (in bytes), and the buffer pointer all satisfy `A::VALUE`.
    pub fn new(offset: u64, aligned_buf: &'a mut [T]) -> ANNResult<Self> {
        Self::assert_is_aligned(aligned_buf.as_ptr() as usize, "buffer pointer")?;
        Self::assert_is_aligned(std::mem::size_of_val(aligned_buf), "buffer length")?;
        Self::assert_is_aligned(offset as usize, "offset")?;
        Ok(Self {
            offset,
            aligned_buf,
            _alignment: PhantomData,
        })
    }

    fn assert_is_aligned(val: usize, kind: &str) -> ANNResult<()> {
        let align = A::VALUE.raw();
        if val.is_multiple_of(align) {
            Ok(())
        } else {
            Err(ANNError::log_disk_io_request_alignment_error(format!(
                "{kind} {val} not aligned to {align}",
            )))
        }
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn aligned_buf(&self) -> &[T] {
        self.aligned_buf
    }

    pub fn aligned_buf_mut(&mut self) -> &mut [T] {
        self.aligned_buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann::ANNErrorKind;
    use diskann_quantization::alloc::{AlignedAllocator, Poly};

    fn aligned_512(len: usize) -> Poly<[u8], AlignedAllocator> {
        Poly::broadcast(0u8, len, AlignedAllocator::A512).unwrap()
    }

    #[test]
    fn aligned_read_carries_offset_and_buffer() {
        let mut buffer = vec![0u8; 512];
        let read = AlignedRead::<u8, A1>::new(512, &mut buffer).unwrap();
        assert_eq!(read.offset(), 512);
        assert_eq!(read.aligned_buf().len(), 512);
    }

    #[test]
    fn aligned_read_buffer_access() {
        let mut buffer = vec![42u8; 512];
        let mut read = AlignedRead::<u8, A1>::new(0, &mut buffer).unwrap();
        assert_eq!(read.aligned_buf()[0], 42);
        read.aligned_buf_mut()[0] = 100;
        assert_eq!(read.aligned_buf()[0], 100);
    }

    #[test]
    fn a512_accepts_fully_aligned_request() {
        let mut buf = aligned_512(512);
        AlignedRead::<u8, A512>::new(0, &mut buf).expect("aligned request should pass");
    }

    #[test]
    fn a1_default_accepts_anything() {
        let mut buffer = vec![0u8; 100];
        AlignedRead::<u8, A1>::new(1, &mut buffer).expect("A1 alignment should accept any request");
    }

    #[test]
    fn rejects_unaligned_buffer_pointer() {
        let mut buf = aligned_512(1024);
        let slice = &mut buf[1..513]; // ptr offset by 1; length 512 ✓; offset 0 ✓
        let err = AlignedRead::<u8, A512>::new(0, slice)
            .expect_err("misaligned buffer pointer should be rejected");
        assert_eq!(err.kind(), ANNErrorKind::DiskIOAlignmentError);
    }

    #[test]
    fn rejects_unaligned_buffer_length() {
        let mut buf = aligned_512(1024);
        let slice = &mut buf[..100]; // ptr ✓; length 100 ✗; offset 0 ✓
        let err = AlignedRead::<u8, A512>::new(0, slice)
            .expect_err("buffer length 100 (not a multiple of 512) should be rejected");
        assert_eq!(err.kind(), ANNErrorKind::DiskIOAlignmentError);
    }

    #[test]
    fn rejects_unaligned_offset() {
        let mut buf = aligned_512(1024);
        let slice = &mut buf[..512]; // ptr ✓; length 512 ✓; offset 1 ✗
        let err = AlignedRead::<u8, A512>::new(1, slice)
            .expect_err("offset 1 (not a multiple of 512) should be rejected");
        assert_eq!(err.kind(), ANNErrorKind::DiskIOAlignmentError);
    }
}
