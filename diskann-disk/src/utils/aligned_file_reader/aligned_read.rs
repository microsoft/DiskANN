/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult};

pub const DISK_IO_ALIGNMENT: usize = 512;

/// Aligned read struct for disk IO, it takes the ownership of the AlignedBoxedSlice and returns the AlignedBoxWithSlice data immutably.
pub struct AlignedRead<'a, T> {
    /// where to read from
    /// offset needs to be aligned with DISK_IO_ALIGNMENT
    offset: u64,

    /// where to read into
    /// aligned_buf and its len need to be aligned with DISK_IO_ALIGNMENT
    aligned_buf: &'a mut [T],
}

impl<'a, T> AlignedRead<'a, T> {
    pub fn new(offset: u64, aligned_buf: &'a mut [T]) -> ANNResult<Self> {
        Self::assert_is_aligned(offset as usize)?;
        Self::assert_is_aligned(std::mem::size_of_val(aligned_buf))?;

        Ok(Self {
            offset,
            aligned_buf,
        })
    }

    fn assert_is_aligned(val: usize) -> ANNResult<()> {
        match val % DISK_IO_ALIGNMENT {
            0 => Ok(()),
            _ => Err(ANNError::log_disk_io_request_alignment_error(format!(
                "The offset or length of AlignedRead request is not {} bytes aligned",
                DISK_IO_ALIGNMENT
            ))),
        }
    }

    /// where to read from
    /// offset needs to be aligned with DISK_IO_ALIGNMENT
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn aligned_buf(&self) -> &[T] {
        self.aligned_buf
    }

    /// where to read into
    /// aligned_buf and its len need to be aligned with DISK_IO_ALIGNMENT
    pub fn aligned_buf_mut(&mut self) -> &mut [T] {
        self.aligned_buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_read_valid() {
        let mut buffer = vec![0u8; 512];
        let aligned_read = AlignedRead::new(0, &mut buffer);
        
        assert!(aligned_read.is_ok());
        let aligned_read = aligned_read.unwrap();
        assert_eq!(aligned_read.offset(), 0);
        assert_eq!(aligned_read.aligned_buf().len(), 512);
    }

    #[test]
    fn test_aligned_read_valid_offset() {
        let mut buffer = vec![0u8; 1024];
        let aligned_read = AlignedRead::new(512, &mut buffer);
        
        assert!(aligned_read.is_ok());
        let aligned_read = aligned_read.unwrap();
        assert_eq!(aligned_read.offset(), 512);
    }

    #[test]
    fn test_aligned_read_invalid_offset() {
        let mut buffer = vec![0u8; 512];
        let aligned_read = AlignedRead::new(100, &mut buffer);
        
        assert!(aligned_read.is_err());
    }

    #[test]
    fn test_aligned_read_invalid_buffer_size() {
        let mut buffer = vec![0u8; 100];
        let aligned_read = AlignedRead::new(0, &mut buffer);
        
        assert!(aligned_read.is_err());
    }

    #[test]
    fn test_aligned_read_buffer_access() {
        let mut buffer = vec![42u8; 512];
        let mut aligned_read = AlignedRead::new(0, &mut buffer).unwrap();
        
        // Test immutable access
        assert_eq!(aligned_read.aligned_buf()[0], 42);
        
        // Test mutable access
        aligned_read.aligned_buf_mut()[0] = 100;
        assert_eq!(aligned_read.aligned_buf()[0], 100);
    }

    #[test]
    fn test_disk_io_alignment_constant() {
        assert_eq!(DISK_IO_ALIGNMENT, 512);
    }
}
