/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Read and write vectors in the DiskANN binary format.
//!
//! The binary format is:
//! - 8-byte header
//!   - `npoints` (u32 LE)
//!   - `ndims` (u32 LE)
//! - Payload: `npoints × ndims` elements of `T`, tightly packed in row-major order
//!
//! The staggered binary allows payloads with a variable number of elements per entry.
//! This format is:
//! - 8-byte header
//!   - `npoints` (u32 LE)
//!   - `total` (u32 LE): The sum of all elements for all `npoints` entries.
//! - Payload:
//!   - `npoints` of u32 LE. These record the size of each element and must sum to `total`.
//!   - `total` elements of type `T` to be chunked according to the previously read sizes.

use std::io::{Read, Seek, Write};

use diskann_wide::{LoHi, SplitJoin};
use thiserror::Error;

use crate::views::rowmajor::{self, Layout, Matrix, MatrixMut};

/// Read a matrix of `T` from the DiskANN binary format (see [module docs](self)).
///
/// Validates that the reader contains enough data before allocating.
pub fn read_bin<T>(reader: &mut (impl Read + Seek)) -> Result<rowmajor::Owned<T>, ReadBinError>
where
    T: bytemuck::Pod,
{
    let metadata = Metadata::read(reader)?;
    let (npoints, ndims) = (metadata.npoints(), metadata.ndims());
    let type_size = std::mem::size_of::<T>();

    let layout = match Layout::<T>::new(npoints, ndims) {
        Ok(layout) => layout,
        Err(_) => {
            return Err(ReadBinError::Overflow {
                npoints: metadata.npoints_u32(),
                ndims: metadata.ndims_u32(),
                type_size,
            })
        }
    };

    let expected_bytes = layout.num_elements() * std::mem::size_of::<T>();
    let available = available_bytes(reader)?;

    if available < expected_bytes as u64 {
        return Err(ReadBinError::SizeMismatch {
            expected: expected_bytes as u64,
            available,
            npoints: metadata.npoints_u32(),
            ndims: metadata.ndims_u32(),
            type_size,
        });
    }

    let mut data =
        rowmajor::Owned::from_element_with_layout(layout, <T as bytemuck::Zeroable>::zeroed());
    reader.read_exact(bytemuck::must_cast_slice_mut::<T, u8>(data.as_mut_slice()))?;
    Ok(data)
}

/// Write a matrix of `T` in the DiskANN binary format (see [module docs](self)).
///
/// Returns the total number of bytes written.
pub fn write_bin<T>(
    data: rowmajor::Ref<'_, T>,
    writer: &mut impl Write,
) -> Result<usize, SaveBinError>
where
    T: bytemuck::Pod,
{
    let metadata =
        Metadata::new(data.nrows(), data.ncols()).map_err(|_| SaveBinError::DimensionOverflow {
            nrows: data.nrows(),
            ncols: data.ncols(),
        })?;
    let bytes = metadata.write(writer)?;
    writer.write_all(bytemuck::must_cast_slice::<T, u8>(data.as_slice()))?;
    Ok(bytes + std::mem::size_of_val(data.as_slice()))
}

/// 8-byte header at the start of a DiskANN binary file: `npoints` and `ndims` as little-endian u32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metadata {
    npoints: u32,
    ndims: u32,
}

impl Metadata {
    /// Construct from any integer types that fit in `u32`.
    pub fn new<T, U>(npoints: T, ndims: U) -> Result<Self, MetadataError<T::Error, U::Error>>
    where
        T: TryInto<u32>,
        U: TryInto<u32>,
    {
        Ok(Self {
            npoints: npoints.try_into().map_err(MetadataError::NumPoints)?,
            ndims: ndims.try_into().map_err(MetadataError::Dim)?,
        })
    }

    /// Number of points as `usize`.
    pub fn npoints(&self) -> usize {
        self.npoints as usize
    }

    /// Number of points as `u32`.
    pub fn npoints_u32(&self) -> u32 {
        self.npoints
    }

    /// Number of dimensions as `usize`.
    pub fn ndims(&self) -> usize {
        self.ndims as usize
    }

    /// Number of dimensions as `u32`.
    pub fn ndims_u32(&self) -> u32 {
        self.ndims
    }

    /// Destructure into (`npoints`, `ndims`) as `usize`.
    pub fn into_dims(&self) -> (usize, usize) {
        (self.npoints(), self.ndims())
    }

    /// Deserialize the 8-byte header from a reader.
    pub fn read<R>(reader: &mut R) -> std::io::Result<Self>
    where
        R: Read,
    {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;

        let LoHi {
            lo: npts_bytes,
            hi: ndims_bytes,
        } = bytes.split();

        let npoints = u32::from_le_bytes(npts_bytes);
        let ndims = u32::from_le_bytes(ndims_bytes);
        Ok(Metadata { npoints, ndims })
    }

    /// Serialize the 8-byte header to a writer. Returns the number of bytes written (always 8).
    pub fn write<W>(&self, writer: &mut W) -> std::io::Result<usize>
    where
        W: Write,
    {
        let bytes: [u8; 8] = LoHi::new(self.npoints.to_le_bytes(), self.ndims.to_le_bytes()).join();
        writer.write_all(&bytes)?;
        Ok(2 * std::mem::size_of::<u32>())
    }
}

#[derive(Debug, Error)]
pub enum MetadataError<T, U> {
    #[error("num points conversion")]
    NumPoints(#[source] T),
    #[error("dim conversion")]
    Dim(#[source] U),
}

/// Error type for [`read_bin`].
#[derive(Debug, Error)]
pub enum ReadBinError {
    /// The reader has fewer bytes remaining than the header declares.
    #[error(
        "binary data too short: header declares {npoints} points × {ndims} dims × {type_size} bytes = \
         {expected} bytes, but only {available} bytes available"
    )]
    SizeMismatch {
        expected: u64,
        available: u64,
        npoints: u32,
        ndims: u32,
        type_size: usize,
    },

    /// The dimensions do not describe a valid allocation (corrupt or malicious header).
    #[error(
        "header dimensions overflow: {npoints} points × {ndims} dims × {type_size} bytes overflows"
    )]
    Overflow {
        npoints: u32,
        ndims: u32,
        type_size: usize,
    },

    /// Underlying IO failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Error type for [`write_bin`].
#[derive(Debug, Error)]
pub enum SaveBinError {
    /// Matrix dimensions exceed `u32::MAX` and cannot be represented in the binary header.
    #[error("dimensions overflow u32: {nrows} rows × {ncols} cols")]
    DimensionOverflow { nrows: usize, ncols: usize },

    /// Underlying IO failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

///////////
// Range //
///////////

/// Read variable-length rows of `T` from a DiskANN staggered binary file.
///
/// The [`Metadata`] header contains the number of rows and the total number of entries
/// (in `ndims`). It is followed by all row lengths as little-endian `u32`s, then all
/// entries of `T` in row order. Trailing bytes, such as groundtruth distances, are left unread.
///
/// # Errors
///
/// Returns an error if the header cannot describe valid allocations, the payload is too
/// short, the row lengths do not sum to the declared total, or reading fails.
pub fn read_staggered_bin<T>(
    reader: &mut (impl Read + Seek),
) -> Result<Vec<Vec<T>>, ReadStaggeredBinError>
where
    T: bytemuck::Pod,
{
    use std::alloc::Layout;

    let metadata = Metadata::read(reader)?;
    let (npoints, total) = (metadata.npoints(), metadata.ndims());

    let overflow = || ReadStaggeredBinError::overflow::<T>(metadata);

    let entry_bytes = Layout::array::<T>(total).map_err(|_| overflow())?.size();
    let length_bytes = Layout::array::<u32>(npoints)
        .map_err(|_| overflow())?
        .size();

    // Verfy that `npoints` doesn't cause us to explode when allocating the `Vec<Vec<_>>`.
    Layout::array::<Vec<T>>(npoints).map_err(|_| overflow())?;

    let expected = entry_bytes.checked_add(length_bytes).ok_or_else(overflow)? as u64;

    let available = available_bytes(reader)?;
    if available < expected {
        return Err(ReadStaggeredBinError::size_mismatch::<T>(
            expected, available, metadata,
        ));
    }

    let mut sizes = vec![0u32; npoints];
    reader.read_exact(bytemuck::must_cast_slice_mut::<u32, u8>(&mut sizes))?;

    // Validate that all the interior lengths are valid.
    let mut remaining = total;
    for (point, size) in sizes.iter_mut().enumerate() {
        // Perform byte correction.
        *size = u32::from_le(*size);

        remaining =
            remaining
                .checked_sub(*size as usize)
                .ok_or(ReadStaggeredBinError::InvalidRowSize {
                    point,
                    size: *size,
                    remaining,
                })?;
    }

    if remaining != 0 {
        return Err(ReadStaggeredBinError::EntryCountMismatch {
            expected: total,
            actual: total - remaining,
        });
    }

    // Interior lengths are valid - read the data!
    let values = sizes
        .into_iter()
        .map(|size| -> std::io::Result<Vec<T>> {
            let mut row = vec![<T as bytemuck::Zeroable>::zeroed(); size as usize];
            reader.read_exact(bytemuck::must_cast_slice_mut::<T, u8>(row.as_mut_slice()))?;
            Ok(row)
        })
        .collect::<std::io::Result<Vec<Vec<T>>>>()?;

    Ok(values)
}

/// Error type for [`read_staggered_bin`].
#[derive(Debug, Error)]
pub enum ReadStaggeredBinError {
    /// The reader has fewer bytes remaining than the header declares.
    #[error(
        "staggered binary data too short: header declares {npoints} row lengths and \
         {total} entries of {type_size} bytes = {expected} bytes, but only {available} bytes available"
    )]
    SizeMismatch {
        expected: u64,
        available: u64,
        npoints: u32,
        total: u32,
        type_size: usize,
    },

    /// The header does not describe valid allocations or its byte count overflows.
    #[error("staggered header overflows: {npoints} rows and {total} entries of {type_size} bytes")]
    Overflow {
        npoints: u32,
        total: u32,
        type_size: usize,
    },

    /// A row requests more entries than remain in the declared total.
    #[error("row {point} declares {size} entries, but only {remaining} entries remain")]
    InvalidRowSize {
        point: usize,
        size: u32,
        remaining: usize,
    },

    /// The row lengths sum to fewer entries than the header declares.
    #[error("header declares {expected} entries, but row lengths sum to {actual}")]
    EntryCountMismatch { expected: usize, actual: usize },

    /// Underlying IO failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl ReadStaggeredBinError {
    fn size_mismatch<T>(expected: u64, available: u64, metadata: Metadata) -> Self {
        Self::SizeMismatch {
            expected,
            available,
            npoints: metadata.npoints_u32(),
            total: metadata.ndims_u32(),
            type_size: std::mem::size_of::<T>(),
        }
    }

    fn overflow<T>(metadata: Metadata) -> Self {
        Self::Overflow {
            npoints: metadata.npoints_u32(),
            total: metadata.ndims_u32(),
            type_size: std::mem::size_of::<T>(),
        }
    }
}

//---------//
// helpers //
//---------//

fn available_bytes(reader: &mut impl Seek) -> std::io::Result<u64> {
    let start = reader.stream_position()?;
    let end = reader.seek(std::io::SeekFrom::End(0))?;
    reader.seek(std::io::SeekFrom::Start(start))?;
    end.checked_sub(start).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("reader ends at byte {end}, before the current position {start}"),
        )
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::assert_contains;

    use super::*;

    #[test]
    fn metadata_read_write_round_trip() {
        let mut buf = Vec::new();
        let metadata = Metadata::new(200u32, 128u32).unwrap();
        metadata.write(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = Metadata::read(&mut cursor).unwrap();
        assert_eq!(loaded, metadata);
    }

    /// This test a pathological case where seeking to the end manages to change the size
    /// of the reader so the returned `end` point is less than where we started. This could
    /// be due to a buggy implementation or changing file size etc.
    ///
    /// Verify that we detect this and don't underflow in out bytes computation.
    #[test]
    fn available_bytes_restores_position_when_reader_shrinks() {
        struct ShrinkingReader(Cursor<Vec<u8>>);

        impl Seek for ShrinkingReader {
            fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
                if position == std::io::SeekFrom::End(0) {
                    self.0.get_mut().truncate(4);
                }
                self.0.seek(position)
            }
        }

        let mut reader = ShrinkingReader(Cursor::new(vec![0; 12]));
        reader.0.set_position(8);

        let error = available_bytes(&mut reader).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
        assert_contains!(error.to_string(), "reader ends at byte 4");
        assert_contains!(error.to_string(), "before the current position 8");
        assert_eq!(reader.0.position(), 8);
    }

    #[test]
    fn round_trip_f32() {
        let mut counter = 1.0f32;
        let matrix = rowmajor::Owned::<f32>::from_fn(3, 4, |_| {
            let v = counter;
            counter += 1.0;
            v
        });

        assert_eq!(
            matrix.as_slice(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );

        let mut buf = Vec::new();
        let written = write_bin(matrix.as_view(), &mut buf).unwrap();
        assert_eq!(written, 8 + 3 * 4 * 4);

        let mut cursor = Cursor::new(&buf);
        let loaded = read_bin::<f32>(&mut cursor).unwrap();
        assert_eq!(loaded.nrows(), 3);
        assert_eq!(loaded.ncols(), 4);
        assert_eq!(loaded.as_slice(), matrix.as_slice());
    }

    #[test]
    fn read_bin_size_mismatch() {
        // Header says 10 points × 4 dims of f32, but only provide 8 bytes of payload
        let mut buf = Vec::new();
        let metadata = Metadata::new(10u32, 4u32).unwrap();
        metadata.write(&mut buf).unwrap();
        buf.extend_from_slice(&[0u8; 8]);

        let mut cursor = Cursor::new(&buf);
        let msg = read_bin::<f32>(&mut cursor).unwrap_err().to_string();

        assert_contains!(msg, "binary data too short");
        assert_contains!(msg, "10 points");
        assert_contains!(msg, "4 dims");
        assert_contains!(msg, "4 bytes = 160 bytes");
        assert_contains!(msg, "only 8 bytes available");
    }

    #[test]
    fn read_bin_overflow() {
        // Header with huge values that overflow usize multiplication
        let mut buf = Vec::new();
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        buf.extend_from_slice(&u32::MAX.to_le_bytes());

        let mut cursor = Cursor::new(&buf);
        let msg = read_bin::<f32>(&mut cursor).unwrap_err().to_string();

        assert_contains!(msg, "header dimensions overflow");
        assert_contains!(msg, "4294967295 points");
        assert_contains!(msg, "4294967295 dims");
        assert_contains!(msg, "4 bytes overflows");
    }

    #[test]
    fn read_bin_error_message_is_informative() {
        let mut buf = Vec::new();
        let metadata = Metadata::new(100u32, 32u32).unwrap();
        metadata.write(&mut buf).unwrap();
        // no payload

        let mut cursor = Cursor::new(&buf);
        let err = read_bin::<f32>(&mut cursor).unwrap_err();
        let msg = err.to_string();

        assert_contains!(msg, "100 points", "missing npoints");
        assert_contains!(msg, "32 dims", "missing ndims");
        assert_contains!(msg, "12800 bytes", "missing expected");
        assert_contains!(msg, "0 bytes available", "missing available");
    }

    //----------------------//
    // `read_staggered_bin` //
    //----------------------//

    fn as_staggered<'a, 'b, T>(x: &'a [&'b [T]]) -> &'a [&'b [T]] {
        x
    }

    fn staggered_data<T: bytemuck::Pod>(rows: &[&[T]]) -> Cursor<Vec<u8>> {
        let sizes = rows
            .iter()
            .map(|row| u32::try_from(row.len()).unwrap())
            .collect::<Vec<_>>();

        let entries = rows.iter().flat_map(|v| *v).copied().collect::<Vec<_>>();
        staggered_data_with_lengths(&sizes, &entries)
    }

    fn staggered_data_with_lengths<T: bytemuck::Pod>(
        sizes: &[u32],
        entries: &[T],
    ) -> Cursor<Vec<u8>> {
        let mut buf = Vec::new();
        Metadata::new(sizes.len(), entries.len())
            .unwrap()
            .write(&mut buf)
            .unwrap();
        for size in sizes {
            buf.extend_from_slice(&size.to_le_bytes());
        }
        buf.extend_from_slice(bytemuck::must_cast_slice::<T, u8>(entries));
        Cursor::new(buf)
    }

    #[test]
    fn read_staggered_bin_rows_and_trailing_data() {
        let rows = as_staggered(&[&[10u16, 20], &[], &[30], &[40, 50]]);

        let data = staggered_data(rows);
        let mut buf = vec![0u8; 3];
        buf.extend_from_slice(data.get_ref());
        let data_end = buf.len() as u64;
        buf.extend_from_slice(&[99u8; 20]);
        let mut cursor = Cursor::new(buf);
        cursor.set_position(3);

        assert_eq!(read_staggered_bin::<u16>(&mut cursor).unwrap(), rows);
        assert_eq!(cursor.position(), data_end);
    }

    #[test]
    fn read_staggered_bin_empty_rows() {
        for rows in [as_staggered(&[]), as_staggered(&[&[], &[], &[]])] {
            let mut cursor = staggered_data(rows);
            assert_eq!(read_staggered_bin::<u32>(&mut cursor).unwrap(), rows);
            assert_eq!(cursor.position(), cursor.get_ref().len() as u64);
        }
    }

    #[test]
    fn read_staggered_bin_zero_sized_entries() {
        let rows = as_staggered(&[&[()], &[], &[(), ()]]);
        let mut cursor = staggered_data(rows);
        assert_eq!(read_staggered_bin::<()>(&mut cursor).unwrap(), rows);
    }

    #[test]
    fn read_staggered_bin_size_mismatch() {
        for available in [0, 7, 8, 19] {
            let mut cursor = staggered_data(&[&[10u32], &[20, 30]]);
            cursor.get_mut().truncate(8 + available);
            let msg = read_staggered_bin::<u32>(&mut cursor)
                .unwrap_err()
                .to_string();

            assert_contains!(msg, "staggered binary data too short");
            assert_contains!(msg, "2 row lengths");
            assert_contains!(msg, "3 entries of 4 bytes = 20 bytes");
            assert_contains!(msg, format!("only {available} bytes available"));
        }
    }

    #[test]
    fn read_staggered_bin_invalid_row_sizes() {
        for (sizes, point, size, remaining) in [
            ([4, 0], 0, 4, 3),
            ([2, 2], 1, 2, 1),
            ([u32::MAX, 0], 0, u32::MAX, 3),
        ] {
            let mut cursor = staggered_data_with_lengths(&sizes, &[10u32, 20, 30]);
            let msg = read_staggered_bin::<u32>(&mut cursor)
                .unwrap_err()
                .to_string();

            assert_contains!(msg, format!("row {point} declares {size} entries"));
            assert_contains!(msg, format!("only {remaining} entries remain"));
        }
    }

    #[test]
    fn read_staggered_bin_entry_count_mismatch() {
        for (sizes, actual) in [(&[][..], 0), (&[0, 0][..], 0), (&[1, 1][..], 2)] {
            let mut cursor = staggered_data_with_lengths(sizes, &[10u32, 20, 30]);
            let msg = read_staggered_bin::<u32>(&mut cursor)
                .unwrap_err()
                .to_string();

            assert_contains!(msg, "header declares 3 entries");
            assert_contains!(msg, format!("row lengths sum to {actual}"));
        }
    }

    #[test]
    fn read_staggered_bin_truncated_header() {
        for len in 0..8 {
            let mut cursor = Cursor::new(vec![0u8; len]);
            let msg = read_staggered_bin::<u32>(&mut cursor)
                .unwrap_err()
                .to_string();
            assert_contains!(msg, "failed to fill whole buffer", "header length: {len}");
        }
    }
}
