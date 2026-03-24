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

use std::io::{Read, Seek, Write};

use diskann_wide::{LoHi, SplitJoin};
use thiserror::Error;

use crate::views::{Matrix, MatrixView};

/// Read a matrix of `T` from the DiskANN binary format (see [module docs](self)).
///
/// Validates that the reader contains enough data before allocating.
pub fn read_bin<T>(reader: &mut (impl Read + Seek)) -> Result<Matrix<T>, ReadBinError>
where
    T: bytemuck::Pod,
{
    let metadata = Metadata::read(reader)?;
    let (npoints, ndims) = (metadata.npoints(), metadata.ndims());
    let type_size = std::mem::size_of::<T>();

    let expected_bytes = npoints
        .checked_mul(ndims)
        .and_then(|n| n.checked_mul(type_size))
        .ok_or(ReadBinError::Overflow {
            npoints: metadata.npoints_u32(),
            ndims: metadata.ndims_u32(),
            type_size,
        })?;

    let data_start = reader.stream_position()?;
    let end = reader.seek(std::io::SeekFrom::End(0))?;
    let available = end - data_start;
    reader.seek(std::io::SeekFrom::Start(data_start))?;

    if available < expected_bytes as u64 {
        return Err(ReadBinError::SizeMismatch {
            expected: expected_bytes as u64,
            available,
            npoints: metadata.npoints_u32(),
            ndims: metadata.ndims_u32(),
            type_size,
        });
    }

    let mut data = Matrix::new(<T as bytemuck::Zeroable>::zeroed(), npoints, ndims);

    reader.read_exact(bytemuck::must_cast_slice_mut::<T, u8>(data.as_mut_slice()))?;
    Ok(data)
}

/// Write a matrix of `T` in the DiskANN binary format (see [module docs](self)).
///
/// Returns the total number of bytes written.
pub fn write_bin<T>(data: MatrixView<'_, T>, writer: &mut impl Write) -> Result<usize, SaveBinError>
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

    /// `npoints * ndims` overflows `usize` (corrupt or malicious header).
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
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::views::Init;

    use super::*;

    #[test]
    fn round_trip_f32() {
        let mut counter = 1.0f32;
        let matrix = Matrix::<f32>::new(
            Init(|| {
                let v = counter;
                counter += 1.0;
                v
            }),
            3,
            4,
        );

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
        let err = read_bin::<f32>(&mut cursor).unwrap_err();

        match err {
            ReadBinError::SizeMismatch {
                expected,
                available,
                npoints,
                ndims,
                type_size,
            } => {
                assert_eq!(expected, 10 * 4 * 4);
                assert_eq!(available, 8);
                assert_eq!(npoints, 10);
                assert_eq!(ndims, 4);
                assert_eq!(type_size, 4);
            }
            other => panic!("expected SizeMismatch, got: {other}"),
        }
    }

    #[test]
    fn read_bin_overflow() {
        // Header with huge values that overflow usize multiplication
        let mut buf = Vec::new();
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        buf.extend_from_slice(&u32::MAX.to_le_bytes());

        let mut cursor = Cursor::new(&buf);
        let err = read_bin::<f32>(&mut cursor).unwrap_err();

        match err {
            ReadBinError::Overflow {
                npoints,
                ndims,
                type_size,
            } => {
                assert_eq!(npoints, u32::MAX);
                assert_eq!(ndims, u32::MAX);
                assert_eq!(type_size, 4);
            }
            other => panic!("expected Overflow, got: {other}"),
        }
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

        assert!(msg.contains("100 points"), "missing npoints: {msg}");
        assert!(msg.contains("32 dims"), "missing ndims: {msg}");
        assert!(msg.contains("12800 bytes"), "missing expected: {msg}");
        assert!(
            msg.contains("0 bytes available"),
            "missing available: {msg}"
        );
    }

    #[test]
    fn metadata_read_write_round_trip() {
        let mut buf = Vec::new();
        let metadata = Metadata::new(200u32, 128u32).unwrap();
        metadata.write(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = Metadata::read(&mut cursor).unwrap();
        assert_eq!(loaded, metadata);
    }
}
