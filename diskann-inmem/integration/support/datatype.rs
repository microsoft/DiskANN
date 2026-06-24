/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::{
    sampling::medoid::ComputeMedoid,
    views::{Matrix, MatrixView},
};
use half::f16;
use serde::{Deserialize, Serialize};
use thiserror::Error;

//////////////
// DataType //
//////////////

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum DataType {
    F32,
    F16,
    U8,
    I8,
}

impl DataType {
    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::U8 => "u8",
            Self::I8 => "i8",
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

pub(crate) trait AsDataType {
    const DATA_TYPE: DataType;
}

#[derive(Debug, Error)]
#[error("wrong data-type: expected {}, got {}", self.expected, self.got)]
pub(crate) struct WrongDataType {
    expected: DataType,
    got: DataType,
}

impl WrongDataType {
    fn new(expected: DataType, got: DataType) -> Self {
        Self { expected, got }
    }
}

///////////
// Slice //
///////////

#[derive(Debug, Clone, Copy)]
pub(crate) enum Slice<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
    U8(&'a [u8]),
    I8(&'a [i8]),
}

impl<'a> Slice<'a> {
    pub(crate) fn data_type(&self) -> DataType {
        match self {
            Self::F32(_) => DataType::F32,
            Self::F16(_) => DataType::F16,
            Self::U8(_) => DataType::U8,
            Self::I8(_) => DataType::I8,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::F32(s) => s.len(),
            Self::F16(s) => s.len(),
            Self::U8(s) => s.len(),
            Self::I8(s) => s.len(),
        }
    }

    pub(crate) fn try_cast<T>(self) -> Result<&'a [T], WrongDataType>
    where
        T: FromSlice,
    {
        T::from_slice(self)
    }
}

pub(crate) trait FromSlice: Sized {
    fn from_slice(slice: Slice<'_>) -> Result<&[Self], WrongDataType>;
}

//////////////
// SliceMut //
//////////////

#[derive(Debug)]
pub(crate) enum SliceMut<'a> {
    F32(&'a mut [f32]),
    F16(&'a mut [f16]),
    U8(&'a mut [u8]),
    I8(&'a mut [i8]),
}

fn try_map<T, U, F, R>(dst: &mut [T], src: &[U], f: F) -> anyhow::Result<()>
where
    T: std::fmt::Display + AsDataType,
    U: std::fmt::Display + AsDataType + Copy,
    F: Fn(U) -> Result<T, R>,
{
    std::iter::zip(dst.iter_mut(), src.iter()).try_for_each(|(d, s)| {
        let converted = match f(*s) {
            Ok(c) => c,
            Err(e) => anyhow::bail!(
                "could not losslessly convert {} {} to {}",
                U::DATA_TYPE,
                s,
                T::DATA_TYPE,
            ),
        };
        *d = converted;
        Ok(())
    })
}

fn f32_to_f16(x: f32) -> Result<f16, ()> {
    let y = f16::from_f32(x);
    let z = f32::from(y);
    if z != x { Err(()) } else { Ok(y) }
}

fn f32_to_u8(x: f32) -> Result<u8, ()> {
    let y = x as u8;
    let z = f32::from(y);
    if z != x { Err(()) } else { Ok(y) }
}

fn f32_to_i8(x: f32) -> Result<i8, ()> {
    let y = x as i8;
    let z = f32::from(y);
    if z != x { Err(()) } else { Ok(y) }
}

fn f16_to_u8(x: f16) -> Result<u8, ()> {
    f32_to_u8(x.into())
}

fn f16_to_i8(x: f16) -> Result<i8, ()> {
    f32_to_i8(x.into())
}

impl<'a> SliceMut<'a> {
    fn len(&self) -> usize {
        match self {
            Self::F32(s) => s.len(),
            Self::F16(s) => s.len(),
            Self::U8(s) => s.len(),
            Self::I8(s) => s.len(),
        }
    }

    pub(crate) fn convert_lossless(&mut self, rhs: Slice<'_>) -> anyhow::Result<()> {
        if self.len() != rhs.len() {
            anyhow::bail!(
                "lhs len {} must be equal to rhs len {}",
                self.len(),
                rhs.len()
            );
        }

        match (self, rhs) {
            (SliceMut::F32(dst), Slice::F32(src)) => dst.copy_from_slice(src),
            (SliceMut::F32(dst), Slice::F16(src)) => try_map(dst, src, |x| x.try_into())?,
            (SliceMut::F32(dst), Slice::U8(src)) => try_map(dst, src, |x| x.try_into())?,
            (SliceMut::F32(dst), Slice::I8(src)) => try_map(dst, src, |x| x.try_into())?,

            (SliceMut::F16(dst), Slice::F32(src)) => try_map(dst, src, f32_to_f16)?,
            (SliceMut::F16(dst), Slice::F16(src)) => dst.copy_from_slice(src),
            (SliceMut::F16(dst), Slice::U8(src)) => try_map(dst, src, |x| x.try_into())?,
            (SliceMut::F16(dst), Slice::I8(src)) => try_map(dst, src, |x| x.try_into())?,

            (SliceMut::U8(dst), Slice::F32(src)) => try_map(dst, src, f32_to_u8)?,
            (SliceMut::U8(dst), Slice::F16(src)) => try_map(dst, src, f16_to_u8)?,
            (SliceMut::U8(dst), Slice::U8(src)) => dst.copy_from_slice(src),
            (SliceMut::U8(dst), Slice::I8(src)) => try_map(dst, src, |x| x.try_into())?,

            (SliceMut::I8(dst), Slice::F32(src)) => try_map(dst, src, f32_to_i8)?,
            (SliceMut::I8(dst), Slice::F16(src)) => try_map(dst, src, f16_to_i8)?,
            (SliceMut::I8(dst), Slice::U8(src)) => try_map(dst, src, |x| x.try_into())?,
            (SliceMut::I8(dst), Slice::I8(src)) => dst.copy_from_slice(src),
        };

        Ok(())
    }
}

/////////////
// Dataset //
/////////////

#[derive(Debug)]
pub(crate) enum Dataset {
    F32(Matrix<f32>),
    F16(Matrix<f16>),
    U8(Matrix<u8>),
    I8(Matrix<i8>),
}

impl Dataset {
    pub(crate) fn nrows(&self) -> usize {
        self.as_view().nrows()
    }

    pub(crate) fn ncols(&self) -> usize {
        self.as_view().ncols()
    }

    pub(crate) fn row(&self, i: usize) -> Option<Slice<'_>> {
        match self {
            Self::F32(m) => m.get_row(i).map(Slice::from),
            Self::F16(m) => m.get_row(i).map(Slice::from),
            Self::U8(m) => m.get_row(i).map(Slice::from),
            Self::I8(m) => m.get_row(i).map(Slice::from),
        }
    }

    pub(crate) fn as_view(&self) -> DatasetView<'_> {
        match self {
            Self::F32(m) => DatasetView::F32(m.as_view()),
            Self::F16(m) => DatasetView::F16(m.as_view()),
            Self::U8(m) => DatasetView::U8(m.as_view()),
            Self::I8(m) => DatasetView::I8(m.as_view()),
        }
    }

    pub(crate) fn as_slice(&self) -> Slice<'_> {
        match self {
            Self::F32(m) => m.as_slice().into(),
            Self::F16(m) => m.as_slice().into(),
            Self::U8(m) => m.as_slice().into(),
            Self::I8(m) => m.as_slice().into(),
        }
    }

    pub(crate) fn medoid(&self) -> Dataset {
        self.as_view().medoid()
    }
}

/////////////////
// DatasetView //
/////////////////

#[derive(Debug, Clone, Copy)]
pub(crate) enum DatasetView<'a> {
    F32(MatrixView<'a, f32>),
    F16(MatrixView<'a, f16>),
    U8(MatrixView<'a, u8>),
    I8(MatrixView<'a, i8>),
}

impl<'a> DatasetView<'a> {
    pub(crate) fn data_type(&self) -> DataType {
        match self {
            Self::F32(_) => DataType::F32,
            Self::F16(_) => DataType::F16,
            Self::U8(_) => DataType::U8,
            Self::I8(_) => DataType::I8,
        }
    }

    pub(crate) fn nrows(&self) -> usize {
        match self {
            Self::F32(m) => m.nrows(),
            Self::F16(m) => m.nrows(),
            Self::U8(m) => m.nrows(),
            Self::I8(m) => m.nrows(),
        }
    }

    pub(crate) fn ncols(&self) -> usize {
        match self {
            Self::F32(m) => m.ncols(),
            Self::F16(m) => m.ncols(),
            Self::U8(m) => m.ncols(),
            Self::I8(m) => m.ncols(),
        }
    }

    pub(crate) fn row(&self, i: usize) -> Option<Slice<'_>> {
        match self {
            Self::F32(m) => m.get_row(i).map(Slice::from),
            Self::F16(m) => m.get_row(i).map(Slice::from),
            Self::U8(m) => m.get_row(i).map(Slice::from),
            Self::I8(m) => m.get_row(i).map(Slice::from),
        }
    }

    pub(crate) fn medoid(&self) -> Dataset {
        match self {
            Self::F32(v) => Matrix::row_vector(Box::from(f32::compute_medoid(*v))).into(),
            Self::F16(v) => Matrix::row_vector(Box::from(f16::compute_medoid(*v))).into(),
            Self::U8(v) => Matrix::row_vector(Box::from(u8::compute_medoid(*v))).into(),
            Self::I8(v) => Matrix::row_vector(Box::from(i8::compute_medoid(*v))).into(),
        }
    }
}

//------//
// Impl //
//------//

macro_rules! define {
    ($T:ty, $variant:ident) => {
        impl AsDataType for $T {
            const DATA_TYPE: DataType = DataType::$variant;
        }

        impl<'a> From<&'a [$T]> for Slice<'a> {
            fn from(s: &'a [$T]) -> Self {
                Self::$variant(s)
            }
        }

        impl<'a> From<&'a mut [$T]> for SliceMut<'a> {
            fn from(s: &'a mut [$T]) -> Self {
                Self::$variant(s)
            }
        }

        impl FromSlice for $T {
            fn from_slice(slice: Slice<'_>) -> Result<&[Self], WrongDataType> {
                if let Slice::$variant(s) = slice {
                    Ok(s)
                } else {
                    Err(WrongDataType::new(DataType::$variant, slice.data_type()))
                }
            }
        }

        impl From<Matrix<$T>> for Dataset {
            fn from(m: Matrix<$T>) -> Self {
                Self::$variant(m)
            }
        }
    };
}

define!(f32, F32);
define!(f16, F16);
define!(u8, U8);
define!(i8, I8);
