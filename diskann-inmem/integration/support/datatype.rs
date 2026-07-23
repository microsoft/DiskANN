/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::{
    sampling::medoid::ComputeMedoid,
    views::{Matrix, MatrixView, MutMatrixView},
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

fn map<T, U, F>(dst: &mut [T], src: &[U], f: F)
where
    T: std::fmt::Display + AsDataType,
    U: std::fmt::Display + AsDataType + Copy,
    F: Fn(U) -> T,
{
    std::iter::zip(dst.iter_mut(), src.iter()).for_each(|(d, s)| {
        *d = f(*s);
    })
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
            Err(_) => anyhow::bail!(
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
            (SliceMut::F32(dst), Slice::F16(src)) => map(dst, src, |x| x.into()),
            (SliceMut::F32(dst), Slice::U8(src)) => map(dst, src, |x| x.into()),
            (SliceMut::F32(dst), Slice::I8(src)) => map(dst, src, |x| x.into()),

            (SliceMut::F16(dst), Slice::F32(src)) => try_map(dst, src, f32_to_f16)?,
            (SliceMut::F16(dst), Slice::F16(src)) => dst.copy_from_slice(src),
            (SliceMut::F16(dst), Slice::U8(src)) => map(dst, src, |x| x.into()),
            (SliceMut::F16(dst), Slice::I8(src)) => map(dst, src, |x| x.into()),

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

    pub(crate) fn preprocess(&mut self, op: &Preprocess) {
        match self {
            Self::F32(m) => op.apply(m.as_mut_view()),
            Self::F16(m) => op.apply(m.as_mut_view()),
            Self::U8(m) => op.apply(m.as_mut_view()),
            Self::I8(m) => op.apply(m.as_mut_view()),
        }
    }
}

/// Preprocess steps for [`Dataset`]s.
///
/// These exist so we can coax `u8` data into a form compatible for testing `i8` data.
#[derive(Debug)]
pub(crate) enum Preprocess {
    // Divide each component by 2.
    Halve,
    // Perform a `floor` operation on the each component.
    Floor,
}

trait Apply<T> {
    fn apply(&self, m: MutMatrixView<'_, T>);
}

impl Apply<f32> for Preprocess {
    fn apply(&self, mut m: MutMatrixView<'_, f32>) {
        match self {
            Self::Halve => m.as_mut_slice().iter_mut().for_each(|v| *v *= 0.5),
            Self::Floor => m.as_mut_slice().iter_mut().for_each(|v| *v = v.floor()),
        }
    }
}

impl Apply<f16> for Preprocess {
    fn apply(&self, mut m: MutMatrixView<'_, f16>) {
        match self {
            Self::Halve => m.as_mut_slice().iter_mut().for_each(|v| {
                *v = f16::from_f32(f32::from(*v) * 0.5);
            }),
            Self::Floor => m.as_mut_slice().iter_mut().for_each(|v| {
                *v = f16::from_f32(f32::from(*v).floor());
            }),
        }
    }
}

impl Apply<u8> for Preprocess {
    fn apply(&self, mut m: MutMatrixView<'_, u8>) {
        match self {
            Self::Halve => m.as_mut_slice().iter_mut().for_each(|v| *v /= 2),
            Self::Floor => {}
        }
    }
}

impl Apply<i8> for Preprocess {
    fn apply(&self, mut m: MutMatrixView<'_, i8>) {
        match self {
            Self::Halve => m.as_mut_slice().iter_mut().for_each(|v| *v /= 2),
            Self::Floor => {}
        }
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

    pub(crate) fn iter(&self) -> Iter<'_> {
        Iter::new(self)
    }
}

pub(crate) struct Iter<'a> {
    view: &'a DatasetView<'a>,
    row: usize,
}

impl<'a> Iter<'a> {
    fn new(view: &'a DatasetView<'a>) -> Self {
        Self { view, row: 0 }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = Slice<'a>;

    fn next(&mut self) -> Option<Slice<'a>> {
        let r = self.view.row(self.row)?;
        self.row += 1;
        Some(r)
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix<T>(data: &[T], nrows: usize, ncols: usize) -> Matrix<T>
    where
        T: Copy,
    {
        Matrix::try_from(Box::from(data), nrows, ncols).unwrap()
    }

    //----------//
    // DataType //
    //----------//

    #[test]
    fn datatype_display() {
        assert_eq!(DataType::F32.to_string(), "f32");
        assert_eq!(DataType::F16.to_string(), "f16");
        assert_eq!(DataType::U8.to_string(), "u8");
        assert_eq!(DataType::I8.to_string(), "i8");
    }

    //-------//
    // Slice //
    //-------//

    #[test]
    fn slice_data_type_and_len() {
        let f: &[f32] = &[1.0, 2.0, 3.0];
        let s = Slice::from(f);
        assert_eq!(s.data_type(), DataType::F32);
        assert_eq!(s.len(), 3);

        let u: &[u8] = &[1, 2];
        assert_eq!(Slice::from(u).data_type(), DataType::U8);
        assert_eq!(Slice::from(u).len(), 2);
    }

    #[test]
    fn slice_try_cast_success() {
        let f: &[f32] = &[1.0, 2.0];
        let s = Slice::from(f);
        let out: &[f32] = s.try_cast().unwrap();
        assert_eq!(out, &[1.0, 2.0]);
    }

    #[test]
    fn slice_try_cast_wrong_type() {
        let f: &[f32] = &[1.0, 2.0];
        let s = Slice::from(f);
        let err = s.try_cast::<u8>().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("u8"), "msg: {msg}");
        assert!(msg.contains("f32"), "msg: {msg}");
    }

    //----------//
    // SliceMut //
    //----------//

    #[test]
    fn convert_lossless_same_type() {
        let mut dst = [0.0f32; 3];
        let src: &[f32] = &[1.0, 2.0, 3.0];
        SliceMut::from(dst.as_mut_slice())
            .convert_lossless(Slice::from(src))
            .unwrap();
        assert_eq!(dst, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn convert_lossless_widening() {
        // u8 -> f32 is always lossless.
        let mut dst = [0.0f32; 3];
        let src: &[u8] = &[1, 2, 250];
        SliceMut::from(dst.as_mut_slice())
            .convert_lossless(Slice::from(src))
            .unwrap();
        assert_eq!(dst, [1.0, 2.0, 250.0]);

        // i8 -> f16 is always lossless.
        let mut dst = [f16::ZERO; 2];
        let src: &[i8] = &[-5, 7];
        SliceMut::from(dst.as_mut_slice())
            .convert_lossless(Slice::from(src))
            .unwrap();
        assert_eq!(dst, [f16::from_f32(-5.0), f16::from_f32(7.0)]);
    }

    #[test]
    fn convert_lossless_narrowing_exact() {
        // Whole-valued, in-range f32 -> u8 is lossless.
        let mut dst = [0u8; 3];
        let src: &[f32] = &[0.0, 12.0, 255.0];
        SliceMut::from(dst.as_mut_slice())
            .convert_lossless(Slice::from(src))
            .unwrap();
        assert_eq!(dst, [0, 12, 255]);
    }

    #[test]
    fn convert_lossless_narrowing_fraction_errors() {
        let mut dst = [0u8; 2];
        let src: &[f32] = &[1.0, 0.5];
        let err = SliceMut::from(dst.as_mut_slice())
            .convert_lossless(Slice::from(src))
            .unwrap_err();
        assert!(err.to_string().contains("losslessly"), "{err}");
    }

    #[test]
    fn convert_lossless_signedness_errors() {
        // Negative i8 cannot fit into u8.
        let mut dst = [0u8; 2];
        let src: &[i8] = &[5, -1];
        assert!(
            SliceMut::from(dst.as_mut_slice())
                .convert_lossless(Slice::from(src))
                .is_err()
        );

        // u8 > 127 cannot fit into i8.
        let mut dst = [0i8; 2];
        let src: &[u8] = &[10, 200];
        assert!(
            SliceMut::from(dst.as_mut_slice())
                .convert_lossless(Slice::from(src))
                .is_err()
        );
    }

    #[test]
    fn convert_lossless_length_mismatch_errors() {
        let mut dst = [0.0f32; 2];
        let src: &[f32] = &[1.0, 2.0, 3.0];
        let err = SliceMut::from(dst.as_mut_slice())
            .convert_lossless(Slice::from(src))
            .unwrap_err();
        assert!(err.to_string().contains("len"), "{err}");
    }

    //---------//
    // Dataset //
    //---------//

    #[test]
    fn dataset_shape_and_views() {
        let ds: Dataset = matrix(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).into();
        assert_eq!(ds.nrows(), 2);
        assert_eq!(ds.ncols(), 3);
        assert_eq!(ds.as_view().data_type(), DataType::F32);
        assert_eq!(ds.as_slice().data_type(), DataType::F32);
        assert_eq!(ds.as_slice().len(), 6);
    }

    #[test]
    fn dataset_medoid_shape() {
        let ds: Dataset = matrix(&[1.0f32, 2.0, 3.0, 4.0], 2, 2).into();
        let medoid = ds.medoid();
        assert_eq!(medoid.nrows(), 1);
        assert_eq!(medoid.ncols(), 2);
    }

    #[test]
    fn dataset_preprocess_halve() {
        let mut ds: Dataset = matrix(&[2.0f32, 4.0, 6.0, 8.0], 2, 2).into();
        ds.preprocess(&Preprocess::Halve);
        let slice: &[f32] = ds.as_slice().try_cast().unwrap();
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }

    //------------//
    // Preprocess //
    //------------//

    #[test]
    fn preprocess_floor_f32() {
        let mut ds: Dataset = matrix(&[1.7f32, 2.2, 3.9, 4.0], 1, 4).into();
        ds.preprocess(&Preprocess::Floor);
        let slice: &[f32] = ds.as_slice().try_cast().unwrap();
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn preprocess_floor_integer_is_noop() {
        let mut ds: Dataset = matrix(&[3u8, 7, 9, 11], 1, 4).into();
        ds.preprocess(&Preprocess::Floor);
        let slice: &[u8] = ds.as_slice().try_cast().unwrap();
        assert_eq!(slice, &[3, 7, 9, 11]);
    }

    #[test]
    fn preprocess_halve_integer() {
        let mut ds: Dataset = matrix(&[4i8, 7, 8, 10], 1, 4).into();
        ds.preprocess(&Preprocess::Halve);
        let slice: &[i8] = ds.as_slice().try_cast().unwrap();
        assert_eq!(slice, &[2, 3, 4, 5]);
    }

    //-------------//
    // DatasetView //
    //-------------//

    #[test]
    fn dataset_view_accessors() {
        let ds: Dataset = matrix(&[1u8, 2, 3, 4, 5, 6], 2, 3).into();
        let view = ds.as_view();
        assert_eq!(view.data_type(), DataType::U8);
        assert_eq!(view.nrows(), 2);
        assert_eq!(view.ncols(), 3);
    }

    #[test]
    fn dataset_view_row() {
        let ds: Dataset = matrix(&[1u8, 2, 3, 4, 5, 6], 2, 3).into();
        let view = ds.as_view();

        let row1: &[u8] = view.row(1).unwrap().try_cast().unwrap();
        assert_eq!(row1, &[4, 5, 6]);

        assert!(view.row(2).is_none());
    }

    #[test]
    fn dataset_view_medoid() {
        let ds: Dataset = matrix(&[1i8, 2, 3, 4], 2, 2).into();
        let medoid = ds.as_view().medoid();
        assert_eq!(medoid.nrows(), 1);
        assert_eq!(medoid.ncols(), 2);
    }
}
