/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::{io::read_bin, views::rowmajor::{self, Matrix, MatrixMut}};
use half::f16;

use super::datatype::{DataType, Dataset, Preprocess, SliceMut};

pub(crate) fn load_and_convert<IO>(
    io: &mut IO,
    src: DataType,
    target: DataType,
    ops: &[Preprocess],
) -> anyhow::Result<Dataset>
where
    IO: std::io::Read + std::io::Seek,
{
    let mut data = match src {
        DataType::F32 => Dataset::from(read_bin::<f32>(io)?),
        DataType::F16 => Dataset::from(read_bin::<f16>(io)?),
        DataType::U8 => Dataset::from(read_bin::<u8>(io)?),
        DataType::I8 => Dataset::from(read_bin::<i8>(io)?),
    };

    for op in ops {
        data.preprocess(op);
    }

    if src == target {
        return Ok(data);
    }

    let dst = match target {
        DataType::F32 => {
            let mut dst = rowmajor::Owned::<f32>::defaulted(data.nrows(), data.ncols()).unwrap();
            SliceMut::from(dst.as_mut_slice()).convert_lossless(data.as_slice())?;
            Dataset::from(dst)
        }
        DataType::F16 => {
            let mut dst = rowmajor::Owned::<f16>::defaulted(data.nrows(), data.ncols()).unwrap();
            SliceMut::from(dst.as_mut_slice()).convert_lossless(data.as_slice())?;
            Dataset::from(dst)
        }
        DataType::U8 => {
            let mut dst = rowmajor::Owned::<u8>::defaulted(data.nrows(), data.ncols()).unwrap();
            SliceMut::from(dst.as_mut_slice()).convert_lossless(data.as_slice())?;
            Dataset::from(dst)
        }
        DataType::I8 => {
            let mut dst = rowmajor::Owned::<i8>::defaulted(data.nrows(), data.ncols()).unwrap();
            SliceMut::from(dst.as_mut_slice()).convert_lossless(data.as_slice())?;
            Dataset::from(dst)
        }
    };

    Ok(dst)
}
