/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::{io::read_bin, views::Matrix};
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
            let mut dst = Matrix::new(0.0f32, data.nrows(), data.ncols());
            SliceMut::from(dst.as_mut_slice()).convert_lossless(data.as_slice())?;
            Dataset::from(dst)
        }
        DataType::F16 => {
            let mut dst = Matrix::new(f16::from_f32(0.0f32), data.nrows(), data.ncols());
            SliceMut::from(dst.as_mut_slice()).convert_lossless(data.as_slice())?;
            Dataset::from(dst)
        }
        DataType::U8 => {
            let mut dst = Matrix::new(0u8, data.nrows(), data.ncols());
            SliceMut::from(dst.as_mut_slice()).convert_lossless(data.as_slice())?;
            Dataset::from(dst)
        }
        DataType::I8 => {
            let mut dst = Matrix::new(0i8, data.nrows(), data.ncols());
            SliceMut::from(dst.as_mut_slice()).convert_lossless(data.as_slice())?;
            Dataset::from(dst)
        }
    };

    Ok(dst)
}
